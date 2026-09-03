"""
Cross-evaluation memo for mass-profile deflection angles on the numpy path.

In the SLaM ``mass_light_dark`` stage every Gaussian of an MGE lens light is fixed --
centre, ``ell_comps``, ``sigma`` and ``intensity`` come from the light-stage instance --
and the whole stack shares one free ``mass_to_light_ratio``. A sampler nevertheless
builds fresh profile objects for every likelihood evaluation, so the identical Faddeeva
field is recomputed each call. This module memoises that field.

Two levers, both keyed on parameter *values* (never on model metadata, which never
reaches profile code):

- **L1** -- any mass profile whose constructor arguments are all scalars: the final
  (y,x) deflection field is stored for that (profile, grid) pair and returned on a hit.
- **L2** -- ``mp.Gaussian`` and the ``lmp`` / ``lmp_linear`` Gaussians that inherit its
  deflections: the key excludes ``mass_to_light_ratio`` and the stored field is the
  **unit-ratio** field, evaluated through the normal path on a copy of the profile whose
  ``mass_to_light_ratio`` is ``1.0``; a hit (and the miss that filled it) returns
  ``mass_to_light_ratio x field``. The deflection is exactly linear in that scalar, so
  the only difference from a direct call is the order of one multiplication -- a
  relative difference at the ulp level, orders of magnitude inside the 1e-12 tolerance
  the unit tests hold it to. ``GaussianGradient`` is *not* linear in a single scalar and
  takes L1 only.

Grids are keyed by **content**, not identity: ``FitDataset.grids`` rebuilds the grid on
every likelihood call, so ``id(grid)`` would miss every time. A shifted, rotated or
re-masked grid changes the bytes and therefore misses, which is correct by construction.

Failure modes are **misses, never stale hits**: anything that cannot be fingerprinted
exactly (a non-scalar constructor argument, a JAX tracer, a grid that is neither numpy
nor a concrete JAX array) falls through to the ordinary call.

__The JAX path__

On JAX the memo is a **trace-time constant fold** rather than a cross-call cache. Under
the JAX likelihood a fixed Gaussian's geometry reaches the profile as Python floats and
the grid as a *concrete* ``jax.Array`` (``Grid2D.subtracted_and_rotated_from`` evaluates
a constant shift-and-rotate eagerly for exactly this reason), while the free
``mass_to_light_ratio`` is a tracer. The Faddeeva subgraph is therefore a constant, but
``jax.jit`` stages every ``jax.numpy`` call whatever its operands, and this stack
disables XLA's constant folding (``--xla_disable_hlo_passes=constant_folding``, set by
``autonerves/jax_wrapper.py``), so nothing else folds it away either.

So the memo folds it explicitly: on a miss it evaluates the unit-ratio field **with
numpy and scipy** on a numpy twin of the grid, stores it in the same dict the numpy path
uses (both backends share entries -- the same bytes), and returns
``mass_to_light_ratio * xp.asarray(field)``. The Faddeeva block leaves the jaxpr, which
is replaced by one constant and one multiply, and the JAX path inherits ``scipy.wofz``
accuracy for fixed geometry rather than the Weideman-32 series.

The cost is one numpy evaluation per (profile geometry, grid) **per trace**, paid at
compile time. Recompilation is not made more likely by this: the embedded constant
changes only when the geometry or the grid changes, and either of those is a different
fit. Anything traced -- a free geometry parameter, or a free ``grid_offset`` that makes
the grid itself a tracer -- fails the concreteness test and falls through to the ordinary
JAX call, so no branch is ever taken on a traced value.

Disable with ``AUTOGALAXY_DEFLECTIONS_MEMO=0``, or in-process with the
``memo_disabled()`` context manager (which a profiling harness needs, since it must be
able to guarantee it restored the previous state). The byte cap defaults to 256 MB and is
overridden with ``AUTOGALAXY_DEFLECTIONS_MEMO_MAX_MB`` (over-sampled grids are large, so
the cap is on bytes, not on the number of entries; eviction is FIFO).
"""

import contextlib
import copy
import hashlib
import inspect
import os
import sys
import weakref

import numpy as np
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from autoarray.structures.decorators.to_vector_yx import VectorYXMaker
from autoarray.structures.grids.irregular_2d import Grid2DIrregular
from autoarray.structures.grids.uniform_2d import Grid2D

from autogalaxy.profiles.mass.abstract.mge import (
    _is_static_scalar as _is_concrete_scalar,
)


class MemoEntry(NamedTuple):
    """
    One memoised deflection field.

    Parameters
    ----------
    field
        The stored (y,x) deflection angles, read-only. For an L2 entry this is the
        unit-``mass_to_light_ratio`` field; for an L1 entry it is the field itself.
    wrapped
        Whether the direct call returned a structure (``VectorYX2D`` /
        ``VectorYX2DIrregular``) rather than a bare ndarray, so a hit reproduces the
        same type.
    """

    field: np.ndarray
    wrapped: bool


_deflections_memo: Dict[str, MemoEntry] = {}

_DEFAULT_MAX_MB = 256.0

_stats = {"hits": 0, "misses": 0, "evictions": 0, "bytes": 0, "jax_folds": 0}

# id(grid) -> (weakref to the grid, its fingerprint, its numpy twin or None). One
# likelihood evaluation calls
# this module once per mass profile with the *same* grid object, so hashing the grid
# bytes per profile would dominate the hit path (~0.2 ms x 30 Gaussians). The weakref
# makes a recycled id a miss rather than a wrong answer. It does assume the grid's
# coordinate array is not mutated in place after it is first fingerprinted, which no
# library path does -- grids are rebuilt, never edited.
_grid_fingerprint_cache: Dict[int, Tuple[Any, str, Any]] = {}

_GRID_FINGERPRINT_CACHE_MAX_ENTRIES = 8

# type -> the names of its constructor arguments, or None if it has none that can be
# read back (``*args`` / ``**kwargs``, or an unreadable signature).
_init_argument_names: Dict[type, Optional[Tuple[str, ...]]] = {}

_gaussian_classes: Optional[Tuple[type, type]] = None

# Set by `memo_disabled()`, which overrides the env switch for the duration of a block.
# A caller that must measure the *uncached* per-call cost of a deflection cannot use the
# env var: this module reads it at call time, but a profiling harness needs a scope it
# can guarantee it restored.
_suspended = False


@contextlib.contextmanager
def memo_disabled():
    """
    Suspend the memo for the duration of the block, whatever the env switch says.

    Restores the previous state on exit, including on an exception. Nesting is safe.
    """
    global _suspended

    previous = _suspended
    _suspended = True
    try:
        yield
    finally:
        _suspended = previous


def memo_enabled() -> bool:
    """
    Whether the deflection memo is active in this process.
    """
    if _suspended:
        return False

    return os.environ.get("AUTOGALAXY_DEFLECTIONS_MEMO", "1") != "0"


def memo_max_bytes() -> int:
    """
    The memo's byte cap, from ``AUTOGALAXY_DEFLECTIONS_MEMO_MAX_MB`` (default 256 MB).

    An unparseable value falls back to the default rather than raising: a bad env var
    must not break a fit.
    """
    try:
        max_mb = float(
            os.environ.get("AUTOGALAXY_DEFLECTIONS_MEMO_MAX_MB", _DEFAULT_MAX_MB)
        )
    except (TypeError, ValueError):
        max_mb = _DEFAULT_MAX_MB

    return int(max(max_mb, 0.0) * 1024 * 1024)


def memo_stats() -> Dict[str, int]:
    """
    The memo's counters: ``hits``, ``misses``, ``evictions``, ``bytes`` stored, the
    number of ``entries``, and ``jax_folds``.

    ``jax_folds`` counts the trace-time numpy evaluations performed for a JAX caller --
    one per (profile geometry, grid) per trace. It is the witness that the fold actually
    happened: a JAX fit of an MGE with 30 fixed Gaussians reports 30 folds on its
    compiling call and none afterwards, and 0 whenever the memo could not key the call.
    """
    return {
        "hits": _stats["hits"],
        "misses": _stats["misses"],
        "evictions": _stats["evictions"],
        "bytes": _stats["bytes"],
        "entries": len(_deflections_memo),
        "jax_folds": _stats["jax_folds"],
    }


def memo_clear() -> None:
    """
    Empty the memo and reset its counters.
    """
    _deflections_memo.clear()
    _grid_fingerprint_cache.clear()
    _stats.update({"hits": 0, "misses": 0, "evictions": 0, "bytes": 0, "jax_folds": 0})


def _is_concrete_array(array, xp=np) -> bool:
    """
    Whether ``array`` is an array whose contents can be read *now*: a numpy array
    always, and on the JAX backend a ``jax.Array`` that is not a tracer.

    The test is positive on both branches, the house style everywhere a traced value
    has to be told apart from a concrete one (``mge._is_static_scalar``,
    ``autoarray.validate.is_concrete_scalar``). It is never
    ``try: np.asarray(...) except``: that reads a tracer's *shape* as success on some
    JAX versions and would turn a traced grid into a silently wrong memo entry.

    ``jax`` is looked up in ``sys.modules`` rather than imported. Nothing can be a
    ``jax.Array`` in a process that has not imported jax, so the lookup is exact, and a
    numpy-only fit never pays an import for a question whose answer is already no.
    """
    if isinstance(array, np.ndarray):
        return True

    if xp is np:
        return False

    if not type(array).__module__.startswith(("jax", "jaxlib")):
        return False

    jax = sys.modules.get("jax")

    if jax is None:
        return False

    return isinstance(array, jax.Array) and not isinstance(array, jax.core.Tracer)


def _concrete_scalar_value(value, xp=np):
    """
    The Python value of a concrete 0-d array, or ``None`` if ``value`` is not one.

    A parameter that a JAX fit holds fixed usually arrives as a Python float, but a
    pytree-native instance can hand a profile a concrete 0-d ``jax.Array`` instead. That
    is as exact a key as the float is, so it is read back with ``float()`` rather than
    making the profile unmemoisable.
    """
    if not _is_concrete_array(value, xp=xp) or isinstance(value, np.ndarray):
        return None

    as_numpy = np.asarray(value)

    if as_numpy.ndim != 0:
        return None

    return float(as_numpy)


def _scalar_token(value, xp=np) -> Optional[str]:
    """
    An exact, hashable token for a constructor-argument value, or None if the value is
    not a scalar (or a tuple / list of scalars) and the profile is therefore not
    memoisable.

    ``repr`` of a Python float round-trips exactly, so the token distinguishes any two
    values the deflection calculation would distinguish. A JAX tracer, an ndarray or a
    nested object all return None, which is what keeps the memo off those profiles.

    A concrete 0-d ``jax.Array`` tokenises as the Python float it holds, so a JAX caller
    and a numpy caller of the same fixed geometry land on the *same* key and share one
    stored field.
    """
    if value is None or isinstance(value, (bool, str)):
        return repr(value)

    if isinstance(value, (int, float)):
        if isinstance(value, float):
            return f"float:{float(value)!r}"
        return f"int:{int(value)!r}"

    if isinstance(value, np.generic):
        return f"np:{value.dtype.str}:{value.item()!r}"

    if isinstance(value, (tuple, list)):
        tokens = [_scalar_token(item, xp=xp) for item in value]
        if any(token is None for token in tokens):
            return None
        return f"{type(value).__name__}({','.join(tokens)})"

    concrete = _concrete_scalar_value(value, xp=xp)

    if concrete is not None:
        return f"float:{concrete!r}"

    return None


def _init_argument_names_from(cls: type) -> Optional[Tuple[str, ...]]:
    """
    The names of ``cls.__init__``'s arguments, cached per class.

    Returns None -- meaning "not memoisable" -- for a constructor taking ``*args`` or
    ``**kwargs``, since an argument that is not a named parameter cannot be read back
    off the instance and a value that is not in the key can silently change the answer.
    """
    if cls in _init_argument_names:
        return _init_argument_names[cls]

    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        _init_argument_names[cls] = None
        return None

    names: List[str] = []

    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            _init_argument_names[cls] = None
            return None
        names.append(name)

    _init_argument_names[cls] = tuple(names)
    return _init_argument_names[cls]


def _profile_token(profile, skip: Tuple[str, ...] = (), xp=np) -> Optional[str]:
    """
    An exact token for a profile's identity: its class plus the values of its
    constructor arguments, read off the instance and excluding the names in ``skip``.

    Returns None if any argument is missing from the instance or is not a scalar, in
    which case the profile is not memoisable. Only constructor arguments are read, so a
    cached array a profile builds on its first call cannot turn a memoisable profile
    into an unmemoisable one halfway through a run.
    """
    cls = type(profile)

    names = _init_argument_names_from(cls)

    if names is None:
        return None

    attributes = vars(profile)

    tokens = [f"{cls.__module__}.{cls.__qualname__}"]

    for name in names:
        if name in skip:
            continue
        if name not in attributes:
            return None
        token = _scalar_token(attributes[name], xp=xp)
        if token is None:
            return None
        tokens.append(f"{name}={token}")

    return "|".join(tokens)


def _numpy_twin_from(grid, values: np.ndarray):
    """
    A numpy grid carrying ``values``, of the same class as ``grid``, or None if that
    class is not one the memo knows how to rebuild.

    The twin is what a JAX-backed call is evaluated on: the deflection body reads only
    the coordinates, so the twin needs nothing else to be faithful. It is built once per
    grid fingerprint and cached beside it. ``Grid2D`` is rebuilt with
    ``over_sample_size=1`` deliberately -- the twin is never over-sampled itself (its
    caller was already handed the over-sampled coordinates when that is what is being
    evaluated), so building a second over-sampled grid inside it would be pure cost.

    An unrecognised grid class returns None, which falls the call through to the direct
    JAX evaluation rather than guessing at a constructor.
    """
    if isinstance(grid, Grid2D):
        return type(grid)(values=values, mask=grid.mask, over_sample_size=1)

    if isinstance(grid, Grid2DIrregular):
        return type(grid)(values=values)

    return None


def _grid_fingerprint_and_twin(grid, xp=np) -> Tuple[Optional[str], Optional[Any]]:
    """
    A content fingerprint of ``grid`` -- the sha256 of its coordinate bytes, its type
    name, shape and dtype, and its ``pixel_scales`` / ``origin`` where the type has them
    -- together with the numpy twin a JAX-backed grid is evaluated on.

    The twin is None for a numpy grid, which is its own twin. A JAX-backed grid is
    accepted only when its coordinates are **concrete**: the device-to-host copy that
    hashing them costs is paid once per grid object (the weakref cache below), at trace
    time. A traced grid, or any other object with no readable coordinates, returns
    ``(None, None)`` -- the profile is then not memoisable, rather than wrongly memoised.
    """
    values = getattr(grid, "array", grid)

    if not _is_concrete_array(values, xp=xp):
        return None, None

    cached = _grid_fingerprint_cache.get(id(grid))

    if cached is not None and cached[0]() is grid:
        return cached[1], cached[2]

    values = np.ascontiguousarray(np.asarray(values))

    digest = hashlib.sha256()
    digest.update(type(grid).__name__.encode())
    digest.update(f"{values.shape}:{values.dtype.str}".encode())

    for name in ("pixel_scales", "origin"):
        try:
            digest.update(f"{name}={getattr(grid, name, None)!r}".encode())
        except Exception:
            digest.update(f"{name}=<unavailable>".encode())

    digest.update(values.tobytes())

    fingerprint = digest.hexdigest()

    # A numpy grid is its own twin; holding a strong reference to it here would defeat
    # the weakref this cache is keyed by, so only a rebuilt one is stored.
    twin = None

    if not isinstance(getattr(grid, "array", grid), np.ndarray):
        twin = _numpy_twin_from(grid, values)

        if twin is None:
            return None, None

    try:
        reference = weakref.ref(grid)
    except TypeError:
        return fingerprint, twin

    if len(_grid_fingerprint_cache) >= _GRID_FINGERPRINT_CACHE_MAX_ENTRIES:
        dead = [
            key for key, entry in _grid_fingerprint_cache.items() if entry[0]() is None
        ]
        for key in dead or [next(iter(_grid_fingerprint_cache))]:
            _grid_fingerprint_cache.pop(key, None)

    _grid_fingerprint_cache[id(grid)] = (reference, fingerprint, twin)

    return fingerprint, twin


def _grid_fingerprint(grid, xp=np) -> Optional[str]:
    """
    The content fingerprint of ``grid`` alone (see
    :func:`_grid_fingerprint_and_twin`).
    """
    return _grid_fingerprint_and_twin(grid, xp=xp)[0]


def _memo_key(tier: str, profile_token: str, grid_fingerprint: str) -> str:
    return hashlib.sha256(
        f"{tier}|{profile_token}|{grid_fingerprint}".encode()
    ).hexdigest()


def _gaussian_class_pair() -> Tuple[type, type]:
    """
    ``(Gaussian, GaussianGradient)``, imported lazily so this module stays importable
    from anywhere in the profile package without a circular import.
    """
    global _gaussian_classes

    if _gaussian_classes is None:
        from autogalaxy.profiles.mass.stellar.gaussian import Gaussian
        from autogalaxy.profiles.mass.stellar.gaussian_gradient import GaussianGradient

        _gaussian_classes = (Gaussian, GaussianGradient)

    return _gaussian_classes


def _takes_ratio_split(profile, xp=np) -> bool:
    """
    Whether ``profile`` takes the L2 (unit-ratio) split: a ``Gaussian`` mass profile
    whose deflections are exactly linear in ``mass_to_light_ratio``.

    ``GaussianGradient`` subclasses ``Gaussian`` but derives its ratio from three
    parameters, so it is excluded and takes L1.

    On JAX the ratio is accepted when it is a **tracer or a 0-d array** as well as when
    it is a concrete scalar: the split exists precisely so a free ratio can multiply a
    stored field, and the memo never branches on the ratio's value -- it only multiplies
    by it. Anything else (an array of per-pixel ratios, say) is not a scale factor and
    is refused.
    """
    gaussian, gaussian_gradient = _gaussian_class_pair()

    if not isinstance(profile, gaussian) or isinstance(profile, gaussian_gradient):
        return False

    ratio = getattr(profile, "mass_to_light_ratio", None)

    if _is_concrete_scalar(ratio):
        return True

    if xp is np:
        return False

    return type(ratio).__module__.startswith(("jax", "jaxlib")) and (
        getattr(ratio, "ndim", None) == 0
    )


def _numpy_profile_from(profile, names: Tuple[str, ...], ratio_split: bool, xp=np):
    """
    A shallow copy of ``profile`` whose every constructor argument is a plain Python
    value, ready to be evaluated with numpy, or None if any of them is not.

    This is the object the trace-time fold evaluates. Its whole purpose is that it
    carries **no tracer anywhere**: a 0-d concrete ``jax.Array`` parameter is read back
    as a float, and the final loop re-checks every argument rather than trusting that
    the token pass covered it -- a tracer smuggled onto the copy would be evaluated by
    numpy into an array of the wrong thing, silently.

    For an L2 profile the copy's ``mass_to_light_ratio`` is 1.0, so what is computed and
    stored is the unit-ratio field the caller then scales by the real (traced) ratio.
    """
    profile_np = copy.copy(profile)

    if ratio_split:
        profile_np.mass_to_light_ratio = 1.0

    attributes = vars(profile_np)

    for name in names:
        if name not in attributes:
            return None

        value = attributes[name]
        concrete = _concrete_scalar_value(value, xp=xp)

        if concrete is not None:
            setattr(profile_np, name, concrete)
        elif isinstance(value, (tuple, list)):
            items = [_concrete_scalar_value(item, xp=xp) for item in value]
            if any(item is not None for item in items):
                setattr(
                    profile_np,
                    name,
                    type(value)(
                        original if item is None else item
                        for original, item in zip(value, items)
                    ),
                )

    for name in names:
        if _scalar_token(vars(profile_np)[name], xp=np) is None:
            return None

    return profile_np


def _unwrap(result) -> Tuple[Optional[np.ndarray], bool]:
    """
    The numpy array inside a deflection result, and whether it arrived inside a
    structure (``VectorYX2D`` / ``VectorYX2DIrregular``) rather than bare.

    Returns ``(None, False)`` for anything else, which is not stored.
    """
    array = getattr(result, "array", None)

    if isinstance(array, np.ndarray):
        return array, True

    if isinstance(result, np.ndarray):
        return result, False

    return None, False


def _store(key: str, field: np.ndarray, wrapped: bool) -> None:
    """
    Store a read-only copy of ``field``, evicting the oldest entries (FIFO) until the
    memo fits its byte cap. A field larger than the whole cap is not stored at all.
    """
    stored = np.array(field)
    stored.setflags(write=False)

    max_bytes = memo_max_bytes()

    if stored.nbytes > max_bytes:
        return

    while _deflections_memo and _stats["bytes"] + stored.nbytes > max_bytes:
        evicted = _deflections_memo.pop(next(iter(_deflections_memo)))
        _stats["bytes"] -= evicted.field.nbytes
        _stats["evictions"] += 1

    _deflections_memo[key] = MemoEntry(field=stored, wrapped=wrapped)
    _stats["bytes"] += stored.nbytes


def _wrapped_result(field: np.ndarray, grid, xp, wrapped: bool):
    """
    ``field`` returned in the shape a direct call would have returned it: re-wrapped for
    ``grid`` by the same maker the ``to_vector_yx`` decorator uses, or bare if the direct
    call was bare.
    """
    if not wrapped:
        return field

    return VectorYXMaker(
        func=lambda _obj, _grid, _xp, **kwargs: field, obj=None, grid=grid, xp=xp
    ).result


def deflections_yx_2d_from(profile, grid, xp=np):
    """
    The deflection angles of ``profile`` on ``grid``, from the memo where the profile's
    parameter values and the grid's contents are both unchanged from a previous call,
    and from ``profile.deflections_yx_2d_from`` otherwise.

    This is the single intercept the summation call sites (``Galaxy`` and ``Basis``) go
    through, so no mass profile needs an override of its own.

    On numpy this is a cross-evaluation cache; on JAX it is a trace-time constant fold,
    evaluated with numpy on a twin of the (concrete) grid and returned as a constant the
    traced ratio multiplies. Both write the same bytes into the same dict. A tracer
    among the key values, or a traced grid, takes neither path: the call falls straight
    through to ``profile.deflections_yx_2d_from``, so nothing here ever branches on a
    traced value.

    Parameters
    ----------
    profile
        The mass profile whose deflection angles are computed.
    grid
        The 2D (y,x) coordinates the deflection angles are computed on.
    xp
        The array backend, numpy or ``jax.numpy``.
    """

    def direct():
        return profile.deflections_yx_2d_from(grid=grid, xp=xp)

    if not memo_enabled():
        return direct()

    ratio_split = _takes_ratio_split(profile, xp=xp)

    skip = ("mass_to_light_ratio",) if ratio_split else ()

    profile_token = _profile_token(profile, skip=skip, xp=xp)

    if profile_token is None:
        return direct()

    grid_fingerprint, twin = _grid_fingerprint_and_twin(grid, xp=xp)

    if grid_fingerprint is None:
        return direct()

    tier = "L2" if ratio_split else "L1"

    key = _memo_key(tier, profile_token, grid_fingerprint)

    entry = _deflections_memo.get(key)

    if entry is not None:
        _stats["hits"] += 1
        field = entry.field if xp is np else xp.asarray(entry.field)
        if ratio_split:
            return _wrapped_result(
                profile.mass_to_light_ratio * field, grid, xp, entry.wrapped
            )
        return _wrapped_result(
            np.array(entry.field) if xp is np else field, grid, xp, entry.wrapped
        )

    _stats["misses"] += 1

    if xp is np:
        if ratio_split:
            unit_profile = copy.copy(profile)
            unit_profile.mass_to_light_ratio = 1.0
            result = unit_profile.deflections_yx_2d_from(grid=grid, xp=xp)
        else:
            result = direct()

        field, wrapped = _unwrap(result)

        if field is None:
            return result

        _store(key, field, wrapped)

        if ratio_split:
            return _wrapped_result(
                profile.mass_to_light_ratio * field, grid, xp, wrapped
            )

        return result

    # The JAX fold: evaluate the field once, now, with numpy and scipy on the numpy twin
    # of the grid, and hand the trace a constant.
    names = _init_argument_names_from(type(profile))

    profile_np = _numpy_profile_from(
        profile, names=names, ratio_split=ratio_split, xp=xp
    )

    if profile_np is None:
        return direct()

    result = profile_np.deflections_yx_2d_from(grid=twin, xp=np)

    field, wrapped = _unwrap(result)

    if field is None:
        return direct()

    _store(key, field, wrapped)
    _stats["jax_folds"] += 1

    field = xp.asarray(field)

    if ratio_split:
        return _wrapped_result(profile.mass_to_light_ratio * field, grid, xp, wrapped)

    return _wrapped_result(field, grid, xp, wrapped)
