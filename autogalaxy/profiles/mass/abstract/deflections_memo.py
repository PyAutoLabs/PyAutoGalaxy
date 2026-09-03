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
exactly (a non-scalar constructor argument, a JAX tracer, a grid without a numpy array)
falls through to the ordinary call. The memo engages only when ``xp is np``; the JAX path
is untouched.

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
import weakref

import numpy as np
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from autoarray.structures.decorators.to_vector_yx import VectorYXMaker


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

_stats = {"hits": 0, "misses": 0, "evictions": 0, "bytes": 0}

# id(grid) -> (weakref to the grid, its fingerprint). One likelihood evaluation calls
# this module once per mass profile with the *same* grid object, so hashing the grid
# bytes per profile would dominate the hit path (~0.2 ms x 30 Gaussians). The weakref
# makes a recycled id a miss rather than a wrong answer. It does assume the grid's
# coordinate array is not mutated in place after it is first fingerprinted, which no
# library path does -- grids are rebuilt, never edited.
_grid_fingerprint_cache: Dict[int, Tuple[Any, str]] = {}

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
    The memo's counters: ``hits``, ``misses``, ``evictions``, ``bytes`` stored and the
    number of ``entries``.
    """
    return {
        "hits": _stats["hits"],
        "misses": _stats["misses"],
        "evictions": _stats["evictions"],
        "bytes": _stats["bytes"],
        "entries": len(_deflections_memo),
    }


def memo_clear() -> None:
    """
    Empty the memo and reset its counters.
    """
    _deflections_memo.clear()
    _grid_fingerprint_cache.clear()
    _stats.update({"hits": 0, "misses": 0, "evictions": 0, "bytes": 0})


def _scalar_token(value) -> Optional[str]:
    """
    An exact, hashable token for a constructor-argument value, or None if the value is
    not a scalar (or a tuple / list of scalars) and the profile is therefore not
    memoisable.

    ``repr`` of a Python float round-trips exactly, so the token distinguishes any two
    values the deflection calculation would distinguish. A JAX tracer, an ndarray or a
    nested object all return None, which is what keeps the memo off those profiles.
    """
    if value is None or isinstance(value, (bool, str)):
        return repr(value)

    if isinstance(value, (int, float)):
        return f"{type(value).__name__}:{value!r}"

    if isinstance(value, np.generic):
        return f"np:{value.dtype.str}:{value.item()!r}"

    if isinstance(value, (tuple, list)):
        tokens = [_scalar_token(item) for item in value]
        if any(token is None for token in tokens):
            return None
        return f"{type(value).__name__}({','.join(tokens)})"

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


def _profile_token(profile, skip: Tuple[str, ...] = ()) -> Optional[str]:
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
        token = _scalar_token(attributes[name])
        if token is None:
            return None
        tokens.append(f"{name}={token}")

    return "|".join(tokens)


def _grid_fingerprint(grid) -> Optional[str]:
    """
    A content fingerprint of ``grid``: the sha256 of its coordinate bytes, its type name,
    shape and dtype, and its ``pixel_scales`` / ``origin`` where the type has them.

    Returns None if the grid carries no numpy coordinate array (a JAX-backed grid, say),
    which makes the profile unmemoisable rather than wrongly memoised.
    """
    values = getattr(grid, "array", grid)

    if not isinstance(values, np.ndarray):
        return None

    cached = _grid_fingerprint_cache.get(id(grid))

    if cached is not None and cached[0]() is grid:
        return cached[1]

    values = np.ascontiguousarray(values)

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

    try:
        reference = weakref.ref(grid)
    except TypeError:
        return fingerprint

    if len(_grid_fingerprint_cache) >= _GRID_FINGERPRINT_CACHE_MAX_ENTRIES:
        dead = [
            key for key, entry in _grid_fingerprint_cache.items() if entry[0]() is None
        ]
        for key in dead or [next(iter(_grid_fingerprint_cache))]:
            _grid_fingerprint_cache.pop(key, None)

    _grid_fingerprint_cache[id(grid)] = (reference, fingerprint)

    return fingerprint


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


def _takes_ratio_split(profile) -> bool:
    """
    Whether ``profile`` takes the L2 (unit-ratio) split: a ``Gaussian`` mass profile
    whose deflections are exactly linear in ``mass_to_light_ratio``.

    ``GaussianGradient`` subclasses ``Gaussian`` but derives its ratio from three
    parameters, so it is excluded and takes L1.
    """
    gaussian, gaussian_gradient = _gaussian_class_pair()

    if not isinstance(profile, gaussian) or isinstance(profile, gaussian_gradient):
        return False

    return isinstance(getattr(profile, "mass_to_light_ratio", None), (int, float))


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

    Parameters
    ----------
    profile
        The mass profile whose deflection angles are computed.
    grid
        The 2D (y,x) coordinates the deflection angles are computed on.
    xp
        The array backend. The memo engages only for numpy; JAX falls straight through.
    """

    def direct():
        return profile.deflections_yx_2d_from(grid=grid, xp=xp)

    if xp is not np or not memo_enabled():
        return direct()

    ratio_split = _takes_ratio_split(profile)

    profile_token = _profile_token(
        profile, skip=("mass_to_light_ratio",) if ratio_split else ()
    )

    if profile_token is None:
        return direct()

    grid_fingerprint = _grid_fingerprint(grid)

    if grid_fingerprint is None:
        return direct()

    tier = "L2" if ratio_split else "L1"

    key = _memo_key(tier, profile_token, grid_fingerprint)

    entry = _deflections_memo.get(key)

    if entry is not None:
        _stats["hits"] += 1
        if ratio_split:
            return _wrapped_result(
                profile.mass_to_light_ratio * entry.field, grid, xp, entry.wrapped
            )
        return _wrapped_result(np.array(entry.field), grid, xp, entry.wrapped)

    _stats["misses"] += 1

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
        return _wrapped_result(profile.mass_to_light_ratio * field, grid, xp, wrapped)

    return result
