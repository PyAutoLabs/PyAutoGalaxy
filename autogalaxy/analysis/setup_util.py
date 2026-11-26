"""
Helpers that wire the compute `aa.Setup` of a likelihood into the dataset analyses.

A dataset analysis (`AnalysisImaging`, `AnalysisInterferometer`) accepts
`setup=None | "auto" | aa.Setup | "<setup_id>"`. The setup is resolved in
`AnalysisDataset.modify_before_fit`, because the rule key of `aa.Setup.auto` needs the
model: a pixelized source (and its mesh) and a parametric / MGE fit have different
designated setups. This module holds the pieces that resolution needs:

- `SETUP_DEFAULT` / `setup_and_use_jax_from`: the default `setup` of an analysis and the precedence
  between `setup` and `use_jax` (an explicit `use_jax` with no `setup` opts out of `"auto"`).
- `setup_input_from`: normalise the `setup` input of an analysis.
- `setup_key_from`: map a dataset and a model to the `(dataset_type, variant)` rule key of
  `aa.Setup.auto`.
- `use_jax_from`: resolve a requested backend the way `af.Analysis.__init__` resolves
  `use_jax` (honouring `PYAUTO_DISABLE_JAX` and a missing JAX install).
- `check_setup_consistency`: run the inversion factory's operator / setup / backend check
  once, outside the likelihood, so a mismatch raises before sampling.
- `setup_header_from` / `setup_banner_from` / `setup_dict_from`: the header and `Setup:`
  lines logged at fit start, and the contents of `files/setup.json`.
- `setup_text_from`: the readable setup (`numba · CPU · sparse operator · fp64`) those
  lines, the "not set" line and the aggregator notice share.
- `setup_environment`: the detected hardware `aa.Setup.auto` chooses for.

The resolution itself is `autogalaxy.analysis.analysis.setup.AnalysisSetupMixin`, shared by every
analysis that takes a `setup`.

The setup decides only how the likelihood is computed (backend, device, operator,
precision, transformer, threads), never the model.
"""

import importlib.util
import logging
import os
import warnings
from typing import Optional, Tuple, Union

import autoarray as aa

from autoarray.setup.setup import (
    EXPLICIT_RULE_ID,
    GPU_A100_NAME,
    HARDWARE_CLASS_CPU,
    HARDWARE_CLASS_GPU,
    MANY_CORE_MIN_CORES,
    Setup,
    rules,
    setup_from,
)

logger = logging.getLogger(__name__)

SETUP_AUTO = "auto"


class _SetupDefault(str):
    """
    The default `setup` of the analyses: a `str` equal to `"auto"` (so it reads as `setup="auto"` in
    signatures and help), told apart from an explicit `setup="auto"` by identity (`is SETUP_DEFAULT`).
    """

    def __reduce__(self):
        return (_setup_default, ())


def _setup_default():
    return SETUP_DEFAULT


SETUP_DEFAULT = _SetupDefault(SETUP_AUTO)


def setup_and_use_jax_from(
    setup, use_jax: Optional[bool], default_use_jax: bool = True
) -> Tuple[Union[None, str, Setup], bool, bool]:
    """
    The `setup` and `use_jax` an analysis is constructed with, after the precedence between them.

    - `setup` passed (an `aa.Setup`, a setup_id string, `"auto"` or `None`): it is used as given.
      `setup=None` is the manual opt-out: the backend is `use_jax`.
    - `setup` not passed and `use_jax` passed (`True` / `False`): the manual opt-out, as `setup=None`.
    - Neither passed: `setup="auto"`.

    A `use_jax` of `None` becomes `default_use_jax`, the backend the analysis used before `"auto"` was the
    default (`False` for `AnalysisWeak`, `True` for the others). For the default `"auto"` it is `True` only
    when JAX is installed, so a JAX-free install does not warn about a backend nobody asked for; the
    resolved setup sets the backend at fit start.

    Returns
    -------
    `(setup, use_jax, setup_defaulted)`: the setup input (with the default replaced by the plain
    `"auto"`), the constructor backend passed to `af.Analysis.__init__` (a `bool`) and whether the setup is
    the implicit default (an unresolved default setup does not stop a pre-fit likelihood evaluation, see
    `autolens.analysis.analysis.setup.raise_if_setup_unresolved`).
    """
    if setup is not SETUP_DEFAULT:
        return setup, default_use_jax if use_jax is None else bool(use_jax), False

    if use_jax is not None:
        return None, bool(use_jax), False

    if default_use_jax:
        default_use_jax = importlib.util.find_spec("jax") is not None

    return SETUP_AUTO, default_use_jax, True

CHOSEN_BY_AUTO = "auto"
CHOSEN_BY_EXPLICIT = "explicit"
CHOSEN_BY_ENV = "PYAUTO_SETUP"

VARIANT_MGE = "mge"
VARIANT_RECTANGULAR = "rectangular"
VARIANT_DELAUNAY = "delaunay"
VARIANT_DELAUNAY_NN = "delaunay_nn"

# When the rule table has no row for a pixelized variant of a dataset type (e.g.
# `interferometer/delaunay_nn`), the nearest variant with a row is used instead.
_VARIANT_FALLBACKS = {
    VARIANT_DELAUNAY_NN: (VARIANT_DELAUNAY, VARIANT_RECTANGULAR),
    VARIANT_DELAUNAY: (VARIANT_RECTANGULAR,),
    VARIANT_RECTANGULAR: (VARIANT_DELAUNAY,),
    VARIANT_MGE: (),
}

# When a model has several pixelizations, the most expensive mesh decides the variant.
_VARIANT_PRIORITY = (VARIANT_DELAUNAY_NN, VARIANT_DELAUNAY, VARIANT_RECTANGULAR)


def setup_input_from(setup) -> Union[None, str, Setup]:
    """
    Normalise the `setup` input of a dataset analysis.

    Returns `None`, the sentinel string `"auto"` (any case), or an `aa.Setup` (an `aa.Setup`
    unchanged, or the setup a setup_id string names, via `setup_from`).

    Raises
    ------
    ValueError
        If `setup` is not `None`, `"auto"`, an `aa.Setup` or a valid setup_id string.
    """
    if isinstance(setup, str) and setup.strip().lower() == SETUP_AUTO:
        return SETUP_AUTO
    return setup_from(setup)


def dataset_type_from(dataset) -> str:
    """
    The `dataset_type` of the setup rule key for a dataset: `"imaging"` or
    `"interferometer"`.

    Raises
    ------
    ValueError
        If the dataset is neither an `aa.Imaging` nor an `aa.Interferometer`.
    """
    if isinstance(dataset, aa.Imaging):
        return "imaging"
    if isinstance(dataset, aa.Interferometer):
        return "interferometer"
    raise ValueError(
        f"A setup can only be chosen automatically for an aa.Imaging or "
        f"aa.Interferometer dataset, got {type(dataset).__name__}."
    )


def mesh_variant_from(mesh_cls: type) -> str:
    """
    The setup rule variant of a pixelization mesh class.

    - `DelaunayNN` (Sibson natural-neighbour interpolation) -> `"delaunay_nn"`.
    - `Delaunay` and its other subclasses (e.g. the k-nearest-neighbour meshes) ->
      `"delaunay"`.
    - The rectangular meshes (`RectangularRTUAdaptDensity` and its subclasses) ->
      `"rectangular"`.

    Any other mesh is treated as `"delaunay"`, the general irregular-mesh rule.
    """
    from autoarray.inversion.mesh.mesh.delaunay import Delaunay
    from autoarray.inversion.mesh.mesh.delaunay_nn import DelaunayNN
    from autoarray.inversion.mesh.mesh.rectangular_rtu_adapt_density import (
        RectangularRTUAdaptDensity,
    )

    if issubclass(mesh_cls, DelaunayNN):
        return VARIANT_DELAUNAY_NN
    if issubclass(mesh_cls, Delaunay):
        return VARIANT_DELAUNAY
    if issubclass(mesh_cls, RectangularRTUAdaptDensity):
        return VARIANT_RECTANGULAR
    return VARIANT_DELAUNAY


def mesh_classes_from(model) -> list:
    """
    The pixelization mesh classes in a model, found as `af.Model` components (free or
    fixed) and as instances anywhere in the model tree.
    """
    classes = []

    if hasattr(model, "models_with_type"):
        classes += [
            m.cls
            for m in model.models_with_type(
                aa.AbstractMesh, include_zero_dimension=True
            )
        ]

    if hasattr(model, "attribute_tuples_with_type"):
        classes += [
            type(value)
            for _, value in model.attribute_tuples_with_type(
                aa.AbstractMesh, ignore_children=False
            )
        ]
    elif isinstance(model, aa.AbstractMesh):
        classes.append(type(model))

    return classes


def variant_from_model(model) -> str:
    """
    The setup rule variant of a model: the variant of its pixelization mesh, or `"mge"`
    when it has no pixelization (parametric light profiles, linear light profiles, MGE).

    When the model has several pixelizations the most expensive mesh decides the variant
    (`delaunay_nn`, then `delaunay`, then `rectangular`).
    """
    variants = {mesh_variant_from(cls) for cls in mesh_classes_from(model)}

    for variant in _VARIANT_PRIORITY:
        if variant in variants:
            return variant

    return VARIANT_MGE


def _known_likelihoods(regime: str = "standard") -> set:
    return {row["likelihood"] for row in rules() if row["regime"] == regime}


def setup_key_and_note_from(
    dataset, model, regime: str = "standard"
) -> Tuple[Tuple[str, str], Optional[str]]:
    """
    The `(dataset_type, variant)` rule key of `aa.Setup.auto` for a dataset and model,
    plus a note when the model's variant has no rule for the dataset type and the nearest
    variant with one was used (`None` otherwise).

    Raises
    ------
    ValueError
        If the dataset type is not supported, or neither the variant nor any fallback has
        a rule.
    """
    dataset_type = dataset_type_from(dataset)
    variant = variant_from_model(model)

    known = _known_likelihoods(regime=regime)

    if f"{dataset_type}/{variant}" in known:
        return (dataset_type, variant), None

    for fallback in _VARIANT_FALLBACKS.get(variant, ()):
        if f"{dataset_type}/{fallback}" in known:
            note = (
                f"no rule for {dataset_type}/{variant}, "
                f"using the nearest rule {dataset_type}/{fallback}"
            )
            return (dataset_type, fallback), note

    raise ValueError(
        f"No setup rule for likelihood {dataset_type}/{variant} in regime {regime!r}, "
        f"and no fallback variant has one. Known likelihoods: {sorted(known)}."
    )


def setup_key_from(dataset, model, regime: str = "standard") -> Tuple[str, str]:
    """
    The `(dataset_type, variant)` rule key of `aa.Setup.auto` for a dataset and model.

    `dataset_type` is `"imaging"` for an `aa.Imaging` and `"interferometer"` for an
    `aa.Interferometer`. `variant` comes from the model's pixelization mesh
    (`"rectangular"`, `"delaunay"`, `"delaunay_nn"`), or is `"mge"` when the model has no
    pixelization. A variant with no rule for the dataset type falls back to the nearest one
    that has a rule (see `setup_key_and_note_from`).
    """
    return setup_key_and_note_from(dataset=dataset, model=model, regime=regime)[0]


def setup_environment():
    """
    The detected hardware (`autonerves.environment.Environment`), or `None` if detection fails.
    """
    try:
        from autonerves.environment import Environment

        return Environment.detect()
    except Exception as e:
        logger.debug(f"Setup: hardware detection failed: {e}")
        return None


def use_jax_from(use_jax: bool) -> bool:
    """
    Resolve a requested JAX backend exactly as `af.Analysis.__init__` resolves `use_jax`:
    `False` when `PYAUTO_DISABLE_JAX=1`, and `False` with a warning when JAX is requested
    but not installed.
    """
    if os.environ.get("PYAUTO_DISABLE_JAX") == "1":
        return False

    if use_jax and importlib.util.find_spec("jax") is None:
        warnings.warn(
            "The setup asks for the 'jax' backend but JAX is not installed. Falling back "
            "to numpy: the fit will run, without JAX acceleration.",
            UserWarning,
            stacklevel=2,
        )
        return False

    return bool(use_jax)


def check_setup_consistency(dataset, use_jax: bool):
    """
    Run the inversion factory's operator / setup / backend check on a dataset once, outside
    the likelihood, so a mismatch raises an `InversionException` now instead of every sample
    being rejected mid-search.

    The check is the one `inversion_imaging_from` / `inversion_interferometer_from` run
    before routing; it reads only `dataset.sparse_operator`, `dataset.setup` and the array
    backend, so nothing heavy is built.
    """
    import numpy as np

    from autoarray.inversion.inversion import factory

    if use_jax:
        import jax.numpy as jnp

        xp = jnp
    else:
        xp = np

    if isinstance(dataset, aa.Imaging):
        factory._check_imaging_setup_consistency(
            dataset=dataset, xp=xp, use_sparse_operator=True
        )
    elif isinstance(dataset, aa.Interferometer):
        factory._check_interferometer_setup_consistency(dataset=dataset, xp=xp)


def chosen_by_from(setup: Setup, auto: bool) -> str:
    """
    How a resolved setup was chosen: `"PYAUTO_SETUP"` when it came from the environment
    variable, `"auto"` when `aa.Setup.auto` chose it from the rule table, else
    `"explicit"`.
    """
    if setup.rule_id == EXPLICIT_RULE_ID:
        return CHOSEN_BY_ENV
    return CHOSEN_BY_AUTO if auto else CHOSEN_BY_EXPLICIT


def hardware_description_from(setup: Setup, environment=None) -> Optional[str]:
    """
    A short description of the hardware a setup runs on: the GPU name for a GPU setup,
    the core count for a CPU setup, or `None` if unknown.
    """
    if environment is None:
        return None
    if setup.device == "gpu":
        return getattr(environment, "gpu_name", None) or "GPU"
    cores = getattr(environment, "cores", None)
    return f"{cores} CPU cores" if cores else None


_BACKEND_TEXT = {"numba": "numba", "jax": "JAX", "numpy": "numpy"}
_OPERATOR_TEXT = {"sparse": "sparse operator", "dense": "dense", "na": "no operator"}
_PRECISION_TEXT = {"mp": "mixed precision"}
_TRANSFORMER_TEXT = {"wtilde": "w-tilde", "nufft": "NUFFT", "dft": "DFT"}

# The `Setup:` / `Parallel:` / `Estimate:` labels of the fit-start block share one width.
LABEL_WIDTH = len("Parallel: ")


def jax_fallback_reason_from(setup: Setup, use_jax: Optional[bool] = None) -> Optional[str]:
    """
    Why a setup designating the `jax` backend runs on numpy instead (`None` when it does not):
    `"JAX disabled"` under `PYAUTO_DISABLE_JAX=1`, else `"JAX not installed"`.

    `use_jax` is the resolved backend of the analysis (`use_jax_from`); `None` means "as
    designated", so there is no fallback.
    """
    if use_jax is None or use_jax or setup.backend != "jax":
        return None
    if os.environ.get("PYAUTO_DISABLE_JAX") == "1":
        return "JAX disabled"
    return "JAX not installed"


def setup_text_from(setup: Setup, use_jax: Optional[bool] = None) -> str:
    """
    The setup tuple in readable tokens, e.g. `numba · CPU · sparse operator · fp64` or
    `numba · CPU · sparse operator · fp64 · w-tilde`.

    `use_jax` is the backend the analysis actually resolved (`use_jax_from`). When the setup
    designates `jax` but `use_jax` is `False` (`PYAUTO_DISABLE_JAX=1` or JAX not installed),
    the text reports what runs, `numpy · CPU`, not the designation. `None` (the default)
    reports the setup as designated.
    """
    if jax_fallback_reason_from(setup=setup, use_jax=use_jax) is not None:
        backend, device = "numpy", "cpu"
    else:
        backend, device = setup.backend, setup.device

    tokens = [
        _BACKEND_TEXT.get(backend, backend),
        device.upper(),
        _OPERATOR_TEXT.get(setup.operator, f"{setup.operator} operator"),
        _PRECISION_TEXT.get(setup.precision, setup.precision),
    ]
    if setup.transformer is not None:
        tokens.append(_TRANSFORMER_TEXT.get(setup.transformer, setup.transformer))
    return " · ".join(tokens)


def rule_likelihood_from(setup: Setup) -> Optional[str]:
    """
    The `<dataset_type>/<variant>` likelihood of the rule that chose a setup (looked up
    by its `decision_id` in the rule table), or `None` when it was not chosen by a rule.
    """
    decision_id = setup.decision_id or setup.rule_id
    if not decision_id or decision_id == EXPLICIT_RULE_ID:
        return None
    for row in rules():
        if row.get("decision_id") == decision_id:
            return row["likelihood"]
    return None


def _intended_hardware_class(environment) -> str:
    """
    The hardware class `aa.Setup.auto` aimed for: `gpu_a100` when `PYAUTO_SETUP=gpu`, or
    when it is unset / `auto` and a GPU is detected; otherwise `cpu_many_core`.
    """
    value = (os.environ.get("PYAUTO_SETUP") or "auto").strip().lower()
    if value == "gpu":
        return HARDWARE_CLASS_GPU
    if value == "cpu":
        return HARDWARE_CLASS_CPU
    return HARDWARE_CLASS_GPU if getattr(environment, "gpu", False) else HARDWARE_CLASS_CPU


_CLASS_TEXT = {HARDWARE_CLASS_CPU: "CPU", HARDWARE_CLASS_GPU: "GPU"}


def extrapolated_reason_from(setup: Setup, environment=None) -> Optional[str]:
    """
    Why an automatically chosen setup is marked extrapolated (`None` when it is not):

    - a CPU rule on fewer than `MANY_CORE_MIN_CORES` cores:
      `8 cores, rules measured on ≥16-core nodes`;
    - an A100 rule on another GPU: `<GPU name>, rules measured on A100`, or on a machine
      with no GPU: `no GPU detected, rules measured on A100`;
    - no rule for the intended hardware class:
      `no GPU rule for this likelihood, the CPU rule is used`.
    """
    if not setup.extrapolated:
        return None

    if environment is not None and setup.hardware_class is not None:
        intended = _intended_hardware_class(environment)
        if intended != setup.hardware_class:
            return (
                f"no {_CLASS_TEXT.get(intended, intended)} rule for this likelihood, "
                f"the {_CLASS_TEXT.get(setup.hardware_class, setup.hardware_class)} "
                f"rule is used"
            )

    if setup.hardware_class == HARDWARE_CLASS_GPU:
        if environment is None:
            return f"rules measured on {GPU_A100_NAME}"
        gpu_name = getattr(environment, "gpu_name", None)
        if not getattr(environment, "gpu", False):
            return f"no GPU detected, rules measured on {GPU_A100_NAME}"
        return f"{gpu_name or 'GPU'}, rules measured on {GPU_A100_NAME}"

    cores = getattr(environment, "cores", None) if environment is not None else None
    measured = f"rules measured on ≥{MANY_CORE_MIN_CORES}-core nodes"
    return f"{cores} cores, {measured}" if cores else measured


def setup_header_from(setup: Setup, chosen_by: str) -> str:
    """
    The header line logged before the `Setup:` line at fit start, saying how the setup
    was chosen:

    - `"auto"`: `Automatic compute setup (CPU/GPU choice from setup="auto"; override with
      PYAUTO_SETUP or setup=...):`
    - `"PYAUTO_SETUP"`: `Compute setup (set by PYAUTO_SETUP=<setup_id>):`
    - `"explicit"`: `Compute setup (set explicitly by setup=...):`
    """
    if chosen_by == CHOSEN_BY_AUTO:
        return (
            'Automatic compute setup (CPU/GPU choice from setup="auto"; '
            "override with PYAUTO_SETUP or setup=...):"
        )
    if chosen_by == CHOSEN_BY_ENV:
        return f"Compute setup (set by PYAUTO_SETUP={setup.setup_id}):"
    return "Compute setup (set explicitly by setup=...):"


def setup_banner_from(
    setup: Setup,
    chosen_by: str,
    environment=None,
    note: Optional[str] = None,
    use_jax: Optional[bool] = None,
) -> str:
    """
    The `Setup:` line logged at fit start (after `setup_header_from`), e.g.

    `Setup:    numba · CPU · sparse operator · fp64   (rule imaging/rectangular ·
    extrapolated: 8 cores, rules measured on ≥16-core nodes)`

    An automatically chosen setup names its rule (`<dataset_type>/<variant>`) and, when
    extrapolated, why (`extrapolated_reason_from`); `note` (e.g. that a fallback rule was
    used) is appended inside the parentheses. An explicit or `PYAUTO_SETUP` setup has
    only the note, if any.

    `use_jax` is the backend the analysis resolved (`use_jax_from`). When a `jax` setup
    falls back to numpy, the line reports what runs and names the designation and why, e.g.

    `Setup:    numpy · CPU · dense · fp64   (rule imaging/mge · designated JAX; JAX disabled)`
    """
    line = f"{'Setup:':<{LABEL_WIDTH}}{setup_text_from(setup, use_jax=use_jax)}"

    details = []
    if chosen_by == CHOSEN_BY_AUTO:
        likelihood = rule_likelihood_from(setup)
        details.append(f"rule {likelihood or setup.rule_id}")
        reason = extrapolated_reason_from(setup=setup, environment=environment)
        if reason:
            details.append(f"extrapolated: {reason}")
    fallback = jax_fallback_reason_from(setup=setup, use_jax=use_jax)
    if fallback:
        designated = _BACKEND_TEXT.get(setup.backend, setup.backend)
        if setup.device != "cpu":
            designated += f" on {setup.device.upper()}"
        details.append(f"designated {designated}; {fallback}")
    if note:
        details.append(note)

    if details:
        line += f"   ({' · '.join(details)})"

    return line


def setup_banner_lines_from(
    setup: Setup,
    chosen_by: str,
    environment=None,
    note: Optional[str] = None,
    use_jax: Optional[bool] = None,
) -> list:
    """
    The header and `Setup:` lines logged at fit start, as two lines logged as two
    consecutive records (so each carries the log prefix and the labels align with the
    `Parallel:` / `Estimate:` lines that follow). `use_jax` is the resolved backend (see
    `setup_banner_from`).
    """
    return [
        setup_header_from(setup=setup, chosen_by=chosen_by),
        setup_banner_from(
            setup=setup,
            chosen_by=chosen_by,
            environment=environment,
            note=note,
            use_jax=use_jax,
        ),
    ]


def setup_not_set_line_from(setup: Setup) -> str:
    """
    The one line logged when an analysis has no `setup`, e.g.

    `Compute setup: not set (auto would choose numba · CPU · sparse operator · fp64;
    pass setup="auto")`
    """
    return (
        f"Compute setup: not set (auto would choose {setup_text_from(setup)}; "
        f'pass setup="auto")'
    )


def setup_dict_from(setup: Setup, chosen_by: str) -> dict:
    """
    The contents of `files/setup.json`: the setup_id, the setup tuple, its provenance and
    how it was chosen.
    """
    return {
        "setup_id": setup.setup_id,
        "backend": setup.backend,
        "device": setup.device,
        "operator": setup.operator,
        "precision": setup.precision,
        "transformer": setup.transformer,
        "rule_id": setup.rule_id,
        "regime": setup.regime,
        "hardware_class": setup.hardware_class,
        "decision_id": setup.decision_id,
        "extrapolated": setup.extrapolated,
        "chosen_by": chosen_by,
    }
