"""
Helpers that wire the compute `aa.Setup` of a likelihood into the dataset analyses.

A dataset analysis (`AnalysisImaging`, `AnalysisInterferometer`) accepts
`setup=None | "auto" | aa.Setup | "<setup_id>"`. The setup is resolved in
`AnalysisDataset.modify_before_fit`, because the rule key of `aa.Setup.auto` needs the
model: a pixelized source (and its mesh) and a parametric / MGE fit have different
designated setups. This module holds the pieces that resolution needs:

- `setup_input_from`: normalise the `setup` input of an analysis.
- `setup_key_from`: map a dataset and a model to the `(dataset_type, variant)` rule key of
  `aa.Setup.auto`.
- `use_jax_from`: resolve a requested backend the way `af.Analysis.__init__` resolves
  `use_jax` (honouring `PYAUTO_DISABLE_JAX` and a missing JAX install).
- `check_setup_consistency`: run the inversion factory's operator / setup / backend check
  once, outside the likelihood, so a mismatch raises before sampling.
- `setup_banner_from` / `setup_dict_from`: the banner line logged at fit start and the
  contents of `files/setup.json`.
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

from autoarray.setup.setup import EXPLICIT_RULE_ID, Setup, rules, setup_from

logger = logging.getLogger(__name__)

SETUP_AUTO = "auto"

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
    A short description of the hardware a setup runs on, for the banner: the GPU name for
    a GPU setup, the core count for a CPU setup, or `None` if unknown.
    """
    if environment is None:
        return None
    if setup.device == "gpu":
        return getattr(environment, "gpu_name", None) or "GPU"
    cores = getattr(environment, "cores", None)
    return f"{cores} CPU cores" if cores else None


def setup_banner_from(
    setup: Setup,
    chosen_by: str,
    environment=None,
    note: Optional[str] = None,
) -> str:
    """
    The setup banner line logged at fit start, e.g.

    `Setup: jax/gpu/dense/fp64 (NVIDIA A100-SXM4-80GB) · rule imaging-rectangular-standard-gpu-a100 ·
    optimal (decision imaging-rectangular-standard-gpu-a100) · override: PYAUTO_SETUP`
    """
    tokens = [setup.backend, setup.device, setup.operator, setup.precision]
    if setup.transformer is not None:
        tokens.append(setup.transformer)

    line = f"Setup: {'/'.join(tokens)}"

    hardware = hardware_description_from(setup=setup, environment=environment)
    if hardware:
        line += f" ({hardware})"

    if chosen_by == CHOSEN_BY_AUTO:
        line += f" · rule {setup.rule_id}"
        quality = "extrapolated" if setup.extrapolated else "optimal"
        line += f" · {quality} (decision {setup.decision_id})"
        line += " · override: PYAUTO_SETUP"
    elif chosen_by == CHOSEN_BY_ENV:
        line += " · rule explicit (PYAUTO_SETUP)"
    else:
        line += " · rule explicit (setup passed to the analysis)"

    if note:
        line += f" · {note}"

    return line


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
