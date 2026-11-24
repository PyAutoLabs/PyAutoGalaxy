import pytest
import numpy as np

import autofit as af
import autogalaxy as ag
from pathlib import Path

directory = Path(__file__).resolve().parent


def test__instance_with_associated_adapt_images_from__galaxy_name_image_dict(
    masked_imaging_7x7,
):
    galaxies = af.ModelInstance()
    galaxies.galaxy = ag.Galaxy(redshift=0.5)
    galaxies.source = ag.Galaxy(redshift=1.0)

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    galaxy_name_image_dict = {
        str(("galaxies", "galaxy")): ag.Array2D.ones(
            shape_native=(3, 3), pixel_scales=1.0
        ),
        str(("galaxies", "source")): ag.Array2D.full(
            fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0
        ),
    }

    adapt_images = ag.AdaptImages(
        galaxy_name_image_dict=galaxy_name_image_dict,
    )

    analysis = ag.AnalysisImaging(
        dataset=masked_imaging_7x7, adapt_images=adapt_images, use_jax=False
    )

    adapt_images = analysis.adapt_images_via_instance_from(instance=instance)

    assert adapt_images.galaxy_image_dict[galaxies.galaxy].native == pytest.approx(
        np.ones((3, 3)), 1.0e-4
    )
    assert adapt_images.galaxy_image_dict[galaxies.source].native == pytest.approx(
        2.0 * np.ones((3, 3)), 1.0e-4
    )


def test__instance_with_associated_adapt_images_from__galaxy_name_image_plane_mesh_grid_dict(
    masked_imaging_7x7,
):
    galaxies = af.ModelInstance()
    galaxies.galaxy = ag.Galaxy(redshift=0.5)
    galaxies.source = ag.Galaxy(redshift=1.0)

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    galaxy_name_image_plane_mesh_grid_dict = {
        str(("galaxies", "galaxy")): ag.Grid2DIrregular(
            values=[(3.0, 3.0), (3.0, 3.0)]
        ),
        str(("galaxies", "source")): ag.Grid2DIrregular(
            values=[(4.0, 4.0), (4.0, 4.0)]
        ),
    }

    adapt_images = ag.AdaptImages(
        galaxy_name_image_plane_mesh_grid_dict=galaxy_name_image_plane_mesh_grid_dict,
    )

    analysis = ag.AnalysisImaging(
        dataset=masked_imaging_7x7, adapt_images=adapt_images, use_jax=False
    )

    adapt_images = analysis.adapt_images_via_instance_from(instance=instance)

    assert adapt_images.galaxy_image_plane_mesh_grid_dict[
        galaxies.galaxy
    ].native == pytest.approx(3.0 * np.ones((2, 2)), 1.0e-4)
    assert adapt_images.galaxy_image_plane_mesh_grid_dict[
        galaxies.source
    ].native == pytest.approx(4.0 * np.ones((2, 2)), 1.0e-4)


class _RaisingGalaxiesResult:
    """
    Result double whose galaxies cannot be built, because materializing the maximum log
    likelihood sample as a model instance fails.
    """

    def __init__(self, error):
        self._error = error

    @property
    def max_log_likelihood_galaxies(self):
        raise self._error


@pytest.mark.parametrize(
    "error",
    [
        AttributeError("no galaxies on this result"),
        af.exc.SamplesException("stored parameters cannot be reconstructed"),
        af.exc.FitException("ell_comps must satisfy e0**2+e1**2 < 1"),
    ],
)
def test__save_results__galaxies_failure_never_kills_the_fit(
    analysis_imaging_7x7, error
):
    """
    `save_results` runs after the search has finished but before `paths.completed()`, so a
    failure writing the (optional) `galaxies.json` must be logged and swallowed rather than
    losing the run its `.completed` marker (PyAutoFit #1535).
    """
    paths = af.DirectoryPaths()

    analysis_imaging_7x7.save_results(
        paths=paths,
        result=_RaisingGalaxiesResult(error),
    )

    assert not (paths._files_path / "galaxies.json").exists()


# --- setup="auto" (likelihood lifecycle 5C) ---------------------------------------------

import json
import logging

import autoarray as aa

from autonerves.environment import Environment

from autogalaxy.analysis import setup_util


def _environment(gpu=False, gpu_name=None, cores=64):
    return Environment(
        gpu=gpu,
        gpu_name=gpu_name,
        gpu_memory_gb=None,
        cores=cores,
        ram_gb=None,
        numba=True,
        nufftax=True,
        site="local",
    )


MANY_CORE = _environment(cores=64)


@pytest.fixture(name="many_core")
def make_many_core(monkeypatch):
    monkeypatch.delenv("PYAUTO_SETUP", raising=False)
    monkeypatch.setattr(Environment, "detect", classmethod(lambda cls: MANY_CORE))


def _model_with_mesh(mesh):
    pixelization = af.Model(ag.Pixelization, mesh=mesh, regularization=ag.reg.Constant)
    return af.Collection(
        galaxies=af.Collection(
            source=af.Model(ag.Galaxy, redshift=1.0, pixelization=pixelization)
        )
    )


def _parametric_model():
    return af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(ag.Galaxy, redshift=0.5, light=ag.lp.Sersic)
        )
    )


@pytest.mark.parametrize(
    "mesh, variant",
    [
        (af.Model(ag.mesh.RectangularUniform, shape=(3, 3)), "rectangular"),
        (ag.mesh.RectangularUniform(shape=(3, 3)), "rectangular"),
        (af.Model(ag.mesh.RectangularRTUAdaptImage), "rectangular"),
        (af.Model(ag.mesh.RectangularBilinearAdaptDensity), "rectangular"),
        (af.Model(ag.mesh.Delaunay, pixels=20), "delaunay"),
        (ag.mesh.Delaunay(pixels=20), "delaunay"),
        (af.Model(ag.mesh.KNearestNeighbor, pixels=20), "delaunay"),
        (af.Model(ag.mesh.DelaunayNN, pixels=20), "delaunay_nn"),
        (ag.mesh.DelaunayNN(pixels=20), "delaunay_nn"),
    ],
)
def test__setup_key_from__imaging_mesh_variants(masked_imaging_7x7, mesh, variant):
    model = _model_with_mesh(mesh)

    assert setup_util.setup_key_from(masked_imaging_7x7, model) == ("imaging", variant)


def test__setup_key_from__pixelization_instance_in_galaxy_instance(masked_imaging_7x7):
    pixelization = ag.Pixelization(
        mesh=ag.mesh.DelaunayNN(pixels=20), regularization=ag.reg.Constant()
    )
    model = af.Collection(
        galaxies=af.Collection(
            source=ag.Galaxy(redshift=1.0, pixelization=pixelization)
        )
    )

    assert setup_util.setup_key_from(masked_imaging_7x7, model) == (
        "imaging",
        "delaunay_nn",
    )


def test__setup_key_from__no_pixelization_is_mge(masked_imaging_7x7):
    assert setup_util.setup_key_from(masked_imaging_7x7, _parametric_model()) == (
        "imaging",
        "mge",
    )

    basis = af.Model(
        ag.lp_basis.Basis,
        profile_list=[af.Model(ag.lp_linear.Gaussian) for _ in range(3)],
    )
    model = af.Collection(
        galaxies=af.Collection(galaxy=af.Model(ag.Galaxy, redshift=0.5, bulge=basis))
    )

    assert setup_util.setup_key_from(masked_imaging_7x7, model) == ("imaging", "mge")


def test__setup_key_from__several_meshes__most_expensive_wins(masked_imaging_7x7):
    model = af.Collection(
        galaxies=af.Collection(
            source_0=af.Model(
                ag.Galaxy,
                redshift=1.0,
                pixelization=af.Model(
                    ag.Pixelization,
                    mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
                    regularization=ag.reg.Constant,
                ),
            ),
            source_1=af.Model(
                ag.Galaxy,
                redshift=1.0,
                pixelization=af.Model(
                    ag.Pixelization,
                    mesh=ag.mesh.Delaunay(pixels=20),
                    regularization=ag.reg.Constant,
                ),
            ),
        )
    )

    assert setup_util.setup_key_from(masked_imaging_7x7, model) == (
        "imaging",
        "delaunay",
    )


def test__setup_key_from__interferometer__missing_variant_falls_back(interferometer_7):
    model = _model_with_mesh(af.Model(ag.mesh.DelaunayNN, pixels=20))

    key, note = setup_util.setup_key_and_note_from(interferometer_7, model)

    assert key == ("interferometer", "delaunay")
    assert "interferometer/delaunay_nn" in note

    assert setup_util.setup_key_from(interferometer_7, _parametric_model()) == (
        "interferometer",
        "mge",
    )


def test__setup_key_from__unsupported_dataset_raises():
    with pytest.raises(ValueError, match="Imaging or aa.Interferometer"):
        setup_util.setup_key_from(object(), _parametric_model())


def test__setup_input_from():
    assert setup_util.setup_input_from(None) is None
    assert setup_util.setup_input_from("auto") == "auto"
    assert setup_util.setup_input_from(" AUTO ") == "auto"

    setup = setup_util.setup_input_from("numba_cpu_sparse_fp64")

    assert isinstance(setup, aa.Setup)
    assert setup.setup_id == "numba_cpu_sparse_fp64"

    explicit = aa.Setup.from_setup_id("jax_gpu_dense_fp64")

    assert setup_util.setup_input_from(explicit) is explicit

    with pytest.raises(ValueError):
        setup_util.setup_input_from("not_a_setup")


def test__analysis__stores_setup_input(masked_imaging_7x7):
    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)

    assert analysis.setup_input is None
    assert analysis.setup is None

    analysis = ag.AnalysisImaging(
        dataset=masked_imaging_7x7, setup="auto", use_jax=False
    )

    assert analysis.setup_input == "auto"

    analysis = ag.AnalysisImaging(
        dataset=masked_imaging_7x7, setup="numba_cpu_dense_fp64", use_jax=False
    )

    assert analysis.setup_input == aa.Setup.from_setup_id("numba_cpu_dense_fp64")


def test__use_jax_from__disable_jax_env(monkeypatch):
    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")

    assert setup_util.use_jax_from(True) is False

    monkeypatch.delenv("PYAUTO_DISABLE_JAX")

    assert setup_util.use_jax_from(False) is False


def test__no_setup__dataset_and_backend_unchanged__would_choose_logged(
    masked_imaging_7x7, many_core, caplog
):
    model = _model_with_mesh(ag.mesh.RectangularUniform(shape=(3, 3)))

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)

    with caplog.at_level(logging.INFO):
        analysis.modify_before_fit(paths=af.DirectoryPaths(), model=model)

    assert analysis.dataset is masked_imaging_7x7
    assert analysis.dataset.setup is None
    assert analysis._use_jax is False
    assert analysis.setup is None

    would_choose = [r.message for r in caplog.records if "auto would choose" in r.message]

    assert would_choose == [
        "Setup: auto would choose numba_cpu_sparse_fp64 "
        "(rule imaging-rectangular-standard-cpu-many-core); pass setup=\"auto\" to use it"
    ]


def test__no_setup__save_attributes_writes_no_setup_json(
    masked_imaging_7x7, tmp_path
):
    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)

    paths = af.DirectoryPaths(name="no_setup", path_prefix=str(tmp_path))
    analysis.save_attributes(paths=paths)

    assert (paths._files_path / "settings.json").exists()
    assert not (paths._files_path / "setup.json").exists()


def test__explicit_setup__save_attributes_writes_setup_json(
    masked_imaging_7x7, many_core, tmp_path
):
    analysis = ag.AnalysisImaging(
        dataset=masked_imaging_7x7, setup="numba_cpu_dense_fp64", use_jax=True
    )
    analysis.modify_before_fit(
        paths=af.DirectoryPaths(), model=_parametric_model()
    )

    paths = af.DirectoryPaths(name="explicit_setup", path_prefix=str(tmp_path))
    analysis.save_attributes(paths=paths)

    with open(paths._files_path / "setup.json") as f:
        setup_dict = json.load(f)

    assert setup_dict == {
        "setup_id": "numba_cpu_dense_fp64",
        "backend": "numba",
        "device": "cpu",
        "operator": "dense",
        "precision": "fp64",
        "transformer": None,
        "rule_id": None,
        "regime": "standard",
        "hardware_class": None,
        "decision_id": None,
        "extrapolated": False,
        "chosen_by": "explicit",
    }


def test__setup_banner_from():
    setup = aa.Setup.from_setup_id(
        "jax_gpu_dense_fp64",
        rule_id="imaging-rectangular-standard-gpu-a100",
        hardware_class="gpu_a100",
        decision_id="imaging-rectangular-standard-gpu-a100",
        extrapolated=False,
    )

    banner = setup_util.setup_banner_from(
        setup=setup,
        chosen_by="auto",
        environment=_environment(gpu=True, gpu_name="NVIDIA A100-SXM4-80GB"),
    )

    assert banner == (
        "Setup: jax/gpu/dense/fp64 (NVIDIA A100-SXM4-80GB) · rule "
        "imaging-rectangular-standard-gpu-a100 · optimal (decision "
        "imaging-rectangular-standard-gpu-a100) · override: PYAUTO_SETUP"
    )

    extrapolated = aa.Setup.from_setup_id(
        "numba_cpu_sparse_fp64_wtilde",
        rule_id="r",
        decision_id="r",
        extrapolated=True,
    )

    banner = setup_util.setup_banner_from(
        setup=extrapolated, chosen_by="auto", environment=_environment(cores=8)
    )

    assert banner.startswith("Setup: numba/cpu/sparse/fp64/wtilde (8 CPU cores)")
    assert "extrapolated (decision r)" in banner
