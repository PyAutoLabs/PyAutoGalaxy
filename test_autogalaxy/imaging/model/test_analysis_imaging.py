from pathlib import Path

import autofit as af
import autogalaxy as ag

from autogalaxy.imaging.model.result import ResultImaging

directory = Path(__file__).resolve().parent


def test__make_result__result_imaging_is_returned(masked_imaging_7x7):
    model = af.Collection(galaxies=af.Collection(galaxy_0=ag.Galaxy(redshift=0.5)))

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)

    search = ag.m.MockSearch(name="test_search")

    result = search.fit(model=model, analysis=analysis)

    assert isinstance(result, ResultImaging)


def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
    masked_imaging_7x7,
):
    galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=0.1))

    model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)

    instance = model.instance_from_unit_vector([])
    fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

    galaxies = analysis.galaxies_via_instance_from(instance=instance)

    fit = ag.FitImaging(dataset=masked_imaging_7x7, galaxies=galaxies)

    assert fit.log_likelihood == fit_figure_of_merit


def test__register_fit_imaging_pytrees__registers_fit_galaxies_and_dataset_model():
    from autoarray.abstract_ndarray import _pytree_registered_classes
    from autoarray.dataset.dataset_model import DatasetModel
    from autogalaxy.galaxy.galaxies import Galaxies
    from autogalaxy.imaging.fit_imaging import FitImaging

    ag.AnalysisImaging._register_fit_imaging_pytrees()

    assert FitImaging in _pytree_registered_classes
    assert DatasetModel in _pytree_registered_classes
    assert Galaxies in _pytree_registered_classes


def test__save_attributes__registers_dataset_fits_in_database(masked_imaging_7x7, tmp_path):
    import numpy as np

    from autofit.aggregator.search_output import SearchOutput
    from autofit.database import Fit, Fits, open_database
    from autogalaxy.aggregator import agg_util
    from autogalaxy.aggregator.imaging.imaging import _imaging_from

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)
    directory_paths = af.DirectoryPaths(
        name="directory", path_prefix=str(tmp_path), is_identifier_in_paths=False
    )
    analysis.save_attributes(paths=directory_paths)
    directory_fit = SearchOutput(directory_paths.output_path)
    (directory_dataset,) = _imaging_from(fit=directory_fit)

    with open_database(str(tmp_path / "fits.sqlite")) as session:
        database_paths = af.DatabasePaths(
            session=session,
            name="database",
            path_prefix=str(tmp_path),
            is_identifier_in_paths=False,
        )
        analysis.save_attributes(paths=database_paths)
        fit_id = database_paths.fit.id
        session.commit()
        session.expunge_all()
        resumed_paths = af.DatabasePaths(
            session=session,
            name="database",
            path_prefix=str(tmp_path),
            is_identifier_in_paths=False,
        )
        analysis.save_attributes(paths=resumed_paths)
        session.commit()
        assert session.query(Fits).count() == 1
        database_fit = session.get(Fit, fit_id)

        with database_fit.value("dataset") as database_hdus, directory_fit.value(
            "dataset"
        ) as directory_hdus:
            assert [hdu.name for hdu in database_hdus] == [
                hdu.name for hdu in directory_hdus
            ]
            for database_hdu, directory_hdu in zip(database_hdus, directory_hdus):
                np.testing.assert_array_equal(database_hdu.data, directory_hdu.data)

        mask, _ = agg_util.mask_header_from(fit=database_fit)
        directory_mask, _ = agg_util.mask_header_from(fit=directory_fit)
        np.testing.assert_array_equal(mask.array, directory_mask.array)
        assert mask.pixel_scales == directory_mask.pixel_scales
        assert mask.origin == directory_mask.origin

        (database_dataset,) = _imaging_from(fit=database_fit)

    np.testing.assert_array_equal(
        database_dataset.data.array, directory_dataset.data.array
    )
    np.testing.assert_array_equal(
        database_dataset.noise_map.array, directory_dataset.noise_map.array
    )
    np.testing.assert_array_equal(
        database_dataset.psf.kernel.array, directory_dataset.psf.kernel.array
    )


# --- setup="auto" (likelihood lifecycle 5C) ---------------------------------------------

import json
import logging

import pytest

import autoarray as aa

from autonerves.environment import Environment


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
A100 = _environment(gpu=True, gpu_name="NVIDIA A100-SXM4-80GB")


@pytest.fixture(autouse=True)
def unset_setup_env(monkeypatch):
    monkeypatch.delenv("PYAUTO_SETUP", raising=False)
    monkeypatch.delenv("PYAUTO_DISABLE_JAX", raising=False)


def _detect(monkeypatch, environment):
    monkeypatch.setattr(Environment, "detect", classmethod(lambda cls: environment))


def _rectangular_model():
    pixelization = af.Model(
        ag.Pixelization,
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant,
    )
    return af.Collection(
        galaxies=af.Collection(
            source=af.Model(ag.Galaxy, redshift=1.0, pixelization=pixelization)
        )
    )


def _mge_model():
    basis = af.Model(
        ag.lp_basis.Basis,
        profile_list=[af.Model(ag.lp_linear.Gaussian) for _ in range(3)],
    )
    return af.Collection(
        galaxies=af.Collection(galaxy=af.Model(ag.Galaxy, redshift=0.5, bulge=basis))
    )


def test__setup_auto__cpu_many_core_rectangular__numba_sparse_operator(
    masked_imaging_7x7, monkeypatch, caplog
):
    pytest.importorskip("numba")
    from autoarray.inversion.inversion.imaging_numba.inversion_imaging_numba_util import (
        SparseLinAlgImagingNumba,
    )

    _detect(monkeypatch, MANY_CORE)

    analysis = ag.AnalysisImaging(
        dataset=masked_imaging_7x7, setup="auto", use_jax=True
    )

    with caplog.at_level(logging.INFO):
        result = analysis.modify_before_fit(
            paths=af.DirectoryPaths(), model=_rectangular_model()
        )

    assert result is analysis
    assert analysis.setup.setup_id == "numba_cpu_sparse_fp64"
    assert analysis.setup_chosen_by == "auto"
    assert analysis._use_jax is False
    assert analysis.dataset.setup == analysis.setup
    assert isinstance(analysis.dataset.sparse_operator, SparseLinAlgImagingNumba)

    banners = [r.message for r in caplog.records if r.message.startswith("Setup: ")]

    assert banners == [
        "Setup: numba/cpu/sparse/fp64 (64 CPU cores) · rule "
        "imaging-rectangular-standard-cpu-many-core · optimal (decision "
        "imaging-rectangular-standard-cpu-many-core) · override: PYAUTO_SETUP"
    ]


def test__setup_auto__gpu_rectangular__jax_dense_no_operator(
    masked_imaging_7x7, monkeypatch
):
    pytest.importorskip("jax")

    _detect(monkeypatch, A100)

    analysis = ag.AnalysisImaging(
        dataset=masked_imaging_7x7, setup="auto", use_jax=False
    )
    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=_rectangular_model())

    assert analysis.setup.setup_id == "jax_gpu_dense_fp64"
    assert analysis.setup.extrapolated is False
    assert analysis._use_jax is True
    assert analysis.dataset.sparse_operator is None


def test__setup_auto__cpu_mge__jax_dense(masked_imaging_7x7, monkeypatch):
    pytest.importorskip("jax")

    _detect(monkeypatch, MANY_CORE)

    analysis = ag.AnalysisImaging(
        dataset=masked_imaging_7x7, setup="auto", use_jax=False
    )
    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=_mge_model())

    assert analysis.setup.setup_id == "jax_cpu_dense_fp64"
    assert analysis._use_jax is True
    assert analysis.dataset.sparse_operator is None


def test__setup_auto__disable_jax_falls_back_to_numpy(masked_imaging_7x7, monkeypatch):
    _detect(monkeypatch, A100)
    monkeypatch.setenv("PYAUTO_DISABLE_JAX", "1")

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7, setup="auto")
    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=_rectangular_model())

    assert analysis.setup.setup_id == "jax_gpu_dense_fp64"
    assert analysis._use_jax is False


def test__setup_auto__pyauto_setup_env_wins(masked_imaging_7x7, monkeypatch, tmp_path):
    _detect(monkeypatch, A100)
    monkeypatch.setenv("PYAUTO_SETUP", "numba_cpu_dense_fp64")

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7, setup="auto")
    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=_rectangular_model())

    assert analysis.setup.setup_id == "numba_cpu_dense_fp64"
    assert analysis.setup_chosen_by == "PYAUTO_SETUP"
    assert analysis._use_jax is False

    paths = af.DirectoryPaths(name="pyauto_setup", path_prefix=str(tmp_path))
    analysis.save_attributes(paths=paths)

    with open(paths._files_path / "setup.json") as f:
        setup_dict = json.load(f)

    assert setup_dict["chosen_by"] == "PYAUTO_SETUP"
    assert setup_dict["rule_id"] == "explicit:PYAUTO_SETUP"
    assert setup_dict["setup_id"] == "numba_cpu_dense_fp64"


def test__explicit_setup__wins_over_pyauto_setup_env(masked_imaging_7x7, monkeypatch):
    _detect(monkeypatch, MANY_CORE)
    monkeypatch.setenv("PYAUTO_SETUP", "jax_cpu_dense_fp64")

    setup = aa.Setup.from_setup_id("numba_cpu_dense_fp64")

    analysis = ag.AnalysisImaging(
        dataset=masked_imaging_7x7, setup=setup, use_jax=True
    )
    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=_mge_model())

    assert analysis.setup is setup
    assert analysis.setup_chosen_by == "explicit"
    assert analysis._use_jax is False


def test__setup_auto__setup_json_written_by_search(masked_imaging_7x7, monkeypatch):
    pytest.importorskip("jax")

    _detect(monkeypatch, MANY_CORE)

    model = af.Collection(galaxies=af.Collection(galaxy_0=ag.Galaxy(redshift=0.5)))

    analysis = ag.AnalysisImaging(
        dataset=masked_imaging_7x7, setup="auto", use_jax=False
    )

    search = ag.m.MockSearch(name="test_search_setup_auto")
    search.fit(model=model, analysis=analysis)

    with open(search.paths._files_path / "setup.json") as f:
        setup_dict = json.load(f)

    assert setup_dict["setup_id"] == "jax_cpu_dense_fp64"
    assert setup_dict["rule_id"] == "imaging-mge-standard-cpu-many-core"
    assert setup_dict["decision_id"] == "imaging-mge-standard-cpu-many-core"
    assert setup_dict["hardware_class"] == "cpu_many_core"
    assert setup_dict["extrapolated"] is False
    assert setup_dict["chosen_by"] == "auto"


def test__check_setup_consistency__numba_operator_under_jax_raises(masked_imaging_7x7):
    pytest.importorskip("jax")
    pytest.importorskip("numba")

    from autogalaxy.analysis import setup_util

    dataset = masked_imaging_7x7.apply_setup("numba_cpu_sparse_fp64")

    setup_util.check_setup_consistency(dataset=dataset, use_jax=False)

    with pytest.raises(aa.exc.InversionException, match="mismatch"):
        setup_util.check_setup_consistency(dataset=dataset, use_jax=True)
