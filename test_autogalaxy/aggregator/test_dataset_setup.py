"""
Aggregator reloads of a dataset reapply the compute setup the search wrote to `files/setup.json`
(`agg_util.dataset_with_setup_from`), so a reloaded fit takes the same operator route as the search.
"""
import logging
from types import SimpleNamespace

import pytest

import autoarray as aa
import autogalaxy as ag

from autogalaxy.aggregator import agg_util
from autogalaxy.aggregator.imaging.imaging import _imaging_from
from autogalaxy.aggregator.interferometer.interferometer import _interferometer_from
from autogalaxy.analysis import setup_util
from autogalaxy.analysis.analysis.dataset import AnalysisDataset
from autogalaxy.interferometer.model.analysis import interferometer_hdu_list_from


def _imaging_hdu_list_from(dataset, monkeypatch):
    """The `dataset.fits` HDU list `AnalysisImaging.save_attributes` writes for `dataset`."""
    captured = []

    monkeypatch.setattr(
        AnalysisDataset,
        "_save_dataset_fits",
        staticmethod(lambda paths, hdu_list: captured.append(hdu_list)),
    )

    analysis = ag.AnalysisImaging(dataset=dataset, use_jax=False)
    analysis.save_attributes(paths=SimpleNamespace(save_json=lambda **kwargs: None))

    monkeypatch.undo()

    return captured[0]


def _fake_fit(values):
    """A `Fit` stand-in whose `value(name)` returns `values.get(name)` (None for a missing file)."""
    return SimpleNamespace(children=[], value=lambda name: values.get(name))


def _setup_dict(setup_id):
    return setup_util.setup_dict_from(
        setup=setup_util.setup_input_from(setup_id),
        chosen_by=setup_util.CHOSEN_BY_EXPLICIT,
    )


def test__imaging_from__no_setup_json__reloads_unchanged(masked_imaging_7x7, monkeypatch):
    hdu_list = _imaging_hdu_list_from(masked_imaging_7x7, monkeypatch)

    dataset = _imaging_from(fit=_fake_fit({"dataset": hdu_list}))[0]

    assert dataset.setup is None
    assert dataset.sparse_operator is None
    assert (dataset.data.array == masked_imaging_7x7.data.array).all()


def test__imaging_from__setup_json__reapplied(masked_imaging_7x7, monkeypatch, caplog):
    pytest.importorskip("numba")

    hdu_list = _imaging_hdu_list_from(masked_imaging_7x7, monkeypatch)

    with caplog.at_level(logging.INFO, logger=agg_util.__name__):
        dataset = _imaging_from(
            fit=_fake_fit(
                {"dataset": hdu_list, "setup": _setup_dict("numba_cpu_sparse_fp64")}
            )
        )[0]

    notices = [r.message for r in caplog.records if r.name == agg_util.__name__]
    assert notices == [
        "Compute setup: reapplied numba_cpu_sparse_fp64 from the search's setup.json"
    ]

    assert dataset.setup.setup_id == "numba_cpu_sparse_fp64"
    assert dataset.sparse_operator is not None

    # The reloaded fit takes the sparse route and reproduces the search's (dense) figure of merit.
    pixelization = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )
    galaxies = [ag.Galaxy(redshift=0.5, pixelization=pixelization)]

    fit = ag.FitImaging(dataset=dataset, galaxies=galaxies)
    fit_dense = ag.FitImaging(dataset=masked_imaging_7x7, galaxies=galaxies)

    assert fit.inversion.dataset.sparse_operator is dataset.sparse_operator
    assert fit.figure_of_merit == pytest.approx(fit_dense.figure_of_merit, rel=1.0e-6)


def test__imaging_from__setup_reapply_fails__one_warning_and_unapplied(
    masked_imaging_7x7, monkeypatch, caplog
):
    hdu_list = _imaging_hdu_list_from(masked_imaging_7x7, monkeypatch)

    def raise_(self, setup, **kwargs):
        raise aa.exc.DatasetException("no JAX here")

    monkeypatch.setattr(aa.Imaging, "apply_setup", raise_)

    with caplog.at_level(logging.WARNING, logger=agg_util.__name__):
        dataset = _imaging_from(
            fit=_fake_fit(
                {"dataset": hdu_list, "setup": _setup_dict("jax_cpu_sparse_fp64")}
            )
        )[0]

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]

    assert len(warnings) == 1
    assert not [r for r in caplog.records if r.message.startswith("Compute setup: reapplied")]
    assert "jax_cpu_sparse_fp64" in warnings[0].message
    assert "no JAX here" in warnings[0].message

    assert dataset.setup is None
    assert dataset.sparse_operator is None


def test__interferometer_from__setup_json__reapplied(interferometer_7):
    pytest.importorskip("finufft")

    values = {
        "dataset": interferometer_hdu_list_from(dataset=interferometer_7),
        "transformer_class": aa.TransformerDFT,
    }

    dataset = _interferometer_from(fit=_fake_fit(values))[0]

    assert dataset.setup is None
    assert dataset.sparse_operator is None

    values["setup"] = _setup_dict("numba_cpu_sparse_fp64_wtilde")

    dataset = _interferometer_from(fit=_fake_fit(values))[0]

    assert dataset.setup.setup_id == "numba_cpu_sparse_fp64_wtilde"
    assert dataset.sparse_operator is not None


def test__dataset_with_setup_from__array_free_dataset__skipped(monkeypatch):
    called = []

    dataset = SimpleNamespace(
        data=None, apply_setup=lambda setup: called.append(setup)
    )

    fit = _fake_fit({"setup": {"setup_id": "numba_cpu_sparse_fp64_wtilde"}})

    assert agg_util.dataset_with_setup_from(fit=fit, dataset=dataset) is dataset
    assert called == []


# The shared setup-resolution mixin.


def test__analysis_dataset__setup_mixin_in_mro_once():
    from autogalaxy.analysis.analysis.analysis import Analysis
    from autogalaxy.analysis.analysis.setup import AnalysisSetupMixin

    for cls in (AnalysisDataset, ag.AnalysisImaging, ag.AnalysisInterferometer):
        mro = cls.__mro__

        assert mro.count(AnalysisSetupMixin) == 1
        assert mro.index(AnalysisSetupMixin) < mro.index(Analysis)

    # Analyses without a setup (e.g. the plain `Analysis` base) do not carry the mixin.
    assert AnalysisSetupMixin not in Analysis.__mro__
