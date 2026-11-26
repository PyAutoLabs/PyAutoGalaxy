"""
A `FactorGraphModel` of default-constructed analyses of different dataset types (e.g. imaging and
interferometer) resolves a different `setup="auto"` per factor. Factors whose setup was defaulted adopt
one shared backend (`adopt_backend`), so the graph fits without pins; explicit setups must still agree.
"""
import importlib.util
import logging

import numpy as np
import pytest

import autoarray as aa
import autofit as af
import autofit.graphical as g
import autogalaxy as ag

from autofit import exc
from autonerves.environment import Environment

MANY_CORE = Environment(
    gpu=False,
    gpu_name=None,
    gpu_memory_gb=None,
    cores=64,
    ram_gb=None,
    numba=True,
    nufftax=True,
    site="local",
)

HAS_JAX = importlib.util.find_spec("jax") is not None


@pytest.fixture(name="many_core")
def make_many_core(monkeypatch):
    monkeypatch.delenv("PYAUTO_SETUP", raising=False)
    monkeypatch.delenv("PYAUTO_DISABLE_JAX", raising=False)
    monkeypatch.setattr(Environment, "detect", classmethod(lambda cls: MANY_CORE))


def _pixelized_model():
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


def _parametric_model():
    return af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(ag.Galaxy, redshift=0.5, light=ag.lp.Sersic)
        )
    )


def _graph(analysis_0, model_0, analysis_1, model_1, **kwargs):
    return g.FactorGraphModel(
        g.AnalysisFactor(model_0, analysis=analysis_0, name="factor_0"),
        g.AnalysisFactor(model_1, analysis=analysis_1, name="factor_1"),
        **kwargs,
    )


def _modify(graph, name):
    return graph.modify_before_fit(
        paths=af.DirectoryPaths(name=name), model=graph.global_prior_model
    )


def _sparse_operator(dataset):
    return dataset.__dict__.get("sparse_operator") or getattr(
        dataset, "_lazy_sparse_operator", None
    )


def test__default_imaging_and_interferometer__adopt_one_backend(
    masked_imaging_7x7, interferometer_7, many_core, caplog
):
    imaging = ag.AnalysisImaging(dataset=masked_imaging_7x7)
    interferometer = ag.AnalysisInterferometer(dataset=interferometer_7)

    assert imaging.setup_is_default and interferometer.setup_is_default

    graph = _graph(imaging, _pixelized_model(), interferometer, _parametric_model())

    with caplog.at_level(logging.INFO):
        _modify(graph, "graph_default_mixed")

    analysis_0 = graph.model_factors[0].analysis
    analysis_1 = graph.model_factors[1].analysis

    for analysis in (analysis_0, analysis_1):
        assert analysis.setup is None
        assert analysis.setup_input is None
        assert analysis.setup_is_default
        assert analysis.is_jax is HAS_JAX

    # The imaging auto setup (numba sparse) attached an operator; adopting the graph backend restores
    # the dataset as given.
    assert analysis_0.dataset is masked_imaging_7x7
    assert analysis_1.dataset is interferometer_7

    assert graph.setup_ids() == {}
    graph.check_setup_agreement()
    graph.check_backend_agreement()

    notes = [r.message for r in caplog.records if "the factor graph runs" in r.message]
    assert len(notes) == 2
    assert 'setup="auto" chose numba · CPU · sparse operator · fp64 for it alone' in notes[0]

    instance = graph.global_prior_model.instance_from_prior_medians()
    log_likelihood = graph.log_likelihood_function(instance)
    assert np.isfinite(float(log_likelihood))


def test__graph_use_jax_false__pushes_numpy_down(
    masked_imaging_7x7, interferometer_7, many_core
):
    imaging = ag.AnalysisImaging(dataset=masked_imaging_7x7)
    interferometer = ag.AnalysisInterferometer(dataset=interferometer_7)

    graph = _graph(
        imaging, _parametric_model(), interferometer, _parametric_model(), use_jax=False
    )
    _modify(graph, "graph_default_numpy")

    for factor in graph.model_factors:
        assert factor.analysis.is_jax is False
        assert factor.analysis.setup is None

    assert graph.is_jax is False
    graph.check_backend_agreement()

    instance = graph.global_prior_model.instance_from_prior_medians()
    assert np.isfinite(graph.log_likelihood_function(instance))


def test__default_factor_follows_explicit_setup(
    masked_imaging_7x7, interferometer_7, many_core
):
    imaging = ag.AnalysisImaging(
        dataset=masked_imaging_7x7, setup="numba_cpu_dense_fp64"
    )
    interferometer = ag.AnalysisInterferometer(dataset=interferometer_7)

    graph = _graph(imaging, _parametric_model(), interferometer, _parametric_model())
    _modify(graph, "graph_explicit_lead")

    assert graph.model_factors[0].analysis.setup.setup_id == "numba_cpu_dense_fp64"
    assert graph.model_factors[1].analysis.setup is None
    assert graph.model_factors[1].analysis.is_jax is False
    assert graph.setup_ids() == {"factor_0": "numba_cpu_dense_fp64"}
    graph.check_backend_agreement()


def test__explicit_disagreeing_setups__still_raise(
    masked_imaging_7x7, interferometer_7, many_core
):
    imaging = ag.AnalysisImaging(
        dataset=masked_imaging_7x7, setup="numba_cpu_dense_fp64"
    )
    interferometer = ag.AnalysisInterferometer(
        dataset=interferometer_7, setup="numba_cpu_sparse_fp64_wtilde"
    )

    graph = _graph(imaging, _parametric_model(), interferometer, _parametric_model())

    with pytest.raises(exc.SearchException, match="setups passed explicitly"):
        _modify(graph, "graph_explicit_bad")


def test__agreeing_default_setups__are_kept(masked_imaging_7x7, many_core):
    graph = _graph(
        ag.AnalysisImaging(dataset=masked_imaging_7x7),
        _pixelized_model(),
        ag.AnalysisImaging(dataset=masked_imaging_7x7),
        _pixelized_model(),
    )
    _modify(graph, "graph_default_agree")

    assert set(graph.setup_ids().values()) == {"numba_cpu_sparse_fp64"}
    assert _sparse_operator(graph.model_factors[0].analysis.dataset) is not None


def test__adopt_backend__explicit_setup_is_never_changed(masked_imaging_7x7, many_core):
    analysis = ag.AnalysisImaging(
        dataset=masked_imaging_7x7, setup="numba_cpu_dense_fp64"
    )
    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=_parametric_model())

    assert analysis.setup_is_default is False
    analysis.adopt_backend(True)

    assert analysis.setup.setup_id == "numba_cpu_dense_fp64"
    assert analysis.is_jax is False
