from pathlib import Path

import autofit as af
import autogalaxy as ag

from autogalaxy.ellipse.model.result import ResultEllipse

directory = Path(__file__).resolve().parent


def test__make_result__result_imaging_is_returned(masked_imaging_7x7):
    ellipse_list = af.Collection(af.Model(ag.Ellipse) for _ in range(2))

    ellipse_list[0].major_axis = 0.2
    ellipse_list[1].major_axis = 0.4

    model = af.Collection(ellipses=ellipse_list)

    analysis = ag.AnalysisEllipse(dataset=masked_imaging_7x7)

    search = ag.m.MockSearch(name="test_search")

    result = search.fit(model=model, analysis=analysis)

    assert isinstance(result, ResultEllipse)


def test__figure_of_merit(
    masked_imaging_7x7,
):
    ellipse_list = af.Collection(af.Model(ag.Ellipse) for _ in range(2))

    ellipse_list[0].major_axis = 0.2
    ellipse_list[1].major_axis = 0.4

    multipole_0_prior_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.1)
    multipole_0_prior_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.1)

    multipole_1_prior_0 = af.UniformPrior(lower_limit=0.0, upper_limit=0.1)
    multipole_1_prior_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.1)

    multipole_list = []

    for i in range(len(ellipse_list)):
        multipole_0 = af.Model(ag.EllipseMultipole)
        multipole_0.m = 1
        multipole_0.multipole_comps.multipole_comps_0 = multipole_0_prior_0
        multipole_0.multipole_comps.multipole_comps_1 = multipole_0_prior_1

        multipole_1 = af.Model(ag.EllipseMultipole)
        multipole_1.m = 4
        multipole_1.multipole_comps.multipole_comps_0 = multipole_1_prior_0
        multipole_1.multipole_comps.multipole_comps_1 = multipole_1_prior_1

        multipole_list.append([multipole_0, multipole_1])

    model = af.Collection(ellipses=ellipse_list, multipoles=multipole_list)

    analysis = ag.AnalysisEllipse(dataset=masked_imaging_7x7, use_jax=False)

    instance = model.instance_from_prior_medians()
    fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

    fit_list = []

    for i in range(len(instance.ellipses)):
        ellipse = instance.ellipses[i]
        multipole_list = instance.multipoles[i]

        fit = ag.FitEllipse(
            dataset=masked_imaging_7x7, ellipse=ellipse, multipole_list=multipole_list
        )

        fit_list.append(fit)

    assert (
        fit_list[0].log_likelihood + fit_list[1].log_likelihood == fit_figure_of_merit
    )
