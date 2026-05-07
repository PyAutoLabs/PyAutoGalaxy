
import autofit as af
import autogalaxy as ag

from autogalaxy.interferometer.model.result import ResultInterferometer
from pathlib import Path


directory = Path(__file__).resolve().parent


def test__make_result__result_interferometer_is_returned(interferometer_7):
    model = af.Collection(galaxies=af.Collection(galaxy_0=ag.Galaxy(redshift=0.5)))

    analysis = ag.AnalysisInterferometer(dataset=interferometer_7, use_jax=False)

    search = ag.m.MockSearch(name="test_search")

    result = search.fit(model=model, analysis=analysis)

    assert isinstance(result, ResultInterferometer)


def test__fit_figure_of_merit__matches_correct_fit_given_galaxy_profiles(
    interferometer_7,
):
    galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=0.1))

    model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

    analysis = ag.AnalysisInterferometer(dataset=interferometer_7, use_jax=False)

    instance = model.instance_from_unit_vector([])
    fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

    galaxies = analysis.galaxies_via_instance_from(instance=instance)

    fit = ag.FitInterferometer(dataset=interferometer_7, galaxies=galaxies)

    assert fit.log_likelihood == fit_figure_of_merit
