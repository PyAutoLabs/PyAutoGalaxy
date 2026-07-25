import autofit as af
import autogalaxy as ag

from autogalaxy.analysis import result as res


def test__result_contains_instance_with_galaxies(
    analysis_imaging_7x7, samples_summary_with_result
):
    result = res.Result(
        samples_summary=samples_summary_with_result,
        analysis=analysis_imaging_7x7,
    )

    assert isinstance(result.instance.galaxies[0], ag.Galaxy)


def test__max_log_likelihood_galaxies_available_as_result(analysis_imaging_7x7):
    galaxy_0 = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=1.0))
    galaxy_1 = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=2.0))

    model = af.Collection(galaxies=af.Collection(galaxy_0=galaxy_0, galaxy_1=galaxy_1))

    max_log_likelihood_instance = model.instance_from_prior_medians()

    search = ag.m.MockSearch(
        name="test_search",
        samples_summary=ag.m.MockSamplesSummary(
            max_log_likelihood_instance=max_log_likelihood_instance
        ),
    )

    result = search.fit(model=model, analysis=analysis_imaging_7x7)

    assert isinstance(result.max_log_likelihood_galaxies[0], ag.Galaxy)
    assert result.max_log_likelihood_galaxies[0].light.intensity == 1.0
    assert result.max_log_likelihood_galaxies[1].light.intensity == 2.0


def test__results_include_pixelization__available_as_property(analysis_imaging_7x7):
    pixelization = ag.m.MockPixelization(mapper=1)

    galaxy_0 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(galaxy_0=galaxy_0, galaxy_1=galaxy_0))

    max_log_likelihood_instance = model.instance_from_prior_medians()

    samples_summary = ag.m.MockSamplesSummary(
        max_log_likelihood_instance=max_log_likelihood_instance
    )

    result = res.ResultDataset(
        samples_summary=samples_summary,
        analysis=analysis_imaging_7x7,
    )

    assert isinstance(
        result.cls_list_from(cls=ag.Pixelization)[0], ag.m.MockPixelization
    )
    assert result.cls_list_from(cls=ag.Pixelization)[0].mapper == 1
