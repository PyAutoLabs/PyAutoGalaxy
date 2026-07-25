import numpy as np
import pytest

import autofit as af
import autogalaxy as ag
from autoarray import Array2D

from autogalaxy.imaging.model.result import ResultImaging


def test__results_include_masked_dataset_and_mask(
    analysis_imaging_7x7, masked_imaging_7x7, samples_summary_with_result
):
    result = ResultImaging(
        samples_summary=samples_summary_with_result,
        analysis=analysis_imaging_7x7,
    )

    assert (result.mask == masked_imaging_7x7.mask).all()
    assert (result.dataset.data == masked_imaging_7x7.data).all()


def test___image_dict(analysis_imaging_7x7):
    galaxies = af.ModelInstance()
    galaxies.galaxy = ag.Galaxy(redshift=0.5)
    galaxies.source = ag.Galaxy(redshift=0.5)

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    samples_summary = ag.m.MockSamplesSummary(max_log_likelihood_instance=instance)

    result = ResultImaging(
        samples_summary=samples_summary, analysis=analysis_imaging_7x7
    )

    image_dict = result.model_image_galaxy_dict

    assert isinstance(image_dict["('galaxies', 'galaxy')"], Array2D)
    assert isinstance(image_dict["('galaxies', 'source')"], Array2D)

    result.instance.galaxies.light = ag.Galaxy(redshift=0.5)

    image_dict = result.model_image_galaxy_dict
    assert (image_dict["('galaxies', 'galaxy')"].native == np.zeros((7, 7))).all()
    assert isinstance(image_dict["('galaxies', 'source')"], Array2D)


def test___linear_light_profiles_in_result(analysis_imaging_7x7):
    galaxies = af.ModelInstance()
    galaxies.galaxy = ag.Galaxy(
        redshift=0.5, bulge=ag.lp_linear.Sersic(centre=(0.05, 0.05))
    )

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    samples_summary = ag.m.MockSamplesSummary(max_log_likelihood_instance=instance)

    result = ResultImaging(
        samples_summary=samples_summary, analysis=analysis_imaging_7x7
    )

    assert not isinstance(
        result.max_log_likelihood_galaxies[0].bulge,
        ag.lp_linear.LightProfileLinear,
    )
    assert result.max_log_likelihood_galaxies[0].bulge.intensity == pytest.approx(
        0.18686846, 1.0e-4
    )
