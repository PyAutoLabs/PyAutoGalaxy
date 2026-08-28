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
