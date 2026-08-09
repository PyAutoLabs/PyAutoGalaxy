import numpy as np
import pytest

import autoarray as aa
import autogalaxy as ag


@pytest.mark.parametrize("over_sample_size", [1, 2, 4])
def test__image_places_total_flux_in_pixel_containing_centre(over_sample_size):
    grid = aa.Grid2D.uniform(
        shape_native=(3, 3),
        pixel_scales=1.0,
        over_sample_size=over_sample_size,
    )

    image = ag.lp.PointSource(centre=(0.2, -0.2), intensity=3.0).image_2d_from(
        grid=grid
    )

    assert image.native == pytest.approx(
        np.array([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]])
    )
    assert np.sum(image) == pytest.approx(3.0)


def test__image_is_zero_when_centre_is_outside_unmasked_grid():
    grid = aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)

    image = ag.lp.PointSource(centre=(2.0, 2.0), intensity=3.0).image_2d_from(grid=grid)

    assert image.native == pytest.approx(np.zeros((3, 3)))


def test__irregular_grid_uses_nearest_sample_as_discrete_delta():
    grid = aa.Grid2DIrregular(values=[(1.0, 1.0), (0.1, -0.2), (-1.0, -1.0)])

    image = ag.lp.PointSource(centre=(0.0, 0.0), intensity=3.0).image_2d_from(grid=grid)

    assert image == pytest.approx([0.0, 3.0, 0.0])


def test__oversampled_psf_convolution_conserves_flux_and_resolves_sub_pixel_shift():
    mask = aa.Mask2D.all_false(shape_native=(11, 11), pixel_scales=1.0)

    over_sample_size = 2
    kernel_size = 9
    coordinates = (np.arange(kernel_size) - (kernel_size - 1) / 2.0) / over_sample_size
    yy, xx = np.meshgrid(-coordinates, coordinates, indexing="ij")
    kernel = np.exp(-0.5 * (yy**2 + xx**2) / 0.8**2)
    kernel = aa.Array2D.no_mask(
        values=kernel / kernel.sum(), pixel_scales=1.0 / over_sample_size
    )
    psf = aa.Convolver(kernel=kernel, convolve_over_sample_size=over_sample_size)

    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=over_sample_size)
    blurring_mask = mask.derive_mask.blurring_from(
        kernel_shape_native=psf.kernel_shape_image_resolution,
        allow_padding=True,
    )
    blurring_grid = aa.Grid2D.from_mask(
        mask=blurring_mask, over_sample_size=over_sample_size
    )

    negative = ag.lp.PointSource(
        centre=(-0.3, -0.3), intensity=3.0
    ).blurred_image_2d_from(grid=grid, blurring_grid=blurring_grid, psf=psf)
    positive = ag.lp.PointSource(
        centre=(0.3, 0.3), intensity=3.0
    ).blurred_image_2d_from(grid=grid, blurring_grid=blurring_grid, psf=psf)

    assert np.sum(negative) == pytest.approx(3.0)
    assert np.sum(positive) == pytest.approx(3.0)

    negative_centroid = np.sum(np.asarray(negative)[:, None] * grid.array, axis=0) / 3.0
    positive_centroid = np.sum(np.asarray(positive)[:, None] * grid.array, axis=0) / 3.0

    assert negative_centroid == pytest.approx((-0.25, -0.25), abs=0.002)
    assert positive_centroid == pytest.approx((0.25, 0.25), abs=0.002)

    galaxy = ag.Galaxy(
        redshift=0.5,
        point_source=ag.lp.PointSource(centre=(0.3, 0.3), intensity=3.0),
    )
    galaxy_image = galaxy.blurred_image_2d_from(
        grid=grid, blurring_grid=blurring_grid, psf=psf
    )

    galaxies = ag.Galaxies(galaxies=[galaxy])
    galaxy_image_dict = galaxies.galaxy_blurred_image_2d_dict_from(
        grid=grid, blurring_grid=blurring_grid, psf=psf
    )

    assert np.asarray(galaxy_image) == pytest.approx(np.asarray(positive))
    assert np.asarray(galaxy_image_dict[galaxy]) == pytest.approx(np.asarray(positive))
