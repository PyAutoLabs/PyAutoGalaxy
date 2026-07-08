from __future__ import division, print_function

import numpy as np
import pytest

import autogalaxy as ag

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__blurred_image_2d_from__single_light_profile__matches_manual_psf_convolution(
    grid_2d_7x7,
    blurring_grid_2d_7x7,
    psf_3x3,
):
    lp = ag.lp.Sersic(intensity=1.0)

    image_2d = lp.image_2d_from(grid=grid_2d_7x7)
    blurring_image_2d = lp.image_2d_from(grid=blurring_grid_2d_7x7)

    blurred_image_2d_manual = psf_3x3.convolved_image_from(
        image=image_2d, blurring_image=blurring_image_2d
    )

    lp_blurred_image_2d = lp.blurred_image_2d_from(
        grid=grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7, psf=psf_3x3
    )

    assert blurred_image_2d_manual.native == pytest.approx(
        lp_blurred_image_2d.native.array, 1.0e-4
    )


def test__blurred_image_2d_from__galaxy_with_operated_and_non_operated_profiles__non_operated_blurred_operated_unblurred(
    grid_2d_7x7,
    blurring_grid_2d_7x7,
    psf_3x3,
):
    light_not_operated = ag.lp.Sersic(intensity=1.0)
    light_operated = ag.lp_operated.Gaussian(intensity=1.0)

    image_2d_not_operated = light_not_operated.image_2d_from(grid=grid_2d_7x7)
    blurring_image_2d_not_operated = light_not_operated.image_2d_from(
        grid=blurring_grid_2d_7x7
    )
    image_2d_operated = light_operated.image_2d_from(grid=grid_2d_7x7)

    galaxy = ag.Galaxy(
        redshift=0.5, light=light_not_operated, light_operated=light_operated
    )

    blurred_image_2d = galaxy.blurred_image_2d_from(
        grid=grid_2d_7x7, psf=psf_3x3, blurring_grid=blurring_grid_2d_7x7
    )

    blurred_image_2d_manual_not_operated = psf_3x3.convolved_image_from(
        image=image_2d_not_operated,
        blurring_image=blurring_image_2d_not_operated,
    )

    assert (
        blurred_image_2d
        == pytest.approx(
            blurred_image_2d_manual_not_operated.array + image_2d_operated.array
        ),
        1.0e-4,
    )


def test__x1_galaxies__padded_image__compare_to_galaxy_images_using_padded_grid_stack(
    grid_2d_7x7,
):
    padded_grid = grid_2d_7x7.padded_grid_from(kernel_shape_native=(3, 3))

    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=1.0))
    g1 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=2.0))
    g2 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=3.0))

    padded_g0_image = g0.image_2d_from(grid=padded_grid)
    padded_g1_image = g1.image_2d_from(grid=padded_grid)
    padded_g2_image = g2.image_2d_from(grid=padded_grid)

    galaxies = ag.Galaxies(galaxies=[g0, g1, g2])

    padded_image = galaxies.padded_image_2d_from(grid=grid_2d_7x7, psf_shape_2d=(3, 3))

    assert padded_image.shape_native == (9, 9)
    assert padded_image == pytest.approx(
        padded_g0_image.array + padded_g1_image.array + padded_g2_image.array, 1.0e-4
    )


def test__unmasked_blurred_image_2d_from__single_light_profile__correct_unmasked_values():
    kernel = ag.Array2D.no_mask(
        values=(np.array([[0.0, 3.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])),
        pixel_scales=1.0,
    )

    psf = ag.Convolver(kernel=kernel)

    mask = ag.Mask2D(
        mask=[[True, True, True], [True, False, True], [True, True, True]],
        pixel_scales=1.0,
        origin=(0.3, 0.3),
    )

    grid = ag.Grid2D.from_mask(mask=mask, over_sample_size=1)

    lp = ag.lp.Sersic(intensity=0.1)

    unmasked_blurred_image_2d = lp.unmasked_blurred_image_2d_from(grid=grid, psf=psf)

    assert unmasked_blurred_image_2d.native == pytest.approx(
        np.array(
            [
                [0.21305245, 0.6141556, 0.10010613],
                [0.18998845, 0.50114327, 0.43959065],
                [0.07793524, 0.16386363, 0.15628968],
            ]
        ),
        1.0e-4,
    )


def test__unmasked_blurred_image_2d_from__galaxy_with_operated_and_non_operated__sum_matches_manual():
    kernel = ag.Array2D.no_mask(
        values=(np.array([[0.0, 3.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])),
        pixel_scales=1.0,
    )
    psf = ag.Convolver(kernel=kernel)

    mask = ag.Mask2D(
        mask=[[True, True, True], [True, False, True], [True, True, True]],
        pixel_scales=1.0,
    )

    grid = ag.Grid2D.from_mask(mask=mask)

    light_not_operated = ag.lp.Gaussian(intensity=1.0)
    light_operated = ag.lp_operated.Gaussian(intensity=1.0)

    padded_grid = grid.padded_grid_from(kernel_shape_native=psf.kernel.shape_native)

    image_2d_not_operated = light_not_operated.image_2d_from(grid=padded_grid)

    blurred_image_2d_not_operated = padded_grid.mask.unmasked_blurred_array_from(
        padded_array=image_2d_not_operated, psf=psf, image_shape=grid.mask.shape
    )

    image_2d_operated = light_operated.image_2d_from(grid=padded_grid)

    image_2d_operated = padded_grid.mask.unmasked_blurred_array_from(
        padded_array=image_2d_operated,
        psf=ag.Convolver.no_blur(pixel_scales=1.0),
        image_shape=grid.mask.shape,
    )

    image_2d_manual = image_2d_operated + blurred_image_2d_not_operated

    galaxy = ag.Galaxy(
        redshift=0.5, light=light_not_operated, light_operated=light_operated
    )

    unmasked_blurred_image_2d = galaxy.unmasked_blurred_image_2d_from(
        grid=grid, psf=psf
    )

    assert unmasked_blurred_image_2d.array == pytest.approx(
        image_2d_manual.array, 1.0e-4
    )


def test__visibilities_from_grid_and_transformer(grid_2d_7x7, transformer_7x7_7):
    lp = ag.lp.Sersic(intensity=1.0)
    lp_visibilities = lp.visibilities_from(
        grid=grid_2d_7x7, transformer=transformer_7x7_7
    )

    image_2d = lp.image_2d_from(grid=grid_2d_7x7)
    visibilities = transformer_7x7_7.visibilities_from(image=image_2d)

    assert visibilities.array == pytest.approx(lp_visibilities.array, 1.0e-4)


def test__blurred_image_2d_list_from__two_non_operated_profiles__each_profile_correctly_blurred(
    grid_2d_7x7,
    blurring_grid_2d_7x7,
    psf_3x3,
):
    lp_0 = ag.lp.Gaussian(intensity=1.0)
    lp_1 = ag.lp.Gaussian(intensity=2.0)

    lp_0_blurred_image_2d = lp_0.blurred_image_2d_from(
        grid=grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7, psf=psf_3x3
    )

    lp_1_blurred_image_2d = lp_1.blurred_image_2d_from(
        grid=grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7, psf=psf_3x3
    )

    gal = ag.Galaxy(redshift=0.5, lp_0=lp_0, lp_1=lp_1)

    blurred_image_2d_list = gal.blurred_image_2d_list_from(
        grid=grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7, psf=psf_3x3
    )

    assert blurred_image_2d_list[0].native == pytest.approx(
        lp_0_blurred_image_2d.native.array, 1.0e-4
    )
    assert blurred_image_2d_list[1].native == pytest.approx(
        lp_1_blurred_image_2d.native.array, 1.0e-4
    )


def test__blurred_image_2d_list_from__non_operated_and_operated_profiles__operated_not_blurred(
    grid_2d_7x7,
    blurring_grid_2d_7x7,
    psf_3x3,
):
    lp_0 = ag.lp.Gaussian(intensity=1.0)
    lp_operated = ag.lp_operated.Gaussian(intensity=3.0)

    lp_0_blurred_image_2d = lp_0.blurred_image_2d_from(
        grid=grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7, psf=psf_3x3
    )

    image_2d_operated = lp_operated.image_2d_from(grid=grid_2d_7x7)

    gal = ag.Galaxy(redshift=0.5, lp_0=lp_0, lp_operated=lp_operated)

    blurred_image_2d_list = gal.blurred_image_2d_list_from(
        grid=grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7, psf=psf_3x3
    )

    assert blurred_image_2d_list[0].native == pytest.approx(
        lp_0_blurred_image_2d.native.array, 1.0e-4
    )
    assert blurred_image_2d_list[1].native == pytest.approx(
        image_2d_operated.native.array, 1.0e-4
    )


def test__unmasked_blurred_image_2d_list_from():
    kernel = ag.Array2D.no_mask(
        values=(np.array([[0.0, 3.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])),
        pixel_scales=1.0,
    )
    psf = ag.Convolver(kernel=kernel)

    mask = ag.Mask2D(
        mask=[[True, True, True], [True, False, True], [True, True, True]],
        pixel_scales=1.0,
    )

    grid = ag.Grid2D.from_mask(mask=mask)

    lp_0 = ag.lp.Sersic(intensity=1.0)
    lp_1 = ag.lp.Sersic(intensity=2.0)

    padded_grid = grid.padded_grid_from(kernel_shape_native=psf.kernel.shape_native)

    manual_blurred_image_0 = lp_0.image_2d_from(grid=padded_grid)
    manual_blurred_image_0 = psf.convolved_image_from(
        image=manual_blurred_image_0, blurring_image=None
    )

    manual_blurred_image_1 = lp_1.image_2d_from(grid=padded_grid)
    manual_blurred_image_1 = psf.convolved_image_from(
        image=manual_blurred_image_1, blurring_image=None
    )

    gal = ag.Galaxy(redshift=0.5, lp_0=lp_0, lp_1=lp_1)

    unmasked_blurred_image_2d_list = gal.unmasked_blurred_image_2d_list_from(
        grid=grid, psf=psf
    )

    assert unmasked_blurred_image_2d_list[0].native.array == pytest.approx(
        manual_blurred_image_0.native.array[1:4, 1:4], 1.0e-4
    )

    assert unmasked_blurred_image_2d_list[1].native.array == pytest.approx(
        manual_blurred_image_1.native.array[1:4, 1:4], 1.0e-4
    )


def test__visibilities_list_from(grid_2d_7x7, transformer_7x7_7):
    lp_0 = ag.lp.Sersic(intensity=1.0)
    lp_1 = ag.lp.Sersic(intensity=2.0)

    lp_0_image = lp_0.image_2d_from(grid=grid_2d_7x7)
    lp_1_image = lp_1.image_2d_from(grid=grid_2d_7x7)

    lp_0_visibilities = transformer_7x7_7.visibilities_from(image=lp_0_image)
    lp_1_visibilities = transformer_7x7_7.visibilities_from(image=lp_1_image)

    gal = ag.Galaxy(redshift=0.5, lp_0=lp_0, lp_1=lp_1)

    visibilities_list = gal.visibilities_list_from(
        grid=grid_2d_7x7, transformer=transformer_7x7_7
    )

    assert (lp_0_visibilities == visibilities_list[0]).all()
    assert (lp_1_visibilities == visibilities_list[1]).all()


def test__galaxy_blurred_image_2d_dict_from(grid_2d_7x7, blurring_grid_2d_7x7, psf_3x3):
    lp_0 = ag.lp.Sersic(intensity=1.0)

    g0 = ag.Galaxy(redshift=0.5, light_profile=lp_0)
    g1 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=2.0))

    light_profile_operated = ag.lp_operated.Gaussian(intensity=3.0)

    g2 = ag.Galaxy(redshift=0.5, light_profile_operated=light_profile_operated)

    galaxies = ag.Galaxies(galaxies=[g1, g0, g2])

    blurred_image_2d_list = galaxies.blurred_image_2d_list_from(
        grid=grid_2d_7x7,
        psf=psf_3x3,
        blurring_grid=blurring_grid_2d_7x7,
    )

    blurred_image_dict = galaxies.galaxy_blurred_image_2d_dict_from(
        grid=grid_2d_7x7,
        psf=psf_3x3,
        blurring_grid=blurring_grid_2d_7x7,
    )

    assert blurred_image_dict[g0].array == pytest.approx(
        blurred_image_2d_list[1].array, 1.0e-4
    )
    assert blurred_image_dict[g1].array == pytest.approx(
        blurred_image_2d_list[0].array, 1.0e-4
    )
    assert blurred_image_dict[g2].array == pytest.approx(
        blurred_image_2d_list[2].array, 1.0e-4
    )

    image_2d = lp_0.image_2d_from(grid=grid_2d_7x7)
    blurring_image_2d = lp_0.image_2d_from(grid=blurring_grid_2d_7x7)

    image_2d_convolved = psf_3x3.convolved_image_from(
        image=image_2d, blurring_image=blurring_image_2d
    )

    assert (blurred_image_dict[g0] == image_2d_convolved).all()

    image_2d_operated = light_profile_operated.image_2d_from(grid=grid_2d_7x7)

    assert (blurred_image_dict[g2] == image_2d_operated).all()


def test__galaxy_visibilities_dict_from(grid_2d_7x7, transformer_7x7_7):
    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=1.0))
    g1 = ag.Galaxy(
        redshift=0.5,
        mass_profile=ag.mp.IsothermalSph(einstein_radius=1.0),
        light_profile=ag.lp.Sersic(intensity=2.0),
    )
    g2 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=3.0))

    galaxies = ag.Galaxies(galaxies=[g1, g0, g2])

    visibilities_list = galaxies.visibilities_list_from(
        grid=grid_2d_7x7, transformer=transformer_7x7_7
    )

    visibilities_dict = galaxies.galaxy_visibilities_dict_from(
        grid=grid_2d_7x7, transformer=transformer_7x7_7
    )

    assert (visibilities_dict[g0] == visibilities_list[1]).all()
    assert (visibilities_dict[g1] == visibilities_list[0]).all()
    assert (visibilities_dict[g2] == visibilities_list[2]).all()


def _oversampled_psf_scene():
    import autoarray as aa

    mask = aa.Mask2D.circular(shape_native=(11, 11), pixel_scales=1.0, radius=3.5)

    s = 2
    n = 9
    c = (np.arange(n) - (n - 1) / 2.0) * (1.0 / s)
    yy, xx = np.meshgrid(-c, c, indexing="ij")
    kernel = np.exp(-0.5 * (yy**2 + xx**2) / 0.8**2)
    kernel = aa.Array2D.no_mask(values=kernel / kernel.sum(), pixel_scales=1.0 / s)
    psf = aa.Convolver(kernel=kernel, convolve_over_sample_size=s)

    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=s)
    blurring_mask = mask.derive_mask.blurring_from(
        kernel_shape_native=psf.kernel_shape_image_resolution, allow_padding=True
    )
    blurring_grid = aa.Grid2D.from_mask(mask=blurring_mask, over_sample_size=s)

    return mask, psf, grid, blurring_grid


def test__blurred_image_2d_from__oversampled_psf__matches_direct_convolver_call():
    # The consumer switch evaluates on the over-sampled grids and hands the
    # sub-block ordered values to the oversampled Convolver — the result must
    # equal calling that (phase-2a tested) API directly.
    mask, psf, grid, blurring_grid = _oversampled_psf_scene()

    galaxy = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.Sersic(
            centre=(0.3, -0.4), intensity=1.0, effective_radius=1.0, sersic_index=2.0
        ),
    )
    galaxies = ag.Galaxies(galaxies=[galaxy])

    blurred = galaxies.blurred_image_2d_from(
        grid=grid, blurring_grid=blurring_grid, psf=psf
    )

    image_sub = galaxy.image_2d_from(grid=grid.over_sampled)
    blurring_sub = galaxy.image_2d_from(grid=blurring_grid.over_sampled)
    direct = psf.convolved_image_from(
        image=image_sub, blurring_image=blurring_sub, mask=mask
    )

    assert np.array(blurred) == pytest.approx(np.array(direct), abs=1.0e-14)

    # The result is at image resolution and differs from the binned-then-convolved
    # (s=1 style) computation — the oversampling is actually doing something.
    assert np.array(blurred).shape == (mask.pixels_in_mask,)


def test__blurred_image_2d_list_and_dict__oversampled_psf__match_scalar_path():
    mask, psf, grid, blurring_grid = _oversampled_psf_scene()

    galaxy_0 = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.Sersic(centre=(0.3, -0.4), intensity=1.0, effective_radius=1.0),
    )
    galaxy_1 = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.Gaussian(centre=(-0.5, 0.2), intensity=2.0, sigma=0.7),
    )
    galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1])

    blurred_list = galaxies.blurred_image_2d_list_from(
        grid=grid, blurring_grid=blurring_grid, psf=psf
    )
    blurred_dict = galaxies.galaxy_blurred_image_2d_dict_from(
        grid=grid, blurring_grid=blurring_grid, psf=psf
    )

    total_from_list = np.sum([np.array(b) for b in blurred_list], axis=0)
    blurred_total = galaxies.blurred_image_2d_from(
        grid=grid, blurring_grid=blurring_grid, psf=psf
    )

    assert total_from_list == pytest.approx(np.array(blurred_total), abs=1.0e-12)

    for galaxy, blurred_scalar in zip([galaxy_0, galaxy_1], blurred_list):
        assert np.array(blurred_dict[galaxy]) == pytest.approx(
            np.array(blurred_scalar), abs=1.0e-14
        )


def test__convolved_padded_image_2d_from__delta_kernel__equals_binned_padded_image():
    # With a delta fine kernel the fine convolution is the identity, so the
    # convolved padded image must equal the binned padded evaluation — testing
    # the padded-frame geometry and bin-down independently of PSF numerics.
    import autoarray as aa

    s = 2
    pixel_scales = 1.0

    grid = aa.Grid2D.uniform(
        shape_native=(7, 7), pixel_scales=pixel_scales, over_sample_size=s
    )

    delta = np.zeros((5, 5))
    delta[2, 2] = 1.0
    psf = aa.Convolver(
        kernel=aa.Array2D.no_mask(values=delta, pixel_scales=pixel_scales / s),
        convolve_over_sample_size=s,
    )

    galaxy = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.Sersic(centre=(0.2, -0.1), intensity=1.0, effective_radius=0.8),
    )
    galaxies = ag.Galaxies(galaxies=[galaxy])

    convolved_padded = galaxies.convolved_padded_image_2d_from(grid=grid, psf=psf)

    kernel_shape = psf.kernel_shape_image_resolution
    padded_shape = (7 + kernel_shape[0] - 1, 7 + kernel_shape[1] - 1)
    padded_mask = aa.Mask2D.all_false(
        shape_native=padded_shape, pixel_scales=pixel_scales, origin=grid.origin
    )
    padded_grid = aa.Grid2D.from_mask(mask=padded_mask, over_sample_size=s)

    image_sub = galaxy.image_2d_from(grid=padded_grid.over_sampled)
    binned = np.array(image_sub).reshape(-1, s**2).mean(axis=1)

    assert np.array(convolved_padded) == pytest.approx(binned, abs=1.0e-14)
