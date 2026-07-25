import numpy as np
import pytest

import autogalaxy as ag


def test__lp_linear_func_list_galaxy_dict__no_linear_profiles__returns_empty_dict(
    lp_0, masked_imaging_7x7
):
    to_inversion = ag.GalaxiesToInversion(
        galaxies=[ag.Galaxy(redshift=0.5)], dataset=masked_imaging_7x7
    )

    lp_linear_func_galaxy_dict = to_inversion.lp_linear_func_list_galaxy_dict

    assert lp_linear_func_galaxy_dict == {}


def test__lp_linear_func_list_galaxy_dict__multiple_galaxies_with_linear_profiles__maps_profiles_to_correct_galaxies(
    lp_0, masked_imaging_7x7
):
    lp_linear_0 = ag.lp_linear.LightProfileLinear()
    lp_linear_1 = ag.lp_linear.LightProfileLinear()
    lp_linear_2 = ag.lp_linear.LightProfileLinear()

    g0 = ag.Galaxy(
        redshift=0.5, lp_0=lp_0, light_linear_0=lp_linear_0, light_linear_1=lp_linear_1
    )
    g1 = ag.Galaxy(redshift=0.5, light_linear=lp_linear_2)

    to_inversion = ag.GalaxiesToInversion(galaxies=[g0, g1], dataset=masked_imaging_7x7)

    lp_linear_func_galaxy_dict = to_inversion.lp_linear_func_list_galaxy_dict

    lp_linear_func_list = list(lp_linear_func_galaxy_dict.keys())

    assert lp_linear_func_galaxy_dict[lp_linear_func_list[0]] == g0
    assert lp_linear_func_galaxy_dict[lp_linear_func_list[1]] == g0
    assert lp_linear_func_galaxy_dict[lp_linear_func_list[2]] == g1

    assert lp_linear_func_list[0].light_profile_list[0] == lp_linear_0
    assert lp_linear_func_list[1].light_profile_list[0] == lp_linear_1
    assert lp_linear_func_list[2].light_profile_list[0] == lp_linear_2


def test__lp_linear_func_list_galaxy_dict__basis_galaxy__groups_profiles_under_basis(
    masked_imaging_7x7
):
    lp_linear_0 = ag.lp_linear.LightProfileLinear()
    lp_linear_1 = ag.lp_linear.LightProfileLinear()
    lp_linear_2 = ag.lp_linear.LightProfileLinear()

    basis = ag.lp_basis.Basis(profile_list=[lp_linear_0, lp_linear_1])

    g0 = ag.Galaxy(redshift=0.5, bulge=basis)
    g1 = ag.Galaxy(redshift=0.5, light_linear=lp_linear_2)

    to_inversion = ag.GalaxiesToInversion(galaxies=[g0, g1], dataset=masked_imaging_7x7)

    lp_linear_func_galaxy_dict = to_inversion.lp_linear_func_list_galaxy_dict

    lp_linear_func_list = list(lp_linear_func_galaxy_dict.keys())

    assert lp_linear_func_galaxy_dict[lp_linear_func_list[0]] == g1
    assert lp_linear_func_galaxy_dict[lp_linear_func_list[1]] == g0

    assert lp_linear_func_list[0].light_profile_list[0] == lp_linear_2
    assert lp_linear_func_list[1].light_profile_list[0] == lp_linear_0
    assert lp_linear_func_list[1].light_profile_list[1] == lp_linear_1


def test__image_plane_mesh_grid_list__with_adapt_images__returns_adapt_mesh_grid(
    masked_imaging_7x7,
):
    pixelization = ag.m.MockPixelization()

    galaxy_pix = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    adapt_images = ag.AdaptImages(
        galaxy_image_dict={galaxy_pix: 2},
        galaxy_image_plane_mesh_grid_dict={galaxy_pix: 3},
    )

    to_inversion = ag.GalaxiesToInversion(
        galaxies=[galaxy_pix], dataset=masked_imaging_7x7, adapt_images=adapt_images
    )

    image_plane_mesh_grid_list = to_inversion.image_plane_mesh_grid_list

    assert image_plane_mesh_grid_list[0] == 3


def test__image_plane_mesh_grid_list__no_adapt_images__returns_none_in_list(
    masked_imaging_7x7,
):
    pixelization = ag.m.MockPixelization()

    galaxy_pix = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    to_inversion = ag.GalaxiesToInversion(
        galaxies=[galaxy_pix],
        dataset=masked_imaging_7x7,
    )

    image_plane_mesh_grid_list = to_inversion.image_plane_mesh_grid_list

    assert image_plane_mesh_grid_list[0] is None


def test__image_plane_mesh_grid_list__no_pixelization_galaxies__returns_none(
    masked_imaging_7x7,
):
    galaxy_no_pix = ag.Galaxy(redshift=0.5)

    to_inversion = ag.GalaxiesToInversion(
        galaxies=[galaxy_no_pix], dataset=masked_imaging_7x7
    )

    image_plane_mesh_grid_list = to_inversion.image_plane_mesh_grid_list

    assert image_plane_mesh_grid_list is None


def test__mapper_galaxy_dict__single_pixelization_galaxy__mapper_has_correct_pixels(
    masked_imaging_7x7,
):
    mesh = ag.mesh.RectangularUniform(shape=(3, 3))
    pixelization = ag.m.MockPixelization(mesh=mesh)

    galaxy_pix = ag.Galaxy(redshift=0.5, pixelization=pixelization)
    galaxy_no_pix = ag.Galaxy(redshift=0.5)

    to_inversion = ag.GalaxiesToInversion(
        galaxies=[galaxy_no_pix, galaxy_pix], dataset=masked_imaging_7x7
    )

    mapper_galaxy_dict = to_inversion.mapper_galaxy_dict

    mapper_list = list(mapper_galaxy_dict.keys())

    assert mapper_list[0].pixels == 9
    assert mapper_galaxy_dict[mapper_list[0]] == galaxy_pix


def test__mapper_galaxy_dict__two_pixelization_galaxies__both_mappers_correct(
    masked_imaging_7x7,
):
    mesh = ag.mesh.RectangularUniform(shape=(3, 3))
    pixelization = ag.m.MockPixelization(mesh=mesh)
    galaxy_pix = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    mesh_2 = ag.mesh.RectangularUniform(shape=(4, 3))
    pixelization_2 = ag.m.MockPixelization(mesh=mesh_2)
    galaxy_pix_2 = ag.Galaxy(redshift=0.5, pixelization=pixelization_2)

    galaxy_no_pix = ag.Galaxy(redshift=0.5)

    to_inversion = ag.GalaxiesToInversion(
        galaxies=[galaxy_no_pix, galaxy_pix, galaxy_no_pix, galaxy_pix_2],
        dataset=masked_imaging_7x7,
    )

    mapper_galaxy_dict = to_inversion.mapper_galaxy_dict

    mapper_list = list(mapper_galaxy_dict.keys())

    assert mapper_list[0].pixels == 9
    assert mapper_list[1].pixels == 12

    assert mapper_galaxy_dict[mapper_list[0]] == galaxy_pix
    assert mapper_galaxy_dict[mapper_list[1]] == galaxy_pix_2


def test__mapper_galaxy_dict__no_pixelization_galaxy__returns_empty_dict(
    masked_imaging_7x7,
):
    galaxy_no_pix = ag.Galaxy(redshift=0.5)

    to_inversion = ag.GalaxiesToInversion(
        galaxies=[galaxy_no_pix], dataset=masked_imaging_7x7
    )

    mapper_galaxy_dict = to_inversion.mapper_galaxy_dict

    assert mapper_galaxy_dict == {}


def test__inversion_imaging_from__linear_light_profile__correct_reconstruction(
    grid_2d_7x7, masked_imaging_7x7
):
    g_linear = ag.Galaxy(
        redshift=0.5, light_linear=ag.lp_linear.Sersic(centre=(0.05, 0.05))
    )

    to_inversion = ag.GalaxiesToInversion(
        dataset=masked_imaging_7x7,
        galaxies=[ag.Galaxy(redshift=0.5), g_linear],
    )

    inversion = to_inversion.inversion

    assert inversion.reconstruction[0] == pytest.approx(0.186868464426, 1.0e-2)


def test__inversion_imaging_from__rectangular_pixelization__reconstructed_data_matches_input(
    grid_2d_7x7, masked_imaging_7x7
):
    pixelization = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=0.0),
    )

    g0 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    to_inversion = ag.GalaxiesToInversion(
        dataset=masked_imaging_7x7,
        galaxies=[ag.Galaxy(redshift=0.5), g0],
    )

    inversion = to_inversion.inversion

    assert inversion.mapped_reconstructed_operated_data == pytest.approx(
        masked_imaging_7x7.data, 1.0e-2
    )


def test__inversion_interferometer_from__linear_light_profile__correct_reconstruction(
    grid_2d_7x7, interferometer_7
):
    g_linear = ag.Galaxy(
        redshift=0.5, light_linear=ag.lp_linear.Sersic(centre=(0.05, 0.05))
    )

    to_inversion = ag.GalaxiesToInversion(
        dataset=interferometer_7,
        galaxies=[ag.Galaxy(redshift=0.5), g_linear],
    )

    inversion = to_inversion.inversion

    assert inversion.reconstruction[0] == pytest.approx(0.04124846952, 1.0e-2)


def test__inversion_interferometer_from__rectangular_pixelization__reconstructed_visibilities_match_input(
    grid_2d_7x7, interferometer_7
):
    interferometer_7.data = ag.Visibilities.ones(shape_slim=(7,))

    pixelization = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(7, 7)),
        regularization=ag.reg.Constant(coefficient=0.0),
    )

    g0 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    to_inversion = ag.GalaxiesToInversion(
        dataset=interferometer_7,
        galaxies=[ag.Galaxy(redshift=0.5), g0],
    )

    inversion = to_inversion.inversion

    assert inversion.mapped_reconstructed_operated_data.real == pytest.approx(
        interferometer_7.data.real, 1.0e-2
    )


def test__raises_exception_if_noise_covariance_input(masked_imaging_covariance_7x7):
    with pytest.raises(ag.exc.InversionException):
        ag.GalaxiesToInversion(
            galaxies=[ag.Galaxy(redshift=0.5)], dataset=masked_imaging_covariance_7x7
        )
