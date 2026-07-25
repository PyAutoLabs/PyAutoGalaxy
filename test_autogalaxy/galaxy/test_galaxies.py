import numpy as np
import pytest
from skimage import measure

import autogalaxy as ag

from autoarray.inversion.mesh.mesh.abstract import AbstractMesh
from autoarray.inversion.regularization.abstract import AbstractRegularization

from autogalaxy.galaxy.galaxies import plane_image_from

from autogalaxy import exc


def test__image_2d_from__single_galaxy__matches_light_profile_image(
    grid_2d_7x7, gal_x1_lp
):
    light_profile = gal_x1_lp.cls_list_from(cls=ag.LightProfile)[0]

    lp_image = light_profile.image_2d_from(grid=grid_2d_7x7)

    galaxies = ag.Galaxies(galaxies=[gal_x1_lp])

    image = galaxies.image_2d_from(grid=grid_2d_7x7)

    assert (image == lp_image).all()


def test__image_2d_from__single_galaxy__matches_galaxy_image(grid_2d_7x7, gal_x1_lp):
    galaxy_image = gal_x1_lp.image_2d_from(grid=grid_2d_7x7)

    galaxies = ag.Galaxies(galaxies=[gal_x1_lp])

    image = galaxies.image_2d_from(grid=grid_2d_7x7)

    assert image == pytest.approx(galaxy_image.array, 1.0e-4)


def test__image_2d_from__two_galaxies__sum_matches_individual_galaxy_images(
    grid_2d_7x7,
):
    # Overwrite one value so intensity in each pixel is different
    grid_2d_7x7[5] = np.array([2.0, 2.0])

    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=1.0))
    g1 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=2.0))

    g0_image = g0.image_2d_from(grid=grid_2d_7x7)
    g1_image = g1.image_2d_from(grid=grid_2d_7x7)

    galaxies = ag.Galaxies(galaxies=[g0, g1])

    image = galaxies.image_2d_from(grid=grid_2d_7x7)

    assert image == pytest.approx(g0_image.array + g1_image.array, 1.0e-4)


def test__image_2d_list_from(grid_2d_7x7):
    # Overwrite one value so intensity in each pixel is different
    grid_2d_7x7[5] = np.array([2.0, 2.0])

    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=1.0))
    g1 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=2.0))

    lp0 = g0.cls_list_from(cls=ag.LightProfile)[0]
    lp1 = g1.cls_list_from(cls=ag.LightProfile)[0]

    lp0_image = lp0.image_2d_from(grid=grid_2d_7x7)
    lp1_image = lp1.image_2d_from(grid=grid_2d_7x7)

    galaxies = ag.Galaxies(galaxies=[g0, g1])

    image = galaxies.image_2d_from(grid=grid_2d_7x7)

    assert image[0] == pytest.approx(lp0_image.array[0] + lp1_image.array[0], 1.0e-4)
    assert image[1] == pytest.approx(lp0_image.array[1] + lp1_image.array[1], 1.0e-4)

    image_of_galaxies = galaxies.image_2d_list_from(grid=grid_2d_7x7)

    assert image_of_galaxies[0][0] == lp0_image[0]
    assert image_of_galaxies[0][1] == lp0_image[1]
    assert image_of_galaxies[1][0] == lp1_image[0]
    assert image_of_galaxies[1][1] == lp1_image[1]


def test__image_2d_from__operated_only_false__returns_only_non_operated_images(
    grid_2d_7x7, lp_0, lp_operated_0
):
    image_2d_not_operated = lp_0.image_2d_from(grid=grid_2d_7x7)

    galaxy_0 = ag.Galaxy(redshift=0.5, light=lp_0, light_operated=lp_operated_0)
    galaxy_1 = ag.Galaxy(
        redshift=1.0, light_operated_0=lp_operated_0, light_operated_1=lp_operated_0
    )
    galaxy_2 = ag.Galaxy(redshift=2.0)

    galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1, galaxy_2])

    image_2d = galaxies.image_2d_from(grid=grid_2d_7x7, operated_only=False)
    assert image_2d == pytest.approx(image_2d_not_operated.array, 1.0e-4)


def test__image_2d_from__operated_only_true__returns_only_operated_images_summed(
    grid_2d_7x7, lp_0, lp_operated_0
):
    image_2d_operated = lp_operated_0.image_2d_from(grid=grid_2d_7x7)

    galaxy_0 = ag.Galaxy(redshift=0.5, light=lp_0, light_operated=lp_operated_0)
    galaxy_1 = ag.Galaxy(
        redshift=1.0, light_operated_0=lp_operated_0, light_operated_1=lp_operated_0
    )
    galaxy_2 = ag.Galaxy(redshift=2.0)

    galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1, galaxy_2])

    image_2d = galaxies.image_2d_from(grid=grid_2d_7x7, operated_only=True)
    assert image_2d == pytest.approx(3.0 * image_2d_operated.array, 1.0e-4)


def test__image_2d_from__operated_only_none__returns_all_images_summed(
    grid_2d_7x7, lp_0, lp_operated_0
):
    image_2d_not_operated = lp_0.image_2d_from(grid=grid_2d_7x7)
    image_2d_operated = lp_operated_0.image_2d_from(grid=grid_2d_7x7)

    galaxy_0 = ag.Galaxy(redshift=0.5, light=lp_0, light_operated=lp_operated_0)
    galaxy_1 = ag.Galaxy(
        redshift=1.0, light_operated_0=lp_operated_0, light_operated_1=lp_operated_0
    )
    galaxy_2 = ag.Galaxy(redshift=2.0)

    galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1, galaxy_2])

    image_2d = galaxies.image_2d_from(grid=grid_2d_7x7, operated_only=None)
    assert image_2d == pytest.approx(
        image_2d_not_operated.array + 3.0 * image_2d_operated.array, 1.0e-4
    )


def test__image_2d_list_from__operated_only_false__non_operated_returned_operated_zeros(
    grid_2d_7x7, lp_0, lp_operated_0
):
    image_2d_not_operated = lp_0.image_2d_from(grid=grid_2d_7x7)

    galaxy_0 = ag.Galaxy(redshift=0.5, light=lp_0, light_operated=lp_operated_0)
    galaxy_1 = ag.Galaxy(
        redshift=1.0, light_operated_0=lp_operated_0, light_operated_1=lp_operated_0
    )
    galaxy_2 = ag.Galaxy(redshift=2.0)

    galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1, galaxy_2])

    image_2d_list = galaxies.image_2d_list_from(grid=grid_2d_7x7, operated_only=False)
    assert image_2d_list[0] == pytest.approx(image_2d_not_operated.array, 1.0e-4)
    assert image_2d_list[1] == pytest.approx(np.zeros((9)), 1.0e-4)
    assert image_2d_list[2] == pytest.approx(np.zeros((9)), 1.0e-4)


def test__image_2d_list_from__operated_only_true__operated_images_returned_non_operated_zeros(
    grid_2d_7x7, lp_0, lp_operated_0
):
    image_2d_operated = lp_operated_0.image_2d_from(grid=grid_2d_7x7)

    galaxy_0 = ag.Galaxy(redshift=0.5, light=lp_0, light_operated=lp_operated_0)
    galaxy_1 = ag.Galaxy(
        redshift=1.0, light_operated_0=lp_operated_0, light_operated_1=lp_operated_0
    )
    galaxy_2 = ag.Galaxy(redshift=2.0)

    galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1, galaxy_2])

    image_2d_list = galaxies.image_2d_list_from(grid=grid_2d_7x7, operated_only=True)
    assert image_2d_list[0] == pytest.approx(image_2d_operated.array, 1.0e-4)
    assert image_2d_list[1] == pytest.approx(2.0 * image_2d_operated.array, 1.0e-4)
    assert image_2d_list[2] == pytest.approx(np.zeros((9)), 1.0e-4)


def test__image_2d_list_from__operated_only_none__sum_of_list_matches_total_image(
    grid_2d_7x7, lp_0, lp_operated_0
):
    image_2d_not_operated = lp_0.image_2d_from(grid=grid_2d_7x7)
    image_2d_operated = lp_operated_0.image_2d_from(grid=grid_2d_7x7)

    galaxy_0 = ag.Galaxy(redshift=0.5, light=lp_0, light_operated=lp_operated_0)
    galaxy_1 = ag.Galaxy(
        redshift=1.0, light_operated_0=lp_operated_0, light_operated_1=lp_operated_0
    )
    galaxy_2 = ag.Galaxy(redshift=2.0)

    galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1, galaxy_2])

    image_2d_list = galaxies.image_2d_list_from(grid=grid_2d_7x7, operated_only=None)
    assert image_2d_list[0] + image_2d_list[1] == pytest.approx(
        image_2d_not_operated.array + 3.0 * image_2d_operated.array, 1.0e-4
    )


def test__galaxy_image_2d_dict_from__no_operated_only__all_galaxies_correct(
    grid_2d_7x7,
):
    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=1.0))
    g1 = ag.Galaxy(
        redshift=0.5,
        mass_profile=ag.mp.IsothermalSph(einstein_radius=1.0),
        light_profile=ag.lp.Sersic(intensity=2.0),
    )
    g2 = ag.Galaxy(redshift=0.5, light_profile=ag.lp_operated.Gaussian(intensity=3.0))

    g0_image = g0.image_2d_from(grid=grid_2d_7x7)
    g1_image = g1.image_2d_from(grid=grid_2d_7x7)
    g2_image = g2.image_2d_from(grid=grid_2d_7x7)

    galaxies = ag.Galaxies(galaxies=[g1, g0, g2])

    galaxy_image_2d_dict = galaxies.galaxy_image_2d_dict_from(grid=grid_2d_7x7)

    assert (galaxy_image_2d_dict[g0] == g0_image).all()
    assert (galaxy_image_2d_dict[g1] == g1_image).all()
    assert (galaxy_image_2d_dict[g2] == g2_image).all()


def test__galaxy_image_2d_dict_from__operated_only_true__non_operated_galaxies_return_zeros(
    grid_2d_7x7,
):
    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=1.0))
    g1 = ag.Galaxy(
        redshift=0.5,
        mass_profile=ag.mp.IsothermalSph(einstein_radius=1.0),
        light_profile=ag.lp.Sersic(intensity=2.0),
    )
    g2 = ag.Galaxy(redshift=0.5, light_profile=ag.lp_operated.Gaussian(intensity=3.0))

    g2_image = g2.image_2d_from(grid=grid_2d_7x7)

    galaxies = ag.Galaxies(galaxies=[g1, g0, g2])

    galaxy_image_2d_dict = galaxies.galaxy_image_2d_dict_from(
        grid=grid_2d_7x7, operated_only=True
    )

    assert (galaxy_image_2d_dict[g0] == np.zeros(shape=(9,))).all()
    assert (galaxy_image_2d_dict[g1] == np.zeros(shape=(9,))).all()
    assert (galaxy_image_2d_dict[g2] == g2_image).all()


def test__galaxy_image_2d_dict_from__operated_only_false__operated_galaxy_returns_zeros(
    grid_2d_7x7,
):
    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.Sersic(intensity=1.0))
    g1 = ag.Galaxy(
        redshift=0.5,
        mass_profile=ag.mp.IsothermalSph(einstein_radius=1.0),
        light_profile=ag.lp.Sersic(intensity=2.0),
    )
    g2 = ag.Galaxy(redshift=0.5, light_profile=ag.lp_operated.Gaussian(intensity=3.0))

    g0_image = g0.image_2d_from(grid=grid_2d_7x7)
    g1_image = g1.image_2d_from(grid=grid_2d_7x7)

    galaxies = ag.Galaxies(galaxies=[g1, g0, g2])

    galaxy_image_2d_dict = galaxies.galaxy_image_2d_dict_from(
        grid=grid_2d_7x7, operated_only=False
    )

    assert (galaxy_image_2d_dict[g0] == g0_image).all()
    assert (galaxy_image_2d_dict[g1] == g1_image).all()
    assert (galaxy_image_2d_dict[g2] == np.zeros(shape=(9,))).all()


def test__convergence_2d_from(grid_2d_7x7):
    g0 = ag.Galaxy(redshift=0.5, mass_profile=ag.mp.IsothermalSph(einstein_radius=1.0))
    g1 = ag.Galaxy(redshift=0.5, mass_profile=ag.mp.IsothermalSph(einstein_radius=2.0))

    g0_convergence = g0.convergence_2d_from(grid=grid_2d_7x7)
    g1_convergence = g1.convergence_2d_from(grid=grid_2d_7x7)

    galaxies = ag.Galaxies(galaxies=[g0, g1])

    convergence = galaxies.convergence_2d_from(grid=grid_2d_7x7)

    assert convergence.array == pytest.approx(
        g0_convergence.array + g1_convergence.array, 1.0e-8
    )


def test__potential_2d_from(grid_2d_7x7):
    g0 = ag.Galaxy(redshift=0.5, mass_profile=ag.mp.IsothermalSph(einstein_radius=1.0))
    g1 = ag.Galaxy(redshift=0.5, mass_profile=ag.mp.IsothermalSph(einstein_radius=2.0))

    g0_potential = g0.potential_2d_from(grid=grid_2d_7x7)
    g1_potential = g1.potential_2d_from(grid=grid_2d_7x7)

    galaxies = ag.Galaxies(galaxies=[g0, g1])

    potential = galaxies.potential_2d_from(grid=grid_2d_7x7)

    assert potential == pytest.approx(g0_potential.array + g1_potential.array, 1.0e-8)


def test__deflections_yx_2d_from(grid_2d_7x7):
    # Overwrite one value so intensity in each pixel is different
    grid_2d_7x7[5] = np.array([2.0, 2.0])

    g0 = ag.Galaxy(redshift=0.5, mass_profile=ag.mp.IsothermalSph(einstein_radius=1.0))
    g1 = ag.Galaxy(redshift=0.5, mass_profile=ag.mp.IsothermalSph(einstein_radius=2.0))

    g0_deflections = g0.deflections_yx_2d_from(grid=grid_2d_7x7)
    g1_deflections = g1.deflections_yx_2d_from(grid=grid_2d_7x7)

    galaxies = ag.Galaxies(galaxies=[g0, g1])

    deflections = galaxies.deflections_yx_2d_from(grid=grid_2d_7x7)

    assert deflections == pytest.approx(
        g0_deflections.array + g1_deflections.array, 1.0e-4
    )


def test__has__no_light_profiles__returns_false():
    galaxies = ag.Galaxies(galaxies=[ag.Galaxy(redshift=0.5)])
    assert galaxies.has(cls=ag.LightProfile) is False


def test__has__single_galaxy_with_light_profile__returns_true():
    galaxies = ag.Galaxies(
        galaxies=[ag.Galaxy(redshift=0.5, light_profile=ag.LightProfile())],
    )
    assert galaxies.has(cls=ag.LightProfile) is True


def test__has__mixed_galaxies_one_with_light_profile__returns_true():
    galaxies = ag.Galaxies(
        galaxies=[
            ag.Galaxy(redshift=0.5, light_profile=ag.LightProfile()),
            ag.Galaxy(redshift=0.5),
        ],
    )
    assert galaxies.has(cls=ag.LightProfile) is True


def test__cls_list_from__no_mass_profiles__returns_empty_list():
    galaxies = ag.Galaxies(galaxies=[ag.Galaxy(redshift=0.5)])

    assert galaxies.cls_list_from(cls=ag.mp.MassProfile) == []


def test__cls_list_from__single_galaxy_single_mass_profile__returns_profile():
    sis_0 = ag.mp.IsothermalSph(einstein_radius=1.0)

    galaxies = ag.Galaxies(galaxies=[ag.Galaxy(redshift=0.5, mass_profile=sis_0)])
    assert galaxies.cls_list_from(cls=ag.mp.MassProfile) == [sis_0]


def test__cls_list_from__two_galaxies_multiple_mass_profiles__returns_all_profiles():
    sis_0 = ag.mp.IsothermalSph(einstein_radius=1.0)
    sis_1 = ag.mp.IsothermalSph(einstein_radius=2.0)
    sis_2 = ag.mp.IsothermalSph(einstein_radius=3.0)

    galaxies = ag.Galaxies(
        galaxies=[
            ag.Galaxy(redshift=0.5, mass_profile_0=sis_0, mass_profile_1=sis_1),
            ag.Galaxy(redshift=0.5, mass_profile_0=sis_2, mass_profile_1=sis_1),
        ],
    )
    assert galaxies.cls_list_from(cls=ag.mp.MassProfile) == [sis_0, sis_1, sis_2, sis_1]


def test__cls_list_from__single_pixelization_galaxy__returns_pixelization():
    pixelization = ag.m.MockPixelization(mapper=1)

    galaxy_pix = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    galaxies = ag.Galaxies(galaxies=[galaxy_pix])

    assert galaxies.cls_list_from(cls=ag.Pixelization)[0].mapper == 1


def test__cls_list_from__two_pixelization_galaxies__returns_both_pixelizations():
    pixelization_0 = ag.m.MockPixelization(mapper=1)
    pixelization_1 = ag.m.MockPixelization(mapper=2)

    galaxy_pix_0 = ag.Galaxy(redshift=0.5, pixelization=pixelization_0)
    galaxy_pix_1 = ag.Galaxy(redshift=0.5, pixelization=pixelization_1)

    galaxies = ag.Galaxies(galaxies=[galaxy_pix_0, galaxy_pix_1])

    assert galaxies.cls_list_from(cls=ag.Pixelization)[0].mapper == 1
    assert galaxies.cls_list_from(cls=ag.Pixelization)[1].mapper == 2


def test__cls_list_from__no_pixelization_galaxy__returns_empty_list():
    galaxy_no_pix = ag.Galaxy(redshift=0.5)

    galaxies = ag.Galaxies(galaxies=[galaxy_no_pix])

    assert galaxies.cls_list_from(cls=ag.Pixelization) == []


def test__plane_image_from(grid_2d_7x7):
    galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=1.0))

    plane_image = plane_image_from(grid=grid_2d_7x7, galaxies=[galaxy], buffer=0.1)

    assert plane_image[0] == pytest.approx(12.5595, 1.0e-3)
