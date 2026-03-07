import numpy as np
from os import path
import pytest

from autoconf.dictable import from_json, output_to_json
import autogalaxy as ag

from autogalaxy import exc


def test__cls_list_from(lp_0, lp_linear_0):
    gal = ag.Galaxy(redshift=0.5, light_0=lp_0)

    cls_list = gal.cls_list_from(cls=ag.LightProfile)

    assert cls_list == [lp_0]

    gal = ag.Galaxy(
        redshift=0.5, light_linear_0=lp_linear_0, light_linear_1=lp_linear_0
    )

    cls_list = gal.cls_list_from(
        cls=ag.LightProfile, cls_filtered=ag.lp_linear.LightProfileLinear
    )

    assert cls_list == []

    cls_list = gal.cls_list_from(cls=ag.LightProfile)

    assert cls_list == [lp_linear_0, lp_linear_0]

    cls_list = gal.cls_list_from(cls=ag.lp_linear.LightProfileLinear)

    assert cls_list == [lp_linear_0, lp_linear_0]


def test__image_2d_from(grid_2d_7x7, gal_x2_lp):
    lp_0_image = gal_x2_lp.light_profile_0.image_2d_from(grid=grid_2d_7x7)
    lp_1_image = gal_x2_lp.light_profile_1.image_2d_from(grid=grid_2d_7x7)

    lp_image = lp_0_image + lp_1_image

    gal_image = gal_x2_lp.image_2d_from(grid=grid_2d_7x7)

    assert gal_image == pytest.approx(lp_image.array, 1.0e-4)


def test__image_2d_from__operated_only_input(grid_2d_7x7, lp_0, lp_operated_0):
    image_2d_not_operated = lp_0.image_2d_from(grid=grid_2d_7x7)
    image_2d_operated = lp_operated_0.image_2d_from(grid=grid_2d_7x7)

    galaxy = ag.Galaxy(redshift=0.5, light=lp_0, light_operated=lp_operated_0)

    image_2d = galaxy.image_2d_from(grid=grid_2d_7x7, operated_only=False)
    assert (image_2d == image_2d_not_operated).all()

    image_2d = galaxy.image_2d_from(grid=grid_2d_7x7, operated_only=True)
    assert (image_2d == image_2d_operated).all()

    image_2d = galaxy.image_2d_from(grid=grid_2d_7x7, operated_only=None)
    assert (image_2d == image_2d_not_operated + image_2d_operated).all()


def test__image_2d_list_from__operated_only_input(grid_2d_7x7, lp_0, lp_operated_0):
    image_2d_not_operated = lp_0.image_2d_from(grid=grid_2d_7x7)
    image_2d_operated = lp_operated_0.image_2d_from(grid=grid_2d_7x7)

    galaxy = ag.Galaxy(redshift=0.5, light=lp_0, light_operated=lp_operated_0)

    image_2d_list = galaxy.image_2d_list_from(grid=grid_2d_7x7, operated_only=False)
    assert (image_2d_list[0] == image_2d_not_operated).all()
    assert (image_2d_list[1] == np.zeros((9))).all()

    image_2d_list = galaxy.image_2d_list_from(grid=grid_2d_7x7, operated_only=True)
    assert (image_2d_list[0] == np.zeros((9))).all()
    assert (image_2d_list[1] == image_2d_operated).all()

    image_2d_list = galaxy.image_2d_list_from(grid=grid_2d_7x7, operated_only=None)
    assert (
        image_2d_list[0] + image_2d_list[1] == image_2d_not_operated + image_2d_operated
    ).all()


def test__luminosity_within_circle(lp_0, lp_1, gal_x2_lp):
    lp_0_luminosity = lp_0.luminosity_within_circle_from(radius=0.5)

    lp_1_luminosity = lp_1.luminosity_within_circle_from(radius=0.5)
    gal_luminosity = gal_x2_lp.luminosity_within_circle_from(radius=0.5)

    assert lp_0_luminosity + lp_1_luminosity == gal_luminosity

    gal_no_lp = ag.Galaxy(redshift=0.5, mass=ag.mp.IsothermalSph())

    assert gal_no_lp.luminosity_within_circle_from(radius=1.0) == None


def test__convergence_2d_from(grid_2d_7x7, mp_0, gal_x1_mp, mp_1, gal_x2_mp):
    mp_0_convergence = gal_x2_mp.mass_profile_0.convergence_2d_from(grid=grid_2d_7x7)

    mp_1_convergence = gal_x2_mp.mass_profile_1.convergence_2d_from(grid=grid_2d_7x7)

    mp_convergence = mp_0_convergence + mp_1_convergence

    gal_convergence = gal_x2_mp.convergence_2d_from(grid=grid_2d_7x7)

    assert gal_convergence[0] == mp_convergence[0]


def test__potential_2d_from(grid_2d_7x7, gal_x2_mp):
    mp_0_potential = gal_x2_mp.mass_profile_0.potential_2d_from(grid=grid_2d_7x7)

    mp_1_potential = gal_x2_mp.mass_profile_1.potential_2d_from(grid=grid_2d_7x7)

    mp_potential = mp_0_potential + mp_1_potential

    gal_potential = gal_x2_mp.potential_2d_from(grid=grid_2d_7x7)

    assert gal_potential[0] == mp_potential[0]


def test__deflections_yx_2d_from(grid_2d_7x7, gal_x2_mp):
    mp_0_deflections = gal_x2_mp.mass_profile_0.deflections_yx_2d_from(grid=grid_2d_7x7)

    mp_1_deflections = gal_x2_mp.mass_profile_1.deflections_yx_2d_from(grid=grid_2d_7x7)

    mp_deflections = mp_0_deflections + mp_1_deflections

    gal_deflections = gal_x2_mp.deflections_yx_2d_from(grid=grid_2d_7x7)

    assert gal_deflections[0, 0] == mp_deflections[0, 0]
    assert gal_deflections[1, 0] == mp_deflections[1, 0]


def test__no_mass_profile__quantities_returned_as_0s_of_shape_grid(
    grid_2d_7x7, mp_0, gal_x1_mp, mp_1, gal_x2_mp
):
    galaxy = ag.Galaxy(redshift=0.5)

    potential = galaxy.potential_2d_from(grid=grid_2d_7x7)

    assert (potential.slim == np.zeros(shape=grid_2d_7x7.shape_slim)).all()

    deflections = galaxy.deflections_yx_2d_from(grid=grid_2d_7x7)

    assert (deflections.slim == np.zeros(shape=(grid_2d_7x7.shape_slim, 2))).all()
    assert (deflections.native == np.zeros(shape=(7, 7, 2))).all()


def test__mass_angular_within_circle_from(mp_0, mp_1, gal_x2_mp):
    mp_0_mass = mp_0.mass_angular_within_circle_from(radius=0.5)

    mp_1_mass = mp_1.mass_angular_within_circle_from(radius=0.5)
    gal_mass = gal_x2_mp.mass_angular_within_circle_from(radius=0.5)

    assert mp_0_mass + mp_1_mass == gal_mass

    gal_no_mp = ag.Galaxy(redshift=0.5, light=ag.lp.SersicSph())

    with pytest.raises(exc.GalaxyException):
        gal_no_mp.mass_angular_within_circle_from(radius=1.0)


def test__light_and_mass_profiles__contained_in_light_and_mass_profile_lists(
    lmp_0, lp_0, lp_1, mp_0
):
    gal_x1_lmp = ag.Galaxy(redshift=0.5, profile=lmp_0)

    assert 1 == len(gal_x1_lmp.cls_list_from(cls=ag.LightProfile))
    assert 1 == len(gal_x1_lmp.cls_list_from(cls=ag.mp.MassProfile))

    assert gal_x1_lmp.cls_list_from(cls=ag.mp.MassProfile)[0] == lmp_0
    assert gal_x1_lmp.cls_list_from(cls=ag.LightProfile)[0] == lmp_0

    gal_multi_profiles = ag.Galaxy(redshift=0.5, profile=lmp_0, light=lp_0, sie=mp_0)

    assert 2 == len(gal_multi_profiles.cls_list_from(cls=ag.LightProfile))
    assert 2 == len(gal_multi_profiles.cls_list_from(cls=ag.mp.MassProfile))


def test__extract_attribute():
    galaxy = ag.Galaxy(redshift=0.5)

    values = galaxy.extract_attribute(cls=ag.LightProfile, attr_name="value")

    assert values == None

    galaxy = ag.Galaxy(
        redshift=0.5,
        lp_0=ag.m.MockLightProfile(value=0.9, value1=(0.0, 1.0)),
        lp_1=ag.m.MockLightProfile(value=0.8, value1=(2.0, 3.0)),
        lp_2=ag.m.MockLightProfile(value=0.7, value1=(4.0, 5.0)),
    )

    values = galaxy.extract_attribute(cls=ag.LightProfile, attr_name="value")

    assert values.in_list == [0.9, 0.8, 0.7]

    values = galaxy.extract_attribute(cls=ag.LightProfile, attr_name="value1")

    assert values.in_list == [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]

    galaxy = ag.Galaxy(
        redshift=0.5,
        lp_3=ag.LightProfile(),
        lp_0=ag.m.MockLightProfile(value=1.0),
        lp_1=ag.m.MockLightProfile(value=2.0),
        mp_0=ag.m.MockMassProfile(value=5.0),
        lp_2=ag.m.MockLightProfile(value=3.0),
    )

    values = galaxy.extract_attribute(cls=ag.LightProfile, attr_name="value")

    assert values.in_list == [1.0, 2.0, 3.0]


def test__image_2d_from__does_not_include_linear_light_profiles(grid_2d_7x7, lp_0):
    lp_linear = ag.lp_linear.Sersic(effective_radius=2.0, sersic_index=2.0)

    galaxy = ag.Galaxy(redshift=0.5, light_0=lp_0, light_linear=lp_linear)

    lp_image = lp_0.image_2d_from(grid=grid_2d_7x7)

    image = galaxy.image_2d_from(grid=grid_2d_7x7)

    assert (image == lp_image).all()


def test__light_profile_2d_quantity_from_grid__symmetric_profiles_give_symmetric_results():
    lp_0 = ag.lp.Sersic(centre=(0.0, 0.0), intensity=1.0)

    lp_1 = ag.lp.Sersic(centre=(100.0, 0.0), intensity=1.0)

    gal_x2_lp = ag.Galaxy(redshift=0.5, light_profile_0=lp_0, light_profile_1=lp_1)

    assert gal_x2_lp.image_2d_from(
        grid=ag.Grid2DIrregular([[0.0, 0.0]])
    ) == pytest.approx(
        gal_x2_lp.image_2d_from(grid=ag.Grid2DIrregular([[100.0, 0.0]])).array, 1.0e-4
    )

    assert gal_x2_lp.image_2d_from(
        grid=ag.Grid2DIrregular([[49.0, 0.0]])
    ) == pytest.approx(
        gal_x2_lp.image_2d_from(grid=ag.Grid2DIrregular([[51.0, 0.0]])).array, 1.0e-4
    )

    lp_0 = ag.lp.Sersic(
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=4.0,
    )

    lp_1 = ag.lp.Sersic(
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=4.0,
        centre=(100, 0),
    )

    lp_2 = ag.lp.Sersic(
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=4.0,
        centre=(0, 100),
    )

    lp_3 = ag.lp.Sersic(
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=4.0,
        centre=(100, 100),
    )

    gal_x4_lp = ag.Galaxy(
        redshift=0.5,
        light_profile_0=lp_0,
        light_profile_1=lp_1,
        light_profile_3=lp_2,
        light_profile_4=lp_3,
    )

    assert gal_x4_lp.image_2d_from(
        grid=ag.Grid2DIrregular([[49.0, 0.0]])
    ) == pytest.approx(
        gal_x4_lp.image_2d_from(grid=ag.Grid2DIrregular([[51.0, 0.0]])).array, 1e-5
    )

    assert gal_x4_lp.image_2d_from(
        grid=ag.Grid2DIrregular([[0.0, 49.0]])
    ) == pytest.approx(
        gal_x4_lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 51.0]])).array, 1e-5
    )

    assert gal_x4_lp.image_2d_from(
        grid=ag.Grid2DIrregular([[100.0, 49.0]])
    ) == pytest.approx(
        gal_x4_lp.image_2d_from(grid=ag.Grid2DIrregular([[100.0, 51.0]])).array, 1e-5
    )

    assert gal_x4_lp.image_2d_from(
        grid=ag.Grid2DIrregular([[49.0, 49.0]])
    ) == pytest.approx(
        gal_x4_lp.image_2d_from(grid=ag.Grid2DIrregular([[51.0, 51.0]])).array, 1e-5
    )


def test__mass_profile_2d_quantity_from_grid__symmetric_profiles_give_symmetric_results():
    mp_0 = ag.mp.Isothermal(ell_comps=(0.333333, 0.0), einstein_radius=1.0)

    mp_1 = ag.mp.Isothermal(
        centre=(100, 0), ell_comps=(0.333333, 0.0), einstein_radius=1.0
    )

    gal_x4_mp = ag.Galaxy(redshift=0.5, mass_profile_0=mp_0, mass_profile_1=mp_1)

    assert gal_x4_mp.convergence_2d_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    ) == pytest.approx(
        gal_x4_mp.convergence_2d_from(grid=ag.Grid2DIrregular([[99.0, 0.0]])).array,
        1.0e-4,
    )

    assert gal_x4_mp.convergence_2d_from(
        grid=ag.Grid2DIrregular([[49.0, 0.0]])
    ) == pytest.approx(
        gal_x4_mp.convergence_2d_from(grid=ag.Grid2DIrregular([[51.0, 0.0]])).array,
        1.0e-4,
    )

    assert gal_x4_mp.potential_2d_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    ) == pytest.approx(
        gal_x4_mp.potential_2d_from(grid=ag.Grid2DIrregular([[99.0, 0.0]])).array, 1e-6
    )

    assert gal_x4_mp.potential_2d_from(
        grid=ag.Grid2DIrregular([[49.0, 0.0]])
    ) == pytest.approx(
        gal_x4_mp.potential_2d_from(grid=ag.Grid2DIrregular([[51.0, 0.0]])).array, 1e-6
    )

    assert gal_x4_mp.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    ) == pytest.approx(
        gal_x4_mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[99.0, 0.0]])).array,
        abs=1e-6,
    )

    assert gal_x4_mp.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[49.0, 0.0]])
    ) == pytest.approx(
        gal_x4_mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[51.0, 0.0]])).array,
        1e-6,
    )

    mp_0 = ag.mp.IsothermalSph(einstein_radius=1.0)

    mp_1 = ag.mp.IsothermalSph(centre=(100, 0), einstein_radius=1.0)

    mp_2 = ag.mp.IsothermalSph(centre=(0, 100), einstein_radius=1.0)

    mp_3 = ag.mp.IsothermalSph(centre=(100, 100), einstein_radius=1.0)

    gal_x4_mp = ag.Galaxy(
        redshift=0.5,
        mass_profile_0=mp_0,
        mass_profile_1=mp_1,
        mass_profile_2=mp_2,
        mass_profile_3=mp_3,
    )

    assert gal_x4_mp.convergence_2d_from(
        grid=ag.Grid2DIrregular([[49.0, 0.0]])
    ) == pytest.approx(
        gal_x4_mp.convergence_2d_from(grid=ag.Grid2DIrregular([[51.0, 0.0]])).array,
        1e-5,
    )

    assert gal_x4_mp.convergence_2d_from(
        grid=ag.Grid2DIrregular([[0.0, 49.0]])
    ) == pytest.approx(
        gal_x4_mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 51.0]])).array,
        1e-5,
    )

    assert gal_x4_mp.convergence_2d_from(
        grid=ag.Grid2DIrregular([[100.0, 49.0]])
    ) == pytest.approx(
        gal_x4_mp.convergence_2d_from(grid=ag.Grid2DIrregular([[100.0, 51.0]])).array,
        1e-5,
    )

    assert gal_x4_mp.convergence_2d_from(
        grid=ag.Grid2DIrregular([[49.0, 49.0]])
    ) == pytest.approx(
        gal_x4_mp.convergence_2d_from(grid=ag.Grid2DIrregular([[51.0, 51.0]])).array,
        1e-5,
    )

    assert gal_x4_mp.potential_2d_from(
        grid=ag.Grid2DIrregular([[49.0, 0.0]])
    ) == pytest.approx(
        gal_x4_mp.potential_2d_from(grid=ag.Grid2DIrregular([[51.0, 0.0]])).array, 1e-5
    )

    assert gal_x4_mp.potential_2d_from(
        grid=ag.Grid2DIrregular([[0.0, 49.0]])
    ) == pytest.approx(
        gal_x4_mp.potential_2d_from(grid=ag.Grid2DIrregular([[0.0, 51.0]])).array, 1e-5
    )

    assert gal_x4_mp.potential_2d_from(
        grid=ag.Grid2DIrregular([[100.0, 49.0]])
    ) == pytest.approx(
        gal_x4_mp.potential_2d_from(grid=ag.Grid2DIrregular([[100.0, 51.0]])).array,
        1e-5,
    )

    assert gal_x4_mp.potential_2d_from(
        grid=ag.Grid2DIrregular([[49.0, 49.0]])
    ) == pytest.approx(
        gal_x4_mp.potential_2d_from(grid=ag.Grid2DIrregular([[51.0, 51.0]])).array, 1e-5
    )

    assert -1.0 * gal_x4_mp.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[49.0, 0.0]])
    )[0, 0] == pytest.approx(
        gal_x4_mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[51.0, 0.0]])).array[
            0, 0
        ],
        1e-5,
    )

    assert 1.0 * gal_x4_mp.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.0, 49.0]])
    )[0, 0] == pytest.approx(
        gal_x4_mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.0, 51.0]])).array[
            0, 0
        ],
        1e-5,
    )

    assert 1.0 * gal_x4_mp.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[100.0, 49.0]])
    )[0, 0] == pytest.approx(
        gal_x4_mp.deflections_yx_2d_from(
            grid=ag.Grid2DIrregular([[100.0, 51.0]])
        ).array[0, 0],
        1e-5,
    )

    assert -1.0 * gal_x4_mp.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[49.0, 49.0]])
    )[0, 0] == pytest.approx(
        gal_x4_mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[51.0, 51.0]])).array[
            0, 0
        ],
        1e-5,
    )

    assert 1.0 * gal_x4_mp.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[49.0, 0.0]])
    )[0, 1] == pytest.approx(
        gal_x4_mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[51.0, 0.0]])).array[
            0, 1
        ],
        1e-5,
    )

    assert -1.0 * gal_x4_mp.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.0, 49.0]])
    )[0, 1] == pytest.approx(
        gal_x4_mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.0, 51.0]])).array[
            0, 1
        ],
        1e-5,
    )

    assert -1.0 * gal_x4_mp.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[100.0, 49.0]])
    )[0, 1] == pytest.approx(
        gal_x4_mp.deflections_yx_2d_from(
            grid=ag.Grid2DIrregular([[100.0, 51.0]])
        ).array[0, 1],
        1e-5,
    )

    assert -1.0 * gal_x4_mp.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[49.0, 49.0]])
    )[0, 1] == pytest.approx(
        gal_x4_mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[51.0, 51.0]])).array[
            0, 1
        ],
        1e-5,
    )


def test__centre_of_profile_in_right_place():
    grid = ag.Grid2D.uniform(shape_native=(7, 7), pixel_scales=1.0)

    galaxy = ag.Galaxy(
        redshift=0.5,
        mass=ag.mp.Isothermal(centre=(1.99999, 0.99999), einstein_radius=1.0),
        mass_0=ag.mp.Isothermal(centre=(1.99999, 0.99999), einstein_radius=1.0),
    )

    convergence = galaxy.convergence_2d_from(grid=grid)
    max_indexes = np.unravel_index(
        convergence.native.argmax(), convergence.shape_native
    )
    assert max_indexes == (1, 4)

    potential = galaxy.potential_2d_from(grid=grid)
    max_indexes = np.unravel_index(potential.native.argmin(), potential.shape_native)
    assert max_indexes == (1, 4)

    deflections = galaxy.deflections_yx_2d_from(grid=grid)
    assert deflections.native[1, 4, 0] > 0
    assert deflections.native[2, 4, 0] < 0
    assert deflections.native[1, 4, 1] > 0
    assert deflections.native[1, 3, 1] < 0

    galaxy = ag.Galaxy(
        redshift=0.5,
        mass=ag.mp.IsothermalSph(centre=(1.99999, 0.99999), einstein_radius=1.0),
        mass_0=ag.mp.IsothermalSph(centre=(1.99999, 0.99999), einstein_radius=1.0),
    )
    convergence = galaxy.convergence_2d_from(grid=grid)
    max_indexes = np.unravel_index(
        convergence.native.argmax(), convergence.shape_native
    )
    assert max_indexes == (1, 4)

    potential = galaxy.potential_2d_from(grid=grid)
    max_indexes = np.unravel_index(potential.native.argmin(), potential.shape_native)
    assert max_indexes == (1, 4)

    deflections = galaxy.deflections_yx_2d_from(grid=grid)
    assert deflections.native[1, 4, 0] > 0
    assert deflections.native[2, 4, 0] < 0
    assert deflections.native[1, 4, 1] > 0
    assert deflections.native[1, 3, 1] < 0


def test__cannot_pass_light_or_mass_list():
    light_list = [ag.lp.Sersic(), ag.lp.Sersic()]

    with pytest.raises(exc.GalaxyException):
        ag.Galaxy(redshift=0.5, light=light_list)

    mass_list = [ag.mp.Sersic(), ag.mp.Sersic()]

    with pytest.raises(exc.GalaxyException):
        ag.Galaxy(redshift=0.5, mass=mass_list)

    with pytest.raises(exc.GalaxyException):
        ag.Galaxy(redshift=0.5, light=light_list, mass=mass_list)


def test__decorator__grid_sub_in__numericas():
    mask = ag.Mask2D(
        mask=[
            [True, True, True, True, True],
            [True, False, False, True, True],
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
        sub_size=2,
    )

    grid = ag.Grid2D.from_mask(mask=mask)

    galaxy = ag.Galaxy(
        redshift=0.5, light=ag.lp.Sersic(centre=(3.0, 3.0), intensity=1.0)
    )

    image = galaxy.image_2d_from(grid=grid)


def test__decorator__oversample_uniform__numerical_values(gal_x1_lp):
    mask = ag.Mask2D(
        mask=[
            [True, True, True, True, True],
            [True, False, False, True, True],
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, True, True, True],
        ],
        pixel_scales=(1.0, 1.0),
    )

    galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=1.0))

    grid = ag.Grid2D.from_mask(mask=mask, over_sample_size=1)

    image = galaxy.image_2d_from(grid=grid)

    assert image[0] == pytest.approx(0.15987224303572964, 1.0e-6)

    grid = ag.Grid2D.from_mask(mask=mask, over_sample_size=2)

    image = galaxy.image_2d_from(grid=grid)

    assert image[0] == pytest.approx(0.17481917162057087, 1.0e-6)
    assert image[1] == pytest.approx(0.391168560508937, 1.0e-6)

    galaxy = ag.Galaxy(
        redshift=0.5, light=ag.lp.Sersic(centre=(3.0, 3.0), intensity=1.0)
    )

    grid = ag.Grid2D.from_mask(mask=mask, over_sample_size=1)

    image = galaxy.image_2d_from(grid=grid)

    assert image[0] == pytest.approx(0.006719704400094508, 1.0e-6)

    grid = ag.Grid2D.from_mask(mask=mask, over_sample_size=2)

    image = galaxy.image_2d_from(grid=grid)

    assert image[0] == pytest.approx(0.006817908632814734, 1.0e-6)
    assert image[1] == pytest.approx(0.013323319136547789, 1.0e-6)


def test__output_to_and_load_from_json():
    json_file = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "galaxy.json"
    )

    g0 = ag.Galaxy(
        redshift=1.0,
        light=ag.lp.Sersic(intensity=1.0),
        mass=ag.mp.Isothermal(einstein_radius=1.0),
        pixelization=ag.Pixelization(
            mesh=ag.mesh.Delaunay(pixels=9),
            regularization=ag.reg.Constant(),
        ),
    )

    output_to_json(g0, file_path=json_file)

    galaxy_from_json = from_json(file_path=json_file)

    assert galaxy_from_json.redshift == 1.0
    assert galaxy_from_json.light.intensity == 1.0
    assert galaxy_from_json.mass.einstein_radius == 1.0
