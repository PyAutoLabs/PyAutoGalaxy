from __future__ import division, print_function
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__image_2d_from__effective_radius_1__grid_at_1__correct_value():
    lp = ag.lp.ElsonFreeFall(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert image == pytest.approx(0.35355, 1e-2)


def test__image_2d_from__intensity_2__scales_linearly_with_intensity():
    lp = ag.lp.ElsonFreeFall(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=2.0,
        effective_radius=1.0,
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert image == pytest.approx(2.0 * 0.35355, 1e-2)


def test__image_2d_from__effective_radius_2__grid_at_1__correct_value():
    lp = ag.lp.ElsonFreeFall(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=2.0,
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert image == pytest.approx(0.71554, 1e-2)


def test__image_2d_from__effective_radius_2__grid_at_3__correct_value():
    lp = ag.lp.ElsonFreeFall(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=2.0,
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 3.0]]))

    assert image == pytest.approx(0.17067, 1e-2)


def test__image_2d_from__spherical_profile__matches_elliptical_with_zero_ellipticity():
    elliptical = ag.lp.ElsonFreeFall(
        ell_comps=(0.0, 0.0), intensity=3.0, effective_radius=2.0
    )
    spherical = ag.lp.ElsonFreeFallSph(intensity=3.0, effective_radius=2.0)

    image_elliptical = elliptical.image_2d_from(grid=grid)
    image_spherical = spherical.image_2d_from(grid=grid)

    assert image_elliptical.array == pytest.approx(image_spherical.array, 1.0e-4)


def test__half_light_radius__eta_4__correct_value():
    lp = ag.lp.ElsonFreeFall(effective_radius=2.0, eta=4.0)

    assert lp.half_light_radius == pytest.approx(1.01964, 1e-2)
