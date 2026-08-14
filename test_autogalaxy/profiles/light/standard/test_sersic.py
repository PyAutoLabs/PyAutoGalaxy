from __future__ import division, print_function
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__image_2d_from__sersic_index_4__intensity_1__correct_value():
    lp = ag.lp.Sersic(
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=4.0,
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert image == pytest.approx(0.351797, 1e-3)


def test__image_2d_from__sersic_index_2__intensity_3__correct_value():
    lp = ag.lp.Sersic(
        ell_comps=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[1.5, 0.0]]))

    assert image == pytest.approx(4.90657319276, 1e-3)


def test__image_2d_from__elliptical__ell_comps_nonzero__correct_value():
    lp = ag.lp.Sersic(
        ell_comps=(0.0, 0.333333),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert image == pytest.approx(5.38066670129, 1e-3)


def test__image_2d_from__spherical_profile__matches_elliptical_with_zero_ellipticity():
    elliptical = ag.lp.Sersic(
        ell_comps=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
    )

    spherical = ag.lp.SersicSph(intensity=3.0, effective_radius=2.0, sersic_index=2.0)

    image_elliptical = elliptical.image_2d_from(grid=grid)
    image_spherical = spherical.image_2d_from(grid=grid)

    assert image_elliptical.array == pytest.approx(image_spherical.array, 1.0e-4)

@pytest.mark.parametrize(
    "ell_comps",
    [(0.0, 0.0), (0.18, 0.22), (-0.31, 0.14), (0.0, 0.9995)],
)
def test__image_2d_from__cartesian_radii_match_transformed_reference_frame(
    ell_comps,
):
    profile = ag.lp.Sersic(
        centre=(0.07, -0.11),
        ell_comps=ell_comps,
        intensity=1.3,
        effective_radius=0.8,
        sersic_index=2.2,
    )
    grid = ag.Grid2DIrregular(
        [[0.13, -0.22], [0.51, 0.37], [-0.42, 0.81], [1.2, -0.63]]
    )

    transformed_grid = profile.transformed_to_reference_frame_grid_from(grid=grid)
    transformed_grid.is_transformed = True

    image = profile.image_2d_from(grid=grid)
    image_via_transformed_grid = profile.image_2d_from(grid=transformed_grid)

    assert image.array == pytest.approx(
        image_via_transformed_grid.array, rel=2.0e-12, abs=2.0e-12
    )

