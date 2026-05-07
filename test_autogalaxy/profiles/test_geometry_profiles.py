from __future__ import division, print_function

from pathlib import Path

import numpy as np
import pytest

import autogalaxy as ag
from autogalaxy.profiles import geometry_profiles

directory = Path(__file__).resolve().parent


def test__cos_and_sin_from_x__zero_ell_comps__cos_1_sin_0():
    elliptical_profile = geometry_profiles.EllProfile(
        centre=(1.0, 1.0), ell_comps=(0.0, 0.0)
    )

    _cos_angle, _sin_angle = elliptical_profile._cos_and_sin_to_x_axis()

    assert _cos_angle == 1.0
    assert _sin_angle == 0.0


def test__cos_and_sin_from_x__ell_comps_45_degrees__cos_and_sin_both_0p707():
    # axis_ratio=0.1, angle=45.0
    elliptical_profile = geometry_profiles.EllProfile(
        centre=(1, 1), ell_comps=(0.8181, 0.0)
    )

    _cos_angle, _sin_angle = elliptical_profile._cos_and_sin_to_x_axis()

    assert _cos_angle == pytest.approx(0.707, 1e-3)
    assert _sin_angle == pytest.approx(0.707, 1e-3)


def test__cos_and_sin_from_x__ell_comps_60_degrees__cos_0p5_sin_0p866():
    # axis_ratio=0.1, angle=60.0
    elliptical_profile = geometry_profiles.EllProfile(
        centre=(1, 1), ell_comps=(0.70856, -0.4090909)
    )

    _cos_angle, _sin_angle = elliptical_profile._cos_and_sin_to_x_axis()

    assert _cos_angle == pytest.approx(0.5, 1e-3)
    assert _sin_angle == pytest.approx(0.866, 1e-3)


def test__eccentric_radii_grid_from__circular_profile__returns_sqrt_2():
    elliptical_profile = geometry_profiles.EllProfile(ell_comps=(0.0, 0.0))

    eccentric_radius = elliptical_profile.eccentric_radii_grid_from(
        grid=ag.Grid2DIrregular([[1.0, 1.0]])
    )

    assert eccentric_radius == pytest.approx(2.0**0.5, 1e-3)


def test__eccentric_radii_grid_from__elliptical_profile__returns_correct_eccentric_radius():
    # eccentric_radius = sqrt(axis_ratio) * sqrt(  x**2 +   (y**2 / axis_ratio**2))
    # eccentric_radius =        sqrt(0.5) * sqrt(1.0**2 + (1.0**2 /        0.5**2))
    # eccentric radius =        sqrt(0.5) * sqrt( 5 ) = 1.58113
    # axis_ratio=0.5, angle=0.0
    elliptical_profile = geometry_profiles.EllProfile(ell_comps=(0.0, 0.333333))

    eccentric_radius = elliptical_profile.eccentric_radii_grid_from(
        ag.Grid2DIrregular([[1.0, 1.0]])
    )

    assert eccentric_radius == pytest.approx(1.58113, 1e-3)


def test__spherical__transform_to_reference_frame__zero_centre__grid_unchanged():
    spherical_profile = geometry_profiles.SphProfile(centre=(0.0, 0.0))

    transformed_grid = spherical_profile.transformed_from_reference_frame_grid_from(
        ag.Grid2DIrregular([[1.0, 1.0]])
    )

    assert (transformed_grid == ag.Grid2DIrregular([[1.0, 1.0]])).all()


def test__spherical__transform_to_reference_frame__equivalent_grid_shift_via_centre():
    spherical_profile1 = geometry_profiles.SphProfile(centre=(0, 0))

    grid1 = spherical_profile1.transformed_to_reference_frame_grid_from(
        grid=ag.Grid2DIrregular([[1.0, 1.0]])
    )

    spherical_profile2 = geometry_profiles.SphProfile(centre=(-1, -1))
    grid2 = spherical_profile2.transformed_to_reference_frame_grid_from(
        grid=ag.Grid2DIrregular([[0.0, 0.0]])
    )

    assert (grid1 == grid2).all()


def test__spherical__transform_to_reference_frame__two_centres_produce_equal_grids():
    spherical_profile1 = geometry_profiles.SphProfile(centre=(1, 1))
    grid1 = spherical_profile1.transformed_to_reference_frame_grid_from(
        grid=ag.Grid2DIrregular([[1.0, 1.0]])
    )

    spherical_profile2 = geometry_profiles.SphProfile(centre=(-1, -1))
    grid2 = spherical_profile2.transformed_to_reference_frame_grid_from(
        grid=ag.Grid2DIrregular([[-1.0, -1.0]])
    )

    assert (grid1 == grid2).all()


def test__spherical__transform_to_and_from_reference_frame__zero_centre__grid_unchanged():
    spherical_profile = geometry_profiles.SphProfile(centre=(0.0, 0.0))

    grid = ag.Grid2DIrregular([[1.0, 1.0]])

    transformed_grid = spherical_profile.transformed_to_reference_frame_grid_from(grid)

    assert transformed_grid == pytest.approx(ag.Grid2DIrregular([[1.0, 1.0]]), 1e-3)


def test__spherical__transform_to_and_from_reference_frame__roundtrip_recovers_original_grid():
    spherical_profile = geometry_profiles.SphProfile(centre=(0.0, 0.0))

    grid_original = ag.Grid2DIrregular([[5.2221, 2.6565]])

    grid_spherical = spherical_profile.transformed_to_reference_frame_grid_from(
        grid_original
    )

    transformed_grid = spherical_profile.transformed_from_reference_frame_grid_from(
        grid_spherical
    )

    assert transformed_grid == pytest.approx(grid_original, 1e-5)


def test__elliptical__transform_grid_to_reference_frame__zero_ell_comps__grid_unchanged():
    elliptical_profile = geometry_profiles.EllProfile(ell_comps=(0.0, 0.0))

    transformed_grid = elliptical_profile.transformed_to_reference_frame_grid_from(
        grid=ag.Grid2DIrregular([[1.0, 1.0]])
    )

    assert transformed_grid == pytest.approx(ag.Grid2DIrregular([[1.0, 1.0]]), 1e-3)


def test__elliptical__transform_grid_from_reference_frame__zero_ell_comps__grid_unchanged():
    elliptical_profile = geometry_profiles.EllProfile(ell_comps=(0.0, 0.0))

    transformed_grid = elliptical_profile.transformed_to_reference_frame_grid_from(
        grid=ag.Grid2DIrregular([[1.0, 1.0]])
    )

    transformed_back_grid = (
        elliptical_profile.transformed_from_reference_frame_grid_from(transformed_grid)
    )

    assert transformed_back_grid == pytest.approx(
        ag.Grid2DIrregular([[1.0, 1.0]]), 1e-3
    )


def test__elliptical__transform_grid_to_reference_frame__90_degree_angle__correct_transformed_grid():
    # axis_ratio=0.1, angle=90.0
    elliptical_profile = geometry_profiles.EllProfile(
        centre=(2.0, 3.0), ell_comps=(0.0, -0.81818181)
    )

    transformed_grid = elliptical_profile.transformed_to_reference_frame_grid_from(
        grid=ag.Grid2DIrregular([[3.0, 4.0]])
    )

    assert transformed_grid == pytest.approx(ag.Grid2DIrregular([[-1.0, 1.0]]), 1e-3)


def test__elliptical__transform_grid_from_reference_frame__90_degree_angle__recovers_original_grid():
    # axis_ratio=0.1, angle=90.0
    elliptical_profile = geometry_profiles.EllProfile(
        centre=(2.0, 3.0), ell_comps=(0.0, -0.81818181)
    )

    transformed_grid = elliptical_profile.transformed_to_reference_frame_grid_from(
        grid=ag.Grid2DIrregular([[3.0, 4.0]])
    )

    transformed_back_grid = (
        elliptical_profile.transformed_from_reference_frame_grid_from(transformed_grid)
    )

    assert transformed_back_grid == pytest.approx(
        ag.Grid2DIrregular([[3.0, 4.0]]), 1e-3
    )


def test__elliptical__transform_grid_to_and_from_reference_frame__roundtrip_recovers_original_grid():
    elliptical_profile = geometry_profiles.EllProfile(ell_comps=(0.818181, 0.0))

    grid_original = ag.Grid2DIrregular([[5.2221, 2.6565]])

    grid_elliptical = elliptical_profile.transformed_to_reference_frame_grid_from(
        grid_original
    )

    transformed_grid = elliptical_profile.transformed_from_reference_frame_grid_from(
        grid_elliptical
    )

    assert transformed_grid == pytest.approx(grid_original, 1e-5)


def test__elliptical__transform_grids_with_mapped_centres__zero_angle__equivalent_grid_shift_via_centre():
    elliptical_profile1 = geometry_profiles.EllProfile(
        ell_comps=(0.0, 0.0), centre=(0, 0)
    )
    grid1 = elliptical_profile1.transformed_to_reference_frame_grid_from(
        grid=ag.Grid2DIrregular([[1.0, 1.0]])
    )

    elliptical_profile2 = geometry_profiles.EllProfile(
        ell_comps=(0.0, 0.0), centre=(-1, -1)
    )
    grid2 = elliptical_profile2.transformed_to_reference_frame_grid_from(
        grid=ag.Grid2DIrregular([[0.0, 0.0]])
    )

    assert (grid1 == grid2).all()


def test__elliptical__transform_grids_with_mapped_centres__55_degree_angle__equivalent_grid_shift_via_centre():
    # axis_ratio=0.1, angle=55.0
    elliptical_profile1 = geometry_profiles.EllProfile(
        centre=(0, 0), ell_comps=(0.76883, -0.27983)
    )
    grid1 = elliptical_profile1.transformed_to_reference_frame_grid_from(
        grid=ag.Grid2DIrregular([[1.0, 1.0]])
    )

    elliptical_profile2 = geometry_profiles.EllProfile(
        centre=(-1, -1), ell_comps=(0.76883, -0.27983)
    )
    grid2 = elliptical_profile2.transformed_to_reference_frame_grid_from(
        grid=ag.Grid2DIrregular([[0.0, 0.0]])
    )

    assert (grid1 == grid2).all()


def test__elliptical__transform_grids_with_mapped_centres__55_degree_angle__two_centres_produce_equal_grids():
    elliptical_profile1 = geometry_profiles.EllProfile(
        centre=(1, 1), ell_comps=(0.76883, -0.27983)
    )
    grid1 = elliptical_profile1.transformed_to_reference_frame_grid_from(
        grid=ag.Grid2DIrregular([[1.0, 1.0]])
    )

    elliptical_profile2 = geometry_profiles.EllProfile(
        centre=(-1, -1), ell_comps=(0.76883, -0.27983)
    )
    grid2 = elliptical_profile2.transformed_to_reference_frame_grid_from(
        grid=ag.Grid2DIrregular([[-1.0, -1.0]])
    )

    assert (grid1 == grid2).all()
