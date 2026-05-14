import numpy as np
import pytest

import autogalaxy as ag


def test__circular_radius():
    ellipse = ag.Ellipse(major_axis=1.0)

    assert ellipse.circular_radius == pytest.approx(6.283185307, 1.0e-4)

    ellipse = ag.Ellipse(major_axis=0.5)

    assert ellipse.circular_radius == pytest.approx(3.141592654, 1.0e-4)


def test__ellipticity():
    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0))

    assert ellipse.ellipticity == 0.0

    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.5))

    assert ellipse.ellipticity == pytest.approx(0.942809041, 1.0e-4)

    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.5, 0.0))

    assert ellipse.ellipticity == pytest.approx(0.942809041, 1.0e-4)

    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.5, 0.5))

    assert ellipse.ellipticity == pytest.approx(0.9851714, 1.0e-4)


def test__minor_axis():
    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    assert ellipse.minor_axis == 1.0

    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=0.5)

    assert ellipse.minor_axis == 0.5

    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.5), major_axis=1.0)

    assert ellipse.minor_axis == pytest.approx(0.33333333, 1.0e-4)


def test__ellipse_radii_from_major_axis():
    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    assert ellipse.ellipse_radii_from_major_axis_from(pixel_scale=1.0)[
        0
    ] == pytest.approx(1.0, 1.0e-4)
    assert ellipse.ellipse_radii_from_major_axis_from(pixel_scale=1.0)[
        1
    ] == pytest.approx(1.0, 1.0e-4)

    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.5), major_axis=1.0)

    assert ellipse.ellipse_radii_from_major_axis_from(pixel_scale=1.0)[
        0
    ] == pytest.approx(1.0, 1.0e-4)
    assert ellipse.ellipse_radii_from_major_axis_from(pixel_scale=1.0)[
        1
    ] == pytest.approx(0.348449654923, 1.0e-4)


def test__angles_from_x0():
    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    assert ellipse.angles_from_x0_from(pixel_scale=1.0)[0] == pytest.approx(0.0, 1.0e-4)
    assert ellipse.angles_from_x0_from(pixel_scale=1.0)[1] == pytest.approx(
        1.25663706143, 1.0e-4
    )

    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.5, 0.5), major_axis=1.0)

    assert ellipse.angles_from_x0_from(pixel_scale=1.0)[0] == pytest.approx(0.0, 1.0e-4)
    assert ellipse.angles_from_x0_from(pixel_scale=1.0)[1] == pytest.approx(
        1.256637061, 1.0e-4
    )


def test__x_from_major_axis():
    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    assert ellipse.x_from_major_axis_from(pixel_scale=1.0)[0] == pytest.approx(
        1.0, 1.0e-4
    )
    assert ellipse.x_from_major_axis_from(pixel_scale=1.0)[1] == pytest.approx(
        0.30901699, 1.0e-4
    )

    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.5, 0.5), major_axis=1.0)

    assert ellipse.x_from_major_axis_from(pixel_scale=1.0)[0] == pytest.approx(
        0.414213562373, 1.0e-4
    )
    assert ellipse.x_from_major_axis_from(pixel_scale=1.0)[1] == pytest.approx(
        0.06898775425, 1.0e-4
    )


def test__y_from_major_axis():
    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    assert ellipse.y_from_major_axis_from(pixel_scale=1.0)[0] == pytest.approx(
        0.0, 1.0e-4
    )
    assert ellipse.y_from_major_axis_from(pixel_scale=1.0)[1] == pytest.approx(
        -0.951056516, 1.0e-4
    )

    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.5, 0.5), major_axis=1.0)

    assert ellipse.y_from_major_axis_from(pixel_scale=1.0)[0] == pytest.approx(
        0.0, 1.0e-4
    )
    assert ellipse.y_from_major_axis_from(pixel_scale=1.0)[1] == pytest.approx(
        -0.2123224755, 1.0e-4
    )


def test__points_from_major_axis():
    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    assert ellipse.points_from_major_axis_from(pixel_scale=1.0)[0][1] == pytest.approx(
        1.0, 1.0e-4
    )
    assert ellipse.points_from_major_axis_from(pixel_scale=1.0)[0][0] == pytest.approx(
        0.0, 1.0e-4
    )
    assert ellipse.points_from_major_axis_from(pixel_scale=1.0)[1][1] == pytest.approx(
        0.30901699, 1.0e-4
    )
    assert ellipse.points_from_major_axis_from(pixel_scale=1.0)[1][0] == pytest.approx(
        -0.951056516, 1.0e-4
    )

    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.5, 0.5), major_axis=1.0)

    assert ellipse.points_from_major_axis_from(pixel_scale=1.0)[0][1] == pytest.approx(
        0.4142135623730, 1.0e-4
    )
    assert ellipse.points_from_major_axis_from(pixel_scale=1.0)[0][0] == pytest.approx(
        0.0, 1.0e-4
    )
    assert ellipse.points_from_major_axis_from(pixel_scale=1.0)[1][1] == pytest.approx(
        0.06898775425, 1.0e-4
    )
    assert ellipse.points_from_major_axis_from(pixel_scale=1.0)[1][0] == pytest.approx(
        -0.2123224755, 1.0e-4
    )


def test__multipole__get_shape_angle__boundary_returns_neg_half_period():
    # The xp.mod rewrite maps boundary case `period/2.0` to `-period/2.0`
    # (closed-open interval), unlike the old while-loop which returned `period/2.0`
    # (open-closed). This test pins the new convention.
    multipole = ag.EllipseMultipole(m=4, multipole_comps=(0.0, 0.0))
    # period = 360 / 4 = 90; choose ellipse.angle() such that the unnormalised
    # offset equals exactly +period/2 = 45.0 degrees.
    # multipole_k_m_and_phi_m_from((0.0, 0.0), m=4) returns phi_m = 0, k = 0.
    # So angle_offset = ellipse.angle() - phi_m = ellipse.angle().
    # We want that to be exactly +45.0 so that after wrap it lands at -45.0.
    # ellipse.angle() depends on ell_comps. Easiest: derive ell_comps via
    # ag.convert.ell_comps_from(axis_ratio=..., angle=45.0).
    ellipse = ag.Ellipse(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.5, angle=45.0),
        major_axis=1.0,
    )
    result = multipole.get_shape_angle(ellipse=ellipse)
    assert result == pytest.approx(-45.0, abs=1e-9)
