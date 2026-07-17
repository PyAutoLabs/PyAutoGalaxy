import numpy as np
import pytest

import autogalaxy as ag
from autogalaxy.interop.coolest import conventions


def test__phi_coolest_from__shifts_and_normalizes():
    assert conventions.phi_coolest_from(angle=90.0) == pytest.approx(0.0)
    assert conventions.phi_coolest_from(angle=135.0) == pytest.approx(45.0)
    assert conventions.phi_coolest_from(angle=45.0) == pytest.approx(-45.0)
    # angle=0 gives phi=-90, which normalizes to the (-90, 90] boundary value.
    assert conventions.phi_coolest_from(angle=0.0) == pytest.approx(90.0)


def test__phi_round_trip__same_ellipse_orientation():
    for angle in [0.0, 30.0, 90.0, 120.0]:
        phi = conventions.phi_coolest_from(angle=angle)
        angle_back = conventions.angle_from_phi_coolest(phi=phi)
        # Orientations are degenerate under 180 deg rotations.
        assert (angle_back - angle) % 180.0 == pytest.approx(0.0, abs=1e-10)


def test__q_phi_and_ell_comps__round_trip():
    ell_comps = ag.convert.ell_comps_from(axis_ratio=0.6, angle=40.0)

    q, phi = conventions.q_phi_from_ell_comps(ell_comps=ell_comps)

    assert q == pytest.approx(0.6)
    assert phi == pytest.approx(-50.0)

    ell_comps_back = conventions.ell_comps_from_q_phi(q=q, phi=phi)

    assert ell_comps_back == pytest.approx(ell_comps)


def test__centre_conversion__swaps_ordering():
    assert conventions.center_from_centre(centre=(1.0, 2.0)) == (2.0, 1.0)
    assert conventions.centre_from_center(center_x=2.0, center_y=1.0) == (1.0, 2.0)


def test__radius_conversion__intermediate_axis():
    assert conventions.radius_intermediate_from(
        radius_major=2.0, q=0.25
    ) == pytest.approx(1.0)
    assert conventions.radius_major_from(
        radius_intermediate=1.0, q=0.25
    ) == pytest.approx(2.0)
