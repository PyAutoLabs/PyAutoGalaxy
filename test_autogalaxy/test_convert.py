import pytest

import autogalaxy as ag


def test__ell_comps_from():
    ell_comps = ag.convert.ell_comps_from(axis_ratio=0.00050025012, angle=0.0)

    assert ell_comps == pytest.approx((0.0, 0.999), abs=1.0e-4)


def test__axis_ratio_and_angle_from():
    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=(0.0, 1.0))

    assert axis_ratio == pytest.approx(0.00050025012, abs=1.0e-4)
    assert angle == pytest.approx(0.0, abs=1.0e-4)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=(1.0, 0.0))

    assert axis_ratio == pytest.approx(0.00050025012, abs=1.0e-4)
    assert angle == pytest.approx(45.0, abs=1.0e-4)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=(0.0, -1.0))

    assert axis_ratio == pytest.approx(0.00050025012, abs=1.0e-4)
    assert angle == pytest.approx(90.0, abs=1.0e-4)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=(-1.0, 0.0))

    assert axis_ratio == pytest.approx(0.00050025012, abs=1.0e-4)
    assert angle == pytest.approx(-45.0, abs=1.0e-4)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=(-1.0, -1.0))

    assert axis_ratio == pytest.approx(0.00050025012, abs=1.0e-4)
    assert angle == pytest.approx(112.5, abs=1.0e-4)


def test__shear_gamma_1_2_from():
    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=0.0)

    assert gamma_1 == pytest.approx(0.05, abs=1.0e-4)
    assert gamma_2 == pytest.approx(0.0, abs=1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=45.0)

    assert gamma_1 == pytest.approx(0.0, abs=1.0e-4)
    assert gamma_2 == pytest.approx(0.05, abs=1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=90.0)

    assert gamma_1 == pytest.approx(-0.05, abs=1.0e-4)
    assert gamma_2 == pytest.approx(0.0, abs=1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=135.0)

    assert gamma_1 == pytest.approx(0.0, abs=1.0e-4)
    assert gamma_2 == pytest.approx(-0.05, abs=1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=180.0)

    assert gamma_1 == pytest.approx(0.05, abs=1.0e-4)
    assert gamma_2 == pytest.approx(0.0, abs=1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=225.0)

    assert gamma_1 == pytest.approx(0.0, abs=1.0e-4)
    assert gamma_2 == pytest.approx(0.05, abs=1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=-45.0)

    assert gamma_1 == pytest.approx(0.0, abs=1.0e-4)
    assert gamma_2 == pytest.approx(-0.05, abs=1.0e-4)


def test__shear_magnitude_and_angle_from():
    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.05, gamma_2=0.0
    )

    assert magnitude == pytest.approx(0.05, abs=1.0e-4)
    assert angle == pytest.approx(0.0, abs=1.0e-4)

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.0, gamma_2=0.05
    )

    assert magnitude == pytest.approx(0.05, abs=1.0e-4)
    assert angle == pytest.approx(45.0, abs=1.0e-4)

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=-0.05, gamma_2=0.0
    )

    assert magnitude == pytest.approx(0.05, abs=1.0e-4)
    assert angle == pytest.approx(90.0, abs=1.0e-4)

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.0, gamma_2=-0.05
    )

    assert magnitude == pytest.approx(0.05, abs=1.0e-4)
    assert angle == pytest.approx(135.0, abs=1.0e-4)

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.05, gamma_2=0.0
    )

    assert magnitude == pytest.approx(0.05, abs=1.0e-4)
    assert angle == pytest.approx(0.0, abs=1.0e-4)

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.0, gamma_2=0.05
    )

    assert magnitude == pytest.approx(0.05, abs=1.0e-4)
    assert angle == pytest.approx(45.0, abs=1.0e-4)

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.05, gamma_2=0.0
    )

    assert magnitude == pytest.approx(0.05, abs=1.0e-4)

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.05, gamma_2=-0.05
    )

    assert magnitude == pytest.approx(0.07071067811865, abs=1.0e-4)
    assert angle == pytest.approx(-22.5, abs=1.0e-4)


def test__multipole_k_m_and_phi_m_from():
    k_m, phi = ag.convert.multipole_k_m_and_phi_m_from(multipole_comps=(0.1, 0.0), m=1)

    assert k_m == pytest.approx(0.1, abs=1e-3)
    assert phi == pytest.approx(90.0, abs=1e-3)

    k_m, phi = ag.convert.multipole_k_m_and_phi_m_from(multipole_comps=(0.0, 0.1), m=1)

    assert k_m == pytest.approx(0.1, abs=1e-3)
    assert phi == pytest.approx(0.0, abs=1e-3)

    k_m, phi = ag.convert.multipole_k_m_and_phi_m_from(multipole_comps=(0.1, 0.0), m=2)

    assert k_m == pytest.approx(0.1, abs=1e-3)
    assert phi == pytest.approx(45.0, abs=1e-3)

    k_m, phi = ag.convert.multipole_k_m_and_phi_m_from(
        multipole_comps=(-0.1, -0.1), m=2
    )

    assert k_m == pytest.approx(0.14142135, abs=1e-3)
    assert phi == pytest.approx(112.5, abs=1e-3)


def test__multipole_comps_from():
    multipole_comps = ag.convert.multipole_comps_from(k_m=0.1, phi_m=90.0, m=1)

    assert multipole_comps == pytest.approx((0.1, 0.0), abs=1e-3)

    multipole_comps = ag.convert.multipole_comps_from(k_m=0.1, phi_m=0.0, m=1)

    assert multipole_comps == pytest.approx((0.0, 0.1), abs=1e-3)

    multipole_comps = ag.convert.multipole_comps_from(k_m=0.1, phi_m=45.0, m=2)

    assert multipole_comps == pytest.approx((0.1, 0.0), abs=1e-3)

    multipole_comps = ag.convert.multipole_comps_from(k_m=0.14142135, phi_m=112.5, m=2)

    assert multipole_comps == pytest.approx((-0.1, -0.1), abs=1e-3)
