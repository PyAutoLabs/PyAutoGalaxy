import autogalaxy as ag
import numpy as np
import pytest


grid_diag = ag.Grid2DIrregular([[0.1, 0.1]])
grid_unit_x = ag.Grid2DIrregular([[0.0, 1.0]])
grid_multi = ag.Grid2DIrregular([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])


def test__convergence_2d_from__gamma_only_is_zero():
    mp = ag.mp.ExternalPotential(gamma_1=0.1, gamma_2=-0.05)
    convergence = mp.convergence_2d_from(grid=grid_multi)
    assert convergence == pytest.approx(np.zeros(3), 1.0e-8)


def test__convergence_2d_from__delta_only_is_zero():
    mp = ag.mp.ExternalPotential(delta_1=0.07, delta_2=0.04)
    convergence = mp.convergence_2d_from(grid=grid_multi)
    assert convergence == pytest.approx(np.zeros(3), 1.0e-8)


def test__convergence_2d_from__tau_only_is_linear_in_xy():
    mp = ag.mp.ExternalPotential(tau_1=0.1, tau_2=0.2)
    grid = ag.Grid2DIrregular([[0.0, 1.0], [0.5, 1.0], [1.0, 0.0]])
    convergence = mp.convergence_2d_from(grid=grid)
    # kappa(x, y) = tau_1 * x + tau_2 * y
    expected = np.array([0.1 * 1.0 + 0.2 * 0.0, 0.1 * 1.0 + 0.2 * 0.5, 0.1 * 0.0 + 0.2 * 1.0])
    assert convergence == pytest.approx(expected, 1.0e-6)


def test__convergence_2d_from__tau_with_nonzero_centre_shifts_origin():
    mp = ag.mp.ExternalPotential(centre=(1.0, 1.0), tau_1=0.1, tau_2=0.0)
    grid = ag.Grid2DIrregular([[1.0, 1.0], [1.0, 2.0]])
    convergence = mp.convergence_2d_from(grid=grid)
    assert convergence == pytest.approx(np.array([0.0, 0.1]), 1.0e-6)


def test__potential_2d_from__gamma_only_matches_external_shear():
    shear = ag.mp.ExternalShear(gamma_1=0.1, gamma_2=-0.05)
    pot = ag.mp.ExternalPotential(gamma_1=0.1, gamma_2=-0.05)
    expected = shear.potential_2d_from(grid=grid_multi)
    actual = pot.potential_2d_from(grid=grid_multi)
    assert actual == pytest.approx(np.asarray(expected), 1.0e-6)


def test__potential_2d_from__tau_only_unit_x_axis():
    mp = ag.mp.ExternalPotential(tau_1=0.05, tau_2=0.0)
    # at (y=0, x=1): r=1, theta=0 -> psi = 0.25 * 1^3 * (0.05 * 1 + 0) = 0.0125
    potential = mp.potential_2d_from(grid=grid_unit_x)
    assert potential == pytest.approx(np.array([0.0125]), 1.0e-6)


def test__potential_2d_from__delta_only_unit_x_axis():
    mp = ag.mp.ExternalPotential(delta_1=0.1, delta_2=0.0)
    # at (y=0, x=1): r=1, theta=0 -> psi = (1/6) * 1^3 * (0.1 * 1 + 0) = 0.1/6
    potential = mp.potential_2d_from(grid=grid_unit_x)
    assert potential == pytest.approx(np.array([0.1 / 6.0]), 1.0e-6)


def test__deflections_yx_2d_from__gamma_only_matches_external_shear():
    shear = ag.mp.ExternalShear(gamma_1=-0.17320, gamma_2=0.1)
    pot = ag.mp.ExternalPotential(gamma_1=-0.17320, gamma_2=0.1)
    grid = ag.Grid2DIrregular([[0.1625, 0.1625]])
    expected = np.asarray(shear.deflections_yx_2d_from(grid=grid))
    actual = np.asarray(pot.deflections_yx_2d_from(grid=grid))
    assert actual == pytest.approx(expected, 1.0e-5)


def test__deflections_yx_2d_from__tau_only_radial_unit_x_axis():
    mp = ag.mp.ExternalPotential(tau_1=0.05, tau_2=0.0)
    # at (y=0, x=1): r=1, theta=0
    #   alpha_r = 0.75 * 1 * 0.05 = 0.0375 ; alpha_theta = 0
    #   alpha_y = 0, alpha_x = 0.0375
    deflections = mp.deflections_yx_2d_from(grid=grid_unit_x)
    assert deflections[0, 0] == pytest.approx(0.0, 1.0e-6)
    assert deflections[0, 1] == pytest.approx(0.0375, 1.0e-6)


def test__deflections_yx_2d_from__delta_only_unit_x_axis():
    mp = ag.mp.ExternalPotential(delta_1=0.1, delta_2=0.0)
    # at (y=0, x=1): alpha_r = 0.5 * 1 * 0.1 = 0.05 ; alpha_theta = 0
    deflections = mp.deflections_yx_2d_from(grid=grid_unit_x)
    assert deflections[0, 0] == pytest.approx(0.0, 1.0e-6)
    assert deflections[0, 1] == pytest.approx(0.05, 1.0e-6)


def test__deflections_yx_2d_from__nonzero_centre_shifts_origin():
    mp = ag.mp.ExternalPotential(centre=(1.0, 1.0), gamma_1=0.1, gamma_2=0.0)
    # at (y=1, x=2) post-shift becomes (y=0, x=1): r=1, theta=0
    #   alpha_r = 1 * (0.1 * cos(0) + 0) = 0.1 ; alpha_theta = 0
    #   alpha_y = 0, alpha_x = 0.1
    grid = ag.Grid2DIrregular([[1.0, 2.0]])
    deflections = mp.deflections_yx_2d_from(grid=grid)
    assert deflections[0, 0] == pytest.approx(0.0, 1.0e-6)
    assert deflections[0, 1] == pytest.approx(0.1, 1.0e-6)


def test__from_magnitudes_and_angles__roundtrip_gamma():
    gamma = 0.1
    theta_gamma = 30.0  # degrees
    mp = ag.mp.ExternalPotential.from_magnitudes_and_angles(
        gamma=gamma, theta_gamma=theta_gamma
    )
    assert mp.gamma_magnitude() == pytest.approx(gamma, 1.0e-6)
    assert mp.gamma_angle() == pytest.approx(theta_gamma, 1.0e-6)


def test__from_magnitudes_and_angles__roundtrip_tau():
    tau = 0.05
    theta_tau = 200.0  # degrees, spin-1 so [0, 360)
    mp = ag.mp.ExternalPotential.from_magnitudes_and_angles(
        tau=tau, theta_tau=theta_tau
    )
    assert mp.tau_magnitude() == pytest.approx(tau, 1.0e-6)
    assert mp.tau_angle() == pytest.approx(theta_tau, 1.0e-6)


def test__from_magnitudes_and_angles__roundtrip_delta():
    delta = 0.02
    theta_delta = 40.0  # degrees, spin-3 so [0, 120)
    mp = ag.mp.ExternalPotential.from_magnitudes_and_angles(
        delta=delta, theta_delta=theta_delta
    )
    assert mp.delta_magnitude() == pytest.approx(delta, 1.0e-6)
    assert mp.delta_angle() == pytest.approx(theta_delta, 1.0e-6)
