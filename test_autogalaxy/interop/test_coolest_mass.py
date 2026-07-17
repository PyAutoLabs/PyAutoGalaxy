import numpy as np
import pytest

import autoarray as aa
import autogalaxy as ag
from autogalaxy import exc
from autogalaxy.interop import coolest


def grid_yx():
    return np.array(
        [
            [1.0, 0.5],
            [-0.3, 0.7],
            [0.2, -1.1],
            [0.8, 0.9],
            [-1.2, -0.4],
        ]
    )


def coolest_power_law_convergence(parameters, grid):
    """
    The COOLEST / lenstronomy PEMD convergence evaluated directly from
    COOLEST-convention parameters:

    kappa(r) = (3 - gamma) / 2 * (theta_E / r)^(gamma - 1),
    r = sqrt(q x'^2 + y'^2 / q),

    with (x', y') the coordinates rotated into the profile frame whose major
    axis lies at position angle phi counter-clockwise from the +y axis.
    """
    x = grid[:, 1] - parameters["center_x"]
    y = grid[:, 0] - parameters["center_y"]

    alpha = np.radians(parameters["phi"] + 90.0)
    x_prime = np.cos(alpha) * x + np.sin(alpha) * y
    y_prime = -np.sin(alpha) * x + np.cos(alpha) * y

    q = parameters["q"]
    gamma = parameters.get("gamma", 2.0)
    r = np.sqrt(q * x_prime**2 + y_prime**2 / q)

    return (3.0 - gamma) / 2.0 * (parameters["theta_E"] / r) ** (gamma - 1.0)


def test__power_law__convergence_matches_coolest_analytic_form():
    profile = ag.mp.PowerLaw(
        centre=(0.1, -0.2),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.6, angle=40.0),
        einstein_radius=1.4,
        slope=2.3,
    )

    profile_dict = coolest.coolest_dict_from_mass(profile=profile)

    assert profile_dict["type"] == "PEMD"

    convergence = profile.convergence_2d_from(grid=aa.Grid2DIrregular(grid_yx()))
    convergence_coolest = coolest_power_law_convergence(
        parameters=profile_dict["parameters"], grid=grid_yx()
    )

    assert np.asarray(convergence) == pytest.approx(convergence_coolest, rel=1e-8)


def test__isothermal__convergence_matches_coolest_analytic_form():
    profile = ag.mp.Isothermal(
        centre=(-0.05, 0.15),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.7, angle=110.0),
        einstein_radius=1.1,
    )

    profile_dict = coolest.coolest_dict_from_mass(profile=profile)

    assert profile_dict["type"] == "SIE"
    assert profile_dict["parameters"]["theta_E"] == pytest.approx(
        2.0 * np.sqrt(0.7) / (1.0 + 0.7) * 1.1
    )

    convergence = profile.convergence_2d_from(grid=aa.Grid2DIrregular(grid_yx()))
    convergence_coolest = coolest_power_law_convergence(
        parameters=profile_dict["parameters"], grid=grid_yx()
    )

    assert np.asarray(convergence) == pytest.approx(convergence_coolest, rel=1e-8)


@pytest.mark.parametrize("axis_ratio,angle,slope", [(0.6, 40.0, 2.3), (0.9, -20.0, 1.7)])
def test__power_law__round_trip_is_numerically_identical(axis_ratio, angle, slope):
    profile = ag.mp.PowerLaw(
        centre=(0.1, -0.2),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=axis_ratio, angle=angle),
        einstein_radius=1.4,
        slope=slope,
    )

    profile_dict = coolest.coolest_dict_from_mass(profile=profile)
    profile_back = coolest.mass_profile_from(
        profile_type=profile_dict["type"], parameters=profile_dict["parameters"]
    )

    grid = aa.Grid2DIrregular(grid_yx())

    assert np.asarray(profile_back.convergence_2d_from(grid=grid)) == pytest.approx(
        np.asarray(profile.convergence_2d_from(grid=grid)), rel=1e-8
    )
    assert np.asarray(
        profile_back.deflections_yx_2d_from(grid=grid)
    ) == pytest.approx(
        np.asarray(profile.deflections_yx_2d_from(grid=grid)), rel=1e-6
    )


def test__isothermal_sph__round_trip():
    profile = ag.mp.IsothermalSph(centre=(0.1, 0.2), einstein_radius=0.8)

    profile_dict = coolest.coolest_dict_from_mass(profile=profile)

    assert profile_dict["parameters"]["q"] == pytest.approx(1.0)
    assert profile_dict["parameters"]["theta_E"] == pytest.approx(0.8)

    profile_back = coolest.mass_profile_from(
        profile_type=profile_dict["type"], parameters=profile_dict["parameters"]
    )

    grid = aa.Grid2DIrregular(grid_yx())

    assert np.asarray(profile_back.convergence_2d_from(grid=grid)) == pytest.approx(
        np.asarray(profile.convergence_2d_from(grid=grid)), rel=1e-8
    )


def test__external_shear__coolest_angle_is_east_of_north():
    # gamma_1 > 0 only: shear aligned with the x axis (ag angle 0) = COOLEST
    # phi_ext of 90 (the (-90, 90] representative of -90).
    profile_dict = coolest.coolest_dict_from_mass(
        profile=ag.mp.ExternalShear(gamma_1=0.05, gamma_2=0.0)
    )
    assert profile_dict["type"] == "ExternalShear"
    assert profile_dict["parameters"]["gamma_ext"] == pytest.approx(0.05)
    assert abs(profile_dict["parameters"]["phi_ext"]) == pytest.approx(90.0)

    # gamma_2 > 0 only: ag angle 45 = COOLEST phi_ext -45.
    profile_dict = coolest.coolest_dict_from_mass(
        profile=ag.mp.ExternalShear(gamma_1=0.0, gamma_2=0.05)
    )
    assert profile_dict["parameters"]["phi_ext"] == pytest.approx(-45.0)


def test__external_shear__round_trip():
    profile = ag.mp.ExternalShear(gamma_1=0.03, gamma_2=-0.04)

    profile_dict = coolest.coolest_dict_from_mass(profile=profile)
    profile_back = coolest.mass_profile_from(
        profile_type=profile_dict["type"], parameters=profile_dict["parameters"]
    )

    assert profile_back.gamma_1 == pytest.approx(0.03)
    assert profile_back.gamma_2 == pytest.approx(-0.04)


def test__nfw__round_trip_requires_sigma_crit_and_is_identical():
    profile = ag.mp.NFW(
        centre=(0.05, -0.1),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        kappa_s=0.12,
        scale_radius=8.0,
    )

    with pytest.raises(exc.ProfileException):
        coolest.coolest_dict_from_mass(profile=profile)

    sigma_crit = 2.5e9

    profile_dict = coolest.coolest_dict_from_mass(
        profile=profile, sigma_crit=sigma_crit
    )

    assert profile_dict["type"] == "NFW"
    assert profile_dict["parameters"]["r_s"] == pytest.approx(np.sqrt(0.8) * 8.0)
    assert profile_dict["parameters"]["rho_c"] == pytest.approx(
        0.12 * sigma_crit / (np.sqrt(0.8) * 8.0)
    )

    profile_back = coolest.mass_profile_from(
        profile_type="NFW",
        parameters=profile_dict["parameters"],
        sigma_crit=sigma_crit,
    )

    assert profile_back.kappa_s == pytest.approx(0.12)
    assert profile_back.scale_radius == pytest.approx(8.0)

    grid = aa.Grid2DIrregular(grid_yx())

    assert np.asarray(profile_back.convergence_2d_from(grid=grid)) == pytest.approx(
        np.asarray(profile.convergence_2d_from(grid=grid)), rel=1e-8
    )


def test__mass_sheet__round_trip_and_off_centre_raises():
    profile_dict = coolest.coolest_dict_from_mass(profile=ag.mp.MassSheet(kappa=0.05))

    assert profile_dict["type"] == "ConvergenceSheet"
    assert profile_dict["parameters"]["kappa_s"] == pytest.approx(0.05)

    profile_back = coolest.mass_profile_from(
        profile_type="ConvergenceSheet", parameters=profile_dict["parameters"]
    )

    assert profile_back.kappa == pytest.approx(0.05)

    with pytest.raises(exc.ProfileException):
        coolest.coolest_dict_from_mass(
            profile=ag.mp.MassSheet(centre=(0.1, 0.0), kappa=0.05)
        )


def test__unsupported_profiles__raise_with_named_error():
    with pytest.raises(exc.ProfileException):
        coolest.coolest_dict_from_mass(profile=ag.mp.gNFW())

    with pytest.raises(exc.ProfileException):
        coolest.mass_profile_from(profile_type="Chameleon", parameters={})
