import numpy as np
import pytest
from scipy.optimize import fsolve

import autogalaxy as ag
from autogalaxy.cosmology.model import Planck15

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from__config_1__y_axis():
    mp = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0, truncation_radius=2.0
    )

    # factor = (4.0 * kappa_s * scale_radius / (r / scale_radius))

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[2.0, 0.0]]))

    factor = (4.0 * 1.0 * 1.0) / (2.0 / 1.0)
    assert deflections[0, 0] == pytest.approx(factor * 0.38209715, abs=1.0e-4)
    assert deflections[0, 1] == pytest.approx(0.0, abs=1.0e-4)


def test__deflections_yx_2d_from__config_1__x_axis():
    mp = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0, truncation_radius=2.0
    )

    factor = (4.0 * 1.0 * 1.0) / (2.0 / 1.0)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.0, 2.0]]))

    assert deflections[0, 0] == pytest.approx(0.0, abs=1.0e-4)
    assert deflections[0, 1] == pytest.approx(factor * 0.38209715, abs=1.0e-4)


def test__deflections_yx_2d_from__config_1__diagonal():
    mp = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0, truncation_radius=2.0
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 1.0]]))

    factor = (4.0 * 1.0 * 1.0) / (np.sqrt(2) / 1.0)
    assert deflections[0, 0] == pytest.approx(
        (1.0 / np.sqrt(2)) * factor * 0.3125838, abs=1.0e-4
    )
    assert deflections[0, 1] == pytest.approx(
        (1.0 / np.sqrt(2)) * factor * 0.3125838, abs=1.0e-4
    )


def test__deflections_yx_2d_from__kappa_s_2():
    mp = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=2.0, scale_radius=1.0, truncation_radius=2.0
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[2.0, 0.0]]))

    factor = (4.0 * 2.0 * 1.0) / (2.0 / 1.0)
    assert deflections[0, 0] == pytest.approx(factor * 0.38209715, abs=1.0e-4)
    assert deflections[0, 1] == pytest.approx(0.0, abs=1.0e-4)


def test__deflections_yx_2d_from__scale_radius_4():
    mp = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=1.0, scale_radius=4.0, truncation_radius=2.0
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([(2.0, 0.0)]))

    assert deflections[0, 0] == pytest.approx(2.1702661386, abs=1.0e-4)
    assert deflections[0, 1] == pytest.approx(0.0, abs=1.0e-4)


def test__convergence_2d_from__grid_2():
    mp = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0, truncation_radius=2.0
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[2.0, 0.0]]))

    assert convergence == pytest.approx(2.0 * 0.046409642, abs=1.0e-4)


def test__convergence_2d_from__grid_diagonal():
    mp = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0, truncation_radius=2.0
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 1.0]]))

    assert convergence == pytest.approx(2.0 * 0.10549515, abs=1.0e-4)


def test__convergence_2d_from__kappa_s_3():
    mp = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=3.0, scale_radius=1.0, truncation_radius=2.0
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[2.0, 0.0]]))

    assert convergence == pytest.approx(6.0 * 0.046409642, abs=1.0e-4)


def test__convergence_2d_from__scale_radius_5():
    mp = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=3.0, scale_radius=5.0, truncation_radius=2.0
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[2.0, 0.0]]))

    assert convergence == pytest.approx(1.51047026, abs=1.0e-4)


def test__potential_2d_from__matches_independent_tnfw_implementation():
    """Potential differences cross-checked against lenstronomy's analytic TNFW."""
    mp = ag.mp.NFWTruncatedSph(
        kappa_s=0.13,
        scale_radius=1.7,
        truncation_radius=3.4,
    )

    potential = mp.potential_2d_from(
        grid=ag.Grid2DIrregular([[0.0, 0.085], [0.0, 0.51], [0.0, 5.1]])
    )
    potential -= potential[0]

    assert potential == pytest.approx(
        [0.0, 0.05804607, 0.88448954],
        abs=1.0e-8,
    )


def test__potential_2d_from__centre_and_scale_radius_are_finite():
    mp = ag.mp.NFWTruncatedSph(
        kappa_s=0.13,
        scale_radius=1.7,
        truncation_radius=3.4,
    )

    potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[0.0, 0.0], [0.0, 1.7]]))

    assert np.all(np.isfinite(potential))
    assert potential[0] == pytest.approx(0.0, abs=1.0e-20)


@pytest.mark.parametrize(
    "tau,radius_over_scale",
    [
        (0.5, 0.3),
        (0.5, 2.0),
        (1.0, 0.3),
        (1.0, 2.0),
        (2.0, 0.3),
        (2.0, 2.0),
        (10.0, 0.3),
        (10.0, 2.0),
    ],
)
def test__potential_2d_from__gradient_matches_analytic_deflections(
    tau, radius_over_scale
):
    scale_radius = 1.7
    radius = radius_over_scale * scale_radius
    step = 1.0e-4 * scale_radius

    mp = ag.mp.NFWTruncatedSph(
        kappa_s=0.13,
        scale_radius=scale_radius,
        truncation_radius=tau * scale_radius,
    )

    radial_grid = ag.Grid2DIrregular([[0.0, radius - step], [0.0, radius + step]])
    potential = mp.potential_2d_from(grid=radial_grid)
    potential_gradient = (potential[1] - potential[0]) / (2.0 * step)

    deflection = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.0, radius]]))[
        0, 1
    ]

    assert potential_gradient == pytest.approx(deflection, rel=1.0e-6)


@pytest.mark.parametrize(
    "tau,radius_over_scale",
    [
        (0.5, 0.3),
        (0.5, 2.0),
        (1.0, 0.3),
        (1.0, 2.0),
        (2.0, 0.3),
        (2.0, 2.0),
        (10.0, 0.3),
        (10.0, 2.0),
    ],
)
def test__potential_2d_from__laplacian_matches_analytic_convergence(
    tau, radius_over_scale
):
    scale_radius = 1.7
    radius = radius_over_scale * scale_radius
    step = 1.0e-3 * scale_radius

    mp = ag.mp.NFWTruncatedSph(
        kappa_s=0.13,
        scale_radius=scale_radius,
        truncation_radius=tau * scale_radius,
    )

    radial_grid = ag.Grid2DIrregular(
        [[0.0, radius - step], [0.0, radius], [0.0, radius + step]]
    )
    potential = mp.potential_2d_from(grid=radial_grid)
    potential_gradient = (potential[2] - potential[0]) / (2.0 * step)
    potential_second_derivative = (
        potential[2] - 2.0 * potential[1] + potential[0]
    ) / step**2.0
    potential_laplacian = potential_second_derivative + potential_gradient / radius

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, radius]]))[0]

    assert potential_laplacian == pytest.approx(2.0 * convergence, rel=1.0e-5)


def test__mass_at_truncation_radius():
    mp = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0, truncation_radius=1.0
    )

    cosmology = ag.m.MockCosmology(
        arcsec_per_kpc=1.0,
        kpc_per_arcsec=1.0,
        critical_surface_density=1.0,
        cosmic_average_density=1.0,
    )

    mass_at_truncation_radius = mp.mass_at_truncation_radius_solar_mass(
        redshift_profile=0.5, redshift_source=1.0, cosmology=cosmology
    )

    assert mass_at_truncation_radius == pytest.approx(0.00009792581, 1.0e-5)


def test__compare_nfw_and_truncated_nfw_with_large_truncation_radius():
    truncated_nfw = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=1.0, scale_radius=4.0, truncation_radius=50000.0
    )

    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=4.0)

    truncated_nfw_convergence = truncated_nfw.convergence_2d_from(
        grid=ag.Grid2DIrregular([[2.0, 2.0], [3.0, 1.0], [-1.0, -9.0]])
    )
    nfw_convergence = nfw.convergence_2d_from(
        grid=ag.Grid2DIrregular([[2.0, 2.0], [3.0, 1.0], [-1.0, -9.0]])
    )

    assert truncated_nfw_convergence == pytest.approx(nfw_convergence, abs=1.0e-4)

    truncated_nfw_deflections = truncated_nfw.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[2.0, 2.0], [3.0, 1.0], [-1.0, -9.0]])
    )
    nfw_deflections = nfw.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[2.0, 2.0], [3.0, 1.0], [-1.0, -9.0]])
    )

    assert truncated_nfw_deflections == pytest.approx(nfw_deflections.array, abs=1.0e-4)

    truncated_nfw_potential = truncated_nfw.potential_2d_from(
        grid=ag.Grid2DIrregular([[2.0, 2.0], [3.0, 1.0], [-1.0, -9.0]])
    )
    nfw_potential = nfw.potential_2d_from(
        grid=ag.Grid2DIrregular([[2.0, 2.0], [3.0, 1.0], [-1.0, -9.0]])
    )

    # The lensing potential is defined only up to an additive constant.
    truncated_nfw_potential -= truncated_nfw_potential[0]
    nfw_potential -= nfw_potential[0]

    assert truncated_nfw_potential == pytest.approx(nfw_potential, abs=1.0e-4)


# ---------------------------------------------------------------------------
# Helpers: reference implementation of the los_pipes unit-conversion formulas
# (convert_units.py from He et al. 2022, MNRAS 511 3046) expressed in terms of
# the PyAutoGalaxy cosmology API.  These are used as ground-truth values in the
# regression tests below.
# ---------------------------------------------------------------------------


def _los_pipes_reference_delta_c(concentration):
    """NFW characteristic overdensity as computed by los_pipes."""
    c = concentration
    return 200.0 / 3.0 * (c**3 / (np.log(1.0 + c) - c / (1.0 + c)))


def _los_pipes_reference_c50(concentration, truncation_factor):
    """Concentration at the truncation overdensity (c50 in los_pipes)."""
    delta_c = _los_pipes_reference_delta_c(concentration)

    def eq(tau):
        return (
            truncation_factor / 3.0 * (tau**3 / (np.log(1.0 + tau) - tau / (1.0 + tau)))
            - delta_c
        )

    return float(fsolve(eq, concentration)[0])


def _los_pipes_reference_convert_to_lens_unit(
    m200, concentration, z_halo, z_source, truncation_factor=100.0
):
    """
    Replicate los_pipes convert_to_lens_unit() using PyAutoGalaxy cosmology.

    Returns (kappa_s, scale_radius_arcsec, truncation_radius_arcsec).
    """
    cosmo = Planck15()

    critical_density = cosmo.critical_density(z_halo)
    kpc_per_arcsec = cosmo.kpc_per_arcsec_from(z_halo)
    sigma_crit = (
        cosmo.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
            z_halo, z_source
        )
    )

    r200_kpc = (m200 / (200.0 * critical_density * (4.0 * np.pi / 3.0))) ** (1.0 / 3.0)
    delta_c = _los_pipes_reference_delta_c(concentration)
    rs_kpc = r200_kpc / concentration
    rho_s = critical_density * delta_c

    kappa_s = rho_s * rs_kpc / sigma_crit
    scale_radius_arcsec = rs_kpc / kpc_per_arcsec

    c50 = _los_pipes_reference_c50(concentration, truncation_factor)
    truncation_radius_arcsec = c50 * scale_radius_arcsec

    return kappa_s, scale_radius_arcsec, truncation_radius_arcsec


def _los_pipes_reference_mass_ratio(concentration, truncation_factor=100.0):
    """Replicate los_pipes scale_c(c): M_tNFW / M_200."""
    tau = _los_pipes_reference_c50(concentration, truncation_factor)
    tau2 = tau**2
    tau_scale = (
        tau2
        / (tau2 + 1.0) ** 2
        * ((tau2 - 1.0) * np.log(tau) + tau * np.pi - (tau2 + 1.0))
    )
    c_scale = np.log(1.0 + concentration) - concentration / (1.0 + concentration)
    return tau_scale / c_scale


# ---------------------------------------------------------------------------
# Tests for _delta_c_from_concentration
# ---------------------------------------------------------------------------


def test__delta_c_from_concentration__standard_value():
    """delta_c formula for c=10 matches the known analytic result."""
    c = 10.0
    expected = 200.0 / 3.0 * (c**3 / (np.log(1.0 + c) - c / (1.0 + c)))
    result = ag.mp.NFWTruncatedSph._delta_c_from_concentration(c)
    assert result == pytest.approx(expected, rel=1.0e-10)


def test__delta_c_from_concentration__various_concentrations():
    """delta_c increases monotonically with concentration."""
    deltas = [
        ag.mp.NFWTruncatedSph._delta_c_from_concentration(c) for c in [5.0, 10.0, 20.0]
    ]
    assert deltas[0] < deltas[1] < deltas[2]


# ---------------------------------------------------------------------------
# Tests for _concentration_at_overdensity_factor
# ---------------------------------------------------------------------------


def test__concentration_at_overdensity_factor__self_consistent():
    """
    For truncation_factor == 200, tau should equal the input concentration because
    both solve the same overdensity condition.
    """
    c = 10.0
    tau = ag.mp.NFWTruncatedSph._concentration_at_overdensity_factor(c, 200.0)
    assert tau == pytest.approx(c, rel=1.0e-4)


def test__concentration_at_overdensity_factor__larger_for_smaller_factor():
    """
    A smaller overdensity threshold means a larger enclosed radius, so tau must
    be larger when truncation_factor < 200.
    """
    c = 10.0
    tau_100 = ag.mp.NFWTruncatedSph._concentration_at_overdensity_factor(c, 100.0)
    tau_50 = ag.mp.NFWTruncatedSph._concentration_at_overdensity_factor(c, 50.0)
    assert tau_50 > tau_100 > c


def test__concentration_at_overdensity_factor__matches_los_pipes_reference():
    c = 10.0
    expected = _los_pipes_reference_c50(c, 100.0)
    result = ag.mp.NFWTruncatedSph._concentration_at_overdensity_factor(c, 100.0)
    assert result == pytest.approx(expected, rel=1.0e-6)


# ---------------------------------------------------------------------------
# Tests for from_m200_concentration
# ---------------------------------------------------------------------------


def test__from_m200_concentration__kappa_s_matches_los_pipes():
    """kappa_s from from_m200_concentration matches los_pipes reference."""
    kappa_s_ref, _, _ = _los_pipes_reference_convert_to_lens_unit(
        1e9, 10.0, 0.5, 1.0, 100.0
    )
    nfw = ag.mp.NFWTruncatedSph.from_m200_concentration(
        centre=(0.0, 0.0),
        m200_solar_mass=1e9,
        concentration=10.0,
        redshift_halo=0.5,
        redshift_source=1.0,
        cosmology=Planck15(),
        truncation_factor=100.0,
    )
    assert nfw.kappa_s == pytest.approx(kappa_s_ref, rel=1.0e-6)


def test__from_m200_concentration__scale_radius_matches_los_pipes():
    """scale_radius_arcsec from from_m200_concentration matches los_pipes reference."""
    _, scale_radius_ref, _ = _los_pipes_reference_convert_to_lens_unit(
        1e9, 10.0, 0.5, 1.0, 100.0
    )
    nfw = ag.mp.NFWTruncatedSph.from_m200_concentration(
        centre=(0.0, 0.0),
        m200_solar_mass=1e9,
        concentration=10.0,
        redshift_halo=0.5,
        redshift_source=1.0,
        cosmology=Planck15(),
        truncation_factor=100.0,
    )
    assert nfw.scale_radius == pytest.approx(scale_radius_ref, rel=1.0e-6)


def test__from_m200_concentration__truncation_radius_matches_los_pipes():
    """truncation_radius from from_m200_concentration matches los_pipes reference."""
    _, _, truncation_radius_ref = _los_pipes_reference_convert_to_lens_unit(
        1e9, 10.0, 0.5, 1.0, 100.0
    )
    nfw = ag.mp.NFWTruncatedSph.from_m200_concentration(
        centre=(0.0, 0.0),
        m200_solar_mass=1e9,
        concentration=10.0,
        redshift_halo=0.5,
        redshift_source=1.0,
        cosmology=Planck15(),
        truncation_factor=100.0,
    )
    assert nfw.truncation_radius == pytest.approx(truncation_radius_ref, rel=1.0e-6)


def test__from_m200_concentration__different_redshifts():
    """from_m200_concentration at a different redshift pair produces valid parameters."""
    nfw = ag.mp.NFWTruncatedSph.from_m200_concentration(
        centre=(0.1, -0.2),
        m200_solar_mass=1e10,
        concentration=7.0,
        redshift_halo=0.3,
        redshift_source=2.0,
        cosmology=Planck15(),
        truncation_factor=100.0,
    )
    kappa_s_ref, rs_ref, rt_ref = _los_pipes_reference_convert_to_lens_unit(
        1e10, 7.0, 0.3, 2.0, 100.0
    )
    assert nfw.kappa_s == pytest.approx(kappa_s_ref, rel=1.0e-6)
    assert nfw.scale_radius == pytest.approx(rs_ref, rel=1.0e-6)
    assert nfw.truncation_radius == pytest.approx(rt_ref, rel=1.0e-6)


def test__from_m200_concentration__centre_is_preserved():
    nfw = ag.mp.NFWTruncatedSph.from_m200_concentration(
        centre=(1.5, -0.3),
        m200_solar_mass=1e9,
        concentration=10.0,
        redshift_halo=0.5,
        redshift_source=1.0,
    )
    assert nfw.centre == (1.5, -0.3)


def test__from_m200_concentration__default_cosmology():
    """from_m200_concentration with no cosmology argument uses Planck15."""
    nfw_default = ag.mp.NFWTruncatedSph.from_m200_concentration(
        m200_solar_mass=1e9, concentration=10.0, redshift_halo=0.5, redshift_source=1.0
    )
    nfw_explicit = ag.mp.NFWTruncatedSph.from_m200_concentration(
        m200_solar_mass=1e9,
        concentration=10.0,
        redshift_halo=0.5,
        redshift_source=1.0,
        cosmology=Planck15(),
    )
    assert nfw_default.kappa_s == pytest.approx(nfw_explicit.kappa_s, rel=1.0e-10)


# ---------------------------------------------------------------------------
# Tests for m200_concentration_from
# ---------------------------------------------------------------------------


def test__m200_concentration_from__round_trip_m200():
    """m200_concentration_from recovers the input M_200 to better than 0.01%."""
    nfw = ag.mp.NFWTruncatedSph.from_m200_concentration(
        m200_solar_mass=1e9,
        concentration=10.0,
        redshift_halo=0.5,
        redshift_source=1.0,
        cosmology=Planck15(),
    )
    m200_recovered, _ = ag.mp.NFWTruncatedSph.m200_concentration_from(
        kappa_s=nfw.kappa_s,
        scale_radius=nfw.scale_radius,
        redshift_halo=0.5,
        redshift_source=1.0,
        cosmology=Planck15(),
    )
    assert m200_recovered == pytest.approx(1e9, rel=1.0e-4)


def test__m200_concentration_from__round_trip_concentration():
    """m200_concentration_from recovers the input concentration to better than 0.01%."""
    nfw = ag.mp.NFWTruncatedSph.from_m200_concentration(
        m200_solar_mass=1e9,
        concentration=10.0,
        redshift_halo=0.5,
        redshift_source=1.0,
        cosmology=Planck15(),
    )
    _, c_recovered = ag.mp.NFWTruncatedSph.m200_concentration_from(
        kappa_s=nfw.kappa_s,
        scale_radius=nfw.scale_radius,
        redshift_halo=0.5,
        redshift_source=1.0,
        cosmology=Planck15(),
    )
    assert c_recovered == pytest.approx(10.0, rel=1.0e-4)


def test__m200_concentration_from__round_trip_high_mass():
    """Round-trip test for a high-mass halo (M_200 = 1e12 M_sun)."""
    nfw = ag.mp.NFWTruncatedSph.from_m200_concentration(
        m200_solar_mass=1e12,
        concentration=5.0,
        redshift_halo=0.3,
        redshift_source=2.0,
        cosmology=Planck15(),
    )
    m200_recovered, c_recovered = ag.mp.NFWTruncatedSph.m200_concentration_from(
        kappa_s=nfw.kappa_s,
        scale_radius=nfw.scale_radius,
        redshift_halo=0.3,
        redshift_source=2.0,
        cosmology=Planck15(),
    )
    assert m200_recovered == pytest.approx(1e12, rel=1.0e-4)
    assert c_recovered == pytest.approx(5.0, rel=1.0e-4)


def test__m200_concentration_from__default_cosmology():
    """m200_concentration_from with no cosmology argument uses Planck15."""
    m200_default, c_default = ag.mp.NFWTruncatedSph.m200_concentration_from(
        kappa_s=0.005781,
        scale_radius=0.279092,
        redshift_halo=0.5,
        redshift_source=1.0,
    )
    m200_explicit, c_explicit = ag.mp.NFWTruncatedSph.m200_concentration_from(
        kappa_s=0.005781,
        scale_radius=0.279092,
        redshift_halo=0.5,
        redshift_source=1.0,
        cosmology=Planck15(),
    )
    assert m200_default == pytest.approx(m200_explicit, rel=1.0e-10)
    assert c_default == pytest.approx(c_explicit, rel=1.0e-10)


# ---------------------------------------------------------------------------
# Tests for mass_ratio_from_concentration_and_truncation_factor
# ---------------------------------------------------------------------------


def test__mass_ratio_from_concentration_and_truncation_factor__matches_los_pipes():
    """Mass ratio from source code matches the los_pipes scale_c(c) reference."""
    c = 10.0
    expected = _los_pipes_reference_mass_ratio(c, 100.0)
    result = ag.mp.NFWTruncatedSph.mass_ratio_from_concentration_and_truncation_factor(
        concentration=c, truncation_factor=100.0
    )
    assert result == pytest.approx(expected, rel=1.0e-6)


def test__mass_ratio_from_concentration_and_truncation_factor__larger_for_smaller_factor():
    """A smaller truncation factor (larger truncation radius) gives a larger mass ratio."""
    c = 10.0
    ratio_100 = (
        ag.mp.NFWTruncatedSph.mass_ratio_from_concentration_and_truncation_factor(
            c, 100.0
        )
    )
    ratio_50 = (
        ag.mp.NFWTruncatedSph.mass_ratio_from_concentration_and_truncation_factor(
            c, 50.0
        )
    )
    assert ratio_50 > ratio_100


def test__mass_ratio_from_concentration_and_truncation_factor__various_concentrations():
    """Spot-check mass ratios at several concentrations against los_pipes reference."""
    for c in [5.0, 10.0, 20.0]:
        expected = _los_pipes_reference_mass_ratio(c, 100.0)
        result = (
            ag.mp.NFWTruncatedSph.mass_ratio_from_concentration_and_truncation_factor(
                c, 100.0
            )
        )
        assert result == pytest.approx(expected, rel=1.0e-6), f"failed for c={c}"


def test__mass_ratio_from_concentration_and_truncation_factor__default_truncation_factor():
    """The default truncation_factor=100 is applied when not specified."""
    c = 10.0
    result_default = (
        ag.mp.NFWTruncatedSph.mass_ratio_from_concentration_and_truncation_factor(c)
    )
    result_explicit = (
        ag.mp.NFWTruncatedSph.mass_ratio_from_concentration_and_truncation_factor(
            c, 100.0
        )
    )
    assert result_default == pytest.approx(result_explicit, rel=1.0e-10)
