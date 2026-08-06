import importlib.util

import numpy as np
import pytest

import autogalaxy as ag

from autogalaxy.profiles.mass.dark.yang24 import _yang24_parameter_ratios_from
from autogalaxy.profiles.mass.dark.yang24_mcr import _sigma_eff_over_m_from

# The `vmapped_deflections_from` path is jax-backed; these tests need jax
# installed to run (it ships via the `[optional]` extras). The NumPy-only
# Python-version matrix has no jax, so skip there rather than fail.
requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="requires jax (installed via the [optional] extras; absent on the NumPy-only matrix env)",
)


def test__parameter_ratios_are_exact_nfw_at_zero_tau():
    rho_s_ratio, r_s_ratio, r_c_ratio = _yang24_parameter_ratios_from(tau=0.0)

    assert rho_s_ratio == pytest.approx(1.0, abs=1.0e-12)
    assert r_s_ratio == pytest.approx(1.0, abs=1.0e-12)
    assert r_c_ratio == 0.0


def test__zero_tau_limit_matches_nfw():
    yang = ag.mp.YangSIDMSph(
        centre=(0.1, -0.2),
        kappa_s=0.2,
        scale_radius=2.0,
        tau=0.0,
    )
    nfw = ag.mp.NFWSph(centre=(0.1, -0.2), kappa_s=0.2, scale_radius=2.0)

    grid = ag.Grid2DIrregular([[0.5, 0.2], [1.0, -0.2], [2.0, 1.0]])

    assert yang.convergence_2d_from(grid=grid) == pytest.approx(
        nfw.convergence_2d_from(grid=grid).array,
        rel=1.0e-8,
    )
    assert yang.deflections_yx_2d_from(grid=grid) == pytest.approx(
        nfw.deflections_yx_2d_from(grid=grid).array,
        rel=1.0e-8,
    )


def test__evolved_halo_has_constant_density_core():
    profile = ag.mp.YangSIDMSph(
        kappa_s=0.2,
        scale_radius=2.0,
        tau=0.5,
    )

    assert profile.core_radius_evolved > 0.0

    density_centre = profile.density_3d_func(ag.ArrayIrregular([1.0e-6]))[0]

    # At the centre the beta = 4 cored form tends to the finite value
    # rho_s * r_s / r_c.
    analytic_centre = (
        profile.rho_s_evolved
        * profile.scale_radius_evolved
        / profile.core_radius_evolved
    )

    assert np.isfinite(density_centre)
    assert density_centre == pytest.approx(analytic_centre, rel=1.0e-4)

    nfw_density = ag.mp.NFWSph(kappa_s=0.2, scale_radius=2.0).density_3d_func(
        ag.ArrayIrregular([1.0e-6])
    )[0]
    assert density_centre < nfw_density


def test__tau_is_truncated_into_fitted_range():
    profile = ag.mp.YangSIDMSph(kappa_s=0.2, scale_radius=2.0, tau=1.7)
    truncated = ag.mp.YangSIDMSph(kappa_s=0.2, scale_radius=2.0, tau=1.0)

    assert profile.tau == 1.0
    assert profile.rho_s_evolved == truncated.rho_s_evolved

    assert ag.mp.YangSIDMSph(kappa_s=0.2, scale_radius=2.0, tau=-0.3).tau == 0.0


def test__lensing_quantities_are_finite_and_positive():
    profile = ag.mp.YangSIDMSph(
        kappa_s=0.2,
        scale_radius=2.0,
        tau=0.5,
    )
    grid = ag.Grid2DIrregular([[0.0, 0.0], [0.2, 0.0], [1.0, 0.0]])

    convergence = np.asarray(profile.convergence_2d_from(grid=grid).array)
    deflections = np.asarray(profile.deflections_yx_2d_from(grid=grid).array)
    potential = np.asarray(profile.potential_2d_from(grid=grid).array)

    assert np.isfinite(convergence).all()
    assert np.isfinite(deflections).all()
    assert np.isfinite(potential).all()
    assert (convergence > 0.0).all()
    assert deflections[0] == pytest.approx(np.array([0.0, 0.0]), abs=1.0e-8)


@requires_jax
def test__vmapped_deflections_match_instance_path_for_zero_tau():
    profile = ag.mp.YangSIDMSph(
        centre=(0.1, -0.2),
        kappa_s=0.2,
        scale_radius=2.0,
        tau=0.0,
    )
    grid = np.array([[0.5, 0.2], [1.0, -0.2], [2.0, 1.0]])

    params = np.array(
        [
            [
                profile.centre[0],
                profile.centre[1],
                profile.rho_s_evolved,
                profile.scale_radius_evolved,
                profile.core_radius_evolved,
            ]
        ]
    )
    mask = np.array([True])

    vmapped = ag.mp.YangSIDMSph.vmapped_deflections_from(
        grid=grid,
        params_batch=params,
        mask=mask,
    )

    np.testing.assert_allclose(
        np.asarray(vmapped),
        profile.deflections_yx_2d_from(grid=ag.Grid2DIrregular(grid)).array,
        rtol=1.0e-6,
        atol=1.0e-8,
    )


@requires_jax
def test__vmapped_deflections_match_instance_path_for_evolved_halo():
    profile = ag.mp.YangSIDMSph(
        centre=(0.1, -0.2),
        kappa_s=0.2,
        scale_radius=2.0,
        tau=0.5,
    )
    grid = np.array([[0.5, 0.2], [1.0, -0.2], [2.0, 1.0]])

    params = np.array(
        [
            [
                profile.centre[0],
                profile.centre[1],
                profile.rho_s_evolved,
                profile.scale_radius_evolved,
                profile.core_radius_evolved,
            ]
        ]
    )
    mask = np.array([True])

    vmapped = ag.mp.YangSIDMSph.vmapped_deflections_from(
        grid=grid,
        params_batch=params,
        mask=mask,
    )

    np.testing.assert_allclose(
        np.asarray(vmapped),
        profile.deflections_yx_2d_from(grid=ag.Grid2DIrregular(grid)).array,
        rtol=5.0e-2,
        atol=1.0e-3,
    )


def test__mcr_constructor_reduces_to_nfw_when_cross_section_is_zero():
    yang = ag.mp.YangSIDMMCRLudlowSph(
        centre=(1.0, 2.0),
        mass_at_200=1.0e9,
        sigma_over_m=0.0,
        t_age=10.0,
        redshift_object=0.6,
        redshift_source=2.5,
    )
    nfw = ag.mp.NFWSph(
        centre=(1.0, 2.0),
        kappa_s=yang.kappa_s,
        scale_radius=yang.scale_radius,
    )

    grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0]])

    assert yang.tau == 0.0
    assert yang.deflections_yx_2d_from(grid=grid) == pytest.approx(
        nfw.deflections_yx_2d_from(grid=grid).array,
        rel=1.0e-8,
    )


def test__mcr_constructor_tau_increases_with_cross_section():
    taus = [
        ag.mp.YangSIDMMCRLudlowSph(
            mass_at_200=1.0e9,
            sigma_over_m=sigma_over_m,
            t_age=10.0,
            redshift_object=0.6,
            redshift_source=2.5,
        ).tau_physical
        for sigma_over_m in [1.0, 10.0, 100.0]
    ]

    assert taus[0] > 0.0
    assert taus[0] < taus[1] < taus[2]
    assert taus[1] == pytest.approx(10.0 * taus[0], rel=1.0e-8)


def test__effective_cross_section_velocity_dependence():
    nu_eff = 20.0

    constant = _sigma_eff_over_m_from(
        sigma_over_m=3.0,
        velocity_exponent=0.0,
        velocity_ref=10.0,
        nu_eff_km_s=nu_eff,
    )
    assert constant == 3.0

    # For sigma ~ v^-a quoted at a reference velocity below the halo's
    # characteristic velocity, the effective cross section is suppressed.
    suppressed = _sigma_eff_over_m_from(
        sigma_over_m=3.0,
        velocity_exponent=1.0,
        velocity_ref=10.0,
        nu_eff_km_s=nu_eff,
    )
    assert suppressed < constant

    # An a = 2 power law has the closed form (v0 / 2 nu)^2 * Gamma(3) / 6.
    analytic = 3.0 * (10.0 / (2.0 * nu_eff)) ** 2 * 2.0 / 6.0
    assert _sigma_eff_over_m_from(
        sigma_over_m=3.0,
        velocity_exponent=2.0,
        velocity_ref=10.0,
        nu_eff_km_s=nu_eff,
    ) == pytest.approx(analytic, rel=1.0e-12)

    with pytest.raises(ValueError):
        _sigma_eff_over_m_from(
            sigma_over_m=3.0,
            velocity_exponent=8.0,
            velocity_ref=10.0,
            nu_eff_km_s=nu_eff,
        )
