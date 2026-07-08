import numpy as np
import pytest

import autogalaxy as ag


def test__zero_interaction_limit_matches_nfw():
    kaplinghat = ag.mp.KaplinghatCoredNFWSph(
        centre=(0.1, -0.2),
        kappa_s=0.2,
        scale_radius=2.0,
        sigma_over_m=0.0,
        t_age=10.0,
    )
    nfw = ag.mp.NFWSph(centre=(0.1, -0.2), kappa_s=0.2, scale_radius=2.0)

    grid = ag.Grid2DIrregular([[0.5, 0.2], [1.0, -0.2], [2.0, 1.0]])

    assert kaplinghat.convergence_2d_from(grid=grid) == pytest.approx(
        nfw.convergence_2d_from(grid=grid).array,
        rel=1.0e-8,
    )
    assert kaplinghat.deflections_yx_2d_from(grid=grid) == pytest.approx(
        nfw.deflections_yx_2d_from(grid=grid).array,
        rel=1.0e-8,
    )


def test__density_and_mass_match_nfw_at_interaction_radius():
    profile = ag.mp.KaplinghatCoredNFWSph(
        kappa_s=0.2,
        scale_radius=2.0,
        interaction_radius=0.4,
    )

    eps = 1.0e-5
    density_inside = profile.density_3d_func(ag.ArrayIrregular([0.4 * (1.0 - eps)]))[0]
    density_outside = profile.density_3d_func(ag.ArrayIrregular([0.4 * (1.0 + eps)]))[0]

    assert density_inside == pytest.approx(density_outside, rel=1.0e-3)


def test__lensing_quantities_are_finite_and_positive():
    profile = ag.mp.KaplinghatCoredNFWSph(
        kappa_s=0.2,
        scale_radius=2.0,
        interaction_radius=0.5,
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


def test__vmapped_deflections_match_instance_path_for_zero_interaction():
    profile = ag.mp.KaplinghatCoredNFWSph(
        centre=(0.1, -0.2),
        kappa_s=0.2,
        scale_radius=2.0,
        interaction_radius=0.0,
    )
    grid = np.array([[0.5, 0.2], [1.0, -0.2], [2.0, 1.0]])

    params = np.array(
        [
            [
                profile.centre[0],
                profile.centre[1],
                profile.kappa_s,
                profile.scale_radius,
                profile.interaction_radius,
                profile.central_density,
                profile.isothermal_radius,
            ]
        ]
    )
    mask = np.array([True])

    vmapped = ag.mp.KaplinghatCoredNFWSph.vmapped_deflections_from(
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


def test__vmapped_deflections_match_instance_path_for_sidm_core():
    profile = ag.mp.KaplinghatCoredNFWSph(
        centre=(0.1, -0.2),
        kappa_s=0.2,
        scale_radius=2.0,
        interaction_radius=0.5,
    )
    grid = np.array([[0.5, 0.2], [1.0, -0.2], [2.0, 1.0]])

    params = np.array(
        [
            [
                profile.centre[0],
                profile.centre[1],
                profile.kappa_s,
                profile.scale_radius,
                profile.interaction_radius,
                profile.central_density,
                profile.isothermal_radius,
            ]
        ]
    )
    mask = np.array([True])

    vmapped = ag.mp.KaplinghatCoredNFWSph.vmapped_deflections_from(
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


def test__mcr_constructor_reduces_to_nfw_when_interaction_is_zero():
    kaplinghat = ag.mp.KaplinghatCoredNFWMCRLudlowSph(
        centre=(1.0, 2.0),
        mass_at_200=1.0e9,
        sigma_over_m=0.0,
        t_age=10.0,
        redshift_object=0.6,
        redshift_source=2.5,
    )
    nfw = ag.mp.NFWSph(
        centre=(1.0, 2.0),
        kappa_s=kaplinghat.kappa_s,
        scale_radius=kaplinghat.scale_radius,
    )

    grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0]])

    assert kaplinghat.interaction_radius == 0.0
    assert kaplinghat.deflections_yx_2d_from(grid=grid) == pytest.approx(
        nfw.deflections_yx_2d_from(grid=grid).array,
        rel=1.0e-8,
    )


def test__mcr_constructor_with_nonzero_scattering_sets_interaction_radius():
    profile = ag.mp.KaplinghatCoredNFWMCRLudlowSph(
        mass_at_200=1.0e9,
        sigma_over_m=10.0,
        t_age=10.0,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    assert profile.interaction_radius > 0.0
    assert profile.isothermal_radius > 0.0
    assert np.isfinite(profile.central_density)
