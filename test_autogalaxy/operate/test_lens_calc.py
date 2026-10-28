import numpy as np
import pytest

import autogalaxy as ag


@pytest.fixture(name="grid_aperture")
def make_grid_aperture():
    # 0.05" pixels, extent +/- 2.6" so a 2.5" aperture about the origin fits.
    return ag.Grid2D.uniform(shape_native=(104, 104), pixel_scales=0.05)


def _nfw_sph_mass_within_circle(kappa_s, scale_radius, radius):
    # Closed-form projected mass of a spherical NFW inside x = R / r_s < 1
    # (Bartelmann 1996), in the convergence units of `ag.mp.NFWSph`.
    x = radius / scale_radius
    h = np.log(x / 2.0) + np.arccosh(1.0 / x) / np.sqrt(1.0 - x**2)
    return 4.0 * np.pi * kappa_s * scale_radius**2 * h


@pytest.mark.parametrize("radius", [1.0, 2.5])
def test__mass_angular_within_circle_via_grid_from__isothermal__matches_quadrature(
    grid_aperture, radius
):
    mass = ag.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0)

    mass_via_grid = ag.mass_angular_within_circle_via_grid_from(
        mass_obj=mass, radius=radius, centre=(0.0, 0.0), grid=grid_aperture
    )

    assert mass_via_grid == pytest.approx(
        mass.mass_angular_within_circle_from(radius=radius), rel=1.0e-3
    )
    assert mass_via_grid == pytest.approx(np.pi * radius, rel=1.0e-3)


@pytest.mark.parametrize("radius", [1.0, 2.5])
def test__mass_angular_within_circle_via_grid_from__nfw_sph__matches_closed_form(
    grid_aperture, radius
):
    # `NFWSph.mass_angular_within_circle_from` currently fails inside `scipy.integrate.quad`
    # (its `convergence_func` returns a length-1 array, not a scalar), so the oracle is the
    # closed-form NFW enclosed mass that this quadrature would integrate to.
    mass = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=0.2, scale_radius=5.0)

    mass_via_grid = ag.mass_angular_within_circle_via_grid_from(
        mass_obj=mass, radius=radius, centre=(0.0, 0.0), grid=grid_aperture
    )

    assert mass_via_grid == pytest.approx(
        _nfw_sph_mass_within_circle(kappa_s=0.2, scale_radius=5.0, radius=radius),
        rel=1.0e-3,
    )


@pytest.mark.parametrize("radius", [0.5, 1.0])
def test__mass_angular_within_circle_via_grid_from__gaussian__matches_closed_form(
    grid_aperture, radius
):
    intensity, sigma, mass_to_light_ratio = 0.7, 0.5, 2.3

    mass = ag.lmp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=intensity,
        sigma=sigma,
        mass_to_light_ratio=mass_to_light_ratio,
    )

    mass_via_grid = ag.mass_angular_within_circle_via_grid_from(
        mass_obj=mass, radius=radius, centre=(0.0, 0.0), grid=grid_aperture
    )

    # Note: the profile's axis ratio is floored at 0.9999, which alone puts the exactly circular
    # closed form ~1e-4 away at large R / sigma; these radii keep that well inside the tolerance.
    closed_form = (
        2.0
        * np.pi
        * mass_to_light_ratio
        * intensity
        * sigma**2
        * (1.0 - np.exp(-(radius**2) / (2.0 * sigma**2)))
    )

    assert mass_via_grid == pytest.approx(closed_form, rel=1.0e-4)


def test__mass_angular_within_circle_via_grid_from__off_centre_aperture(grid_aperture):
    mass = ag.mp.Isothermal(centre=(0.3, -0.2), einstein_radius=1.0)

    mass_via_grid = ag.mass_angular_within_circle_via_grid_from(
        mass_obj=mass, radius=1.0, centre=(0.3, -0.2), grid=grid_aperture
    )

    assert mass_via_grid == pytest.approx(np.pi, rel=1.0e-3)


def test__mass_angular_within_circle_via_grid_from__basis__equals_sum_of_members(
    grid_aperture,
):
    gaussian_0 = ag.lmp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        sigma=0.3,
        mass_to_light_ratio=2.0,
    )
    gaussian_1 = ag.lmp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(-0.05, 0.1),
        intensity=0.4,
        sigma=0.8,
        mass_to_light_ratio=2.0,
    )

    basis = ag.lp_basis.Basis(profile_list=[gaussian_0, gaussian_1])

    mass_basis = ag.mass_angular_within_circle_via_grid_from(
        mass_obj=basis, radius=1.0, centre=(0.0, 0.0), grid=grid_aperture
    )
    mass_0 = ag.mass_angular_within_circle_via_grid_from(
        mass_obj=gaussian_0, radius=1.0, centre=(0.0, 0.0), grid=grid_aperture
    )
    mass_1 = ag.mass_angular_within_circle_via_grid_from(
        mass_obj=gaussian_1, radius=1.0, centre=(0.0, 0.0), grid=grid_aperture
    )

    assert mass_basis == pytest.approx(mass_0 + mass_1, rel=1.0e-10)


def test__mass_angular_within_circle_via_grid_from__galaxy_and_galaxies__equal_profile(
    grid_aperture,
):
    mass = ag.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0)

    galaxy = ag.Galaxy(redshift=0.5, mass=mass)
    galaxies = ag.Galaxies(galaxies=[galaxy])

    mass_profile = ag.mass_angular_within_circle_via_grid_from(
        mass_obj=mass, radius=1.0, centre=(0.0, 0.0), grid=grid_aperture
    )
    mass_galaxy = ag.mass_angular_within_circle_via_grid_from(
        mass_obj=galaxy, radius=1.0, centre=(0.0, 0.0), grid=grid_aperture
    )
    mass_galaxies = ag.mass_angular_within_circle_via_grid_from(
        mass_obj=galaxies, radius=1.0, centre=(0.0, 0.0), grid=grid_aperture
    )

    assert mass_galaxy == pytest.approx(mass_profile, rel=1.0e-10)
    assert mass_galaxies == pytest.approx(mass_profile, rel=1.0e-10)


def test__mass_angular_within_circle_via_grid_from__circle_beyond_grid__raises(
    grid_aperture,
):
    mass = ag.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0)

    with pytest.raises(ValueError):
        ag.mass_angular_within_circle_via_grid_from(
            mass_obj=mass, radius=3.0, centre=(0.0, 0.0), grid=grid_aperture
        )
