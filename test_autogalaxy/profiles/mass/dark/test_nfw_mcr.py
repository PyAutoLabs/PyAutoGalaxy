import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__mass_and_concentration_consistent_with_normal_nfw():

    from autogalaxy.cosmology.model import FlatLambdaCDM

    cosmology = FlatLambdaCDM(H0=70.0, Om0=0.3)

    mp = ag.mp.NFWMCRDuffySph(
        centre=(1.0, 2.0),
        mass_at_200=1.0e9,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    mass_at_200_via_mass = mp.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_mass = mp.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    nfw_kappa_s = ag.mp.NFWSph(
        centre=(1.0, 2.0),
        kappa_s=mp.kappa_s,
        scale_radius=mp.scale_radius,
    )

    mass_at_200_via_kappa_s = nfw_kappa_s.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_kappa_s = nfw_kappa_s.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    # We are using the NFWTruncatedSph to check that the mass gives a consistent kappa_s, given certain radii.

    assert mass_at_200_via_kappa_s == mass_at_200_via_mass
    assert concentration_via_kappa_s == concentration_via_mass

    assert mp.centre == (1.0, 2.0)

    assert mp.axis_ratio() == 1.0

    assert mp.angle() == 0.0

    assert mp.inner_slope == 1.0

    assert mp.scale_radius == pytest.approx(0.273382, 1.0e-3)


def test__mass_and_concentration_consistent_with_normal_nfw__scatter_0():

    from autogalaxy.cosmology.model import FlatLambdaCDM

    cosmology = FlatLambdaCDM(H0=70.0, Om0=0.3)

    mp = ag.mp.NFWMCRLudlowSph(
        centre=(1.0, 2.0),
        mass_at_200=1.0e9,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    mass_at_200_via_mass = mp.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_mass = mp.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    nfw_kappa_s = ag.mp.NFWSph(
        centre=(1.0, 2.0),
        kappa_s=mp.kappa_s,
        scale_radius=mp.scale_radius,
    )

    mass_at_200_via_kappa_s = nfw_kappa_s.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_kappa_s = nfw_kappa_s.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    # We uare using the NFWTruncatedSph to check the mass gives a conosistnt kappa_s, given certain radii.

    assert mass_at_200_via_kappa_s == mass_at_200_via_mass
    assert concentration_via_kappa_s == concentration_via_mass

    assert mp.centre == (1.0, 2.0)
    assert mp.axis_ratio() == 1.0
    assert mp.angle() == 0.0
    assert mp.inner_slope == 1.0
    assert mp.scale_radius == pytest.approx(0.21157, 1.0e-3)

    deflections_ludlow = mp.deflections_yx_2d_from(grid=grid)
    deflections = nfw_kappa_s.deflections_yx_2d_from(grid=grid)

    assert (deflections_ludlow == deflections).all()


def test__same_as_above_but_elliptical():

    from autogalaxy.cosmology.model import FlatLambdaCDM

    cosmology = FlatLambdaCDM(H0=70.0, Om0=0.3)

    mp = ag.mp.NFWMCRLudlow(
        centre=(1.0, 2.0),
        ell_comps=(0.1, 0.2),
        mass_at_200=1.0e9,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    mass_at_200_via_mass = mp.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_mass = mp.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    nfw_kappa_s = ag.mp.NFW(
        centre=(1.0, 2.0),
        ell_comps=(0.1, 0.2),
        kappa_s=mp.kappa_s,
        scale_radius=mp.scale_radius,
    )

    mass_at_200_via_kappa_s = nfw_kappa_s.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_kappa_s = nfw_kappa_s.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    # We uare using the NFWTruncatedSph to check the mass gives a conosistnt kappa_s, given certain radii.

    assert mass_at_200_via_kappa_s == mass_at_200_via_mass
    assert concentration_via_kappa_s == concentration_via_mass

    assert mp.centre == (1.0, 2.0)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=(0.1, 0.2))

    assert mp.axis_ratio() == axis_ratio
    assert mp.angle() == angle
    assert mp.inner_slope == 1.0

    assert mp.scale_radius == pytest.approx(0.211578, 1.0e-3)

    deflections_ludlow = mp.deflections_yx_2d_from(grid=grid)
    deflections = nfw_kappa_s.deflections_yx_2d_from(grid=grid)

    assert (deflections_ludlow == deflections).all()


def test__same_as_above_but_generalized_elliptical():

    from autogalaxy.cosmology.model import FlatLambdaCDM

    cosmology = FlatLambdaCDM(H0=70.0, Om0=0.3)

    mp = ag.mp.gNFWMCRLudlow(
        centre=(1.0, 2.0),
        ell_comps=(0.1, 0.2),
        mass_at_200=1.0e9,
        inner_slope=2.0,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    mass_at_200_via_mass = mp.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_mass = mp.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    nfw_kappa_s = ag.mp.gNFW(
        centre=(1.0, 2.0),
        ell_comps=(0.1, 0.2),
        kappa_s=mp.kappa_s,
        scale_radius=mp.scale_radius,
        inner_slope=2.0,
    )

    mass_at_200_via_kappa_s = nfw_kappa_s.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_kappa_s = nfw_kappa_s.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    # We uare using the NFWTruncatedSph to check the mass gives a conosistnt kappa_s, given certain radii.

    assert mass_at_200_via_kappa_s == mass_at_200_via_mass
    assert concentration_via_kappa_s == concentration_via_mass

    assert mp.centre == (1.0, 2.0)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=(0.1, 0.2))

    assert mp.axis_ratio() == axis_ratio
    assert mp.angle() == angle
    assert mp.inner_slope == 2.0

    assert mp.scale_radius == pytest.approx(0.21157, 1.0e-3)

    deflections_ludlow = mp.deflections_yx_2d_from(grid=grid)
    deflections = nfw_kappa_s.deflections_yx_2d_from(grid=grid)

    assert (deflections_ludlow == deflections).all()


def test__same_as_above_but_cored_nfw__spherical():

    from autogalaxy.cosmology.model import FlatLambdaCDM

    cosmology = FlatLambdaCDM(H0=70.0, Om0=0.3)

    mp = ag.mp.cNFWMCRLudlowSph(
        centre=(1.0, 2.0),
        mass_at_200=1.0e9,
        f_c=0.01,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    mass_at_200_via_mass = mp.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_mass = mp.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    cnfw_kappa_s = ag.mp.cNFWSph(
        centre=(1.0, 2.0),
        kappa_s=mp.kappa_s,
        scale_radius=mp.scale_radius,
        core_radius=mp.core_radius,
    )

    mass_at_200_via_kappa_s = cnfw_kappa_s.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_kappa_s = cnfw_kappa_s.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    # We are using the NFWTruncatedSph to check the mass gives a consistent kappa_s, given certain radii.

    assert mass_at_200_via_kappa_s == mass_at_200_via_mass
    assert concentration_via_kappa_s == concentration_via_mass

    assert mp.centre == (1.0, 2.0)

    assert mp.axis_ratio() == 1.0

    assert mp.angle() == 0.0

    assert mp.scale_radius == pytest.approx(0.21158, 1.0e-3)

    assert mp.core_radius == pytest.approx(0.0021158, 1.0e-3)

    deflections_ludlow = mp.deflections_yx_2d_from(grid=grid)
    deflections = cnfw_kappa_s.deflections_yx_2d_from(grid=grid)

    assert (deflections_ludlow == deflections).all()


def test__same_as_above_but_cored_nfw__elliptical():

    from autogalaxy.cosmology.model import FlatLambdaCDM

    cosmology = FlatLambdaCDM(H0=70.0, Om0=0.3)

    mp = ag.mp.cNFWMCRLudlow(
        centre=(1.0, 2.0),
        ell_comps=(0.1, 0.2),
        mass_at_200=1.0e9,
        f_c=0.01,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    mass_at_200_via_mass = mp.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_mass = mp.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    cnfw_kappa_s = ag.mp.cNFW(
        centre=(1.0, 2.0),
        ell_comps=(0.1, 0.2),
        kappa_s=mp.kappa_s,
        scale_radius=mp.scale_radius,
        core_radius=mp.core_radius,
    )

    mass_at_200_via_kappa_s = cnfw_kappa_s.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_kappa_s = cnfw_kappa_s.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    # We are using the NFWTruncatedSph to check the mass gives a consistent kappa_s, given certain radii.

    assert mass_at_200_via_kappa_s == mass_at_200_via_mass
    assert concentration_via_kappa_s == concentration_via_mass

    assert mp.centre == (1.0, 2.0)

    assert mp.ell_comps == (0.1, 0.2)

    assert mp.axis_ratio() == cnfw_kappa_s.axis_ratio()

    assert mp.angle() == cnfw_kappa_s.angle()

    assert mp.scale_radius == pytest.approx(0.21158, 1.0e-3)

    assert mp.core_radius == pytest.approx(0.0021158, 1.0e-3)

    deflections_ludlow = mp.deflections_yx_2d_from(grid=grid)
    deflections = cnfw_kappa_s.deflections_yx_2d_from(grid=grid)

    assert (deflections_ludlow == deflections).all()


def _gnfw_mcr_mass_3d_within_r200(mp, redshift_object, redshift_source):
    from scipy.integrate import quad

    from autogalaxy.cosmology.model import Planck15
    from autogalaxy.profiles.mass.dark import mcr_util

    cosmology = Planck15()

    _, _, radius_at_200 = mcr_util.kappa_s_and_scale_radius_for_ludlow(
        mass_at_200=mp.mass_at_200,
        scatter_sigma=0.0,
        redshift_object=redshift_object,
        redshift_source=redshift_source,
    )

    critical_surface_density = (
        cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
            redshift_0=redshift_object, redshift_1=redshift_source, xp=np
        )
    )
    kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=redshift_object, xp=np)

    scale_radius_kpc = float(mp.scale_radius) * kpc_per_arcsec
    rho_s = float(mp.kappa_s) * critical_surface_density / scale_radius_kpc
    gamma = float(mp.inner_slope)

    integral = quad(
        lambda x: x ** (2.0 - gamma) * (1.0 + x) ** (gamma - 3.0),
        0.0,
        radius_at_200 / scale_radius_kpc,
    )[0]

    return 4.0 * np.pi * rho_s * scale_radius_kpc**3 * integral


@pytest.mark.parametrize("inner_slope", [0.5, 1.0, 1.5])
def test__gnfw_mcr_ludlow__mass_within_r200_is_mass_at_200(inner_slope):

    mp = ag.mp.gNFWMCRLudlow(
        mass_at_200=1.0e12,
        inner_slope=inner_slope,
        redshift_object=0.5,
        redshift_source=1.0,
    )

    mass_3d = _gnfw_mcr_mass_3d_within_r200(
        mp, redshift_object=0.5, redshift_source=1.0
    )

    assert mass_3d == pytest.approx(1.0e12, rel=1.0e-3)


def test__gnfw_mcr_ludlow__inner_slope_1_is_nfw_mcr_ludlow():

    gnfw = ag.mp.gNFWMCRLudlow(
        mass_at_200=1.0e12, inner_slope=1.0, redshift_object=0.5, redshift_source=1.0
    )
    nfw = ag.mp.NFWMCRLudlow(
        mass_at_200=1.0e12, redshift_object=0.5, redshift_source=1.0
    )

    assert gnfw.kappa_s == pytest.approx(nfw.kappa_s, rel=1.0e-8)
    assert gnfw.scale_radius == pytest.approx(nfw.scale_radius, rel=1.0e-8)


@pytest.mark.parametrize("inner_slope", [0.5, 1.5])
def test__gnfw_mcr_ludlow__jax_jit_matches_numpy(inner_slope):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from autogalaxy.profiles.mass.dark import mcr_util

    kappa_s_np, scale_radius_np, _ = mcr_util.kappa_s_and_scale_radius_for_ludlow_gnfw(
        mass_at_200=1.0e12,
        scatter_sigma=0.0,
        redshift_object=0.5,
        redshift_source=1.0,
        inner_slope=inner_slope,
    )

    @jax.jit
    def f(mass_at_200, inner_slope):
        return mcr_util.kappa_s_and_scale_radius_for_ludlow_gnfw(
            mass_at_200=mass_at_200,
            scatter_sigma=0.0,
            redshift_object=0.5,
            redshift_source=1.0,
            inner_slope=inner_slope,
        )

    kappa_s_jax, scale_radius_jax, _ = f(jnp.array(1.0e12), jnp.array(inner_slope))

    assert float(kappa_s_jax) == pytest.approx(kappa_s_np, rel=1.0e-4)
    assert float(scale_radius_jax) == pytest.approx(scale_radius_np, rel=1.0e-4)
