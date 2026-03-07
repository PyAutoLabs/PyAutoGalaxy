import numpy as np
import warnings


def kappa_s_and_scale_radius_for_duffy(mass_at_200, redshift_object, redshift_source):
    """
    Computes the AutoGalaxy NFW parameters (kappa_s, scale_radius) for an NFW halo of the given
    mass, enforcing the Duffy '08 mass-concentration relation.

    Interprets mass as *`M_{200c}`*, not `M_{200m}`.
    """
    from autogalaxy.cosmology.model import Planck15

    cosmology = Planck15()

    # Msun / kpc^3  (no units conversion needed)
    cosmic_average_density = cosmology.critical_density(redshift_object, xp=np)

    # Msun / kpc^2
    critical_surface_density = (
        cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
            redshift_0=redshift_object,
            redshift_1=redshift_source,
            xp=np,
        )
    )

    # kpc / arcsec
    kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=redshift_object, xp=np)

    # r200 in kpc
    radius_at_200 = (
        mass_at_200 / (200.0 * cosmic_average_density * (4.0 * np.pi / 3.0))
    ) ** (1.0 / 3.0)

    # Duffy+2008 mass–concentration (as in your code)
    coefficient = 5.71 * (1.0 + redshift_object) ** (-0.47)
    concentration = coefficient * (mass_at_200 / 2.952465309e12) ** (-0.084)

    de_c = (
        200.0
        / 3.0
        * (
            concentration**3
            / (np.log(1.0 + concentration) - concentration / (1.0 + concentration))
        )
    )

    scale_radius_kpc = radius_at_200 / concentration
    rho_s = cosmic_average_density * de_c  # Msun / kpc^3
    kappa_s = rho_s * scale_radius_kpc / critical_surface_density  # dimensionless
    scale_radius = scale_radius_kpc / kpc_per_arcsec  # arcsec

    return kappa_s, scale_radius, radius_at_200


def _ludlow16_cosmology_callback(
    mass_at_200,
    redshift_object,
    redshift_source,
):
    """
    Pure NumPy / Python function.
    Must NEVER see JAX tracers.
    """

    import numpy as np
    from colossus.cosmology import cosmology as col_cosmology
    from colossus.halo.concentration import concentration as col_concentration
    from autogalaxy.cosmology.model import Planck15

    # -----------------------
    # Colossus cosmology
    # -----------------------
    col_cosmo = col_cosmology.setCosmology("planck15")

    m_input = mass_at_200 * col_cosmo.h
    concentration = col_concentration(
        m_input,
        "200c",
        redshift_object,
        model="ludlow16",
    )

    # -----------------------
    # AutoGalaxy cosmology (no astropy.units)
    # -----------------------
    cosmology = Planck15()

    # Msun / kpc^3 (your xp drop-in should return this directly)
    cosmic_average_density = cosmology.critical_density(redshift_object, xp=np)

    # Msun / kpc^2
    critical_surface_density = (
        cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
            redshift_0=redshift_object,
            redshift_1=redshift_source,
            xp=np,
        )
    )

    # kpc / arcsec
    kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=redshift_object, xp=np)

    return (
        np.float64(concentration),
        np.float64(cosmic_average_density),
        np.float64(critical_surface_density),
        np.float64(kpc_per_arcsec),
    )


def ludlow16_cosmology_jax(
    mass_at_200,
    redshift_object,
    redshift_source,
):
    """
    JAX-safe wrapper around Colossus + Astropy cosmology.
    """
    import jax
    import jax.numpy as jnp
    from jax import ShapeDtypeStruct

    return jax.pure_callback(
        _ludlow16_cosmology_callback,
        (
            ShapeDtypeStruct((), jnp.float64),  # concentration
            ShapeDtypeStruct((), jnp.float64),  # rho_crit(z)
            ShapeDtypeStruct((), jnp.float64),  # Sigma_crit
            ShapeDtypeStruct((), jnp.float64),  # kpc/arcsec
        ),
        mass_at_200,
        redshift_object,
        redshift_source,
        vmap_method="sequential",
    )


def kappa_s_and_scale_radius_for_ludlow(
    mass_at_200,
    scatter_sigma,
    redshift_object,
    redshift_source,
):

    if isinstance(mass_at_200, (float, np.ndarray, np.float64)):
        xp = np
    else:
        import jax.numpy as jnp

        xp = jnp

    # ------------------------------------
    # Cosmology + concentration (callback)
    # ------------------------------------

    if xp is np:
        (
            concentration,
            cosmic_average_density,
            critical_surface_density,
            kpc_per_arcsec,
        ) = _ludlow16_cosmology_callback(
            mass_at_200,
            redshift_object,
            redshift_source,
        )
    else:
        (
            concentration,
            cosmic_average_density,
            critical_surface_density,
            kpc_per_arcsec,
        ) = ludlow16_cosmology_jax(
            mass_at_200,
            redshift_object,
            redshift_source,
        )

    # Apply scatter (JAX-safe)
    concentration = 10.0 ** (xp.log10(concentration) + scatter_sigma * 0.15)

    # ------------------------------------
    # JAX-native algebra
    # ------------------------------------
    radius_at_200 = (
        mass_at_200 / (200.0 * cosmic_average_density * (4.0 * xp.pi / 3.0))
    ) ** (1.0 / 3.0)

    de_c = (
        200.0
        / 3.0
        * (
            concentration**3
            / (xp.log(1.0 + concentration) - concentration / (1.0 + concentration))
        )
    )

    scale_radius_kpc = radius_at_200 / concentration
    rho_s = cosmic_average_density * de_c
    kappa_s = rho_s * scale_radius_kpc / critical_surface_density
    scale_radius = scale_radius_kpc / kpc_per_arcsec

    return kappa_s, scale_radius, radius_at_200


def kappa_s_scale_radius_and_core_radius_for_ludlow(
    mass_at_200, scatter_sigma, f_c, redshift_object, redshift_source
):
    """
    Computes the AutoGalaxy cNFW parameters (kappa_s, scale_radius, core_radius) for a cored NFW halo of the given
    mass, enforcing the Penarrubia '12 mass-concentration relation.

    Interprets mass as *`M_{200c}`*, not `M_{200m}`.

    f_c = core_radius / scale radius
    """

    if isinstance(mass_at_200, (float, np.ndarray, np.float64)):
        xp = np
    else:
        import jax.numpy as jnp

        xp = jnp

        # ------------------------------------
        # Cosmology + concentration (callback)
        # ------------------------------------

    if xp is np:
        (
            concentration,
            cosmic_average_density,
            critical_surface_density,
            kpc_per_arcsec,
        ) = _ludlow16_cosmology_callback(
            mass_at_200,
            redshift_object,
            redshift_source,
        )
    else:
        (
            concentration,
            cosmic_average_density,
            critical_surface_density,
            kpc_per_arcsec,
        ) = ludlow16_cosmology_jax(
            mass_at_200,
            redshift_object,
            redshift_source,
        )

    # Apply scatter (JAX-safe)
    concentration = 10.0 ** (xp.log10(concentration) + scatter_sigma * 0.15)

    # ------------------------------------
    # JAX-native algebra
    # ------------------------------------
    radius_at_200 = (
        mass_at_200 / (200.0 * cosmic_average_density * (4.0 * xp.pi / 3.0))
    ) ** (
        1.0 / 3.0
    )  # r200

    mcr_penarrubia = (
        f_c**2 * xp.log(1 + concentration / f_c)
        + (1 - 2 * f_c) * xp.log(1 + concentration)
    ) / (1 + f_c) ** 2 - concentration / (
        (1 + concentration) * (1 - f_c)
    )  # mass concentration relation (Penarrubia+2012)

    scale_radius_kpc = radius_at_200 / concentration  # scale radius in kpc
    rho_0 = mass_at_200 / (4 * xp.pi * scale_radius_kpc**3 * mcr_penarrubia)
    kappa_s = rho_0 * scale_radius_kpc / critical_surface_density  # kappa_s
    scale_radius = scale_radius_kpc / kpc_per_arcsec  # scale radius in arcsec
    core_radius = f_c * scale_radius  # core radius in arcsec

    return kappa_s, scale_radius, core_radius, radius_at_200
