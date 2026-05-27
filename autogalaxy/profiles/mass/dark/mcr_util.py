import numpy as np


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


def ludlow16_cosmology(mass_at_200, redshift_object, redshift_source, xp=np):
    """
    Ludlow et al. 2016 concentration plus the three cosmological quantities
    needed to convert it into AutoGalaxy NFW parameters
    (``kappa_s``, ``scale_radius``).

    The concentration is computed via ``ludlow16.ludlow16_concentration``
    (a JAX-native port of ``colossus.halo.concentration.modelLudlow16``).
    The cosmology quantities flow through the autogalaxy ``Planck15``
    cosmology, which already supports both numpy and JAX via its own
    ``xp`` parameter.

    Mass is in Msun (physical) — colossus internally works in Msun/h, so
    the conversion is applied here.

    Returns
    -------
    concentration : scalar xp array
    cosmic_average_density : scalar xp array
        Critical density at ``redshift_object``, in Msun / kpc^3.
    critical_surface_density : scalar xp array
        Sigma_crit between the lens and source planes, in Msun / kpc^2.
    kpc_per_arcsec : scalar xp array
        Angular-diameter scale at ``redshift_object``, in kpc / arcsec.
    """
    from autogalaxy.cosmology.model import Planck15
    from autogalaxy.profiles.mass.dark.ludlow16 import (
        ludlow16_concentration,
        PLANCK15_COSMOLOGY,
    )

    cosmology = Planck15()
    h = PLANCK15_COSMOLOGY["h"]

    concentration = ludlow16_concentration(
        mass_at_200 * h,
        redshift_object,
        xp=xp,
        **PLANCK15_COSMOLOGY,
    )

    cosmic_average_density = cosmology.critical_density(redshift_object, xp=xp)
    critical_surface_density = (
        cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
            redshift_0=redshift_object,
            redshift_1=redshift_source,
            xp=xp,
        )
    )
    kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=redshift_object, xp=xp)

    return (
        concentration,
        cosmic_average_density,
        critical_surface_density,
        kpc_per_arcsec,
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

    (
        concentration,
        cosmic_average_density,
        critical_surface_density,
        kpc_per_arcsec,
    ) = ludlow16_cosmology(
        mass_at_200,
        redshift_object,
        redshift_source,
        xp=xp,
    )

    # Apply scatter (JAX-safe)
    concentration = 10.0 ** (xp.log10(concentration) + scatter_sigma * 0.15)

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

    (
        concentration,
        cosmic_average_density,
        critical_surface_density,
        kpc_per_arcsec,
    ) = ludlow16_cosmology(
        mass_at_200,
        redshift_object,
        redshift_source,
        xp=xp,
    )

    # Apply scatter (JAX-safe)
    concentration = 10.0 ** (xp.log10(concentration) + scatter_sigma * 0.15)

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

    if xp is np and mcr_penarrubia <= 0:
        import warnings

        warnings.warn(
            f"cNFW Penarrubia MCR integral factor is non-positive "
            f"({mcr_penarrubia:.4e}) for f_c={f_c:.4f}, "
            f"concentration={concentration:.2f}. This produces an unphysical "
            f"negative kappa_s. The cored NFW parameterisation is not valid "
            f"for core fractions this large (f_c > ~0.18 for typical "
            f"concentrations). Clamping to a small positive value.",
            stacklevel=3,
        )
        mcr_penarrubia = 1e-10

    scale_radius_kpc = radius_at_200 / concentration  # scale radius in kpc
    rho_0 = mass_at_200 / (4 * xp.pi * scale_radius_kpc**3 * mcr_penarrubia)
    kappa_s = rho_0 * scale_radius_kpc / critical_surface_density  # kappa_s
    scale_radius = scale_radius_kpc / kpc_per_arcsec  # scale radius in arcsec
    core_radius = f_c * scale_radius  # core radius in arcsec

    return kappa_s, scale_radius, core_radius, radius_at_200
