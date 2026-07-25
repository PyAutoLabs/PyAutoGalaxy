from typing import Tuple

from autogalaxy.profiles.mass.dark.gnfw import gNFWSph

import numpy as np
from autogalaxy import cosmology as cosmo


def kappa_s_and_scale_radius(
    cosmology,
    virial_mass,
    concentration,
    overdens,
    redshift_object,
    redshift_source,
    inner_slope,
):
    from scipy.integrate import quad

    # Msun / kpc^3 (no astropy.units)
    critical_density = cosmology.critical_density(redshift_object, xp=np)

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

    if overdens == 0:
        x = cosmology.Om(redshift_object, xp=np) - 1.0
        overdens = 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2  # Bryan & Norman (1998)

    # r_vir in kpc
    virial_radius = (
        virial_mass / (overdens * critical_density * (4.0 * np.pi / 3.0))
    ) ** (1.0 / 3.0)

    # scale radius of gNFW in kpc
    scale_radius_kpc = virial_radius / concentration

    ##############################
    def integrand(r):
        return (r**2 / r**inner_slope) * (1.0 + r / scale_radius_kpc) ** (
            inner_slope - 3.0
        )

    de_c = (
        (overdens / 3.0)
        * (virial_radius**3 / scale_radius_kpc**inner_slope)
        / quad(integrand, 0.0, virial_radius)[0]
    )
    ##############################

    rho_s = critical_density * de_c  # Msun / kpc^3
    kappa_s = rho_s * scale_radius_kpc / critical_surface_density  # dimensionless
    scale_radius = scale_radius_kpc / kpc_per_arcsec  # arcsec

    return kappa_s, scale_radius, virial_radius, overdens


class gNFWVirialMassgNFWConcSph(gNFWSph):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        log10m_vir: float = 12.0,
        c_gNFW: float = 10.0,
        overdens: float = 0.0,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
        inner_slope: float = 1.0,
    ):
        """
        Spherical gNFW profile initialized with the virial mass and c_gNFW concentration of the halo.

        The virial radius of the halo is defined as the radius at which the density of the halo
        equals overdens * the critical density of the Universe. r_vir = (3*m_vir/4*pi*overdens*critical_density)^1/3.

        If the overdens parameter is set to 0, the virial overdensity of Bryan & Norman (1998) will be used.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        log10m_vir
            The log10(virial mass) of the dark matter halo.
        c_gNFW
            The c_gNFW concentration of the dark matter halo
        overdens
            The spherical overdensity used to define the virial radius of the dark matter
            halo: r_vir = (3*m_vir/4*pi*overdens*critical_density)^1/3. If this parameter is set to 0, the virial
            overdensity of Bryan & Norman (1998) will be used.
        redshift_object
            Lens redshift.
        redshift_source
            Source redshift.
        inner_slope
            The inner slope of the dark matter halo's gNFW density profile.
        """

        self.log10m_vir = log10m_vir
        self.c_gNFW = c_gNFW
        self.redshift_object = redshift_object
        self.redshift_source = redshift_source
        self.inner_slope = inner_slope

        (
            kappa_s,
            scale_radius,
            virial_radius,
            overdens,
        ) = kappa_s_and_scale_radius(
            cosmology=cosmo.Planck15(),
            virial_mass=10**log10m_vir,
            concentration=c_gNFW,
            overdens=overdens,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
            inner_slope=inner_slope,
        )

        self.virial_radius = virial_radius
        self.overdens = overdens

        super().__init__(
            centre=centre,
            kappa_s=kappa_s,
            inner_slope=inner_slope,
            scale_radius=scale_radius,
        )
