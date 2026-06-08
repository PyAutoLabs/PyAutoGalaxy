from typing import Tuple

import numpy as np
from scipy.optimize import brentq

from autogalaxy.profiles.mass.dark import mcr_util
from autogalaxy.profiles.mass.dark.kaplinghat import KaplinghatCoredNFWSph


MSUN_G = 1.98847e33
KPC_CM = 3.0856775814913673e21
GYR_S = 3.15576e16
G_KPC_KM2_S2_MSUN = 4.30091727003628e-6


def _nfw_physical_density_from(radius_kpc, rho_s, scale_radius_kpc):
    x = np.maximum(radius_kpc / scale_radius_kpc, 1.0e-12)
    return rho_s / (x * (1.0 + x) ** 2)


def _nfw_physical_mass_from(radius_kpc, rho_s, scale_radius_kpc):
    x = np.maximum(radius_kpc / scale_radius_kpc, 1.0e-12)
    mass_factor = np.where(
        x < 1.0e-4,
        0.5 * x**2 - (2.0 / 3.0) * x**3,
        np.log1p(x) - x / (1.0 + x),
    )
    return 4.0 * np.pi * rho_s * scale_radius_kpc**3 * mass_factor


def _interaction_radius_kpc_from(
    sigma_over_m,
    t_age,
    rho_s,
    scale_radius_kpc,
    radius_at_200,
):
    if sigma_over_m <= 0.0 or t_age <= 0.0:
        return 0.0

    t_seconds = t_age * GYR_S

    def rate_times_age(radius_kpc):
        density = _nfw_physical_density_from(
            radius_kpc=radius_kpc,
            rho_s=rho_s,
            scale_radius_kpc=scale_radius_kpc,
        )
        mass = _nfw_physical_mass_from(
            radius_kpc=radius_kpc,
            rho_s=rho_s,
            scale_radius_kpc=scale_radius_kpc,
        )
        velocity_km_s = np.sqrt(
            G_KPC_KM2_S2_MSUN * mass / np.maximum(radius_kpc, 1.0e-12)
        )
        density_g_cm3 = density * MSUN_G / KPC_CM**3
        velocity_cm_s = velocity_km_s * 1.0e5

        return sigma_over_m * density_g_cm3 * velocity_cm_s * t_seconds

    r_min = min(scale_radius_kpc * 1.0e-8, radius_at_200 * 1.0e-10)
    r_max = radius_at_200

    if rate_times_age(r_min) < 1.0:
        return 0.0

    if rate_times_age(r_max) > 1.0:
        return r_max

    return brentq(lambda r: rate_times_age(r) - 1.0, r_min, r_max)


class KaplinghatCoredNFWMCRLudlowSph(KaplinghatCoredNFWSph):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        mass_at_200: float = 1e9,
        sigma_over_m: float = 1.0,
        t_age: float = 10.0,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):
        self.mass_at_200 = mass_at_200
        self.sigma_over_m = sigma_over_m
        self.t_age = t_age
        self.redshift_object = redshift_object
        self.redshift_source = redshift_source

        (
            concentration,
            cosmic_average_density,
            critical_surface_density,
            kpc_per_arcsec,
        ) = mcr_util.ludlow16_cosmology(
            mass_at_200=mass_at_200,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
            xp=np,
        )

        radius_at_200 = (
            mass_at_200 / (200.0 * cosmic_average_density * (4.0 * np.pi / 3.0))
        ) ** (1.0 / 3.0)
        scale_radius_kpc = radius_at_200 / concentration

        de_c = (
            200.0
            / 3.0
            * (
                concentration**3
                / (np.log(1.0 + concentration) - concentration / (1.0 + concentration))
            )
        )
        rho_s = cosmic_average_density * de_c
        kappa_s = rho_s * scale_radius_kpc / critical_surface_density
        scale_radius = scale_radius_kpc / kpc_per_arcsec

        interaction_radius_kpc = _interaction_radius_kpc_from(
            sigma_over_m=sigma_over_m,
            t_age=t_age,
            rho_s=rho_s,
            scale_radius_kpc=scale_radius_kpc,
            radius_at_200=radius_at_200,
        )
        interaction_radius = interaction_radius_kpc / kpc_per_arcsec

        super().__init__(
            centre=centre,
            kappa_s=kappa_s,
            scale_radius=scale_radius,
            sigma_over_m=sigma_over_m,
            t_age=t_age,
            interaction_radius=interaction_radius,
        )
