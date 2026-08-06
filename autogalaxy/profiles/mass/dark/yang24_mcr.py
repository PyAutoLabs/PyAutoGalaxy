import math
from typing import Tuple

import numpy as np

from autogalaxy.profiles.mass.dark import mcr_util
from autogalaxy.profiles.mass.dark.yang24 import YangSIDMSph


MSUN_G = 1.98847e33
KPC_CM = 3.0856775814913673e21
GYR_S = 3.15576e16
G_KPC_KM2_S2_MSUN = 4.30091727003628e-6
G_CGS = 6.6743e-8

# x = r / r_s at which the NFW circular velocity peaks.
NFW_X_AT_V_MAX = 2.1625815870646098

# The calibrated constant C of the Yang et al. (2024) collapse-time formula.
COLLAPSE_CALIBRATION_C = 0.75


def _nu_eff_km_s_from(rho_s, scale_radius_kpc):
    """
    The effective velocity dispersion nu_eff = 0.64 * V_max of the halo's CDM
    baseline NFW profile (Yang et al. 2024; Outmezguine et al. 2023), in km/s.

    ``rho_s`` is in Msun / kpc^3 and ``scale_radius_kpc`` in kpc.
    """
    x = NFW_X_AT_V_MAX
    mass_factor = np.log1p(x) - x / (1.0 + x)
    v_max_km_s = np.sqrt(
        4.0
        * np.pi
        * G_KPC_KM2_S2_MSUN
        * rho_s
        * scale_radius_kpc**2
        * mass_factor
        / x
    )
    return 0.64 * v_max_km_s


def _sigma_eff_over_m_from(sigma_over_m, velocity_exponent, velocity_ref, nu_eff_km_s):
    """
    The effective cross section per unit mass <sigma v^5> / <v^5> of
    Outmezguine et al. (2023), for a power-law velocity dependence

        sigma(v) = sigma_over_m * (v / velocity_ref)^(-velocity_exponent)

    averaged over a Maxwell-Boltzmann relative-velocity distribution
    f(v) ~ v^2 exp(-v^2 / (4 nu_eff^2)). The average is closed form:

        sigma_eff/m = (sigma/m) (velocity_ref / (2 nu_eff))^a Gamma(4 - a/2) / 6

    and requires ``velocity_exponent < 8`` for the average to converge. At
    ``velocity_exponent = 0`` this reduces to the constant cross section.
    """
    a = float(velocity_exponent)

    if a == 0.0:
        return sigma_over_m

    if a >= 8.0:
        raise ValueError(
            "The <sigma v^5> Maxwell-Boltzmann average diverges for a "
            f"velocity_exponent >= 8 (input value {a})."
        )

    return (
        sigma_over_m
        * (velocity_ref / (2.0 * nu_eff_km_s)) ** a
        * math.gamma(4.0 - a / 2.0)
        / 6.0
    )


def _tau_from(sigma_eff_over_m, t_age, rho_s, scale_radius_kpc):
    """
    The dimensionless gravothermal time tau = t_age / t_c, via the Yang et al.
    (2024) collapse timescale of the CDM baseline halo:

        t_c = (150 / C) / [ (sigma_eff/m) rho_s r_s sqrt(4 pi G rho_s) ]

    with C = 0.75. ``sigma_eff_over_m`` is in cm^2/g, ``t_age`` in Gyr,
    ``rho_s`` in Msun / kpc^3 and ``scale_radius_kpc`` in kpc.
    """
    if sigma_eff_over_m <= 0.0 or t_age <= 0.0:
        return 0.0

    rho_s_g_cm3 = rho_s * MSUN_G / KPC_CM**3
    scale_radius_cm = scale_radius_kpc * KPC_CM

    collapse_rate = (
        sigma_eff_over_m
        * rho_s_g_cm3
        * scale_radius_cm
        * np.sqrt(4.0 * np.pi * G_CGS * rho_s_g_cm3)
    )
    t_c_seconds = (150.0 / COLLAPSE_CALIBRATION_C) / collapse_rate

    return float(t_age * GYR_S / t_c_seconds)


class YangSIDMMCRLudlowSph(YangSIDMSph):
    r"""
    The Yang et al. (2024) parametric SIDM halo constructed from physical
    inputs, with the CDM baseline NFW parameters set by the Ludlow et al.
    (2016) mass-concentration relation.

    The dimensionless gravothermal time is solved per halo as
    :math:`\tau = t_{\rm age} / t_c` with

    .. math::
        t_c = \frac{150 / C}{(\sigma_{\rm eff}/m) \rho_s r_s
              \sqrt{4 \pi G \rho_s}}, \quad C = 0.75,

    where :math:`(\rho_s, r_s)` are the halo's CDM baseline NFW parameters in
    physical units. The effective cross section
    :math:`\sigma_{\rm eff}/m` follows Outmezguine et al. (2023): a power-law
    velocity dependence :math:`\sigma \propto v^{-a}` averaged over a
    Maxwell-Boltzmann distribution at :math:`\nu_{\rm eff} = 0.64 V_{\rm max}`.

    Parameters
    ----------
    centre
        The (y,x) arc-second coordinates of the profile centre.
    mass_at_200
        The mass of the halo enclosing 200 times the cosmic average density,
        in solar masses.
    sigma_over_m
        The self-interaction cross section per unit mass, in cm^2 / g,
        evaluated at the reference velocity ``velocity_ref``.
    velocity_exponent
        The power-law index a of the velocity-dependent cross section
        sigma(v) = sigma_over_m * (v / velocity_ref)^(-a). Zero gives a
        velocity-independent cross section (must be < 8 for the
        Maxwell-Boltzmann average to converge).
    velocity_ref
        The reference velocity, in km/s, at which ``sigma_over_m`` is quoted.
        Irrelevant when ``velocity_exponent`` is zero.
    t_age
        The age of the halo in Gyr, from which tau = t_age / t_c is computed.
        For multi-plane systems construct each halo with the age appropriate to
        its redshift plane.
    redshift_object
        The halo redshift, setting the Ludlow et al. (2016) concentration and
        the arc-second unit conversions.
    redshift_source
        The source redshift, setting the critical surface density.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        mass_at_200: float = 1e9,
        sigma_over_m: float = 1.0,
        velocity_exponent: float = 0.0,
        velocity_ref: float = 10.0,
        t_age: float = 10.0,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):
        self.mass_at_200 = mass_at_200
        self.sigma_over_m = sigma_over_m
        self.velocity_exponent = velocity_exponent
        self.velocity_ref = velocity_ref
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

        self.nu_eff = float(
            _nu_eff_km_s_from(rho_s=rho_s, scale_radius_kpc=scale_radius_kpc)
        )
        self.sigma_eff_over_m = float(
            _sigma_eff_over_m_from(
                sigma_over_m=sigma_over_m,
                velocity_exponent=velocity_exponent,
                velocity_ref=velocity_ref,
                nu_eff_km_s=self.nu_eff,
            )
        )

        # The un-truncated tau (the base class truncates into the fitted
        # [0, 1] range), kept for diagnostics such as core-collapse fractions.
        self.tau_physical = _tau_from(
            sigma_eff_over_m=self.sigma_eff_over_m,
            t_age=t_age,
            rho_s=rho_s,
            scale_radius_kpc=scale_radius_kpc,
        )
        tau = self.tau_physical

        super().__init__(
            centre=centre,
            kappa_s=kappa_s,
            scale_radius=scale_radius,
            tau=tau,
        )
