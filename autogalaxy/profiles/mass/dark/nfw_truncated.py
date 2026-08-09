import numpy as np
from typing import Optional, Tuple

import autoarray as aa

from autogalaxy.cosmology.model import LensingCosmology
from autogalaxy.profiles.mass.dark.abstract import AbstractgNFW, coord_func_f_from


def coord_func_k_from(grid_radius, tau, xp=np):
    return xp.log(
        xp.divide(
            grid_radius,
            xp.sqrt(xp.square(grid_radius) + xp.square(tau)) + tau,
        )
    )


def coord_func_m_from(grid_radius, tau, xp=np):
    f_r = coord_func_f_from(grid_radius=grid_radius, xp=xp)
    k_r = coord_func_k_from(grid_radius=grid_radius, tau=tau, xp=xp)

    return (tau**2.0 / (tau**2.0 + 1.0) ** 2.0) * (
        ((tau**2.0 + 2.0 * grid_radius**2.0 - 1.0) * f_r)
        + (xp.pi * tau)
        + ((tau**2.0 - 1.0) * xp.log(tau))
        + (
            xp.sqrt(grid_radius**2.0 + tau**2.0)
            * (((tau**2.0 - 1.0) / tau) * k_r - xp.pi)
        )
    )


def potential_func_sph_from(grid_radius, tau, xp=np):
    r"""Return the dimensionless analytic tNFW lensing potential.

    This is equation (18) of Baltz, Marshall & Oguri (2009) for their
    :math:`n=1` smoothly truncated NFW profile.  The returned function is the
    paper's dimensionless radial term; ``NFWTruncatedSph.potential_2d_from``
    supplies the PyAutoGalaxy normalization :math:`2\kappa_s r_s^2`.

    The potential is defined only up to an additive constant.  We retain the
    paper's convention, in which the central potential is zero, but use its
    small-radius series through :math:`x=10^{-1}`.  Direct evaluation there
    subtracts large, nearly equal terms and loses precision in JAX float32.
    """
    grid_radius = xp.real(grid_radius)
    series_radius = xp.maximum(grid_radius, 1.0e-12)
    use_small_radius_series = grid_radius <= 1.0e-1

    # Keep every branch of the closed form finite when JAX traces ``where``.
    # Values below the switch are replaced by the series before returning.
    grid_radius = xp.where(use_small_radius_series, 1.0e-1, grid_radius)

    u = xp.square(grid_radius)
    tau_squared = xp.square(tau)
    root = xp.sqrt(tau_squared + u)

    radius_lt = xp.sqrt(xp.where(grid_radius < 1.0, 1.0 - u, 0.25))
    radius_gt = xp.sqrt(xp.where(grid_radius > 1.0, u - 1.0, 0.25))
    f_lt = xp.arctanh(radius_lt) / radius_lt
    f_gt = xp.arctan(radius_gt) / radius_gt
    f_r = xp.where(grid_radius < 1.0, f_lt, xp.where(grid_radius > 1.0, f_gt, 1.0))

    l_r = coord_func_k_from(grid_radius=grid_radius, tau=tau, xp=xp)

    inverse_radius = 1.0 / grid_radius
    inverse_radius_lt = xp.where(grid_radius < 1.0, inverse_radius, 1.5)
    inverse_radius_gt = xp.where(grid_radius > 1.0, inverse_radius, 0.5)
    cos_lt = -xp.square(xp.arccosh(inverse_radius_lt))
    cos_gt = xp.square(xp.arccos(inverse_radius_gt))
    cos_term = xp.where(
        grid_radius < 1.0,
        cos_lt,
        xp.where(grid_radius > 1.0, cos_gt, 0.0),
    )

    potential = (
        2.0 * tau_squared * xp.pi * (tau - root + tau * xp.log(tau + root))
        + 2.0 * (tau_squared - 1.0) * tau * root * l_r
        + tau_squared * (tau_squared - 1.0) * xp.square(l_r)
        + 4.0 * tau_squared * (u - 1.0) * f_r
        + tau_squared * (tau_squared - 1.0) * cos_term
        + tau_squared
        * ((tau_squared - 1.0) * xp.log(tau) - tau_squared - 1.0)
        * xp.log(u)
        - tau_squared
        * (
            (tau_squared - 1.0) * xp.log(tau) * xp.log(4.0 * tau)
            + 2.0 * xp.log(tau / 2.0)
            - 2.0 * tau * (tau - xp.pi) * xp.log(2.0 * tau)
        )
    )

    potential = potential / xp.square(tau_squared + 1.0)

    small_radius_log = xp.log(2.0 / series_radius)
    small_radius_coefficient_2 = (
        small_radius_log * xp.square(tau_squared + 1.0)
        - tau_squared * xp.log(tau)
        + tau_squared
        - xp.pi * tau
        + xp.log(tau)
        + 1.0
    )
    small_radius_coefficient_2 /= 2.0 * xp.square(tau_squared + 1.0)
    small_radius_coefficient_4 = (
        small_radius_log * (3.0 * tau**6.0 + 5.0 * tau**4.0 + tau_squared - 1.0)
        - tau**6.0
        - tau**4.0
        + tau_squared * xp.log(tau)
        + xp.pi * tau
        - xp.log(tau)
    )
    small_radius_coefficient_4 /= 16.0 * tau_squared * xp.square(tau_squared + 1.0)
    small_radius_coefficient_6 = (
        small_radius_log * (20.0 * tau**8.0 + 28.0 * tau**6.0 - 4.0 * tau_squared + 4.0)
        - 9.0 * tau**8.0
        - 11.0 * tau**6.0
        - 4.0 * tau_squared * xp.log(tau)
        + tau_squared
        - 4.0 * xp.pi * tau
        + 4.0 * xp.log(tau)
        - 1.0
    )
    small_radius_coefficient_6 /= 192.0 * tau**4.0 * xp.square(tau_squared + 1.0)
    potential_small_radius = (
        xp.square(series_radius) * small_radius_coefficient_2
        + series_radius**4.0 * small_radius_coefficient_4
        + series_radius**6.0 * small_radius_coefficient_6
    )

    return xp.where(use_small_radius_series, potential_small_radius, potential)


class NFWTruncatedSph(AbstractgNFW):
    r"""
    Spherical truncated NFW (tNFW) dark matter halo profile (Baltz, Marshall & Oguri 2009).

    The tNFW profile introduces a smooth truncation of the NFW density at a truncation
    radius :math:`r_t`, characterised by the dimensionless truncation ratio
    :math:`\tau = r_t / r_s`:

    .. math::

        \rho(r) = \frac{\rho_s}{(r/r_s)(1 + r/r_s)^2}
                  \left(\frac{\tau^2}{\tau^2 + (r/r_s)^2}\right)

    where :math:`r_s` is the scale radius and :math:`\rho_s` is related to the
    dimensionless convergence normalisation via
    :math:`\kappa_s = \rho_s r_s / \Sigma_{\rm crit}`.

    The projected convergence is given by:

    .. math::

        \kappa(x) = 2 \kappa_s \, L(x, \tau)

    where :math:`L(x, \tau)` is the auxiliary function defined in Baltz et al. (2009)
    (implemented here as ``coord_func_l``).

    References
    ----------
    - Baltz, Marshall & Oguri 2009, JCAP, 2009, 015  (arXiv:0705.0682)
    - Navarro, Frenk & White 1997, ApJ, 490, 493
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        kappa_s: float = 0.05,
        scale_radius: float = 1.0,
        truncation_radius: float = 2.0,
    ):
        r"""
        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        kappa_s
            The dimensionless convergence normalisation
            (:math:`\kappa_s = \rho_s r_s / \Sigma_{\rm crit}`).
        scale_radius
            The NFW scale radius :math:`r_s`, as an angle on the sky in arcseconds.
        truncation_radius
            The truncation radius :math:`r_t`, as an angle on the sky in arcseconds.
            The dimensionless truncation ratio is :math:`\tau = r_t / r_s`.
        """
        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            kappa_s=kappa_s,
            inner_slope=1.0,
            scale_radius=scale_radius,
        )

        self.truncation_radius = truncation_radius
        self.tau = self.truncation_radius / self.scale_radius

    @aa.decorators.to_vector_yx
    @aa.decorators.transform
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        eta = xp.multiply(
            1.0 / self.scale_radius,
            self.radial_grid_from(grid=grid, xp=xp, **kwargs).array,
        )

        deflection_grid = xp.multiply(
            (4.0 * self.kappa_s * self.scale_radius / eta),
            self.deflection_func_sph(grid_radius=eta),
        )

        return self._cartesian_grid_via_radial_from(
            grid=grid, radius=deflection_grid, xp=xp
        )

    def deflection_func_sph(self, grid_radius, xp=np):
        grid_radius = grid_radius + 0j
        return xp.real(self.coord_func_m(grid_radius=grid_radius, xp=xp))

    def convergence_func(self, grid_radius: float, xp=np) -> float:
        grid_radius = ((1.0 / self.scale_radius) * grid_radius) + 0j
        return xp.real(
            2.0 * self.kappa_s * self.coord_func_l(grid_radius=grid_radius.array, xp=xp)
        )

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """Calculate the analytic lensing potential of the spherical tNFW profile."""
        eta = xp.multiply(
            1.0 / self.scale_radius,
            self.radial_grid_from(grid=grid, xp=xp, **kwargs).array,
        )

        return (
            2.0
            * self.kappa_s
            * self.scale_radius**2.0
            * potential_func_sph_from(
                grid_radius=eta,
                tau=self.tau,
                xp=xp,
            )
        )

    def coord_func_k(self, grid_radius, xp=np):
        return coord_func_k_from(grid_radius, self.tau, xp=xp)

    def coord_func_l(self, grid_radius, xp=np):
        f_r = self.coord_func_f(grid_radius=grid_radius, xp=xp)
        g_r = self.coord_func_g(grid_radius=grid_radius, xp=xp)
        k_r = self.coord_func_k(grid_radius=grid_radius, xp=xp)

        return xp.divide(self.tau**2.0, (self.tau**2.0 + 1.0) ** 2.0) * (
            ((self.tau**2.0 + 1.0) * g_r)
            + (2 * f_r)
            - (xp.pi / (xp.sqrt(self.tau**2.0 + grid_radius**2.0)))
            + (
                (
                    (self.tau**2.0 - 1.0)
                    / (self.tau * (xp.sqrt(self.tau**2.0 + grid_radius**2.0)))
                )
                * k_r
            )
        )

    def coord_func_m(self, grid_radius, xp=np):
        return coord_func_m_from(grid_radius, self.tau, xp=xp)

    @staticmethod
    def radial_deflection_from(r, params, xp):
        kappa_s, scale_radius, truncation_radius = params[0], params[1], params[2]
        eta = (r / scale_radius) + 0j
        tau = truncation_radius / scale_radius
        m = xp.real(coord_func_m_from(eta, tau, xp=xp))
        return (4.0 * kappa_s * scale_radius / xp.real(eta)) * m

    @staticmethod
    def _delta_c_from_concentration(concentration: float) -> float:
        """
        NFW characteristic overdensity delta_c for a given concentration.

        This is the standard NFW normalisation:

            delta_c = (200/3) * c^3 / (ln(1+c) - c/(1+c))

        Parameters
        ----------
        concentration
            NFW concentration parameter c = r_200 / r_s.
        """
        return (
            200.0
            / 3.0
            * (
                concentration**3
                / (np.log(1.0 + concentration) - concentration / (1.0 + concentration))
            )
        )

    @staticmethod
    def _concentration_at_overdensity_factor(
        concentration: float,
        truncation_factor: float,
    ) -> float:
        """
        Solve for the concentration-like parameter ``tau`` at which the mean enclosed
        density of the NFW equals ``truncation_factor`` times the critical density.

        For a truncation factor of 100, this finds ``r_100`` expressed as ``r_100 / r_s``.
        The truncation radius of the tNFW profile is then ``tau * r_s``.

        Parameters
        ----------
        concentration
            NFW concentration parameter c = r_200 / r_s.
        truncation_factor
            Overdensity threshold that defines the truncation radius.  The
            truncation radius is the sphere within which the mean enclosed density
            equals ``truncation_factor`` times the critical density.  The default
            value of 100 sets truncation at r_100.
        """
        from scipy.optimize import fsolve

        delta_c = NFWTruncatedSph._delta_c_from_concentration(concentration)

        def equation(tau):
            return (
                truncation_factor
                / 3.0
                * (tau**3 / (np.log(1.0 + tau) - tau / (1.0 + tau)))
                - delta_c
            )

        return float(fsolve(equation, concentration, full_output=False)[0])

    @classmethod
    def from_m200_concentration(
        cls,
        centre: Tuple[float, float] = (0.0, 0.0),
        m200_solar_mass: float = 1e9,
        concentration: float = 10.0,
        redshift_halo: float = 0.5,
        redshift_source: float = 1.0,
        cosmology: Optional[LensingCosmology] = None,
        truncation_factor: float = 100.0,
    ) -> "NFWTruncatedSph":
        """
        Construct an ``NFWTruncatedSph`` from the halo virial mass M_200 and
        concentration rather than the lensing parameters (kappa_s, scale_radius,
        truncation_radius).

        The conversion follows the standard NFW lensing procedure (He et al. 2022,
        MNRAS 511 3046):

        1. Derive the NFW scale radius and characteristic density from M_200, the
           concentration, and the critical density at ``redshift_halo``.
        2. Convert to the dimensionless convergence ``kappa_s`` using the critical
           surface density between ``redshift_halo`` and ``redshift_source``.
        3. Express the scale radius in arc-seconds using the angular diameter
           distance to ``redshift_halo``.
        4. Set the truncation radius to ``r_t`` where the mean enclosed density
           equals ``truncation_factor`` times the critical density (default is
           r_100 for ``truncation_factor=100``).

        Parameters
        ----------
        centre
            The (y, x) arc-second coordinates of the profile centre.
        m200_solar_mass
            Virial mass M_200 in solar masses.
        concentration
            NFW concentration parameter c = r_200 / r_s.
        redshift_halo
            Redshift of the line-of-sight halo.
        redshift_source
            Redshift of the lensed background source.
        cosmology
            Cosmology used for distance and density calculations.  Defaults to
            Planck15 if not supplied.
        truncation_factor
            Overdensity threshold defining the truncation radius.  The default
            value of 100 sets the truncation at r_100.
        """
        from autogalaxy.cosmology.model import Planck15

        if cosmology is None:
            cosmology = Planck15()

        critical_density = cosmology.critical_density(redshift_halo)
        kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=redshift_halo)
        critical_surface_density = cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
            redshift_0=redshift_halo,
            redshift_1=redshift_source,
        )

        r200_kpc = (
            m200_solar_mass / (200.0 * critical_density * (4.0 * np.pi / 3.0))
        ) ** (1.0 / 3.0)

        delta_c = cls._delta_c_from_concentration(concentration)
        rs_kpc = r200_kpc / concentration
        rho_s = critical_density * delta_c

        kappa_s = rho_s * rs_kpc / critical_surface_density
        scale_radius = rs_kpc / kpc_per_arcsec

        tau = cls._concentration_at_overdensity_factor(concentration, truncation_factor)
        truncation_radius = tau * scale_radius

        return cls(
            centre=centre,
            kappa_s=kappa_s,
            scale_radius=scale_radius,
            truncation_radius=truncation_radius,
        )

    @staticmethod
    def m200_concentration_from(
        kappa_s: float,
        scale_radius: float,
        redshift_halo: float,
        redshift_source: float,
        cosmology: Optional[LensingCosmology] = None,
    ) -> Tuple[float, float]:
        """
        Recover the virial mass M_200 and concentration from lensing parameters.

        This is the inverse of :meth:`from_m200_concentration`.  Given the
        dimensionless convergence ``kappa_s`` and the scale radius in arc-seconds,
        the characteristic NFW density and scale radius in kpc are recovered, and
        the concentration is solved numerically from the NFW overdensity equation.

        Parameters
        ----------
        kappa_s
            Dimensionless NFW convergence normalisation = rho_s * r_s / Sigma_crit.
        scale_radius
            NFW scale radius in arc-seconds.
        redshift_halo
            Redshift of the halo.
        redshift_source
            Redshift of the background source.
        cosmology
            Cosmology used for distance and density calculations.  Defaults to
            Planck15 if not supplied.

        Returns
        -------
        Tuple[float, float]
            ``(m200_solar_mass, concentration)``.
        """
        from scipy.optimize import fsolve
        from autogalaxy.cosmology.model import Planck15

        if cosmology is None:
            cosmology = Planck15()

        critical_density = cosmology.critical_density(redshift_halo)
        kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=redshift_halo)
        critical_surface_density = cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
            redshift_0=redshift_halo,
            redshift_1=redshift_source,
        )

        rs_kpc = scale_radius * kpc_per_arcsec
        rho_s = kappa_s * critical_surface_density / rs_kpc
        delta_c = rho_s / critical_density

        def equation(c):
            return 200.0 / 3.0 * (c**3 / (np.log(1.0 + c) - c / (1.0 + c))) - delta_c

        concentration = float(fsolve(equation, 10.0)[0])
        r200_kpc = concentration * rs_kpc
        m200 = 200.0 * (4.0 / 3.0 * np.pi) * critical_density * r200_kpc**3

        return m200, concentration

    @staticmethod
    def mass_ratio_from_concentration_and_truncation_factor(
        concentration: float,
        truncation_factor: float = 100.0,
    ) -> float:
        """
        Mass ratio of a truncated NFW halo to its untruncated M_200 value.

        The truncated NFW mass is:

            M_tNFW = M_200 * tau_scale / c_scale

        where:
            tau_scale = tau^2/(tau^2+1)^2 * ((tau^2-1)*ln(tau) + tau*pi - (tau^2+1))
            c_scale   = ln(1+c) - c/(1+c)

        and ``tau`` is the solution to the ``_concentration_at_overdensity_factor``
        equation for the given concentration and truncation factor.

        This is the function tabulated and cubic-spline interpolated as the
        ``scale_c(c)`` function in the los_pipes simulation code (He et al. 2022).

        Parameters
        ----------
        concentration
            NFW concentration parameter c = r_200 / r_s.
        truncation_factor
            Overdensity threshold defining the truncation radius (default 100).
        """
        tau = NFWTruncatedSph._concentration_at_overdensity_factor(
            concentration, truncation_factor
        )

        tau2 = tau**2
        tau_scale = (
            tau2
            / (tau2 + 1.0) ** 2
            * ((tau2 - 1.0) * np.log(tau) + tau * np.pi - (tau2 + 1.0))
        )
        c_scale = np.log(1.0 + concentration) - concentration / (1.0 + concentration)

        return tau_scale / c_scale

    def mass_at_truncation_radius_solar_mass(
        self,
        redshift_profile,
        redshift_source,
        redshift_of_cosmic_average_density="profile",
        cosmology: LensingCosmology = None,
        xp=np,
    ):
        from autogalaxy.cosmology.model import Planck15

        cosmology = cosmology or Planck15()

        mass_at_200 = self.mass_at_200_solar_masses(
            redshift_object=redshift_profile,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
            xp=xp,
        )

        return (
            mass_at_200
            * (self.tau**2.0 / (self.tau**2.0 + 1.0) ** 2.0)
            * (
                ((self.tau**2.0 - 1) * np.log(self.tau))
                + (self.tau * np.pi)
                - (self.tau**2.0 + 1)
            )
        )
