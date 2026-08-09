from typing import Tuple

import numpy as np
from scipy.integrate import quad

import autoarray as aa

from autogalaxy.profiles.mass.dark.abstract import DarkProfile
from autogalaxy.profiles.mass.dark.nfw import NFWSph
from autogalaxy.profiles.mass.dark.kaplinghat import (
    _nfw_radial_deflection_from,
    _trapezoid_from,
)
from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.profiles import validate


def _yang24_parameter_ratios_from(tau):
    """
    The Yang et al. (2024) fitting functions for how the cored-NFW parameters
    evolve with the dimensionless gravothermal time ``tau = t / t_c``
    (arXiv:2305.16176, Section 2).

    Returns the ratios (rho_s / rho_s0, r_s / r_s0, r_c / r_s0) relative to the
    halo's CDM baseline NFW parameters (rho_s0, r_s0). All three fits are exact
    at ``tau = 0`` (ratios 1, 1, 0), recovering the NFW profile.
    """
    tau = float(tau)

    log_term = np.log(tau + 0.001) / np.log(0.001)

    rho_s_ratio = (
        2.033
        + 0.7381 * tau
        + 7.264 * tau**5
        - 12.73 * tau**7
        + 9.915 * tau**9
        + (1.0 - 2.033) * log_term
    )
    r_s_ratio = (
        0.7178
        - 0.1026 * tau
        + 0.2474 * tau**2
        - 0.4079 * tau**3
        + (1.0 - 0.7178) * log_term
    )
    r_c_ratio = (
        2.555 * np.sqrt(tau)
        - 3.632 * tau
        + 2.131 * tau**2
        - 1.415 * tau**3
        + 0.4683 * tau**4
    )

    return rho_s_ratio, r_s_ratio, max(r_c_ratio, 0.0)


def _yang24_density_3d_from(radii, rho_s, r_s, r_c, xp):
    """
    The beta = 4 cored-NFW density of Yang et al. (2024):

        rho(r) = rho_s / [ ((r^4 + r_c^4)^(1/4) / r_s) (1 + r / r_s)^2 ]

    Closed form in any array backend; reduces exactly to NFW as r_c -> 0.
    """
    r = xp.maximum(radii, 1.0e-12)
    r_tilde = (r**4 + r_c**4) ** 0.25
    return rho_s / ((r_tilde / r_s) * (1.0 + r / r_s) ** 2)


class YangSIDMSph(MassProfile, DarkProfile):
    r"""
    Spherical SIDM halo following the parametric gravothermal-evolution model of
    Yang, Nadler, Yu & Zhong (2024), arXiv:2305.16176.

    The density is the beta = 4 cored-NFW form

    .. math::
        \rho(r) = \frac{\rho_s}{\left[(r^4 + r_c^4)^{1/4} / r_s\right]
                  \left(1 + r / r_s\right)^2}

    whose parameters :math:`(\rho_s, r_s, r_c)` evolve along the universal
    gravothermal sequence as a function of the dimensionless time
    :math:`\tau = t / t_c`, via the Yang et al. (2024) fitting functions. At
    ``tau = 0`` the profile is exactly the NFW profile of the halo's CDM
    baseline ``(kappa_s, scale_radius)``; as ``tau`` grows the halo first forms
    a core (core expansion) and then contracts towards core collapse.

    The fits are calibrated for ``tau`` in [0, 1]; input values are truncated
    into that range, following the truncation applied in Yang et al. (2024)
    for deep collapse.

    Parameters
    ----------
    centre
        The (y,x) arc-second coordinates of the profile centre.
    kappa_s
        The overall normalization of the CDM baseline NFW halo
        (rho_s0 * scale_radius / critical surface density).
    scale_radius
        The arc-second scale radius r_s0 of the CDM baseline NFW halo.
    tau
        The dimensionless gravothermal evolution time t / t_c, where t_c is the
        halo's core-collapse timescale. Use `YangSIDMMCRLudlowSph` to compute
        ``tau`` from physical inputs (cross section, halo age and mass).
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        kappa_s: float = 0.05,
        scale_radius: float = 1.0,
        tau: float = 0.0,
    ):
        super().__init__(centre=centre, ell_comps=(0.0, 0.0))

        self.kappa_s = kappa_s
        validate.validate_scale_radius(scale_radius=scale_radius)
        self.scale_radius = scale_radius
        self.tau = min(max(float(tau), 0.0), 1.0)

        rho_s_ratio, r_s_ratio, r_c_ratio = _yang24_parameter_ratios_from(
            tau=self.tau
        )

        self.rho_s_evolved = rho_s_ratio * kappa_s / scale_radius
        self.scale_radius_evolved = r_s_ratio * scale_radius
        self.core_radius_evolved = r_c_ratio * scale_radius

        self._nfw = NFWSph(
            centre=(0.0, 0.0),
            kappa_s=kappa_s,
            scale_radius=scale_radius,
        )

    def _density_3d_from_radius(self, radii):
        radii = np.asarray(radii, dtype=float)
        return _yang24_density_3d_from(
            radii=radii,
            rho_s=self.rho_s_evolved,
            r_s=self.scale_radius_evolved,
            r_c=self.core_radius_evolved,
            xp=np,
        )

    def density_3d_func(self, r, xp=np):
        radii = r.array if hasattr(r, "array") else r
        if xp is not np:
            radii = np.asarray(radii)
        return self._density_3d_from_radius(radii)

    def convergence_func(self, grid_radius, xp=np):
        radii = (
            grid_radius.array
            if hasattr(grid_radius, "array")
            else np.asarray(grid_radius)
        )
        scalar_input = np.ndim(radii) == 0
        radii = np.atleast_1d(np.asarray(radii, dtype=float))

        if self.tau <= 0.0:
            values = self._nfw.convergence_func(aa.ArrayIrregular(radii), xp=np)
            return values[0] if scalar_input else values

        z_max = max(
            500.0 * self.scale_radius_evolved, 50.0 * self.core_radius_evolved
        )

        def convergence_at_radius(radius):
            radius = float(max(radius, 1.0e-8))
            integral = quad(
                lambda z: self._density_3d_from_radius(np.sqrt(radius**2 + z**2)),
                0.0,
                z_max,
                epsrel=1.0e-5,
                limit=100,
            )[0]
            return 2.0 * integral

        convergence = np.array([convergence_at_radius(radius) for radius in radii])
        return convergence[0] if scalar_input else convergence

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        radii = self.radial_grid_from(grid=grid, xp=np, **kwargs)
        return self.convergence_func(grid_radius=radii, xp=np)

    def radial_deflection_from_radius(self, radius):
        radius = float(radius)

        if radius <= 1.0e-8:
            return 0.0

        if self.tau <= 0.0:
            return float(
                np.sqrt(
                    np.sum(
                        self._nfw.deflections_yx_2d_from(
                            grid=aa.Grid2DIrregular([[radius, 0.0]])
                        ).array[0]
                        ** 2.0
                    )
                )
            )

        mass_2d = quad(
            lambda r: self.convergence_func(aa.ArrayIrregular([r]))[0] * r,
            0.0,
            radius,
            epsrel=1.0e-4,
            limit=100,
        )[0]
        return 2.0 * mass_2d / radius

    @staticmethod
    def radial_deflection_from(r, params, xp):
        rho_s = params[0]
        r_s = params[1]
        r_c = params[2]

        r = xp.asarray(r)
        r_safe = xp.maximum(r, 1.0e-8)

        z_max = xp.maximum(500.0 * r_s, 50.0 * r_c)
        z_unit = xp.linspace(1.0e-5, 1.0, 160)
        z = z_max * z_unit**3
        u = xp.linspace(0.0, 1.0, 64)

        projected_radii = xp.maximum(r_safe[:, None] * u[None, :], 1.0e-6)
        three_d_radii = xp.sqrt(
            projected_radii[:, :, None] ** 2 + z[None, None, :] ** 2
        )

        density = _yang24_density_3d_from(
            radii=three_d_radii,
            rho_s=rho_s,
            r_s=r_s,
            r_c=r_c,
            xp=xp,
        )
        convergence = 2.0 * _trapezoid_from(density, x=z, axis=-1, xp=xp)

        mass_integral = r_safe**2 * _trapezoid_from(
            convergence * u[None, :], x=u, axis=-1, xp=xp
        )
        numerical = 2.0 * mass_integral / r_safe
        analytic_nfw = _nfw_radial_deflection_from(
            r=r_safe,
            kappa_s=rho_s * r_s,
            scale_radius=r_s,
            xp=xp,
        )

        radial_deflection = xp.where(
            r_c > 0.0,
            numerical,
            analytic_nfw,
        )

        return xp.where(r > 1.0e-8, radial_deflection, 0.0)

    @aa.decorators.to_vector_yx
    @aa.decorators.transform
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        theta = self.radial_grid_from(grid=grid, xp=np, **kwargs).array
        deflection_r = np.array(
            [self.radial_deflection_from_radius(radius) for radius in theta]
        )

        return self._cartesian_grid_via_radial_from(
            grid=grid,
            radius=deflection_r,
            xp=np,
            **kwargs,
        )

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        theta = self.radial_grid_from(grid=grid, xp=np, **kwargs).array

        potential = np.array(
            [
                quad(
                    self.radial_deflection_from_radius,
                    0.0,
                    max(float(radius), 1.0e-8),
                    epsrel=1.0e-4,
                    limit=100,
                )[0]
                for radius in theta
            ]
        )

        return potential
