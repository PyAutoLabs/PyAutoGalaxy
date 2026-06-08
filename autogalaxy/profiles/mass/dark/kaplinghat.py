import functools
from typing import Tuple

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

import autoarray as aa

from autogalaxy.profiles.mass.dark.abstract import DarkProfile
from autogalaxy.profiles.mass.dark.nfw import NFWSph
from autogalaxy.profiles.mass.abstract.abstract import MassProfile


@functools.lru_cache(maxsize=1)
def _isothermal_lane_emden_table():
    x_min = 1.0e-5
    x_max = 200.0

    def ode(x, y):
        h, dh_dx, m = y
        density = np.exp(-h)
        return [dh_dx, density - 2.0 * dh_dx / x, x**2 * density]

    h0 = x_min**2 / 6.0
    dh0 = x_min / 3.0
    m0 = x_min**3 / 3.0

    x_eval = np.geomspace(x_min, x_max, 4096)
    sol = solve_ivp(
        ode,
        (x_min, x_max),
        (h0, dh0, m0),
        t_eval=x_eval,
        rtol=1.0e-9,
        atol=1.0e-11,
    )

    if not sol.success:
        raise RuntimeError("Could not tabulate the isothermal Lane-Emden solution.")

    return sol.t, sol.y[0], sol.y[2]


def _interp_lane_emden(x):
    x_table, h_table, m_table = _isothermal_lane_emden_table()
    x = np.asarray(x, dtype=float)
    h = np.interp(x, x_table, h_table)
    m = np.interp(x, x_table, m_table)
    return h, m


def _nfw_density_from(r, kappa_s, scale_radius):
    x = np.maximum(r / scale_radius, 1.0e-12)
    return (kappa_s / scale_radius) / (x * (1.0 + x) ** 2)


def _nfw_mass_3d_within_radius_from(r, kappa_s, scale_radius):
    x = np.maximum(r / scale_radius, 1.0e-12)
    mass_factor = np.where(
        x < 1.0e-4,
        0.5 * x**2 - (2.0 / 3.0) * x**3,
        np.log1p(x) - x / (1.0 + x),
    )
    return 4.0 * np.pi * (kappa_s / scale_radius) * scale_radius**3 * mass_factor


def _trapezoid_from(y, x, axis, xp):
    if hasattr(xp, "trapezoid"):
        return xp.trapezoid(y, x=x, axis=axis)
    return xp.trapz(y, x=x, axis=axis)


def _interp_lane_emden_xp(x, xp):
    x_table, h_table, _ = _isothermal_lane_emden_table()
    return xp.interp(x, xp.asarray(x_table), xp.asarray(h_table))


def _nfw_radial_deflection_from(r, kappa_s, scale_radius, xp):
    x = xp.maximum(r / scale_radius, 1.0e-8)

    below = x < 1.0
    above = x > 1.0
    x_below = xp.minimum(x, 1.0 - 1.0e-8)
    x_above = xp.maximum(x, 1.0 + 1.0e-8)

    f_below = xp.arccosh(1.0 / x_below) / xp.sqrt(1.0 - x_below**2)
    f_above = xp.arccos(1.0 / x_above) / xp.sqrt(x_above**2 - 1.0)
    f_at_one = 1.0
    f = xp.where(below, f_below, xp.where(above, f_above, f_at_one))

    g = xp.log(x / 2.0) + f
    return 4.0 * kappa_s * scale_radius * g / x


def _kaplinghat_density_3d_from_radius(
    radii,
    kappa_s,
    scale_radius,
    interaction_radius,
    central_density,
    isothermal_radius,
    xp,
):
    x_nfw = xp.maximum(radii / scale_radius, 1.0e-12)
    nfw_density = (kappa_s / scale_radius) / (x_nfw * (1.0 + x_nfw) ** 2)

    safe_isothermal_radius = xp.maximum(isothermal_radius, 1.0e-12)
    x_iso = xp.maximum(radii / safe_isothermal_radius, 1.0e-5)
    h = _interp_lane_emden_xp(x_iso, xp=xp)
    safe_central_density = xp.where(xp.isfinite(central_density), central_density, 0.0)
    iso_density = safe_central_density * xp.exp(-h)

    use_iso = (
        (interaction_radius > 0.0)
        & (isothermal_radius > 0.0)
        & xp.isfinite(central_density)
        & (radii < interaction_radius)
    )

    return xp.where(use_iso, iso_density, nfw_density)


def _matched_isothermal_parameters_from(interaction_radius, kappa_s, scale_radius):
    rho_1 = _nfw_density_from(
        r=interaction_radius,
        kappa_s=kappa_s,
        scale_radius=scale_radius,
    )
    mass_1 = _nfw_mass_3d_within_radius_from(
        r=interaction_radius,
        kappa_s=kappa_s,
        scale_radius=scale_radius,
    )

    target = mass_1 / (4.0 * np.pi * rho_1 * interaction_radius**3)

    x_table, h_table, m_table = _isothermal_lane_emden_table()
    ratio = m_table * np.exp(h_table) / x_table**3

    if target <= ratio[0]:
        x_1 = x_table[0]
    elif target >= ratio[-1]:
        x_1 = x_table[-1]
    else:
        x_1 = brentq(
            lambda x: (
                np.interp(x, x_table, m_table)
                * np.exp(np.interp(x, x_table, h_table))
                / x**3
                - target
            ),
            x_table[0],
            x_table[-1],
        )

    h_1 = np.interp(x_1, x_table, h_table)
    central_density = rho_1 * np.exp(h_1)
    isothermal_radius = interaction_radius / x_1

    return central_density, isothermal_radius


class KaplinghatCoredNFWSph(MassProfile, DarkProfile):
    r"""
    Spherical SIDM-motivated cored-NFW halo following the Kaplinghat, Tulin
    and Yu (2016) isothermal-Jeans construction.

    The profile is NFW outside ``interaction_radius``. Inside that radius it
    uses the self-gravitating isothermal solution of the spherical Jeans-Poisson
    equation, matched to the NFW envelope in both density and enclosed mass.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        kappa_s: float = 0.05,
        scale_radius: float = 1.0,
        sigma_over_m: float = 1.0,
        t_age: float = 10.0,
        interaction_radius: float = None,
    ):
        super().__init__(centre=centre, ell_comps=(0.0, 0.0))

        self.kappa_s = kappa_s
        self.scale_radius = scale_radius
        self.sigma_over_m = sigma_over_m
        self.t_age = t_age

        if interaction_radius is None:
            strength = max(float(sigma_over_m) * float(t_age), 0.0)
            interaction_radius = scale_radius * strength / (100.0 + strength)

        self.interaction_radius = max(float(interaction_radius), 0.0)

        if self.interaction_radius > 0.0:
            (
                self.central_density,
                self.isothermal_radius,
            ) = _matched_isothermal_parameters_from(
                interaction_radius=self.interaction_radius,
                kappa_s=self.kappa_s,
                scale_radius=self.scale_radius,
            )
        else:
            self.central_density = np.inf
            self.isothermal_radius = 0.0

        self._nfw = NFWSph(
            centre=(0.0, 0.0),
            kappa_s=kappa_s,
            scale_radius=scale_radius,
        )

    def _density_3d_from_radius(self, radii):
        radii = np.asarray(radii, dtype=float)

        nfw_density = _nfw_density_from(
            r=radii,
            kappa_s=self.kappa_s,
            scale_radius=self.scale_radius,
        )

        if self.interaction_radius <= 0.0:
            return nfw_density

        x = np.maximum(radii / self.isothermal_radius, 1.0e-5)
        h, _ = _interp_lane_emden(x)
        iso_density = self.central_density * np.exp(-h)

        return np.where(radii < self.interaction_radius, iso_density, nfw_density)

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

        if self.interaction_radius <= 0.0:
            values = self._nfw.convergence_func(aa.ArrayIrregular(radii), xp=np)
            return values[0] if scalar_input else values

        z_max = max(500.0 * self.scale_radius, 50.0 * self.interaction_radius)

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

        if self.interaction_radius <= 0.0:
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
        kappa_s = params[0]
        scale_radius = params[1]
        interaction_radius = params[2]
        central_density = params[3]
        isothermal_radius = params[4]

        r = xp.asarray(r)
        r_safe = xp.maximum(r, 1.0e-8)

        z_max = xp.maximum(500.0 * scale_radius, 50.0 * interaction_radius)
        z_unit = xp.linspace(1.0e-5, 1.0, 160)
        z = z_max * z_unit**3
        u = xp.linspace(0.0, 1.0, 64)

        projected_radii = xp.maximum(r_safe[:, None] * u[None, :], 1.0e-6)
        three_d_radii = xp.sqrt(
            projected_radii[:, :, None] ** 2 + z[None, None, :] ** 2
        )

        density = _kaplinghat_density_3d_from_radius(
            radii=three_d_radii,
            kappa_s=kappa_s,
            scale_radius=scale_radius,
            interaction_radius=interaction_radius,
            central_density=central_density,
            isothermal_radius=isothermal_radius,
            xp=xp,
        )
        convergence = 2.0 * _trapezoid_from(density, x=z, axis=-1, xp=xp)

        mass_integral = r_safe**2 * _trapezoid_from(
            convergence * u[None, :], x=u, axis=-1, xp=xp
        )
        numerical = 2.0 * mass_integral / r_safe
        analytic_nfw = _nfw_radial_deflection_from(
            r=r_safe,
            kappa_s=kappa_s,
            scale_radius=scale_radius,
            xp=xp,
        )

        radial_deflection = xp.where(
            interaction_radius > 0.0,
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
