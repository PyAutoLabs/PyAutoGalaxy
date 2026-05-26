import numpy as np

from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.profiles.mass.stellar.abstract import StellarProfile


class Gaussian(MassProfile, StellarProfile):
    r"""
    Elliptical Gaussian stellar mass profile.

    The convergence of the Gaussian mass profile is proportional to the Gaussian surface
    brightness scaled by a mass-to-light ratio:

    .. math::

        \kappa(R) = \Upsilon \, \frac{I}{2\pi\sigma^2}
                    \exp\!\left(-\frac{R^2}{2\sigma^2}\right)

    where :math:`\Upsilon` is the mass-to-light ratio (``mass_to_light_ratio``),
    :math:`I` is the overall intensity normalisation (``intensity``), :math:`\sigma` is
    the Gaussian width (``sigma``), and :math:`R` is the elliptical radius
    :math:`R^2 = x^2 + y^2/q^2` with axis ratio :math:`q`.

    Deflection angles are computed analytically via the Faddeeva (scaled complementary
    error) function :math:`w(z)` following Shajib (2019).

    References
    ----------
    - Shajib 2019, MNRAS, 488, 1387  (arXiv:1906.08263)
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        sigma: float = 1.0,
        mass_to_light_ratio: float = 1.0,
    ):
        r"""
        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall intensity normalisation :math:`I` of the Gaussian (electrons per second).
        sigma
            The Gaussian width :math:`\sigma` in arcseconds.
        mass_to_light_ratio
            The mass-to-light ratio :math:`\Upsilon` in solar units.
        """

        super(MassProfile, self).__init__(centre=centre, ell_comps=ell_comps)
        self.mass_to_light_ratio = mass_to_light_ratio
        self.intensity = intensity
        self.sigma = sigma

    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        return self.deflections_2d_via_analytic_from(grid=grid, xp=xp, **kwargs)

    @aa.decorators.to_vector_yx
    @aa.decorators.transform(rotate_back=True)
    def deflections_2d_via_analytic_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        deflections = (
            self.mass_to_light_ratio
            * self.intensity
            * self.sigma
            * xp.sqrt((2 * xp.pi) / (1.0 - self.axis_ratio(xp) ** 2.0))
            * self.zeta_from(grid=grid, xp=xp)
        )

        return xp.vstack((-1.0 * xp.imag(deflections), xp.real(deflections))).T

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """
        return self.convergence_func(
            self.eccentric_radii_grid_from(grid=grid, xp=xp, **kwargs), xp=xp
        )

    def convergence_func(self, grid_radius: float, xp=np) -> float:
        return self.mass_to_light_ratio * self.image_2d_via_radii_from(
            grid_radius, xp=xp
        )

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        from autogalaxy.profiles.mass.abstract.mge import MGEDecomposer

        radii_min = self.sigma / 100.0
        radii_max = self.sigma * 20.0
        sigmas = xp.exp(xp.linspace(xp.log(radii_min), xp.log(radii_max), 30))
        mge_decomp = MGEDecomposer(mass_profile=self)
        return mge_decomp.potential_2d_via_mge_from(
            grid=grid, xp=xp, sigma_log_list=sigmas,
            ellipticity_convention="circularised", three_D=False,
        )

    def image_2d_via_radii_from(self, grid_radii: np.ndarray, xp=np):
        """Calculate the intensity of the Gaussian light profile on a grid of radial coordinates.

        Parameters
        ----------
        grid_radii
            The radial distance from the centre of the profile. for each coordinate on the grid.

        Note: sigma is divided by sqrt(q) here.
        """
        return xp.multiply(
            self.intensity,
            xp.exp(
                -0.5
                * xp.square(
                    xp.divide(
                        grid_radii.array, self.sigma / xp.sqrt(self.axis_ratio(xp))
                    )
                )
            ),
        )

    def axis_ratio(self, xp=np):
        axis_ratio = super().axis_ratio(xp=xp)
        return xp.where(axis_ratio < 0.9999, axis_ratio, 0.9999)

    def zeta_from(self, grid: aa.type.Grid2DLike, xp=np):
        q = xp.asarray(self.axis_ratio(xp), dtype=xp.float64)
        q2 = q * q

        y = xp.asarray(grid.array[:, 0], dtype=xp.float64)
        x = xp.asarray(grid.array[:, 1], dtype=xp.float64)

        ind_pos_y = y >= 0

        scale = q / (
            xp.asarray(self.sigma, dtype=xp.float64)
            * xp.sqrt(xp.asarray(2.0, dtype=xp.float64) * (1.0 - q2))
        )

        xs = x * scale
        ys = xp.abs(y) * scale

        z1 = xs + 1j * ys
        z2 = q * xs + 1j * ys / q

        exp_term = xp.exp(-(xs * xs) * (1.0 - q2) - (ys * ys) * (1.0 / q2 - 1.0))

        core = -1j * (self.wofz(z1, xp=xp) - exp_term * self.wofz(z2, xp=xp))

        return xp.where(ind_pos_y, core, xp.conj(core))

    @staticmethod
    def wofz(z, xp=np):
        """
        JAX-compatible Faddeeva function w(z) = exp(-z^2) * erfc(-i z)
        Based on the Poppe–Wijers / Zaghloul–Ali rational approximations.
        Valid for all complex z. JIT + autodiff safe.
        """

        z = xp.asarray(z, dtype=xp.complex128)
        x = xp.real(z)
        y = xp.imag(z)

        r2 = x * x + y * y
        y2 = y * y
        z2 = z * z

        sqrt_pi = xp.asarray(xp.sqrt(xp.pi), dtype=xp.float64)
        inv_sqrt_pi = xp.asarray(1.0 / sqrt_pi, dtype=xp.float64)

        # ---------- Large-|z| continued fraction ----------
        r1_s1 = xp.asarray([2.5, 2.0, 1.5, 1.0, 0.5], dtype=xp.float64)

        t = z
        for c in r1_s1:
            t = z - c / t

        w_large = 1j * inv_sqrt_pi / t

        # ---------- Region 5 ----------
        U5 = xp.asarray(
            [1.320522, 35.7668, 219.031, 1540.787, 3321.990, 36183.31], dtype=xp.float64
        )
        V5 = xp.asarray(
            [1.841439, 61.57037, 364.2191, 2186.181, 9022.228, 24322.84, 32066.6],
            dtype=xp.float64,
        )

        t = inv_sqrt_pi
        for u in U5:
            t = u + z2 * t

        s = xp.asarray(1.0, dtype=xp.float64)
        for v in V5:
            s = v + z2 * s

        real_exp = xp.clip(-xp.real(z2), None, 700.0)
        imag_exp = -xp.imag(z2)
        w5 = (
            xp.exp(real_exp + 1j * imag_exp) + 1j * z * t / s
        )  # clip prevents overflow error

        # ---------- Region 6 ----------
        U6 = xp.asarray(
            [5.9126262, 30.180142, 93.15558, 181.92853, 214.38239, 122.60793],
            dtype=xp.float64,
        )
        V6 = xp.asarray(
            [
                10.479857,
                53.992907,
                170.35400,
                348.70392,
                457.33448,
                352.73063,
                122.60793,
            ],
            dtype=xp.float64,
        )

        t = inv_sqrt_pi
        for u in U6:
            t = u - 1j * z * t

        s = xp.asarray(1.0, dtype=xp.float64)
        for v in V6:
            s = v - 1j * z * s

        w6 = t / s

        # ---------- Region logic ----------
        reg1 = (r2 >= 62.0) | ((r2 >= 30.0) & (r2 < 62.0) & (y2 >= 1e-13))
        reg2 = ((r2 >= 30) & (r2 < 62) & (y2 < 1e-13)) | (
            (r2 >= 2.5) & (r2 < 30) & (y2 < 0.072)
        )

        w = w6
        w = xp.where(reg2, w5, w)
        w = xp.where(reg1, w_large, w)

        return w
