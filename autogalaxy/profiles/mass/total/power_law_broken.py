import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile


class PowerLawBroken(MassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        inner_slope: float = 1.5,
        outer_slope: float = 2.5,
        break_radius: float = 0.01,
    ):
        """
        Ell, homoeoidal mass model with an inner_slope
        and outer_slope, continuous in density across break_radius.
        Position angle is defined to be zero on x-axis and
        +ve angle rotates the lens anticlockwise

        The grid variable is a tuple of (theta_1, theta_2), where
        each theta_1, theta_2 is itself a 2D array of the x and y
        coordinates respectively.~
        """

        super().__init__(centre=centre, ell_comps=ell_comps)

        self.einstein_radius = einstein_radius
        self.einstein_radius_elliptical = np.sqrt(self.axis_ratio()) * einstein_radius
        self.break_radius = break_radius
        self.inner_slope = inner_slope
        self.outer_slope = outer_slope

        # Parameters defined in the notes
        self.nu = break_radius / self.einstein_radius_elliptical
        self.dt = (2 - self.inner_slope) / (2 - self.outer_slope)

        # Normalisation (eq. 5)
        if self.nu < 1:
            self.kB = (2 - self.inner_slope) / (
                (2 * self.nu**2)
                * (1 + self.dt * (self.nu ** (self.outer_slope - 2) - 1))
            )
        else:
            self.kB = (2 - self.inner_slope) / (2 * self.nu**2)

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Returns the dimensionless density kappa=Sigma/Sigma_c (eq. 1)
        """

        # Ell radius
        radius = xp.hypot(grid.array[:, 1] * self.axis_ratio(xp), grid.array[:, 0])

        # Inside break radius
        kappa_inner = self.kB * (self.break_radius / radius) ** self.inner_slope

        # Outside break radius
        kappa_outer = self.kB * (self.break_radius / radius) ** self.outer_slope

        return kappa_inner * (radius <= self.break_radius) + kappa_outer * (
            radius > self.break_radius
        )

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        from autogalaxy.profiles.mass.abstract.mge import MGEDecomposer

        radii_min = self.break_radius / 100.0
        radii_max = self.einstein_radius * 20.0
        sigmas = xp.exp(xp.linspace(xp.log(radii_min), xp.log(radii_max), 30))
        mge_decomp = MGEDecomposer(mass_profile=self)
        return mge_decomp.potential_2d_via_mge_from(
            grid=grid, xp=xp, sigma_log_list=sigmas,
            ellipticity_convention="major", three_D=False,
        )

    @aa.decorators.to_vector_yx
    @aa.decorators.transform(rotate_back=True)
    def deflections_yx_2d_from(self, grid, xp=np, max_terms=20, **kwargs):
        """
        Returns the complex deflection angle from eq. 18 and 19
        """

        # Rotate coordinates
        z = grid.array[:, 1] + 1j * grid.array[:, 0]

        # Ell radius
        R = xp.hypot(z.real * self.axis_ratio(xp), z.imag)

        # Factors common to eq. 18 and 19
        factors = (
            2
            * self.kB
            * (self.break_radius**2)
            / (self.axis_ratio(xp) * z * (2 - self.inner_slope))
        )

        # Hypergeometric functions
        # (in order of appearance in eq. 18 and 19)
        # These can also be computed with scipy.special.hyp2f1(), it's
        # much slower can be a useful test
        F1 = self.hyp2f1_series(
            self.inner_slope, self.axis_ratio(xp), R, z, max_terms=max_terms, xp=xp
        )
        F2 = self.hyp2f1_series(
            self.inner_slope,
            self.axis_ratio(xp),
            self.break_radius,
            z,
            max_terms=max_terms,
            xp=xp,
        )
        F3 = self.hyp2f1_series(
            self.outer_slope, self.axis_ratio(xp), R, z, max_terms=max_terms, xp=xp
        )
        F4 = self.hyp2f1_series(
            self.outer_slope,
            self.axis_ratio(xp),
            self.break_radius,
            z,
            max_terms=max_terms,
            xp=xp,
        )

        # theta < break radius (eq. 18)
        inner_part = factors * F1 * (self.break_radius / R) ** (self.inner_slope - 2)

        # theta > break radius (eq. 19)
        outer_part = factors * (
            F2
            + self.dt * (((self.break_radius / R) ** (self.outer_slope - 2)) * F3 - F4)
        )

        # Combine and take the conjugate
        deflections = (
            inner_part * (R <= self.break_radius) + outer_part * (R > self.break_radius)
        ).conjugate()

        return xp.vstack((xp.imag(deflections), xp.real(deflections))).T

    @staticmethod
    def hyp2f1_series(t, q, r, z, max_terms=20, xp=np):
        """
        Computes eq. 26 for a radius r, slope t,
        axis ratio q, and coordinates z.
        """

        # u from eq. 25
        q_ = (1 - q**2) / (q**2)
        u = 0.5 * (1 - xp.sqrt(1 - q_ * (r / z) ** 2))

        # First coefficient
        a_n = 1.0

        # Storage for sum
        F = xp.zeros_like(z, dtype="complex64")

        for n in range(max_terms):
            F += a_n * (u**n)
            a_n *= ((2 * n) + 4 - (2 * t)) / ((2 * n) + 4 - t)

        return F


class PowerLawBrokenSph(PowerLawBroken):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        inner_slope: float = 1.5,
        outer_slope: float = 2.5,
        break_radius: float = 0.01,
    ):
        """
        Ell, homoeoidal mass model with an inner_slope
        and outer_slope, continuous in density across break_radius.
        Position angle is defined to be zero on x-axis and
        +ve angle rotates the lens anticlockwise

        The grid variable is a tuple of (theta_1, theta_2), where
        each theta_1, theta_2 is itself a 2D array of the x and y
        coordinates respectively.~
        """

        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            einstein_radius=einstein_radius,
            inner_slope=inner_slope,
            outer_slope=outer_slope,
            break_radius=break_radius,
        )
