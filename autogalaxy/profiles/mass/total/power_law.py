import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.total.power_law_core import PowerLawCore
from autogalaxy import convert as al_convert

import jax
import jax.numpy as jnp


class PowerLaw(PowerLawCore):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
    ):
        """
        Represents an elliptical power-law density distribution.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        einstein_radius
            The arc-second Einstein radius.
        slope
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        """

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            einstein_radius=einstein_radius,
            slope=slope,
            core_radius=0.0,
        )

    @aa.grid_dec.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):

        alpha = self.deflections_yx_2d_from(
            grid=aa.Grid2DIrregular(grid), xp=xp, **kwargs
        )

        alpha_x = alpha[:, 1]
        alpha_y = alpha[:, 0]

        x = grid.array[:, 1] - self.centre[1]
        y = grid.array[:, 0] - self.centre[0]

        return (x * alpha_x + y * alpha_y) / (3 - self.slope)

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        For coordinates (0.0, 0.0) the analytic calculation of the deflection angle gives a NaN. Therefore,
        coordinates at (0.0, 0.0) are shifted slightly to (1.0e-8, 1.0e-8).

        This code is an adaption of Tessore & Metcalf 2015:
        https://arxiv.org/abs/1507.01819

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        slope = self.slope - 1.0
        einstein_radius = (
            2.0 / (self.axis_ratio(xp) ** -0.5 + self.axis_ratio(xp) ** 0.5)
        ) * self.einstein_radius

        factor = xp.divide(1.0 - self.axis_ratio(xp), 1.0 + self.axis_ratio(xp))
        b = xp.multiply(einstein_radius, xp.sqrt(self.axis_ratio(xp)))
        angle = xp.arctan2(
            grid.array[:, 0], xp.multiply(self.axis_ratio(xp), grid.array[:, 1])
        )  # Note, this angle is not the position angle
        z = xp.add(
            xp.multiply(xp.cos(angle), 1 + 0j), xp.multiply(xp.sin(angle), 0 + 1j)
        )

        R = xp.sqrt(
            (self.axis_ratio(xp) * grid.array[:, 1]) ** 2
            + grid.array[:, 0] ** 2
            + 1e-16
        )

        if xp.__name__.startswith("jax"):

            from .jax_utils import omega

            zh = omega(z, slope, factor, n_terms=20, xp=xp)

            complex_angle = (
                2.0 * b / (1.0 + self.axis_ratio(xp)) * (b / R) ** (slope - 1.0) * zh
            )
        else:

            from scipy import special

            complex_angle = (
                2.0
                * b
                / (1.0 + self.axis_ratio(xp))
                * (b / R) ** (slope - 1.0)
                * z
                * special.hyp2f1(1.0, 0.5 * slope, 2.0 - 0.5 * slope, -factor * z**2)
            )

        deflection_y = complex_angle.imag
        deflection_x = complex_angle.real

        rescale_factor = (self.ellipticity_rescale(xp)) ** (slope - 1)

        deflection_y *= rescale_factor
        deflection_x *= rescale_factor

        return self.rotated_grid_from_reference_frame_from(
            grid=xp.vstack((deflection_y, deflection_x)).T, xp=xp
        )

    def convergence_func(self, grid_radius: float, xp=np) -> float:
        return self.einstein_radius_rescaled(xp) * grid_radius.array ** (
            -(self.slope - 1)
        )

    @staticmethod
    def potential_func(u, y, x, axis_ratio, slope, core_radius, xp=np):
        _eta_u = xp.sqrt((u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u)))))
        return (
            (_eta_u / u)
            * ((3.0 - slope) * _eta_u) ** -1.0
            * _eta_u ** (3.0 - slope)
            / ((1 - (1 - axis_ratio**2) * u) ** 0.5)
        )


class PowerLawSph(PowerLaw):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
    ):
        """
        Represents a spherical power-law density distribution.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius
            The arc-second Einstein radius.
        slope
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        """

        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            einstein_radius=einstein_radius,
            slope=slope,
        )

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        eta = self.radial_grid_from(grid=grid, xp=xp, **kwargs).array
        deflection_r = (
            2.0
            * self.einstein_radius_rescaled(xp)
            * xp.divide(
                xp.power(eta, (3.0 - self.slope)),
                xp.multiply((3.0 - self.slope), eta),
            )
        )

        return self._cartesian_grid_via_radial_from(
            grid=grid, radius=deflection_r, xp=xp
        )



class PowerLawMultiBand(PowerLawCore):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        axis_ratio: float = 0.99,
        position_angle: float = 0.0,
        einstein_radius: float = 1.0,
        slope: float = 2.0,
        shift_centre_0: float = 0.0,
        shift_centre_1: float = 0.0,
        shift_angle: float = 0.0,
        #ell_comps: Tuple[float, float] = (0.0, 0.0),
    ):
        """
        Represents an elliptical power-law density distribution.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        einstein_radius
            The arc-second Einstein radius.
        slope
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        """

        centre = (centre[0] + shift_centre_0, centre[1] + shift_centre_1)
        position_angle = position_angle + shift_angle
        ell_comps = al_convert.ell_comps_from(axis_ratio=axis_ratio, angle=position_angle, xp=jnp)

        #ell_comps = (ell_comps[0].astype(float), ell_comps[1].astype(float))
        # print for eager mode and debug.print for traced JAX execution.
        # print(f"PowerLawMultiBand ell_comps (python print): {ell_comps}")
        # jax.debug.print(
        #     "PowerLawMultiBand ell_comps (jax.debug.print): {}",
        #     ell_comps,
        # )

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            einstein_radius=einstein_radius,
            slope=slope,
            core_radius=0.0,
        )

        self.shift_centre_0 = shift_centre_0
        self.shift_centre_1 = shift_centre_1
        self.shift_angle = shift_angle
        self.position_angle_value = position_angle
        self.axis_ratio_value = axis_ratio
        
        

    @aa.grid_dec.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):

        alpha = self.deflections_yx_2d_from(
            grid=aa.Grid2DIrregular(grid), xp=xp, **kwargs
        )

        alpha_x = alpha[:, 1]
        alpha_y = alpha[:, 0]

        x = grid.array[:, 1] - self.centre[1]
        y = grid.array[:, 0] - self.centre[0]

        return (x * alpha_x + y * alpha_y) / (3 - self.slope)

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        For coordinates (0.0, 0.0) the analytic calculation of the deflection angle gives a NaN. Therefore,
        coordinates at (0.0, 0.0) are shifted slightly to (1.0e-8, 1.0e-8).

        This code is an adaption of Tessore & Metcalf 2015:
        https://arxiv.org/abs/1507.01819

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        slope = self.slope - 1.0
        einstein_radius = (
            2.0 / (self.axis_ratio(xp) ** -0.5 + self.axis_ratio(xp) ** 0.5)
        ) * self.einstein_radius

        factor = xp.divide(1.0 - self.axis_ratio(xp), 1.0 + self.axis_ratio(xp))
        b = xp.multiply(einstein_radius, xp.sqrt(self.axis_ratio(xp)))
        angle = xp.arctan2(
            grid.array[:, 0], xp.multiply(self.axis_ratio(xp), grid.array[:, 1])
        )  # Note, this angle is not the position angle
        z = xp.add(
            xp.multiply(xp.cos(angle), 1 + 0j), xp.multiply(xp.sin(angle), 0 + 1j)
        )

        R = xp.sqrt(
            (self.axis_ratio(xp) * grid.array[:, 1]) ** 2
            + grid.array[:, 0] ** 2
            + 1e-16
        )

        if xp.__name__.startswith("jax"):

            from .jax_utils import omega

            zh = omega(z, slope, factor, n_terms=20, xp=xp)

            complex_angle = (
                2.0 * b / (1.0 + self.axis_ratio(xp)) * (b / R) ** (slope - 1.0) * zh
            )
        else:

            from scipy import special

            complex_angle = (
                2.0
                * b
                / (1.0 + self.axis_ratio(xp))
                * (b / R) ** (slope - 1.0)
                * z
                * special.hyp2f1(1.0, 0.5 * slope, 2.0 - 0.5 * slope, -factor * z**2)
            )

        deflection_y = complex_angle.imag
        deflection_x = complex_angle.real

        rescale_factor = (self.ellipticity_rescale(xp)) ** (slope - 1)

        deflection_y *= rescale_factor
        deflection_x *= rescale_factor

        return self.rotated_grid_from_reference_frame_from(
            grid=xp.vstack((deflection_y, deflection_x)).T, xp=xp
        )

    def convergence_func(self, grid_radius: float, xp=np) -> float:
        return self.einstein_radius_rescaled(xp) * grid_radius.array ** (
            -(self.slope - 1)
        )

    @staticmethod
    def potential_func(u, y, x, axis_ratio, slope, core_radius, xp=np):
        _eta_u = xp.sqrt((u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u)))))
        return (
            (_eta_u / u)
            * ((3.0 - slope) * _eta_u) ** -1.0
            * _eta_u ** (3.0 - slope)
            / ((1 - (1 - axis_ratio**2) * u) ** 0.5)
        )
