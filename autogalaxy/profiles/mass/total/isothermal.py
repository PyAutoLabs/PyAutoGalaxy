import numpy as np

from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.total.power_law import PowerLaw


def psi_from(grid, axis_ratio, core_radius, xp=np):
    """
    Returns the $\Psi$ term in expressions for the calculation of the deflection of an elliptical isothermal mass
    distribution. This is used in the `Isothermal` and `Chameleon` `MassProfile`'s.

    The expression for Psi is:

    $\Psi = \sqrt(q^2(s^2 + x^2) + y^2)$

    Parameters
    ----------
    grid
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    axis_ratio
            Ratio of profiles ellipse's minor and major axes (b/a)
    core_radius
        The radius of the inner core

    Returns
    -------
    float
        The value of the Psi term.

    """
    return xp.sqrt(
        (axis_ratio**2.0 * (grid.array[:, 1] ** 2.0 + core_radius**2.0))
        + grid.array[:, 0] ** 2.0
        + 1e-16
    )


class Isothermal(PowerLaw):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
    ):
        """
        Represents an elliptical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope = 2.0.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        einstein_radius
            The arc-second Einstein radius.
        """

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            einstein_radius=einstein_radius,
            slope=2.0,
        )

    def axis_ratio(self, xp=np):
        axis_ratio = super().axis_ratio(xp=xp)
        return xp.minimum(axis_ratio, 0.99999)

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        factor = (
            2.0
            * self.einstein_radius_rescaled(xp)
            * self.axis_ratio(xp)
            / xp.sqrt(1 - self.axis_ratio(xp) ** 2)
        )

        psi = psi_from(
            grid=grid, axis_ratio=self.axis_ratio(xp), core_radius=0.0, xp=xp
        )

        deflection_y = xp.arctanh(
            xp.divide(
                xp.multiply(xp.sqrt(1 - self.axis_ratio(xp) ** 2), grid.array[:, 0]),
                psi,
            )
        )
        deflection_x = xp.arctan(
            xp.divide(
                xp.multiply(xp.sqrt(1 - self.axis_ratio(xp) ** 2), grid.array[:, 1]),
                psi,
            )
        )
        return self.rotated_grid_from_reference_frame_from(
            grid=xp.multiply(factor, xp.vstack((deflection_y, deflection_x)).T),
            xp=xp,
            **kwargs,
        )

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    def shear_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the (gamma_y, gamma_x) shear vector field on a grid of (y,x) arc-second coordinates.

        The result is returned as a `ShearYX2D` dats structure, which has shape [total_shear_vectors, 2], where
        entries for [:,0] are the gamma_2 values and entries for [:,1] are the gamma_1 values.

        Note therefore that this convention means the FIRST entries in the array are the gamma_2 values and the SECOND
        entries are the gamma_1 values.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        convergence = self.convergence_2d_from(grid=grid, xp=xp, **kwargs)

        gamma_2 = (
            -2
            * convergence.array
            * xp.divide(
                grid.array[:, 1] * grid.array[:, 0],
                grid.array[:, 1] ** 2 + grid.array[:, 0] ** 2,
            )
        )
        gamma_1 = -convergence.array * xp.divide(
            grid.array[:, 1] ** 2 - grid.array[:, 0] ** 2,
            grid.array[:, 1] ** 2 + grid.array[:, 0] ** 2,
        )

        shear_field = self.rotated_grid_from_reference_frame_from(
            grid=xp.vstack((gamma_2, gamma_1)).T, xp=xp, angle=self.angle(xp) * 2
        )

        return aa.VectorYX2DIrregular(values=shear_field, grid=grid)

    def convergence_func(self, grid_radius: float, xp=np) -> float:
        return self.einstein_radius_rescaled(xp) / grid_radius.array


class IsothermalSph(Isothermal):
    def __init__(
        self, centre: Tuple[float, float] = (0.0, 0.0), einstein_radius: float = 1.0
    ):
        """
        Represents a spherical isothermal density distribution, which is equivalent to the spherical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius
            The arc-second Einstein radius.
        """
        super().__init__(
            centre=centre, ell_comps=(0.0, 0.0), einstein_radius=einstein_radius
        )

    def axis_ratio(self, xp=np):
        return 1.0

    @aa.over_sample
    @aa.grid_dec.to_array
    @aa.grid_dec.transform
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the potential on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        eta = self.elliptical_radii_grid_from(grid=grid, xp=xp, **kwargs)
        return 2.0 * self.einstein_radius_rescaled(xp) * eta

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        return self._cartesian_grid_via_radial_from(
            grid=grid,
            xp=xp,
            radius=xp.full(grid.shape[0], 2.0 * self.einstein_radius_rescaled(xp)),
            **kwargs,
        )
