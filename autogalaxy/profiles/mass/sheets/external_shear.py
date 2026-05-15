import numpy as np

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile

from autogalaxy import convert
import jax.numpy as jnp


class ExternalShear(MassProfile):
    def __init__(self, gamma_1: float = 0.0, gamma_2: float = 0.0):
        """
        An `ExternalShear` term, to model the line-of-sight contribution of other galaxies / satellites.

        The shear angle is defined in the direction of stretching of the image. Therefore, if an object located \
        outside the lens is responsible for the shear, it will be offset 90 degrees from the value of angle.

        Parameters
        ----------
        gamma
        """

        super().__init__(centre=(0.0, 0.0), ell_comps=(0.0, 0.0))
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2

    def magnitude(self, xp=np):
        return convert.shear_magnitude_from(
            gamma_1=self.gamma_1, gamma_2=self.gamma_2, xp=xp
        )

    def angle(self, xp=np):
        return convert.shear_angle_from(
            gamma_1=self.gamma_1, gamma_2=self.gamma_2, xp=xp
        )

    def convergence_func(self, grid_radius: float) -> float:
        return 0.0

    def average_convergence_of_1_radius(self):
        return 0.0

    @aa.grid_dec.to_array
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        return xp.zeros(shape=grid.shape[0])

    @aa.grid_dec.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        shear_angle = (
            self.angle(xp) - 90
        )  ##to be onsistent with autolens deflection angle calculation
        phig = xp.deg2rad(shear_angle)
        shear_amp = self.magnitude(xp=xp)
        phicoord = xp.arctan2(grid.array[:, 0], grid.array[:, 1])
        rcoord = xp.sqrt(grid.array[:, 0] ** 2.0 + grid.array[:, 1] ** 2.0)

        return -0.5 * shear_amp * rcoord**2 * xp.cos(2 * (phicoord - phig))

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        deflection_y = -xp.multiply(self.magnitude(xp=xp), grid.array[:, 0])
        deflection_x = xp.multiply(self.magnitude(xp=xp), grid.array[:, 1])
        return self.rotated_grid_from_reference_frame_from(
            grid=xp.vstack((deflection_y, deflection_x)).T,
            xp=xp,
        )
    


class ExternalShearMultiBand(MassProfile):
    def __init__(self, centre_0: float = 0.0, centre_1: float = 0.0, magnitude: float = 0.0, position_angle: float = 0.0, shift_centre_0: float = 0.0, shift_centre_1: float = 0.0, shift_angle: float = 0.0):
        """
        An `ExternalShear` term, to model the line-of-sight contribution of other galaxies / satellites.

        The shear angle is defined in the direction of stretching of the image. Therefore, if an object located \
        outside the lens is responsible for the shear, it will be offset 90 degrees from the value of angle.

        Parameters
        ----------
        gamma
        """

        centre_0 = centre_0 + shift_centre_0
        centre_1 = centre_1 + shift_centre_1
        position_angle = position_angle + shift_angle

        gamma_1, gamma_2 = convert.shear_gamma_1_2_from(
            magnitude=magnitude, angle=position_angle, xp=jnp
        )

        super().__init__(centre=(centre_0, centre_1), ell_comps=(0.0, 0.0))
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2

        self.shift_centre_0 = shift_centre_0
        self.shift_centre_1 = shift_centre_1
        self.shift_angle = shift_angle
        self.magnitude_value = magnitude
        self.position_angle_value = position_angle

    def magnitude(self, xp=np):
        return convert.shear_magnitude_from(
            gamma_1=self.gamma_1, gamma_2=self.gamma_2, xp=xp
        )

    def angle(self, xp=np):
        return convert.shear_angle_from(
            gamma_1=self.gamma_1, gamma_2=self.gamma_2, xp=xp
        )

    def convergence_func(self, grid_radius: float) -> float:
        return 0.0

    def average_convergence_of_1_radius(self):
        return 0.0

    @aa.grid_dec.to_array
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        return xp.zeros(shape=grid.shape[0])

    @aa.grid_dec.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        shear_angle = (
            self.angle(xp) - 90
        )  ##to be onsistent with autolens deflection angle calculation
        phig = xp.deg2rad(shear_angle)
        shear_amp = self.magnitude(xp=xp)
        phicoord = xp.arctan2(grid.array[:, 0], grid.array[:, 1])
        rcoord = xp.sqrt(grid.array[:, 0] ** 2.0 + grid.array[:, 1] ** 2.0)

        return -0.5 * shear_amp * rcoord**2 * xp.cos(2 * (phicoord - phig))

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        deflection_y = -xp.multiply(self.magnitude(xp=xp), grid.array[:, 0])
        deflection_x = xp.multiply(self.magnitude(xp=xp), grid.array[:, 1])
        return self.rotated_grid_from_reference_frame_from(
            grid=xp.vstack((deflection_y, deflection_x)).T,
            xp=xp,
        )
