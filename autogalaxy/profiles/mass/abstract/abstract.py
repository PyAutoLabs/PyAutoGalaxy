import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.geometry_profiles import EllProfile


class MassProfile(EllProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
    ):
        """
        Abstract class for elliptical mass profiles.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        """
        super().__init__(centre=centre, ell_comps=ell_comps)

    def deflections_yx_2d_from(self, grid):
        raise NotImplementedError

    def deflections_2d_via_potential_2d_from(self, grid):
        potential = self.potential_2d_from(grid=grid)

        deflections_y_2d = np.gradient(
            potential.native.array, grid.native.array[:, 0, 0], axis=0
        )
        deflections_x_2d = np.gradient(
            potential.native.array, grid.native.array[0, :, 1], axis=1
        )

        return aa.Grid2D(
            values=np.stack((deflections_y_2d, deflections_x_2d), axis=-1),
            mask=grid.mask,
        )

    def convergence_2d_from(self, grid, xp=np):
        raise NotImplementedError

    def convergence_func(self, grid_radius: float) -> float:
        raise NotImplementedError

    def potential_2d_from(self, grid):
        raise NotImplementedError

    def potential_func(self, u, y, x):
        raise NotImplementedError

    def mass_integral(self, x, xp=np):
        return 2 * xp.pi * x * self.convergence_func(grid_radius=aa.ArrayIrregular(x))

    @property
    def ellipticity_rescale(self):
        return NotImplementedError()

    def mass_angular_within_circle_from(self, radius: float):
        """
        Integrate the mass profiles's convergence profile to compute the total mass within a circle of
        specified radius. This is centred on the mass profile.

        Parameters
        ----------
        radius : dim.Length
            The radius of the circle to compute the dimensionless mass within.
        """
        from scipy.integrate import quad

        return quad(self.mass_integral, a=0.0, b=radius)[0]

    def density_between_circular_annuli(
        self, inner_annuli_radius: float, outer_annuli_radius: float
    ):
        """
        Calculate the mass between two circular annuli and compute the density by dividing by the annuli surface
        area.

        The value returned by the mass integral is dimensionless, therefore the density between annuli is returned in \
        unit_label of inverse radius squared. A conversion factor can be specified to convert this to a physical value \
        (e.g. the critical surface mass density).

        Parameters
        ----------
        inner_annuli_radius
            The radius of the inner annulus outside of which the density are estimated.
        outer_annuli_radius
            The radius of the outer annulus inside of which the density is estimated.
        """
        annuli_area = (np.pi * outer_annuli_radius**2.0) - (
            np.pi * inner_annuli_radius**2.0
        )

        outer_mass = self.mass_angular_within_circle_from(radius=outer_annuli_radius)

        inner_mass = self.mass_angular_within_circle_from(radius=inner_annuli_radius)

        return (outer_mass - inner_mass) / annuli_area

    @property
    def average_convergence_of_1_radius(self):
        """
        The radius a critical curve forms for this mass profile, e.g. where the mean convergence is equal to 1.0.

        In case of ellipitical mass profiles, the 'average' critical curve is used, whereby the convergence is \
        rescaled into a circle using the axis ratio.

        This radius corresponds to the Einstein radius of the mass profile, and is a property of a number of \
        mass profiles below.
        """
        from scipy.optimize import root_scalar

        def func(radius):
            return (
                self.mass_angular_within_circle_from(radius=radius)
                - np.pi * radius**2.0
            )

        return self.ellipticity_rescale() * root_scalar(func, bracket=[1e-4, 1e4]).root

    def extract_attribute(self, cls, attr_name):
        """
        Returns an attribute of a class and its children profiles in the galaxy as a `ValueIrregular`
        or `Grid2DIrregular` object.

        For example, if a galaxy has two light profiles and we want the `LightProfile` axis-ratios, the following:

        `galaxy.extract_attribute(cls=LightProfile, name="axis_ratio"`

        would return:

        ArrayIrregular(values=[axis_ratio_0, axis_ratio_1])

        If a galaxy has three mass profiles and we want the `MassProfile` centres, the following:

        `galaxy.extract_attribute(cls=MassProfile, name="centres"`

         would return:

        GridIrregular2D(grid=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1), (centre_y_2, centre_x_2)])

        This is used for visualization, for example plotting the centres of all light profiles colored by their profile.
        """

        if isinstance(self, cls):
            if hasattr(self, attr_name):
                attribute = getattr(self, attr_name)

                if isinstance(attribute, float):
                    return aa.ArrayIrregular(values=[attribute])
                if isinstance(attribute, tuple):
                    return aa.Grid2DIrregular(values=[attribute])
