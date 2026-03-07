import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.profiles.mass.stellar.abstract import StellarProfile

from autogalaxy.profiles.mass.total.isothermal import psi_from


class Chameleon(MassProfile, StellarProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.02,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The elliptical Chamelon mass profile.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        core_radius_0 : the core size of the first elliptical cored Isothermal profile.
        core_radius_1 : core_radius_0 + core_radius_1 is the core size of the second elliptical cored Isothermal profile.
            We use core_radius_1 here is to avoid negative values.

        Profile form:
            mass_to_light_ratio * intensity *\
                (1.0 / Sqrt(x^2 + (y/q)^2 + core_radius_0^2) - 1.0 / Sqrt(x^2 + (y/q)^2 + (core_radius_0 + core_radius_1)**2.0))
        """

        super(Chameleon, self).__init__(centre=centre, ell_comps=ell_comps)
        super(MassProfile, self).__init__(centre=centre, ell_comps=ell_comps)
        self.mass_to_light_ratio = mass_to_light_ratio
        self.intensity = intensity
        self.core_radius_0 = core_radius_0
        self.core_radius_1 = core_radius_1

    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        return self.deflections_2d_via_analytic_from(grid=grid, xp=xp, **kwargs)

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    def deflections_2d_via_analytic_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.
        Following Eq. (15) and (16), but the parameters are slightly different.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        factor = (
            2.0
            * self.mass_to_light_ratio
            * self.intensity
            / (1 + self.axis_ratio(xp))
            * self.axis_ratio(xp)
            / xp.sqrt(1.0 - self.axis_ratio(xp) ** 2.0)
        )

        core_radius_0 = xp.sqrt(
            (4.0 * self.core_radius_0**2.0) / (1.0 + self.axis_ratio(xp)) ** 2
        )
        core_radius_1 = xp.sqrt(
            (4.0 * self.core_radius_1**2.0) / (1.0 + self.axis_ratio(xp)) ** 2
        )

        psi0 = psi_from(
            grid=grid, axis_ratio=self.axis_ratio(xp), core_radius=core_radius_0, xp=xp
        )
        psi1 = psi_from(
            grid=grid, axis_ratio=self.axis_ratio(xp), core_radius=core_radius_1, xp=xp
        )

        deflection_y0 = xp.arctanh(
            xp.divide(
                xp.multiply(
                    xp.sqrt(1.0 - self.axis_ratio(xp) ** 2.0), grid.array[:, 0]
                ),
                xp.add(psi0, self.axis_ratio(xp) ** 2.0 * core_radius_0),
            )
        )

        deflection_x0 = xp.arctan(
            xp.divide(
                xp.multiply(
                    xp.sqrt(1.0 - self.axis_ratio(xp) ** 2.0), grid.array[:, 1]
                ),
                xp.add(psi0, core_radius_0),
            )
        )

        deflection_y1 = xp.arctanh(
            xp.divide(
                xp.multiply(
                    xp.sqrt(1.0 - self.axis_ratio(xp) ** 2.0), grid.array[:, 0]
                ),
                xp.add(psi1, self.axis_ratio(xp) ** 2.0 * core_radius_1),
            )
        )

        deflection_x1 = xp.arctan(
            xp.divide(
                xp.multiply(
                    xp.sqrt(1.0 - self.axis_ratio(xp) ** 2.0), grid.array[:, 1]
                ),
                xp.add(psi1, core_radius_1),
            )
        )

        deflection_y = xp.subtract(deflection_y0, deflection_y1)
        deflection_x = xp.subtract(deflection_x0, deflection_x1)

        return self.rotated_grid_from_reference_frame_from(
            xp.multiply(factor, xp.vstack((deflection_y, deflection_x)).T), xp=xp
        )

    @aa.over_sample
    @aa.grid_dec.to_array
    @aa.grid_dec.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """Calculate the projected convergence at a given set of arc-second gridded coordinates.
        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.
        """
        return self.convergence_func(
            self.elliptical_radii_grid_from(grid=grid, xp=xp, **kwargs), xp=xp
        )

    def convergence_func(self, grid_radius: float, xp=np) -> float:
        return self.mass_to_light_ratio * self.image_2d_via_radii_from(
            grid_radius, xp=xp
        )

    @aa.grid_dec.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        return xp.zeros(shape=grid.shape[0])

    def image_2d_via_radii_from(self, grid_radii: np.ndarray, xp=np):
        """Calculate the intensity of the Chamelon light profile on a grid of radial coordinates.

        Parameters
        ----------
        grid_radii
            The radial distance from the centre of the profile. for each coordinate on the grid.
        """

        axis_ratio_factor = (1.0 + self.axis_ratio(xp)) ** 2.0

        return xp.multiply(
            self.intensity / (1 + self.axis_ratio(xp)),
            xp.add(
                xp.divide(
                    1.0,
                    xp.sqrt(
                        xp.add(
                            xp.square(grid_radii.array),
                            (4.0 * self.core_radius_0**2.0) / axis_ratio_factor,
                        )
                    ),
                ),
                -xp.divide(
                    1.0,
                    xp.sqrt(
                        xp.add(
                            xp.square(grid_radii.array),
                            (4.0 * self.core_radius_1**2.0) / axis_ratio_factor,
                        )
                    ),
                ),
            ),
        )

    def axis_ratio(self, xp=np):
        axis_ratio = super().axis_ratio(xp=xp)
        return axis_ratio if axis_ratio < 0.99999 else 0.99999


class ChameleonSph(Chameleon):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.02,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The spherica; Chameleon mass profile.

        Profile form:
            mass_to_light_ratio * intensity *\
                (1.0 / Sqrt(x^2 + (y/q)^2 + core_radius_0^2) - 1.0 / Sqrt(x^2 + (y/q)^2 + (core_radius_0 + core_radius_1)**2.0))

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        core_radius_0 : the core size of the first elliptical cored Isothermal profile.
        core_radius_1 : core_radius_0 + core_radius_1 is the core size of the second elliptical cored Isothermal profile.
            We use core_radius_1 here is to avoid negative values.
       """

        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            core_radius_0=core_radius_0,
            core_radius_1=core_radius_1,
            mass_to_light_ratio=mass_to_light_ratio,
        )
