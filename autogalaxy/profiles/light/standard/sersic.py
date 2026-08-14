"""
Sersic light profiles.

The Sersic profile is one of the most widely used models for describing the surface brightness of galaxies.
It has the functional form:

    I(r) = I_eff * exp{ -b_n * [(r / r_eff)^(1/n) - 1] }

where `r_eff` is the effective (half-light) radius, `n` is the Sersic index controlling concentration, and
`b_n` is derived from `n` to ensure that `r_eff` encloses half the total flux.

Special cases: n=1 is the exponential (disk) profile, n=4 is the de Vaucouleurs (bulge) profile.

This module provides both elliptical (`Sersic`) and spherical (`SersicSph`) variants.
"""

import numpy as np

from numpy import seterr
from typing import Optional, Tuple

import autoarray as aa

from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.light.decorators import (
    check_operated_only,
)
from autogalaxy.profiles import validate


class AbstractSersic(LightProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
    ):
        """
        Abstract base class for elliptical Sersic light profiles.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        effective_radius
            The circular radius containing half the light of this light profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        """
        super().__init__(centre=centre, ell_comps=ell_comps, intensity=intensity)
        self.effective_radius = effective_radius
        validate.validate_sersic_index(sersic_index=sersic_index)
        self.sersic_index = sersic_index

    @property
    def elliptical_effective_radius(self) -> float:
        """
        The `effective_radius` of a Sersic light profile is defined as the circular effective radius, which is the
        radius within which a circular aperture contains half the profile's total integrated light.

        For elliptical systems, this will not robustly capture the light profile's elliptical shape.

        The elliptical effective radius instead describes the major-axis radius of the ellipse containing
        half the light, and may be more appropriate for highly flattened systems like disk galaxies.
        """
        return self.effective_radius / xp.sqrt(self.axis_ratio(xp))

    @property
    def sersic_constant(self) -> float:
        """
        A parameter derived from Sersic index which ensures that effective radius contains 50% of the profile's
        total integrated light.
        """
        return (
            (2 * self.sersic_index)
            - (1.0 / 3.0)
            + (4.0 / (405.0 * self.sersic_index))
            + (46.0 / (25515.0 * self.sersic_index**2))
            + (131.0 / (1148175.0 * self.sersic_index**3))
            - (2194697.0 / (30690717750.0 * self.sersic_index**4))
        )

    def image_2d_via_radii_from(self, radius: np.ndarray) -> np.ndarray:
        """
        Returns the 2D image of the Sersic light profile from a grid of coordinates which are the radial distances of
        each coordinate from the its `centre`.

        Parameters
        ----------
        grid_radii
            The radial distances from the centre of the profile, for each coordinate on the grid.
        """
        return self._intensity * xp.exp(
            -self.sersic_constant
            * (((radius / self.effective_radius) ** (1.0 / self.sersic_index)) - 1)
        )


class Sersic(AbstractSersic, LightProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
    ):
        """
        The elliptical Sersic light profile.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        """
        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )

    def image_2d_via_radii_from(
        self, grid_radii: np.ndarray, xp=np, **kwargs
    ) -> np.ndarray:
        """
        Returns the 2D image of the Sersic light profile from a grid of coordinates which are the radial distances of
        each coordinate from the its `centre`.

        Parameters
        ----------
        grid_radii
            The radial distances from the centre of the profile, for each coordinate on the grid.
        """
        seterr(all="ignore")
        return xp.multiply(
            self._intensity,
            xp.exp(
                xp.multiply(
                    -self.sersic_constant,
                    xp.add(
                        xp.power(
                            xp.divide(grid_radii.array, self.effective_radius),
                            1.0 / self.sersic_index,
                        ),
                        -1,
                    ),
                )
            ),
        )

    @aa.decorators.to_array
    def _eccentric_radii_grid_from_cartesian(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ) -> np.ndarray:
        """Return eccentric radii without converting ``ell_comps`` to polar form."""
        ell_comps_y, ell_comps_x = self.ell_comps
        ell_comps_norm = xp.sqrt(
            xp.maximum(
                xp.add(xp.square(ell_comps_y), xp.square(ell_comps_x)),
                1.0e-12,
            )
        )
        ell_comps_scale = xp.minimum(1.0, 0.999 / ell_comps_norm)

        ell_comps_y = xp.multiply(ell_comps_y, ell_comps_scale)
        ell_comps_x = xp.multiply(ell_comps_x, ell_comps_scale)
        ell_comps_norm_squared = xp.add(xp.square(ell_comps_y), xp.square(ell_comps_x))

        y = xp.add(grid.array[:, 0], -self.centre[0])
        x = xp.add(grid.array[:, 1], -self.centre[1])

        numerator = xp.add(
            xp.multiply(
                xp.add(1.0 + ell_comps_norm_squared, -2.0 * ell_comps_x),
                xp.square(x),
            ),
            xp.add(
                xp.multiply(-4.0 * ell_comps_y, xp.multiply(x, y)),
                xp.multiply(
                    xp.add(1.0 + ell_comps_norm_squared, 2.0 * ell_comps_x),
                    xp.square(y),
                ),
            ),
        )

        return xp.sqrt(xp.divide(numerator, 1.0 - ell_comps_norm_squared))

    @aa.over_sample
    @aa.decorators.to_array
    @check_operated_only
    def image_2d_from(
        self,
        grid: aa.type.Grid2DLike,
        xp=np,
        operated_only: Optional[bool] = None,
        **kwargs,
    ) -> aa.Array2D:
        """
        Returns the Sersic light profile's 2D image from a 2D grid of Cartesian (y,x) coordinates.

        If the coordinates have not been transformed to the profile's geometry (e.g. translated to the
        profile `centre`), this is performed automatically.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the Sersic evaluated at every (y,x) coordinate on the transformed grid.
        """

        if getattr(grid, "is_transformed", False):
            grid_radii = self.eccentric_radii_grid_from(grid=grid, xp=xp, **kwargs)
        else:
            grid_radii = self._eccentric_radii_grid_from_cartesian(
                grid=grid, xp=xp, **kwargs
            )

        return self.image_2d_via_radii_from(grid_radii=grid_radii, xp=xp, **kwargs)


class SersicSph(Sersic):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
    ):
        """
        The spherical Sersic light profile.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the of the light profile.
        """
        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )
