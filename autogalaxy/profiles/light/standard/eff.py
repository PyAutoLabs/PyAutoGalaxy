import numpy as np
from typing import Optional, Tuple

import autoarray as aa

from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.light.decorators import (
    check_operated_only,
)


class ElsonFreeFall(LightProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        eta: float = 1.5,
    ):
        """
        The elliptical Elson, Fall and Freeman (EFF) light profile, which is commonly used to represent the clumps of
        Lyman-alpha emitter galaxies (see https://arxiv.org/abs/1708.08854).

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
        eta
            Scales the intensity gradient of the profile.
        """

        super().__init__(centre=centre, ell_comps=ell_comps, intensity=intensity)

        self.effective_radius = effective_radius
        self.eta = eta

    def image_2d_via_radii_from(self, grid_radii: np.ndarray, xp=np) -> np.ndarray:
        """
        Returns the 2D image of the Sersic light profile from a grid of coordinates which are the radial distances of
        each coordinate from the its `centre`.

        Parameters
        ----------
        grid_radii
            The radial distances from the centre of the profile, for each coordinate on the grid.
        """
        np.seterr(all="ignore")
        return self._intensity * (1 + (grid_radii / self.effective_radius) ** 2) ** (
            -self.eta
        )

    @aa.over_sample
    @aa.decorators.to_array
    @check_operated_only
    @aa.decorators.transform
    def image_2d_from(
        self,
        grid: aa.type.Grid2DLike,
        xp=np,
        operated_only: Optional[bool] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Returns the Eff light profile's 2D image from a 2D grid of Cartesian (y,x) coordinates.

        If the coordinates have not been transformed to the profile's geometry (e.g. translated to the
        profile `centre`), this is performed automatically.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the Eff evaluated at every (y,x) coordinate on the transformed grid.
        """
        return self.image_2d_via_radii_from(
            self.eccentric_radii_grid_from(grid=grid, xp=xp, **kwargs)
        )

    @property
    def half_light_radius(self) -> float:
        return self.effective_radius * np.sqrt(0.5 ** (1.0 / (1.0 - self.eta)) - 1.0)


class ElsonFreeFallSph(ElsonFreeFall):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        eta: float = 1.5,
    ):
        """
        The spherical Elson, Fall and Freeman (EFF) light profile, which is commonly used to represent the clumps of
        Lyman-alpha emitter galaxies (see https://arxiv.org/abs/1708.08854).

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        eta
            Scales the intensity gradient of the profile.
        """

        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            eta=eta,
        )
