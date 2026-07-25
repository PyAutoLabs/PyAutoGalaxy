import numpy as np
from typing import Optional, Tuple

import autoarray as aa

from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.light.decorators import (
    check_operated_only,
)


class Moffat(LightProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        alpha: float = 0.5,
        beta: float = 2.0,
    ):
        """
        The elliptical Moffat light profile, which is commonly used to model the Point Spread Function of
        Astronomy observations.

        This form of the MOffat profile is a reparameterizaiton of the original formalism given by
        https://ui.adsabs.harvard.edu/abs/1969A%26A.....3..455M/abstract. The actual profile itself is identical.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        alpha
            Scales the overall size of the Moffat profile and for a PSF typically corresponds to the FWHM / 2.
        beta
            Scales the wings at the outskirts of the Moffat profile, where smaller values imply heavier wings and it
            tends to a Gaussian as beta goes to infinity.
        """

        super().__init__(centre=centre, ell_comps=ell_comps, intensity=intensity)
        self.alpha = alpha
        self.beta = beta

    def image_2d_via_radii_from(self, grid_radii: np.ndarray, xp=np) -> np.ndarray:
        """
        Returns the 2D image of the Moffat light profile from a grid of coordinates which are the radial distance of
        each coordinate from the its `centre`.

        Note: sigma is divided by sqrt(q) here.

        Parameters
        ----------
        grid_radii
            The radial distances from the centre of the profile, for each coordinate on the grid.
        """
        return xp.multiply(
            self._intensity,
            xp.power(
                1
                + xp.square(
                    xp.divide(
                        grid_radii.array, self.alpha / xp.sqrt(self.axis_ratio(xp))
                    )
                ),
                -self.beta,
            ),
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
        Returns the Moffat light profile's 2D image from a 2D grid of Cartesian (y,x) coordinates.

        If the coordinates have not been transformed to the profile's geometry (e.g. translated to the
        profile `centre`), this is performed automatically.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the Moffat evaluated at every (y,x) coordinate on the transformed grid.
        """

        return self.image_2d_via_radii_from(
            self.eccentric_radii_grid_from(grid=grid, xp=xp, **kwargs)
        )


class MoffatSph(Moffat):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        alpha: float = 0.5,
        beta: float = 2.0,
    ):
        """
        The spherical Moffat light profile:

        This form of the MOffat profile is a reparameterizaiton of the original formalism given by
        https://ui.adsabs.harvard.edu/abs/1969A%26A.....3..455M/abstract. The actual profile itself is identical.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        alpha
            Scales the overall size of the Moffat profile and for a PSF typically corresponds to the FWHM / 2.
        beta
            Scales the wings at the outskirts of the Moffat profile, where smaller values imply heavier wings and it
            tends to a Gaussian as beta goes to infinity.
        """

        super().__init__(centre=centre, intensity=intensity, alpha=alpha, beta=beta)
