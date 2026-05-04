import numpy as np
from typing import Optional, Tuple

import autoarray as aa

from autogalaxy.profiles.light.decorators import (
    check_operated_only,
)
from autogalaxy.profiles.light.standard.shapelets.abstract import AbstractShapelet


class ShapeletExponential(AbstractShapelet):
    def __init__(
        self,
        n: int,
        m: int,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 1.0,
        beta: float = 1.0,
    ):
        """
        Shapelets where the basis function is defined according to an Exponential using a polar (r,theta) grid of
        coordinates.

        Shapelets are defined according to:

          https://arxiv.org/abs/astro-ph/0105178

        Shapelets are described in the context of strong lens modeling in:

          https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.3066T/abstract

        Parameters
        ----------
        n
            The n order of the shapelets basis function.
        m
            The m order of the shapelets basis function in the x-direction.
        centre
            The (y,x) arc-second coordinates of the profile (shapelet) centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        beta
            The characteristic length scale of the shapelet basis function, defined in arc-seconds.
        """

        self.n = n
        self.m = m

        super().__init__(
            centre=centre, ell_comps=ell_comps, beta=beta, intensity=intensity
        )

    @property
    def coefficient_tag(self) -> str:
        return f"n_{self.n}_m_{self.m}"

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
        Returns the Exponential Shapelet light profile's 2D image from a 2D grid of Exponential (y,x) coordinates.

        If the coordinates have not been transformed to the profile's geometry (e.g. translated to the
        profile `centre`), this is performed automatically.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the Exponential Shapelet evaluated at every (y,x) coordinate on the transformed grid.
        """
        from scipy.special import genlaguerre

        # factorial backend switch
        if xp is np:
            from scipy.special import factorial
        else:
            from jax.scipy.special import factorial

        radial = (grid.array[:, 0] ** 2 + grid.array[:, 1] ** 2) / self.beta
        theta = xp.arctan(grid.array[:, 1] / grid.array[:, 0])

        prefactor = (
            1.0
            / xp.sqrt(2 * xp.pi)
            / self.beta
            * (self.n + 0.5) ** (-1 - xp.abs(self.m))
            * (-1) ** (self.n + self.m)
            * xp.sqrt(
                factorial(self.n - xp.abs(self.m)) / 2 * self.n
                + 1 / factorial(self.n + xp.abs(self.m))
            )
        )

        laguerre = genlaguerre(n=self.n - xp.abs(self.m), alpha=2 * xp.abs(self.m))
        shapelet = laguerre(radial / (self.n + 0.5))

        return xp.abs(
            prefactor
            * xp.exp(-radial / (2 * self.n + 1))
            * radial ** (xp.abs(self.m))
            * shapelet
            * xp.cos(self.m * theta)
            + -1.0j * xp.sin(self.m * theta)
        )


class ShapeletExponentialSph(ShapeletExponential):
    def __init__(
        self,
        n: int,
        m: int,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 1.0,
        beta: float = 1.0,
    ):
        """
        Shapelets where the basis function is defined according to a Exponential (r,theta) grid of coordinates.

        Shapelets are defined according to:

          https://arxiv.org/abs/astro-ph/0105178

        Shapelets are described in the context of strong lens modeling in:

          https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.3066T/abstract

        Parameters
        ----------
        n_y
            The order of the shapelets basis function in the y-direction.
        n_x
            The order of the shapelets basis function in the x-direction.
        centre
            The (y,x) arc-second coordinates of the profile (shapelet) centre.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        beta
            The characteristic length scale of the shapelet basis function, defined in arc-seconds.
        """

        super().__init__(
            n=n,
            m=m,
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            beta=beta,
        )
