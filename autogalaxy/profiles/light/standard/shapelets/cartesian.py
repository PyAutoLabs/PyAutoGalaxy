import numpy as np
from typing import Optional, Tuple

import autoarray as aa

from autogalaxy.profiles.light.decorators import (
    check_operated_only,
)
from autogalaxy.profiles.light.standard.shapelets.abstract import AbstractShapelet


def hermite_phys(n: int, x, xp=np):
    """
    Physicists' Hermite polynomial H_n(x), compatible with NumPy and JAX via `xp`.

    Recurrence:
      H_0(x) = 1
      H_1(x) = 2x
      H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    H0 = xp.ones_like(x)
    if n == 0:
        return H0

    H1 = 2.0 * x
    if n == 1:
        return H1

    Hnm1 = H0
    Hn = H1
    for k in range(1, n):
        Hnp1 = 2.0 * x * Hn - 2.0 * float(k) * Hnm1
        Hnm1, Hn = Hn, Hnp1
    return Hn


class ShapeletCartesian(AbstractShapelet):
    def __init__(
        self,
        n_y: int,
        n_x: int,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 1.0,
        beta: float = 1.0,
    ):
        """
        Shapelets where the basis function is defined according to a Cartesian (y,x) grid of coordinates.

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
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        beta
            The characteristic length scale of the shapelet basis function, defined in arc-seconds.
        """

        self.n_y = n_y
        self.n_x = n_x

        super().__init__(
            centre=centre, ell_comps=ell_comps, beta=beta, intensity=intensity
        )

    @property
    def coefficient_tag(self) -> str:
        return f"n_y_{self.n_y}_n_x_{self.n_x}"

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
        Returns the Cartesian Shapelet light profile's 2D image from a 2D grid of Cartesian (y,x) coordinates.
        """

        # factorial backend switch
        if xp is np:
            from scipy.special import factorial
        else:
            from jax.scipy.special import factorial

        y = grid.array[:, 0]
        x = grid.array[:, 1]

        # Apply axis-ratio stretching (minor axis)
        q = self.axis_ratio(xp)
        y_ell = y / q
        x_ell = x

        # Evaluate Hermite polynomials (JAX-safe)
        shapelet_y = hermite_phys(self.n_y, y_ell / self.beta, xp=xp)
        shapelet_x = hermite_phys(self.n_x, x_ell / self.beta, xp=xp)

        gaussian = xp.exp(-0.5 * (x_ell**2 + y_ell**2) / (self.beta**2))

        norm = self.beta * xp.sqrt(
            (2.0 ** (self.n_x + self.n_y))
            * xp.pi
            * factorial(self.n_y)
            * factorial(self.n_x)
        )

        return self._intensity * (shapelet_y * shapelet_x * gaussian) / norm


class ShapeletCartesianSph(ShapeletCartesian):
    def __init__(
        self,
        n_y: int,
        n_x: int,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 1.0,
        beta: float = 1.0,
    ):
        """
        Shapelets where the basis function is defined according to a Cartesian (y,x) grid of coordinates.

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
            n_y=n_y,
            n_x=n_x,
            centre=centre,
            ell_comps=(0.0, 0.0),
            beta=beta,
            intensity=intensity,
        )
