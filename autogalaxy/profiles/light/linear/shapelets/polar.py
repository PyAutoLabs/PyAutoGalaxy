from typing import Tuple

from autogalaxy.profiles.light import standard as lp

from autogalaxy.profiles.light.linear.abstract import LightProfileLinear


class ShapeletPolar(lp.ShapeletPolar, LightProfileLinear):
    def __init__(
        self,
        n: int,
        m: int,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        beta: float = 1.0,
    ):
        """
        Shapelets where the basis function is defined according to a Polar (r,theta) grid of coordinates.

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
        q
            The axis-ratio of the elliptical coordinate system, where a perfect circle has q=1.0.
        phi
            The position angle (in degrees) of the elliptical coordinate system, measured counter-clockwise from the
            positive x-axis.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        beta
            The characteristic length scale of the shapelet basis function, defined in arc-seconds.
        """

        super().__init__(
            n=n, m=m, centre=centre, ell_comps=ell_comps, beta=beta, intensity=1.0
        )


class ShapeletPolarSph(ShapeletPolar):
    def __init__(
        self,
        n: int,
        m: int,
        centre: Tuple[float, float] = (0.0, 0.0),
        phi: float = 0.0,
        beta: float = 1.0,
    ):
        """
        Shapelets where the basis function is defined according to a Polar (r,theta) grid of coordinates.

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
        phi
            The position angle (in degrees) of the elliptical coordinate system, measured counter-clockwise from the
            positive x-axis.
        beta
            The characteristic length scale of the shapelet basis function, defined in arc-seconds.
        """

        super().__init__(n=n, m=m, centre=centre, beta=beta)
