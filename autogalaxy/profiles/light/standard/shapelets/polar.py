import numpy as np
from typing import Optional, Tuple

import autoarray as aa

from autogalaxy.profiles.light.decorators import (
    check_operated_only,
)
from autogalaxy import convert
from autogalaxy.profiles.light.standard.shapelets.abstract import AbstractShapelet


def genlaguerre_jax(n, alpha, x):
    """
    Generalized (associated) Laguerre polynomial L_n^alpha(x)
    calculated using the explicit summation formula, optimized for JAX vectorization.

    Parameters:
        n (int): Degree of the polynomial (static Python integer).
        alpha (Numeric): Parameter alpha > -1.
        x (Array): Input array (evaluation points).
    """
    import jax.numpy as jnp
    from jax.scipy.special import gammaln

    # 0. Input Validation (Requires static Python int n)
    if not isinstance(n, int) or n < 0:
        # Use Python's math.isnan/isinf check if n is float, otherwise type error
        raise ValueError(
            f"Degree n must be a non-negative Python integer (static), got {n}."
        )

    # Base Case L0
    if n == 0:
        return jnp.ones_like(x)

    # 1. Generate k values for summation range [0, 1, 2, ..., n]
    k_values = jnp.arange(n + 1)  # (n+1,)

    # 2. Reshape inputs for broadcasting (x: (M, 1), k: (1, n+1))
    x_expanded = jnp.expand_dims(x, axis=-1)
    k_values_expanded = jnp.expand_dims(k_values, axis=0)

    # --- A. Binomial Factor (BF) Calculation ---
    # BF = exp( log( (n+alpha)! / ((n-k)! * (alpha+k)!) ) )

    log_N_plus_alpha_fact = gammaln(n + alpha + 1)

    log_BF_k = (
        log_N_plus_alpha_fact
        - gammaln(n - k_values + 1)  # log( (n-k)! )
        - gammaln(alpha + k_values + 1)  # log( (alpha+k)! )
    )

    BF_k = jnp.exp(log_BF_k)  # Shape: (n+1,)

    # --- B. Term Factor (TF) Calculation ---
    # TF = (-x)^k / k!

    # Note: jnp.math.gamma(k_values + 1) is equivalent to k! in log-gamma space
    TF_k = jnp.power(-x_expanded, k_values_expanded) / jnp.exp(
        gammaln(k_values_expanded + 1)
    )
    # TF_k Shape: (M, n+1)

    # --- C. Final Summation ---
    # Sum over the last axis (axis=1), which corresponds to k
    # BF_k broadcasts over the M dimension of TF_k
    return jnp.sum(BF_k * TF_k, axis=1)


class ShapeletPolar(AbstractShapelet):
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

        self.n = int(n)
        self.m = int(m)

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
        Returns the Polar Shapelet light profile's 2D image from a 2D grid of Polar (y,x) coordinates.

        If the coordinates have not been transformed to the profile's geometry (e.g. translated to the
        profile `centre`), this is performed automatically.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the Polar Shapelet evaluated at every (y,x) coordinate on the transformed grid.
        """
        if xp is np:

            from scipy.special import factorial

        else:

            from jax.scipy.special import factorial

        const = (
            ((-1) ** ((self.n - xp.abs(self.m)) // 2))
            * xp.sqrt(
                factorial((self.n - xp.abs(self.m)) // 2)
                / factorial((self.n + xp.abs(self.m)) // 2)
            )
            / self.beta
            / xp.sqrt(xp.pi)
        )
        y = grid.array[:, 0]
        x = grid.array[:, 1]

        rsq = (x**2 + (y / self.axis_ratio(xp)) ** 2) / self.beta**2
        theta = xp.arctan2(y, x)

        m_abs = abs(self.m)
        n_laguerre = (self.n - m_abs) // 2

        if xp is np:

            from scipy.special import genlaguerre

            laguerre = genlaguerre(
                n=(self.n - xp.abs(self.m)) / 2.0, alpha=xp.abs(self.m)
            )
            laguerre_vals = laguerre(rsq)

        else:

            laguerre_vals = genlaguerre_jax(n=n_laguerre, alpha=m_abs, x=rsq)

        radial = rsq ** (xp.abs(self.m) / 2.0) * xp.exp(-rsq / 2.0) * laguerre_vals

        m = self.m

        azimuthal = xp.where(
            m == 0,
            xp.ones_like(theta),
            xp.where(
                m > 0,
                xp.sin(-m * theta),
                xp.cos(-m * theta),
            ),
        )

        return self._intensity * const * radial * azimuthal


class ShapeletPolarSph(ShapeletPolar):
    def __init__(
        self,
        n: int,
        m: int,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 1.0,
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
            intensity=intensity,
            beta=beta,
        )
