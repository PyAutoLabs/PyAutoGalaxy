"""
Elliptical Gaussian light profile with m=3 and m=4 Fourier multipole perturbations
applied to the eccentric radius before evaluating the Gaussian profile.

With both ``multipole_*_comps`` set to ``(0.0, 0.0)`` (the default), the profile
reduces exactly to ``Gaussian``.
"""

from typing import Optional, Tuple

import numpy as np

import autoarray as aa

from autogalaxy.profiles.light.decorators import check_operated_only
from autogalaxy.profiles.light.standard._multipole_mixin import (
    _LightProfileMultipoleMixin,
)
from autogalaxy.profiles.light.standard.gaussian import Gaussian


class GaussianMultipole(_LightProfileMultipoleMixin, Gaussian):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        sigma: float = 1.0,
        multipole_3_comps: Tuple[float, float] = (0.0, 0.0),
        multipole_4_comps: Tuple[float, float] = (0.0, 0.0),
    ):
        """
        The elliptical Gaussian light profile with m=3 and m=4 Fourier multipole
        perturbations on the eccentric radius.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate
            system. The multipole perturbation is applied to the eccentric radius and
            therefore follows this ellipticity.
        intensity
            Overall intensity normalisation of the light profile.
        sigma
            The sigma value of the Gaussian.
        multipole_3_comps
            The ``(cos, sin)`` components of the m=3 Fourier perturbation. Defaults to
            ``(0.0, 0.0)`` which reduces the profile to ``Gaussian``.
        multipole_4_comps
            The ``(cos, sin)`` components of the m=4 Fourier perturbation. Defaults to
            ``(0.0, 0.0)`` which reduces the profile to ``Gaussian``.
        """
        super().__init__(
            centre=centre, ell_comps=ell_comps, intensity=intensity, sigma=sigma
        )
        self.multipole_3_comps = multipole_3_comps
        self.multipole_4_comps = multipole_4_comps

    def image_2d_via_radii_from(
        self, grid_radii: np.ndarray, xp=np, **kwargs
    ) -> np.ndarray:
        """
        Returns the 2D Gaussian image evaluated at the input radial values.

        Unlike ``Gaussian.image_2d_via_radii_from``, this override accepts a raw backend
        array (the output of ``perturbed_radii_from``) rather than an autoarray-wrapped
        grid, since the perturbation step strips the wrapper.
        """
        return xp.multiply(
            self._intensity,
            xp.exp(
                -0.5
                * xp.square(
                    xp.divide(grid_radii, self.sigma / xp.sqrt(self.axis_ratio(xp)))
                )
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
    ) -> aa.Array2D:
        """
        Returns the 2D image of the multipole-perturbed Gaussian profile.
        """
        perturbed_radii = self.perturbed_radii_from(grid=grid, xp=xp, **kwargs)
        return self.image_2d_via_radii_from(grid_radii=perturbed_radii, xp=xp, **kwargs)
