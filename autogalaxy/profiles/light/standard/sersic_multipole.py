"""
Elliptical Sersic light profile with m=3 and m=4 Fourier multipole perturbations
applied to the eccentric radius before evaluating the Sersic profile.

With both ``multipole_*_comps`` set to ``(0.0, 0.0)`` (the default), the profile
reduces exactly to ``Sersic``.
"""

from numpy import seterr
from typing import Optional, Tuple

import numpy as np

import autoarray as aa

from autogalaxy.profiles.light.decorators import check_operated_only
from autogalaxy.profiles.light.standard._multipole_mixin import (
    _LightProfileMultipoleMixin,
)
from autogalaxy.profiles.light.standard.sersic import Sersic


class SersicMultipole(_LightProfileMultipoleMixin, Sersic):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
        multipole_3_comps: Tuple[float, float] = (0.0, 0.0),
        multipole_4_comps: Tuple[float, float] = (0.0, 0.0),
    ):
        """
        The elliptical Sersic light profile with m=3 and m=4 Fourier multipole
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
        effective_radius
            The circular radius containing half the light of this profile (in the
            unperturbed limit ``multipole_*_comps = (0, 0)``).
        sersic_index
            Controls the concentration of the profile.
        multipole_3_comps
            The ``(cos, sin)`` components of the m=3 Fourier perturbation. Defaults to
            ``(0.0, 0.0)`` which reduces the profile to ``Sersic``.
        multipole_4_comps
            The ``(cos, sin)`` components of the m=4 Fourier perturbation. Defaults to
            ``(0.0, 0.0)`` which reduces the profile to ``Sersic``.
        """
        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )
        self.multipole_3_comps = multipole_3_comps
        self.multipole_4_comps = multipole_4_comps

    def image_2d_via_radii_from(
        self, grid_radii: np.ndarray, xp=np, **kwargs
    ) -> np.ndarray:
        """
        Returns the 2D Sersic image evaluated at the input radial values.

        Unlike ``Sersic.image_2d_via_radii_from``, this override accepts a raw backend
        array (the output of ``perturbed_radii_from``) rather than an autoarray-wrapped
        grid, since the perturbation step strips the wrapper.
        """
        seterr(all="ignore")
        return xp.multiply(
            self._intensity,
            xp.exp(
                xp.multiply(
                    -self.sersic_constant,
                    xp.add(
                        xp.power(
                            xp.divide(grid_radii, self.effective_radius),
                            1.0 / self.sersic_index,
                        ),
                        -1,
                    ),
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
        Returns the 2D image of the multipole-perturbed Sersic profile.
        """
        perturbed_radii = self.perturbed_radii_from(grid=grid, xp=xp, **kwargs)
        return self.image_2d_via_radii_from(grid_radii=perturbed_radii, xp=xp, **kwargs)
