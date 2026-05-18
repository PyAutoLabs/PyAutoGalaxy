"""
Multipole light profiles.

These profiles extend the standard Sersic and Gaussian light profiles by perturbing the
eccentric radius with a Fourier series at orders m=3 and m=4 before evaluating the
underlying radial profile:

    r_perturbed = r * (1 + c_3 cos(3 theta) + s_3 sin(3 theta)
                          + c_4 cos(4 theta) + s_4 sin(4 theta))

where ``theta`` is the polar angle in the profile's elliptical reference frame. With all
multipole components set to zero (the default), the profile reduces exactly to its base
class (``Sersic`` or ``Gaussian``).

The (cos, sin) components are passed as a 2-tuple ``multipole_<m>_comps = (c, s)``,
matching the parameter convention used by ``EllipseMultipole`` and ``PowerLawMultipole``.

Note: this is the "elliptical multipole" flavour — the perturbation rides on the
already-elliptical ``grid_radii``, so the deformation follows the profile's ellipticity.
This is distinct from the polar-frame perturbation in ``EllipseMultipole``.
"""

from numpy import seterr
from typing import Optional, Tuple

import numpy as np

import autoarray as aa

from autogalaxy.profiles.light.decorators import check_operated_only
from autogalaxy.profiles.light.standard.gaussian import Gaussian
from autogalaxy.profiles.light.standard.sersic import Sersic


class _LightProfileMultipoleMixin:
    """
    Shared `perturbed_radii_from` helper for light profiles with m=3 and m=4 Fourier
    perturbations applied to the eccentric radius.

    Concrete subclasses must set ``multipole_3_comps`` and ``multipole_4_comps`` on
    ``self`` and inherit from a profile that provides ``eccentric_radii_grid_from``.
    """

    multipole_3_comps: Tuple[float, float]
    multipole_4_comps: Tuple[float, float]

    def perturbed_radii_from(
        self,
        grid: aa.type.Grid2DLike,
        xp=np,
        **kwargs,
    ) -> np.ndarray:
        """
        Returns the eccentric radii of ``grid`` perturbed by the m=3 and m=4 Fourier
        multipoles, as a raw backend array (numpy or jax.numpy).

        The result is floored at 1e-8 to keep the Sersic ``r ** (1/n)`` evaluation
        finite when a large multipole component would otherwise drive the radius
        through zero.
        """
        grid_radii = self.eccentric_radii_grid_from(grid=grid, xp=xp, **kwargs)
        grid_radii_array = (
            grid_radii.array if hasattr(grid_radii, "array") else grid_radii
        )
        y = grid.array[:, 0]
        x = grid.array[:, 1]
        theta = xp.arctan2(y, x)
        c3, s3 = self.multipole_3_comps
        c4, s4 = self.multipole_4_comps
        perturbation = (
            1.0
            + c3 * xp.cos(3.0 * theta)
            + s3 * xp.sin(3.0 * theta)
            + c4 * xp.cos(4.0 * theta)
            + s4 * xp.sin(4.0 * theta)
        )
        return xp.maximum(xp.multiply(grid_radii_array, perturbation), 1e-8)


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
