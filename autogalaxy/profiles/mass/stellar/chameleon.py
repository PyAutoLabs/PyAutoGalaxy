import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.profiles.mass.stellar.abstract import StellarProfile

from autogalaxy.profiles.mass.total.isothermal import psi_from


class Chameleon(MassProfile, StellarProfile):
    r"""
    Elliptical Chameleon stellar mass profile (Dutton et al. 2011).

    The Chameleon profile is the difference of two cored isothermal (pseudo-Jaffe) profiles
    with core radii :math:`s_0` and :math:`s_0 + s_1`, providing a flexible approximation
    to a variety of stellar light profiles:

    .. math::

        \kappa(\xi) = \Upsilon \, I \left(
            \frac{1}{\sqrt{\xi^2 + s_0^2}}
            - \frac{1}{\sqrt{\xi^2 + (s_0 + s_1)^2}}
        \right)

    where :math:`\xi^2 = x^2 + (y/q)^2` is the elliptical radius, :math:`\Upsilon` is
    the mass-to-light ratio (``mass_to_light_ratio``), :math:`I` is the intensity
    normalisation (``intensity``), :math:`s_0` = ``core_radius_0``, and
    :math:`s_1` = ``core_radius_1``.

    Deflection angles are computed analytically via the cored isothermal deflection
    formula (Eq. 15–16 of Dutton et al. 2011).

    References
    ----------
    - Dutton, Brewer, Marshall et al. 2011, MNRAS, 417, 1621
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.02,
        mass_to_light_ratio: float = 1.0,
    ):
        r"""
        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall intensity normalisation :math:`I` (electrons per second).
        core_radius_0
            Core radius :math:`s_0` of the first cored isothermal component (arcseconds).
        core_radius_1
            Additional core size increment :math:`s_1` such that the second component has
            core radius :math:`s_0 + s_1` (arcseconds).  Using an increment avoids
            negative parameter values.
        mass_to_light_ratio
            The mass-to-light ratio :math:`\Upsilon` in solar units.
        """

        super(Chameleon, self).__init__(centre=centre, ell_comps=ell_comps)
        super(MassProfile, self).__init__(centre=centre, ell_comps=ell_comps)
        self.mass_to_light_ratio = mass_to_light_ratio
        self.intensity = intensity
        self.core_radius_0 = core_radius_0
        self.core_radius_1 = core_radius_1

    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        return self.deflections_2d_via_analytic_from(grid=grid, xp=xp, **kwargs)

    @aa.decorators.to_vector_yx
    @aa.decorators.transform(rotate_back=True)
    def deflections_2d_via_analytic_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.
        Following Eq. (15) and (16), but the parameters are slightly different.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        factor = (
            2.0
            * self.mass_to_light_ratio
            * self.intensity
            / (1 + self.axis_ratio(xp))
            * self.axis_ratio(xp)
            / xp.sqrt(1.0 - self.axis_ratio(xp) ** 2.0)
        )

        core_radius_0 = xp.sqrt(
            (4.0 * self.core_radius_0**2.0) / (1.0 + self.axis_ratio(xp)) ** 2
        )
        core_radius_1 = xp.sqrt(
            (4.0 * self.core_radius_1**2.0) / (1.0 + self.axis_ratio(xp)) ** 2
        )

        psi0 = psi_from(
            grid=grid, axis_ratio=self.axis_ratio(xp), core_radius=core_radius_0, xp=xp
        )
        psi1 = psi_from(
            grid=grid, axis_ratio=self.axis_ratio(xp), core_radius=core_radius_1, xp=xp
        )

        deflection_y0 = xp.arctanh(
            xp.divide(
                xp.multiply(
                    xp.sqrt(1.0 - self.axis_ratio(xp) ** 2.0), grid.array[:, 0]
                ),
                xp.add(psi0, self.axis_ratio(xp) ** 2.0 * core_radius_0),
            )
        )

        deflection_x0 = xp.arctan(
            xp.divide(
                xp.multiply(
                    xp.sqrt(1.0 - self.axis_ratio(xp) ** 2.0), grid.array[:, 1]
                ),
                xp.add(psi0, core_radius_0),
            )
        )

        deflection_y1 = xp.arctanh(
            xp.divide(
                xp.multiply(
                    xp.sqrt(1.0 - self.axis_ratio(xp) ** 2.0), grid.array[:, 0]
                ),
                xp.add(psi1, self.axis_ratio(xp) ** 2.0 * core_radius_1),
            )
        )

        deflection_x1 = xp.arctan(
            xp.divide(
                xp.multiply(
                    xp.sqrt(1.0 - self.axis_ratio(xp) ** 2.0), grid.array[:, 1]
                ),
                xp.add(psi1, core_radius_1),
            )
        )

        deflection_y = xp.subtract(deflection_y0, deflection_y1)
        deflection_x = xp.subtract(deflection_x0, deflection_x1)

        return xp.multiply(factor, xp.vstack((deflection_y, deflection_x)).T)

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """Calculate the projected convergence at a given set of arc-second gridded coordinates.
        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.
        """
        return self.convergence_func(
            self.elliptical_radii_grid_from(grid=grid, xp=xp, **kwargs), xp=xp
        )

    def convergence_func(self, grid_radius: float, xp=np) -> float:
        return self.mass_to_light_ratio * self.image_2d_via_radii_from(
            grid_radius, xp=xp
        )

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        from autogalaxy.profiles.mass.abstract.mge import MGEDecomposer

        radii_min = self.core_radius_0 / 10.0
        radii_max = self.core_radius_1 * 200.0
        sigmas = xp.exp(xp.linspace(xp.log(radii_min), xp.log(radii_max), 30))
        mge_decomp = MGEDecomposer(mass_profile=self)
        return mge_decomp.potential_2d_via_mge_from(
            grid=grid, xp=xp, sigma_log_list=sigmas,
            ellipticity_convention="circularised", three_D=False,
        )

    def image_2d_via_radii_from(self, grid_radii: np.ndarray, xp=np):
        """Calculate the intensity of the Chamelon light profile on a grid of radial coordinates.

        Parameters
        ----------
        grid_radii
            The radial distance from the centre of the profile. for each coordinate on the grid.
        """

        axis_ratio_factor = (1.0 + self.axis_ratio(xp)) ** 2.0

        return xp.multiply(
            self.intensity / (1 + self.axis_ratio(xp)),
            xp.add(
                xp.divide(
                    1.0,
                    xp.sqrt(
                        xp.add(
                            xp.square(grid_radii.array),
                            (4.0 * self.core_radius_0**2.0) / axis_ratio_factor,
                        )
                    ),
                ),
                -xp.divide(
                    1.0,
                    xp.sqrt(
                        xp.add(
                            xp.square(grid_radii.array),
                            (4.0 * self.core_radius_1**2.0) / axis_ratio_factor,
                        )
                    ),
                ),
            ),
        )

    def axis_ratio(self, xp=np):
        axis_ratio = super().axis_ratio(xp=xp)
        return axis_ratio if axis_ratio < 0.99999 else 0.99999


class ChameleonSph(Chameleon):
    r"""
    Spherical Chameleon stellar mass profile.

    A special case of :class:`Chameleon` with no ellipticity (:math:`q = 1`).
    The convergence is:

    .. math::

        \kappa(r) = \Upsilon \, I \left(
            \frac{1}{\sqrt{r^2 + s_0^2}}
            - \frac{1}{\sqrt{r^2 + (s_0 + s_1)^2}}
        \right)

    References
    ----------
    - Dutton, Brewer, Marshall et al. 2011, MNRAS, 417, 1621
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.02,
        mass_to_light_ratio: float = 1.0,
    ):
        r"""
        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall intensity normalisation :math:`I` (electrons per second).
        core_radius_0
            Core radius :math:`s_0` of the first cored isothermal component (arcseconds).
        core_radius_1
            Additional core size increment :math:`s_1` such that the second component has
            core radius :math:`s_0 + s_1` (arcseconds).
        mass_to_light_ratio
            The mass-to-light ratio :math:`\Upsilon` in solar units.
        """

        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            core_radius_0=core_radius_0,
            core_radius_1=core_radius_1,
            mass_to_light_ratio=mass_to_light_ratio,
        )
