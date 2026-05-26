import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.dark.abstract import AbstractgNFW
from autogalaxy.profiles.mass import MGEDecomposer


class gNFW(AbstractgNFW):
    r"""
    Elliptical generalised NFW (gNFW) dark matter halo profile with a free inner slope.

    The three-dimensional density profile is:

    .. math::

        \rho(r) = \frac{\rho_s}{(r/r_s)^{\gamma}(1 + r/r_s)^{3-\gamma}}

    where :math:`\gamma` is the inner logarithmic slope (``inner_slope``).  For
    :math:`\gamma = 1` this reduces to the standard :class:`NFW` profile.

    The projected convergence is computed by numerical line-of-sight integration:

    .. math::

        \kappa(\xi) = 2 \kappa_s \, (\xi/r_s)^{1-\gamma}
        \left[
            (1 + \xi/r_s)^{\gamma-3}
            + (3 - \gamma) \int_0^1 \frac{(y + \xi/r_s)^{\gamma-4}
            \left(1 - \sqrt{1 - y^2}\right)}{1} \, \mathrm{d}y
        \right]

    Deflection angles are computed via a Multi-Gaussian Expansion (MGE) decomposition
    following Shajib (2019).

    References
    ----------
    - Wyithe, Turner & Spergel 2001, ApJ, 555, 504
    - Shajib 2019, MNRAS, 488, 1387  (arXiv:1906.08263)
    """

    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        return self.deflections_2d_via_mge_from(grid=grid, xp=xp, **kwargs)

    def deflections_2d_via_mge_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        radii_min = self.scale_radius / 20000.0
        radii_max = self.scale_radius * 200.0
        log_sigmas = xp.linspace(xp.log(radii_min), xp.log(radii_max), 30)
        sigmas = xp.exp(log_sigmas)

        mge_decomp = MGEDecomposer(mass_profile=self)

        deflections_via_mge = mge_decomp.deflections_2d_via_mge_from(
            grid=grid,
            xp=xp,
            sigma_log_list=sigmas,
            ellipticity_convention="major",
            three_D=True,
        )
        return deflections_via_mge

    def convergence_func(self, grid_radius: float, xp=np) -> float:

        from scipy.integrate import quad

        def integral_y(y, eta):
            return (y + eta) ** (self.inner_slope - 4) * (1 - xp.sqrt(1 - y**2))

        grid_radius = xp.array((1.0 / self.scale_radius) * grid_radius.array)

        for index in range(grid_radius.shape[0]):
            integral_y_value = quad(
                integral_y,
                a=0.0,
                b=1.0,
                args=grid_radius[index],
                epsrel=gNFW.epsrel,
            )[0]

            grid_radius[index] = (
                2.0
                * self.kappa_s
                * (grid_radius[index] ** (1 - self.inner_slope))
                * (
                    (1 + grid_radius[index]) ** (self.inner_slope - 3)
                    + ((3 - self.inner_slope) * integral_y_value)
                )
            )

        return grid_radius


class gNFWSph(gNFW):
    r"""
    Spherical generalised NFW (gNFW) dark matter halo profile with a free inner slope.

    A special case of :class:`gNFW` with no ellipticity (:math:`q = 1`).  The 3-D
    density and projected convergence follow the same gNFW expressions as the
    elliptical variant but evaluated on a circular radial grid.

    .. math::

        \rho(r) = \frac{\rho_s}{(r/r_s)^{\gamma}(1 + r/r_s)^{3-\gamma}}

    References
    ----------
    - Wyithe, Turner & Spergel 2001, ApJ, 555, 504
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        kappa_s: float = 0.05,
        inner_slope: float = 1.0,
        scale_radius: float = 1.0,
    ):
        r"""
        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        kappa_s
            The overall normalization of the dark matter halo
            (:math:`\kappa_s = \rho_s r_s / \Sigma_{\rm crit}`).
        inner_slope
            The inner logarithmic slope :math:`\gamma` of the dark matter density profile.
        scale_radius
            The NFW scale radius :math:`r_s`, as an angle on the sky in arcseconds.
        """

        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            kappa_s=kappa_s,
            inner_slope=inner_slope,
            scale_radius=scale_radius,
        )

    pass
