import numpy as np

from typing import Tuple

from autogalaxy.profiles.mass.dark.abstract import AbstractgNFW
from autogalaxy.profiles.mass.abstract.mge import MGEDecomposer

import autoarray as aa


class cNFW(AbstractgNFW):
    def __init__(
            self,
            centre: Tuple[float, float] = (0.0, 0.0),
            ell_comps: Tuple[float, float] = (0.0, 0.0),
            kappa_s: float = 0.05,
            scale_radius: float = 1.0,
            core_radius: float = 0.01,
    ):
        """
        Represents a cored NFW density distribution

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        kappa_s
            The overall normalization of the dark matter halo
            (kappa_s = (rho_0 * D_d * scale_radius)/lensing_critical_density)
        scale_radius
            The cored NFW scale radius `theta_s`, as an angle on the sky in arcseconds.
        core_radius
            The cored NFW core radius `theta_c`, as an angle on the sky in arcseconds.
        """

        super().__init__(centre=centre, ell_comps=ell_comps)

        self.kappa_s = kappa_s
        self.scale_radius = scale_radius
        self.core_radius = core_radius


    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        return self.deflections_2d_via_mge_from(grid=grid, xp=xp, **kwargs)

    def deflections_2d_via_mge_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        radii_min = self.scale_radius / 1000.0
        radii_max = self.scale_radius * 200.0
        log_sigmas = xp.linspace(xp.log(radii_min), xp.log(radii_max), 20)
        sigmas = xp.exp(log_sigmas)

        mge_decomp = MGEDecomposer(mass_profile=self)

        deflections_via_mge = mge_decomp.deflections_2d_via_mge_from(
            grid=grid,
            xp=xp,
            sigma_log_list=sigmas,
            ellipticity_convention='major',
            three_D=True
        )
        return deflections_via_mge

    def density_3d_func(self, r, xp=np):

        rho_at_scale_radius = (
            self.kappa_s / self.scale_radius
        )  # density parameter of 3D gNFW

        return (
            rho_at_scale_radius
            * self.scale_radius**3.0
            * (r.array + self.core_radius) ** (-1.0)
            * (r.array + self.scale_radius) ** (-2.0)
        )


class cNFWSph(cNFW):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        kappa_s: float = 0.05,
        scale_radius: float = 1.0,
        core_radius: float = 0.01,
    ):
        """
        Represents a spherical cored NFW density distribution

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        kappa_s
            The overall normalization of the dark matter halo
            (kappa_s = (rho_0 * D_d * scale_radius)/lensing_critical_density)
        scale_radius
            The cored NFW scale radius `theta_s`, as an angle on the sky in arcseconds.
        core_radius
            The cored NFW core radius `theta_c`, as an angle on the sky in arcseconds.
        """

        super().__init__(centre=centre, ell_comps=(0.0, 0.0))

        self.kappa_s = kappa_s
        self.scale_radius = scale_radius
        self.core_radius = core_radius

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        The input grid of (y,x) coordinates are transformed to a coordinate system centred on the profile centre with
        and rotated based on the position angle defined from its `ell_comps` (this is described fully below).

        The numerical backend can be selected via the ``xp`` argument, allowing this
        method to be used with both NumPy and JAX (e.g. inside ``jax.jit``-compiled
        code). This is described fully later in this example.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        xp
            The numerical backend to use, either `numpy` or `jax.numpy`.
        """
        theta = self.radial_grid_from(grid=grid, xp=xp, **kwargs).array
        theta = xp.maximum(theta, 1e-8)

        factor = 4.0 * self.kappa_s * self.scale_radius**2

        deflection_r = (
            factor
            * (
                self.F_func(theta, self.scale_radius, xp=xp)
                - self.F_func(theta, self.core_radius, xp=xp)
                - (self.scale_radius - self.core_radius)
                * self.dev_F_func(theta, self.scale_radius, xp=xp)
            )
            / (theta * (self.scale_radius - self.core_radius) ** 2)
        )

        return self._cartesian_grid_via_radial_from(
            grid=grid,
            radius=deflection_r,
            xp=xp,
            **kwargs,
        )

    def F_func(self, theta, radius, xp=np):

        F = theta * 0.0

        # theta < radius
        mask1 = (theta > 0) & (theta < radius)

        # theta > radius
        mask2 = theta > radius

        F = xp.where(
            mask1,
            (
                radius / 2 * xp.log(2 * radius / theta)
                - xp.sqrt(radius**2 - theta**2)
                * xp.arctanh(xp.sqrt((radius - theta) / (radius + theta)))
            ),
            F,
        )

        F = xp.where(
            mask2,
            (
                radius / 2 * xp.log(2 * radius / theta)
                + xp.sqrt(theta**2 - radius**2)
                * xp.arctan(xp.sqrt((theta - radius) / (theta + radius)))
            ),
            F,
        )

        return 2 * radius * F

    def dev_F_func(self, theta, radius, xp=np):

        dev_F = theta * 0.0

        mask1 = (theta > 0) & (theta < radius)
        mask2 = theta == radius
        mask3 = theta > radius

        dev_F = xp.where(
            mask1,
            (
                radius * xp.log(2 * radius / theta)
                - (2 * radius**2 - theta**2)
                / xp.sqrt(radius**2 - theta**2)
                * xp.arctanh(xp.sqrt((radius - theta) / (radius + theta)))
            ),
            dev_F,
        )

        dev_F = xp.where(
            mask2,
            radius * (xp.log(2) - 1 / 2),
            dev_F,
        )

        dev_F = xp.where(
            mask3,
            (
                radius * xp.log(2 * radius / theta)
                + (theta**2 - 2 * radius**2)
                / xp.sqrt(theta**2 - radius**2)
                * xp.arctan(xp.sqrt((theta - radius) / (theta + radius)))
            ),
            dev_F,
        )

        return 2 * dev_F

    @aa.grid_dec.to_array
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Convergence (dimensionless surface mass density) for the cored NFW profile.
        This is not yet implemented for `cNFWSph`.
        """
        return xp.zeros(shape=grid.shape[0])

    @aa.grid_dec.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Lensing potential for the cored NFW profile.
        This is not yet implemented for `cNFWSph`.
        """
        return xp.zeros(shape=grid.shape[0])
