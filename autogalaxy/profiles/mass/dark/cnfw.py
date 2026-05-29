import numpy as np

from typing import Tuple

from autogalaxy.profiles.mass.dark.abstract import AbstractgNFW
from autogalaxy.profiles.mass.abstract.mge import MGEDecomposer

import autoarray as aa


def F_func_from(theta, radius, xp=np):
    F = theta * 0.0

    mask1 = (theta > 0) & (theta <= radius)
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


def dev_F_func_from(theta, radius, xp=np):
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


class cNFW(AbstractgNFW):
    r"""
    Elliptical cored NFW (cNFW) dark matter halo profile.

    The three-dimensional density profile introduces a constant-density core of
    radius :math:`r_c` that suppresses the central cusp of the standard NFW profile:

    .. math::

        \rho(r) = \frac{\rho_0 \, r_s^3}{(r + r_c)(r + r_s)^2}

    where :math:`r_c` is the core radius (``core_radius``) and :math:`r_s` is the
    scale radius (``scale_radius``).  In the limit :math:`r_c \to 0` the profile
    approaches the standard NFW density.

    The convergence and deflection angles are computed via a Multi-Gaussian
    Expansion (MGE) decomposition following Shajib (2019).

    References
    ----------
    - Read, Agertz & Collins 2016, MNRAS, 459, 2573
    - Shajib 2019, MNRAS, 488, 1387  (arXiv:1906.08263)
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        kappa_s: float = 0.05,
        scale_radius: float = 1.0,
        core_radius: float = 0.01,
    ):
        r"""
        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        kappa_s
            The overall normalization of the dark matter halo
            (:math:`\kappa_s = \rho_0 D_d r_s / \Sigma_{\rm crit}`).
        scale_radius
            The cored NFW scale radius :math:`r_s`, as an angle on the sky in arcseconds.
        core_radius
            The cored NFW core radius :math:`r_c`, as an angle on the sky in arcseconds.
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
            ellipticity_convention="major",
            three_D=True,
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

    def convergence_func(self, grid_radius, xp=np):
        """
        Radial projected convergence kappa(r), reusing the MGE-of-3D-density decomposition
        (the same machinery `convergence_2d_from` uses with `three_D=True`) evaluated at the
        radial coordinate `grid_radius`.

        cNFW has no closed-form radial convergence helper, so this delegates to the MGE
        Gaussian sum. This hook is reached by `MGEDecomposer.decompose_convergence_via_mge`
        (`three_D=False`, not used by cNFW) and by radial mass integration (`mass_integral`
        -> `mass_angular_within_circle_from` -> Einstein radius).

        The result is the q-independent radial profile (like `NFW.convergence_func`):
        ellipticity is re-introduced by the MGE machinery elsewhere, so no `sigmas_factor`
        rescale is applied (`sigmas_factor=1.0`). Verified to match `convergence_2d_from` for
        the spherical case and to be q-independent for the elliptical case.
        """
        radii = (
            grid_radius.array
            if hasattr(grid_radius, "array")
            else xp.asarray(grid_radius)
        )
        # Track scalar input so we return a scalar (matching other `convergence_func`
        # implementations). `mass_integral` -> scipy.quad feeds scalar radii and a length-1
        # array return would warn (and eventually error) on the array->scalar conversion.
        scalar_input = xp.ndim(radii) == 0
        radii = xp.atleast_1d(radii)
        radii_min = self.scale_radius / 1000.0
        radii_max = self.scale_radius * 200.0
        sigmas = xp.exp(xp.linspace(xp.log(radii_min), xp.log(radii_max), 20))
        mge_decomp = MGEDecomposer(mass_profile=self)
        convergence = mge_decomp._convergence_2d_via_mge_from(
            grid_radii=radii,
            xp=xp,
            sigma_log_list=sigmas,
            three_D=True,
            sigmas_factor=1.0,
        )
        return convergence[0] if scalar_input else convergence

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        radii_min = self.scale_radius / 1000.0
        radii_max = self.scale_radius * 200.0
        sigmas = xp.exp(xp.linspace(xp.log(radii_min), xp.log(radii_max), 20))
        mge_decomp = MGEDecomposer(mass_profile=self)
        return mge_decomp.convergence_2d_via_mge_from(
            grid=grid,
            xp=xp,
            sigma_log_list=sigmas,
            ellipticity_convention="major",
            three_D=True,
        )

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        radii_min = self.scale_radius / 1000.0
        radii_max = self.scale_radius * 200.0
        sigmas = xp.exp(xp.linspace(xp.log(radii_min), xp.log(radii_max), 20))
        mge_decomp = MGEDecomposer(mass_profile=self)
        return mge_decomp.potential_2d_via_mge_from(
            grid=grid,
            xp=xp,
            sigma_log_list=sigmas,
            ellipticity_convention="major",
            three_D=True,
        )


class cNFWSph(cNFW):
    r"""
    Spherical cored NFW (cNFW) dark matter halo profile.

    A special case of :class:`cNFW` with no ellipticity (:math:`q = 1`).  The 3-D
    density and projected convergence follow the same cored-NFW expressions with
    an analytic deflection-angle formula available for the spherical case:

    .. math::

        \rho(r) = \frac{\rho_0 \, r_s^3}{(r + r_c)(r + r_s)^2}

    References
    ----------
    - Read, Agertz & Collins 2016, MNRAS, 459, 2573
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        kappa_s: float = 0.05,
        scale_radius: float = 1.0,
        core_radius: float = 0.01,
    ):
        r"""
        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        kappa_s
            The overall normalization of the dark matter halo
            (:math:`\kappa_s = \rho_0 D_d r_s / \Sigma_{\rm crit}`).
        scale_radius
            The cored NFW scale radius :math:`r_s`, as an angle on the sky in arcseconds.
        core_radius
            The cored NFW core radius :math:`r_c`, as an angle on the sky in arcseconds.
        """

        super().__init__(centre=centre, ell_comps=(0.0, 0.0))

        self.kappa_s = kappa_s
        self.scale_radius = scale_radius
        self.core_radius = core_radius

    @aa.decorators.to_vector_yx
    @aa.decorators.transform
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
        return F_func_from(theta, radius, xp=xp)

    def dev_F_func(self, theta, radius, xp=np):
        return dev_F_func_from(theta, radius, xp=xp)

    @staticmethod
    def radial_deflection_from(r, params, xp):
        kappa_s, scale_radius, core_radius = params[0], params[1], params[2]
        theta = xp.maximum(r, 1e-8)
        factor = 4.0 * kappa_s * scale_radius**2
        return (
            factor
            * (
                F_func_from(theta, scale_radius, xp=xp)
                - F_func_from(theta, core_radius, xp=xp)
                - (scale_radius - core_radius)
                * dev_F_func_from(theta, scale_radius, xp=xp)
            )
            / (theta * (scale_radius - core_radius) ** 2)
        )

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        radii_min = self.scale_radius / 1000.0
        radii_max = self.scale_radius * 200.0
        sigmas = xp.exp(xp.linspace(xp.log(radii_min), xp.log(radii_max), 20))
        mge_decomp = MGEDecomposer(mass_profile=self)
        return mge_decomp.convergence_2d_via_mge_from(
            grid=grid,
            xp=xp,
            sigma_log_list=sigmas,
            ellipticity_convention="major",
            three_D=True,
        )

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        radii_min = self.scale_radius / 1000.0
        radii_max = self.scale_radius * 200.0
        sigmas = xp.exp(xp.linspace(xp.log(radii_min), xp.log(radii_max), 20))
        mge_decomp = MGEDecomposer(mass_profile=self)
        return mge_decomp.potential_2d_via_mge_from(
            grid=grid,
            xp=xp,
            sigma_log_list=sigmas,
            ellipticity_convention="major",
            three_D=True,
        )
