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
        """
        Returns the deflection angles via the MGE decomposition of the 3D density, plus
        the central point-like term for the mass inside the smallest Gaussian (see
        :meth:`_inner_mass_deficit_from`).
        """
        mge_decomp = MGEDecomposer(mass_profile=self)

        deflections_via_mge = mge_decomp.deflections_2d_via_mge_from(
            grid=grid,
            xp=xp,
            sigma_log_list=self._mge_sigmas_from(xp=xp),
            ellipticity_convention="major",
            three_D=True,
        )

        return deflections_via_mge + self._deflections_2d_inner_mass_from(
            grid=grid, xp=xp
        )

    def _mge_sigmas_from(self, xp=np):
        """
        The widths of the 30 log-spaced Gaussians of the MGE decomposition of the 3D
        density, from ``scale_radius / 20000`` to ``200 * scale_radius``.
        """
        radii_min = self.scale_radius / 20000.0
        radii_max = self.scale_radius * 200.0
        return xp.exp(xp.linspace(xp.log(radii_min), xp.log(radii_max), 30))

    def _enclosed_mass_3d_from(self, radius, xp=np):
        r"""
        Returns the exact 3D mass inside a sphere of ``radius``, in the convergence units
        of the projected mass (:math:`\int \kappa \, dA`):

        .. math::

            M(<r) = 4 \pi \kappa_s r_s^2 F(x), \qquad
            F(x) = \int_0^x t^{2-\gamma} (1+t)^{\gamma-3} \, dt, \qquad x = r / r_s.

        The substitution :math:`t = x s^{1/(3-\gamma)}` gives

        .. math::

            F(x) = \frac{x^{3-\gamma}}{3-\gamma}
                   \int_0^1 \left(1 + x s^{1/(3-\gamma)}\right)^{\gamma-3} \, ds,

        whose integrand is smooth (the cusp is absorbed), so a fixed 32-node
        Gauss-Legendre rule is accurate to double precision for any
        :math:`\gamma < 3`, on both array backends.
        """
        nodes, weights = np.polynomial.legendre.leggauss(32)
        nodes = xp.asarray(0.5 * (nodes + 1.0))
        weights = xp.asarray(0.5 * weights)

        exponent = 3.0 - self.inner_slope
        x = radius / self.scale_radius

        f_x = (
            x**exponent
            / exponent
            * xp.sum(weights * (1.0 + x * nodes ** (1.0 / exponent)) ** (-exponent))
        )

        return 4.0 * np.pi * self.kappa_s * self.scale_radius**2.0 * f_x

    def _inner_mass_deficit_from(self, xp=np):
        r"""
        Returns the mass the MGE decomposition of the 3D density misses at its inner
        cut-off, in the convergence units of the projected mass.

        The Gaussians represent the density only down to the smallest width
        :math:`\sigma_{\rm min}`; inside it the expansion flattens, while the gNFW cusp
        keeps a mass :math:`\propto \sigma_{\rm min}^{3-\gamma}`. For a steep cusp
        (:math:`\gamma \gtrsim 1.5`) that mass is no longer negligible: without this term
        the deflections are biased low by 0.8% at :math:`\gamma = 2`, 6% at 2.4 and 40% at
        2.8 (PyAutoGalaxy#729).

        The deficit is the exact 3D mass inside a reference radius
        :math:`r_* = 10 \sigma_{\rm min}` minus the 3D mass the Gaussians hold inside it,
        each Gaussian of projected amplitude :math:`A_j` and width :math:`\sigma_j`
        contributing

        .. math::

            2 \pi A_j \sigma_j^2 \left[ {\rm erf}(u_j / \sqrt{2})
            - \sqrt{2 / \pi} \, u_j e^{-u_j^2 / 2} \right],
            \qquad u_j = r_* / \sigma_j.

        Above a few :math:`\sigma_{\rm min}` the expansion follows the density, so the
        difference does not depend on :math:`r_*` (it is flat from 5 to 30
        :math:`\sigma_{\rm min}`). Both masses scale with the axis ratio for an
        elliptical profile (major-axis convention).
        """
        sigmas = self._mge_sigmas_from(xp=xp)

        amps, _ = MGEDecomposer(mass_profile=self).decompose_convergence_via_mge(
            sigma_log_list=sigmas, three_D=True, xp=xp
        )

        radius_ref = 10.0 * sigmas[0]
        u = radius_ref / sigmas

        if xp is np:
            from scipy.special import erf
        else:
            from jax.scipy.special import erf

        mass_mge = xp.sum(
            2.0
            * np.pi
            * amps
            * sigmas**2.0
            * (erf(u / np.sqrt(2.0)) - np.sqrt(2.0 / np.pi) * u * xp.exp(-0.5 * u**2.0))
        )

        mass_true = self._enclosed_mass_3d_from(radius=radius_ref, xp=xp)

        return self.axis_ratio(xp=xp) * (mass_true - mass_mge)

    @aa.decorators.to_vector_yx
    @aa.decorators.transform
    def _deflections_2d_inner_mass_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ):
        """
        Returns the deflection angles of the mass inside the MGE inner cut-off
        (:meth:`_inner_mass_deficit_from`), placed as a uniform disc of radius
        ``sigma_min`` at the centre: exactly a point mass, ``M / (pi R)``, for every
        ``R >= sigma_min`` (``scale_radius / 20000``), and finite with a finite gradient
        at ``R = 0``.
        """
        mass = self._inner_mass_deficit_from(xp=xp)
        core_radius_squared = self._mge_sigmas_from(xp=xp)[0] ** 2.0

        y = xp.asarray(grid.array[:, 0], dtype=xp.float64)
        x = xp.asarray(grid.array[:, 1], dtype=xp.float64)

        factor = mass / (np.pi * xp.maximum(y * y + x * x, core_radius_squared))

        return self.rotated_grid_from_reference_frame_from(
            xp.vstack((factor * y, factor * x)).T, xp=xp
        )

    def _potential_2d_inner_mass_from(self, grid: aa.type.Grid2DLike, xp=np):
        """
        Returns the lensing potential of the uniform-disc inner-mass term of
        :meth:`_deflections_2d_inner_mass_from`, zero at the centre like the MGE
        potential, on a grid already in the profile's reference frame.
        """
        mass = self._inner_mass_deficit_from(xp=xp)
        core_radius_squared = self._mge_sigmas_from(xp=xp)[0] ** 2.0

        y = xp.asarray(grid.array[:, 0], dtype=xp.float64)
        x = xp.asarray(grid.array[:, 1], dtype=xp.float64)
        radii_squared = y * y + x * x

        return (mass / (2.0 * np.pi)) * (
            xp.minimum(radii_squared, core_radius_squared) / core_radius_squared
            + xp.log(
                xp.maximum(radii_squared, core_radius_squared) / core_radius_squared
            )
        )

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Returns the 2D lensing potential via the same MGE decomposition used for the
        deflection angles, plus the same inner-mass term, so that ``grad(psi)`` matches
        ``deflections_yx_2d_from``. The inner-mass term is exactly consistent; the MGE
        part integrates the deflections along radial lines with a fixed quadrature,
        which for ``inner_slope`` above ~2 leaves ``grad(psi)`` and the deflections
        apart at the 0.1-1% level.

        This implementation is inherited by every elliptical and spherical gNFW/NFW variant
        (``NFW``, ``gNFWSph``, and the MCR / Virial-mass subclasses), none of which have a
        closed-form elliptical potential. The MGE decomposer expands the 3D density into a
        Gaussian sum and integrates each Gaussian's analytic potential.
        """
        mge_decomp = MGEDecomposer(mass_profile=self)

        potential_mge = mge_decomp.potential_2d_via_mge_from(
            grid=grid,
            xp=xp,
            sigma_log_list=self._mge_sigmas_from(xp=xp),
            ellipticity_convention="major",
            three_D=True,
        )

        return potential_mge + self._potential_2d_inner_mass_from(grid=grid, xp=xp)

    def convergence_func(self, grid_radius: float, xp=np) -> float:
        """
        Returns the gNFW convergence at the input elliptical radii (see the class docstring
        for the closed-form expression and its line-of-sight integral I(eta), eta = xi / r_s).

        Small inputs integrate I(eta) with ``scipy.integrate.quad`` once per radius, in the
        numerically stable form of ``_convergence_integral_from``. When there are at least
        ``max(convergence_table_min_points, n_nodes)`` finite positive radii in a 1D NumPy
        input, I(eta) is instead tabulated with the same stable integral (at
        ``convergence_table_epsrel``) on a log-spaced eta grid of
        ``convergence_table_nodes_per_decade`` nodes per decade spanning those radii, and
        interpolated in (log eta, log I) with a cubic spline (see ``AbstractgNFW`` for the
        accuracy of this scheme); both paths agree to ~1e-7. Radii that are zero, negative or
        non-finite are always integrated per radius with the original integrand form, so their
        output is unchanged.
        """

        from scipy.integrate import quad

        def integral_y(y, eta):
            return (y + eta) ** (self.inner_slope - 4) * (1 - xp.sqrt(1 - y**2))

        grid_radius = xp.array((1.0 / self.scale_radius) * grid_radius.array)

        def convergence_from(eta, integral_y_value):
            return (
                2.0
                * self.kappa_s
                * (eta ** (1 - self.inner_slope))
                * (
                    (1 + eta) ** (self.inner_slope - 3)
                    + ((3 - self.inner_slope) * integral_y_value)
                )
            )

        loop_indexes = range(grid_radius.shape[0])

        if (
            xp is np
            and grid_radius.ndim == 1
            and np.issubdtype(grid_radius.dtype, np.floating)
        ):
            usable = np.isfinite(grid_radius) & (grid_radius > 0.0)
            eta = grid_radius[usable]

            if eta.size > 0:
                log_eta = np.log(eta)
                log_min, log_max = log_eta.min(), log_eta.max()
                pad = self.convergence_table_pad_decades * np.log(10.0)
                total_decades = (log_max - log_min) / np.log(10.0) + 2.0 * (
                    self.convergence_table_pad_decades
                )
                n_nodes = max(
                    int(
                        np.ceil(total_decades * self.convergence_table_nodes_per_decade)
                    )
                    + 1,
                    self.convergence_table_min_nodes,
                )

                if eta.size >= max(self.convergence_table_min_points, n_nodes):
                    from scipy.interpolate import CubicSpline

                    log_eta_nodes = np.linspace(log_min - pad, log_max + pad, n_nodes)
                    integral_nodes = np.array(
                        [
                            self._convergence_integral_from(
                                eta=np.exp(log_eta_node),
                                epsrel=self.convergence_table_epsrel,
                            )
                            for log_eta_node in log_eta_nodes
                        ]
                    )

                    spline = CubicSpline(log_eta_nodes, np.log(integral_nodes))

                    grid_radius[usable] = convergence_from(eta, np.exp(spline(log_eta)))

                    loop_indexes = np.flatnonzero(~usable)

        for index in loop_indexes:
            eta_index = grid_radius[index]

            if np.ndim(eta_index) == 0 and np.isfinite(eta_index) and eta_index > 0.0:
                integral_y_value = self._convergence_integral_from(
                    eta=eta_index, epsrel=gNFW.epsrel
                )
            else:
                integral_y_value = quad(
                    integral_y,
                    a=0.0,
                    b=1.0,
                    args=eta_index,
                    epsrel=gNFW.epsrel,
                )[0]

            grid_radius[index] = convergence_from(grid_radius[index], integral_y_value)

        return grid_radius

    def _convergence_integral_from(self, eta: float, epsrel: float) -> float:
        """
        Returns I(eta) = int_0^1 (y + eta)^(gamma - 4) (1 - sqrt(1 - y^2)) dy for one finite,
        positive eta, as used by both the per-radius path and the interpolation-table nodes of
        ``convergence_func``.

        The integral is evaluated after substituting y = sin(t), which gives the smooth integrand
        (sin t + eta)^(gamma - 4) 2 sin^2(t / 2) cos(t) on [0, pi / 2]: this removes both the
        square-root endpoint singularity at y = 1 and the cancellation in 1 - sqrt(1 - y^2) at
        small y. Breakpoints at eta, 10 eta, 100 eta, ... (below 1) resolve the peak of width
        ~eta at small eta. No absolute tolerance is used because I(eta) is tiny at large eta.
        At the class ``epsrel`` (1.49e-5) this is accurate to ~1e-10 relative in the
        convergence for 0 <= gamma <= 2.5 and 1e-5 <= eta <= 1e3.
        """
        from scipy.integrate import quad

        def integrand(t, eta):
            return (
                (np.sin(t) + eta) ** (self.inner_slope - 4)
                * 2.0
                * np.sin(0.5 * t) ** 2
                * np.cos(t)
            )

        points = [eta * 10.0**k for k in range(12) if eta * 10.0**k < 1.0]

        return quad(
            integrand,
            a=0.0,
            b=0.5 * np.pi,
            args=(eta,),
            epsrel=epsrel,
            epsabs=0.0,
            limit=200,
            points=points or None,
        )[0]


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
