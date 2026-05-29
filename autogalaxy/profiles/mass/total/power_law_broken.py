import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile


class PowerLawBroken(MassProfile):
    r"""Broken elliptical power-law mass profile with inner and outer slopes.

    The convergence has a power-law form with different slopes inside and
    outside a break radius :math:`\theta_{\rm b}`, matched to be continuous
    at the break:

    .. math::

        \kappa(R) = \begin{cases}
            \kappa_{\rm b}
            \left(\dfrac{\theta_{\rm b}}{R}\right)^{\gamma_{\rm in}}
            & R \leq \theta_{\rm b} \\[6pt]
            \kappa_{\rm b}
            \left(\dfrac{\theta_{\rm b}}{R}\right)^{\gamma_{\rm out}}
            & R > \theta_{\rm b}
        \end{cases}

    where :math:`\gamma_{\rm in}` and :math:`\gamma_{\rm out}` are the inner and
    outer logarithmic slopes, :math:`\theta_{\rm b}` is the break radius, and
    :math:`\kappa_{\rm b}` is the convergence at the break radius (set by the
    normalisation condition on the Einstein radius).

    Parameters
    ----------
    centre : (float, float)
        (y, x) arc-second coordinates of the profile centre.
    ell_comps : (float, float)
        Ellipticity components (e1, e2) of the elliptical coordinate system.
    einstein_radius : float
        Einstein radius in arcseconds.
    inner_slope : float
        Logarithmic density slope :math:`\gamma_{\rm in}` inside the break radius.
    outer_slope : float
        Logarithmic density slope :math:`\gamma_{\rm out}` outside the break radius.
    break_radius : float
        Break radius :math:`\theta_{\rm b}` in arcseconds separating the two power-law regimes.

    References
    ----------
    Du, Metcalf & Barkana (2020), MNRAS, 495, 4209.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        inner_slope: float = 1.5,
        outer_slope: float = 2.5,
        break_radius: float = 0.01,
    ):
        """
        Ell, homoeoidal mass model with an inner_slope
        and outer_slope, continuous in density across break_radius.
        Position angle is defined to be zero on x-axis and
        +ve angle rotates the lens anticlockwise

        The grid variable is a tuple of (theta_1, theta_2), where
        each theta_1, theta_2 is itself a 2D array of the x and y
        coordinates respectively.~
        """

        super().__init__(centre=centre, ell_comps=ell_comps)

        self.einstein_radius = einstein_radius
        self.einstein_radius_elliptical = np.sqrt(self.axis_ratio()) * einstein_radius
        self.break_radius = break_radius
        self.inner_slope = inner_slope
        self.outer_slope = outer_slope

        # Parameters defined in the notes
        self.nu = break_radius / self.einstein_radius_elliptical
        self.dt = (2 - self.inner_slope) / (2 - self.outer_slope)

        # Normalisation (eq. 5)
        if self.nu < 1:
            self.kB = (2 - self.inner_slope) / (
                (2 * self.nu**2)
                * (1 + self.dt * (self.nu ** (self.outer_slope - 2) - 1))
            )
        else:
            self.kB = (2 - self.inner_slope) / (2 * self.nu**2)

    def _convergence(self, radii, xp=np):
        """
        Returns the dimensionless density kappa=Sigma/Sigma_c (eq. 1) as a function of the
        (elliptical or circular) radial coordinate `radii`.

        Shared by `convergence_2d_from` (which passes the elliptical radius) and the
        `convergence_func` hook (which passes the radial coordinate directly). `radii` is a
        plain array/scalar (callers unwrap `aa.ArrayIrregular` to `.array` first), so the
        boolean masks `(radii <= break_radius)` return raw numpy bool arrays.
        """

        # Inside break radius
        kappa_inner = self.kB * (self.break_radius / radii) ** self.inner_slope

        # Outside break radius
        kappa_outer = self.kB * (self.break_radius / radii) ** self.outer_slope

        return kappa_inner * (radii <= self.break_radius) + kappa_outer * (
            radii > self.break_radius
        )

    def convergence_func(self, grid_radius, xp=np):
        # Unwrap `aa.ArrayIrregular` -> plain array so `_convergence` returns a plain array
        # (matching e.g. `Isothermal.convergence_func`). `mass_integral` -> scipy.quad cannot
        # consume an `aa.ArrayIrregular` return.
        radii = grid_radius.array if hasattr(grid_radius, "array") else grid_radius
        return self._convergence(radii, xp=xp)

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Returns the dimensionless density kappa=Sigma/Sigma_c (eq. 1)
        """

        # Ell radius
        radius = xp.hypot(grid.array[:, 1] * self.axis_ratio(xp), grid.array[:, 0])

        return self._convergence(radius, xp=xp)

    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        The lensing potential is not available for the broken power law.

        It would be computed by decomposing the projected convergence into Gaussians via
        `potential_2d_via_mge_from` (`three_D=False`), but that decomposition integrates the
        convergence along a *complex* contour and therefore requires an analytic convergence
        profile. The broken power law's convergence is piecewise — a slope discontinuity at
        `break_radius` — so it is non-analytic and the MGE potential is numerically invalid
        (wrong by many orders of magnitude). A correct potential would integrate this
        profile's own analytic deflection field (eq. 18-19) along radial lines, which is not
        yet implemented.

        `convergence_2d_from`, `deflections_yx_2d_from`, and `convergence_func` (and hence the
        Einstein-radius / enclosed-mass integrals) are all available and correct.
        """
        raise NotImplementedError(
            "PowerLawBroken.potential_2d_from is not implemented: the MGE potential "
            "decomposition requires an analytic convergence profile, but the broken power "
            "law's convergence is piecewise (a kink at break_radius). Use deflections / "
            "convergence instead, or integrate the analytic deflections for the potential."
        )

    @aa.decorators.to_vector_yx
    @aa.decorators.transform(rotate_back=True)
    def deflections_yx_2d_from(self, grid, xp=np, max_terms=20, **kwargs):
        """
        Returns the complex deflection angle from eq. 18 and 19
        """

        # Rotate coordinates
        z = grid.array[:, 1] + 1j * grid.array[:, 0]

        # Ell radius
        R = xp.hypot(z.real * self.axis_ratio(xp), z.imag)

        # Factors common to eq. 18 and 19
        factors = (
            2
            * self.kB
            * (self.break_radius**2)
            / (self.axis_ratio(xp) * z * (2 - self.inner_slope))
        )

        # Hypergeometric functions
        # (in order of appearance in eq. 18 and 19)
        # These can also be computed with scipy.special.hyp2f1(), it's
        # much slower can be a useful test
        F1 = self.hyp2f1_series(
            self.inner_slope, self.axis_ratio(xp), R, z, max_terms=max_terms, xp=xp
        )
        F2 = self.hyp2f1_series(
            self.inner_slope,
            self.axis_ratio(xp),
            self.break_radius,
            z,
            max_terms=max_terms,
            xp=xp,
        )
        F3 = self.hyp2f1_series(
            self.outer_slope, self.axis_ratio(xp), R, z, max_terms=max_terms, xp=xp
        )
        F4 = self.hyp2f1_series(
            self.outer_slope,
            self.axis_ratio(xp),
            self.break_radius,
            z,
            max_terms=max_terms,
            xp=xp,
        )

        # theta < break radius (eq. 18)
        inner_part = factors * F1 * (self.break_radius / R) ** (self.inner_slope - 2)

        # theta > break radius (eq. 19)
        outer_part = factors * (
            F2
            + self.dt * (((self.break_radius / R) ** (self.outer_slope - 2)) * F3 - F4)
        )

        # Combine and take the conjugate
        deflections = (
            inner_part * (R <= self.break_radius) + outer_part * (R > self.break_radius)
        ).conjugate()

        return xp.vstack((xp.imag(deflections), xp.real(deflections))).T

    @staticmethod
    def hyp2f1_series(t, q, r, z, max_terms=20, xp=np):
        """
        Computes eq. 26 for a radius r, slope t,
        axis ratio q, and coordinates z.
        """

        # u from eq. 25
        q_ = (1 - q**2) / (q**2)
        u = 0.5 * (1 - xp.sqrt(1 - q_ * (r / z) ** 2))

        # First coefficient
        a_n = 1.0

        # Storage for sum
        F = xp.zeros_like(z, dtype="complex64")

        for n in range(max_terms):
            F += a_n * (u**n)
            a_n *= ((2 * n) + 4 - (2 * t)) / ((2 * n) + 4 - t)

        return F


class PowerLawBrokenSph(PowerLawBroken):
    r"""Broken spherical power-law mass profile with inner and outer slopes.

    The spherical limit of :class:`PowerLawBroken`.  The convergence is:

    .. math::

        \kappa(r) = \begin{cases}
            \kappa_{\rm b}
            \left(\dfrac{\theta_{\rm b}}{r}\right)^{\gamma_{\rm in}}
            & r \leq \theta_{\rm b} \\[6pt]
            \kappa_{\rm b}
            \left(\dfrac{\theta_{\rm b}}{r}\right)^{\gamma_{\rm out}}
            & r > \theta_{\rm b}
        \end{cases}

    where :math:`r` is the circular projected radius, :math:`\theta_{\rm b}` is
    the break radius, and :math:`\kappa_{\rm b}` is the convergence at the
    break radius.

    Parameters
    ----------
    centre : (float, float)
        (y, x) arc-second coordinates of the profile centre.
    einstein_radius : float
        Einstein radius in arcseconds.
    inner_slope : float
        Logarithmic density slope :math:`\gamma_{\rm in}` inside the break radius.
    outer_slope : float
        Logarithmic density slope :math:`\gamma_{\rm out}` outside the break radius.
    break_radius : float
        Break radius :math:`\theta_{\rm b}` in arcseconds separating the two power-law regimes.

    References
    ----------
    Du, Metcalf & Barkana (2020), MNRAS, 495, 4209.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        inner_slope: float = 1.5,
        outer_slope: float = 2.5,
        break_radius: float = 0.01,
    ):
        """
        Ell, homoeoidal mass model with an inner_slope
        and outer_slope, continuous in density across break_radius.
        Position angle is defined to be zero on x-axis and
        +ve angle rotates the lens anticlockwise

        The grid variable is a tuple of (theta_1, theta_2), where
        each theta_1, theta_2 is itself a 2D array of the x and y
        coordinates respectively.~
        """

        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            einstein_radius=einstein_radius,
            inner_slope=inner_slope,
            outer_slope=outer_slope,
            break_radius=break_radius,
        )
