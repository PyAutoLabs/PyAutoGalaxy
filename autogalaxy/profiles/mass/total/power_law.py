import math
import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.total.power_law_core import PowerLawCore

# Relative truncation error of the numpy omega series, as a bound on the geometric tail
# sum_{n >= N} f^n = f^N / (1 - f) with f the second flattening (1 - q) / (1 + q). Two orders
# of magnitude tighter than the rtol 1e-6 the deflection pins are checked at, so the series
# is at least as accurate as the complex `hyp2f1` it replaces on every physical axis ratio.
_OMEGA_SERIES_RTOL = 1e-10


def _omega_n_terms_from(factor: float, rtol: float = _OMEGA_SERIES_RTOL) -> int:
    """
    The number of terms of the Tessore & Metcalf (2015, eq. 29) omega series needed for its
    truncation error to be below `rtol`, as a plain Python int.

    Every term of the recurrence is bounded by `factor ** n` in magnitude (the ratio
    `(2n - (2 - t)) / (2n + (2 - t))` is below one for all slopes `t` in (0, 2)), so the tail
    after `N` terms is at most `factor ** N / (1 - factor)`; the smallest `N` with that bound
    below `rtol` is returned. The count follows the axis ratio: 11 terms at q = 0.8, 39 at
    q = 0.3, 124 at q = 0.1 (rtol 1e-10). A fixed count fails for q below ~0.25, the recorded
    hazard `component.power-law.series-vs-hyp2f1-divergence`.

    Parameters
    ----------
    factor
        The second flattening `f = (1 - q) / (1 + q)` of the ellipse with axis ratio `q`.
    rtol
        The bound on the relative truncation error of the series.
    """
    factor = float(factor)
    if factor <= 0.0:
        return 1
    if factor >= 1.0:
        raise ValueError(
            f"The power-law omega series does not converge for factor={factor} "
            f"(axis_ratio must be > 0)."
        )
    return max(2, math.ceil((math.log(rtol) + math.log1p(-factor)) / math.log(factor)))


def _omega_series_from(z: np.ndarray, slope: float, factor: float, n_terms: int) -> np.ndarray:
    """
    Numpy evaluation of the angular part of the elliptical power-law deflection, the omega
    series of Tessore & Metcalf (2015, eq. 29) truncated after `n_terms` terms. The same
    recurrence `jax_utils.omega` evaluates with `jax.lax.scan` on the JAX path, written as a
    polynomial in `w = -factor * z ** 2` with real coefficients

        omega = z * sum_n a_n w ** n,   a_0 = 1,   a_n = a_{n-1} (2n - (2 - t)) / (2n + (2 - t))

    and evaluated by Horner's rule, which is one complex multiply-add per term over the grid.

    Parameters
    ----------
    z
        `exp(i * phi)` where `phi` is the elliptical angle of every coordinate on the grid.
    slope
        The internal power-law slope `t = slope - 1`.
    factor
        The second flattening `f = (1 - q) / (1 + q)`.
    n_terms
        The number of terms of the series to sum (see `_omega_n_terms_from`).
    """
    two_minus_slope = 2.0 - slope

    coefficients = np.empty(n_terms)
    coefficients[0] = 1.0
    for n in range(1, n_terms):
        two_n = 2.0 * n
        coefficients[n] = (
            coefficients[n - 1] * (two_n - two_minus_slope) / (two_n + two_minus_slope)
        )

    w = -factor * z * z

    polynomial = np.full(z.shape, coefficients[-1], dtype=np.complex128)
    for coefficient in coefficients[-2::-1]:
        polynomial *= w
        polynomial += coefficient

    return z * polynomial


class PowerLaw(PowerLawCore):
    r"""Elliptical power-law (EPL / PEMD) mass profile.

    The convergence of the elliptical power-law is:

    .. math::

        \kappa(R) = \frac{3 - \gamma}{2}
                    \left(\frac{\theta_{\rm E}}{R}\right)^{\gamma - 1}

    where :math:`\gamma` is the logarithmic density slope, :math:`\theta_{\rm E}`
    is the Einstein radius, and :math:`R` is the elliptical radius.  The
    isothermal case corresponds to :math:`\gamma = 2`.

    Parameters
    ----------
    centre : (float, float)
        (y, x) arc-second coordinates of the profile centre.
    ell_comps : (float, float)
        Ellipticity components (e1, e2) of the elliptical coordinate system.
    einstein_radius : float
        Einstein radius in arcseconds.
    slope : float
        Logarithmic density slope :math:`\gamma`; shallower profiles have
        lower values, steeper profiles have higher values.

    References
    ----------
    Tessore & Metcalf (2015), A&A, 580, A79.
    Schneider, Ehlers & Falco (1992), *Gravitational Lenses*, Springer.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
    ):
        """
        Represents an elliptical power-law density distribution.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        einstein_radius
            The arc-second Einstein radius.
        slope
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        """

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            einstein_radius=einstein_radius,
            slope=slope,
            core_radius=0.0,
        )

    @aa.decorators.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):

        alpha = self.deflections_yx_2d_from(
            grid=aa.Grid2DIrregular(grid), xp=xp, **kwargs
        )

        alpha_x = alpha[:, 1]
        alpha_y = alpha[:, 0]

        x = grid.array[:, 1] - self.centre[1]
        y = grid.array[:, 0] - self.centre[0]

        return (x * alpha_x + y * alpha_y) / (3 - self.slope)

    @aa.decorators.to_vector_yx
    @aa.decorators.transform(rotate_back=True)
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        For coordinates (0.0, 0.0) the analytic calculation of the deflection angle gives a NaN. Therefore,
        coordinates at (0.0, 0.0) are shifted slightly to (1.0e-8, 1.0e-8).

        This code is an adaption of Tessore & Metcalf 2015:
        https://arxiv.org/abs/1507.01819

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        slope = self.slope - 1.0
        axis_ratio = self.axis_ratio(xp)

        einstein_radius = (
            2.0 / (axis_ratio**-0.5 + axis_ratio**0.5)
        ) * self.einstein_radius_major_from(xp)

        factor = (1.0 - axis_ratio) / (1.0 + axis_ratio)
        b = einstein_radius * xp.sqrt(axis_ratio)

        y = grid.array[:, 0]
        qx = axis_ratio * grid.array[:, 1]

        R = xp.sqrt(qx**2 + y**2 + 1e-16)

        if xp.__name__.startswith("jax"):

            from .jax_utils import omega

            angle = xp.arctan2(y, qx)  # Note, this angle is not the position angle
            z = xp.add(
                xp.multiply(xp.cos(angle), 1 + 0j), xp.multiply(xp.sin(angle), 0 + 1j)
            )

            zh = omega(z, slope, factor, n_terms=20, xp=xp)

        else:

            # `z = exp(i * angle)` is the unit vector (q x + i y) / |q x + i y|, formed without
            # the per-coordinate arctan2, cos and sin; the exact centre, where that vector has
            # no direction, takes z = 1 as arctan2(0, 0) = 0 did.
            r_ell = np.hypot(qx, y)
            on_centre = r_ell == 0.0
            z = (qx + 1j * y) / np.where(on_centre, 1.0, r_ell)
            z = np.where(on_centre, 1.0 + 0j, z)

            zh = _omega_series_from(
                z, slope, factor, n_terms=_omega_n_terms_from(factor)
            )

        prefactor = (
            2.0
            * b
            / (1.0 + axis_ratio)
            * self.ellipticity_rescale(xp) ** (slope - 1.0)
            * (b / R) ** (slope - 1.0)
        )

        complex_angle = prefactor * zh

        return xp.vstack((complex_angle.imag, complex_angle.real)).T

    def convergence_func(self, grid_radius: float, xp=np) -> float:
        return self.einstein_radius_rescaled(xp) * grid_radius.array ** (
            -(self.slope - 1)
        )

    @staticmethod
    def potential_func(u, y, x, axis_ratio, slope, core_radius, xp=np):
        _eta_u = xp.sqrt((u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u)))))
        return (
            (_eta_u / u)
            * ((3.0 - slope) * _eta_u) ** -1.0
            * _eta_u ** (3.0 - slope)
            / ((1 - (1 - axis_ratio**2) * u) ** 0.5)
        )


class PowerLawIntermediate(PowerLaw):
    r"""Elliptical power-law mass profile with an intermediate-axis Einstein radius.

    Identical mass distribution to :class:`PowerLaw`, but the ``einstein_radius``
    parameter follows the *intermediate-axis* convention used by the COOLEST
    standard, lenstronomy and herculens: radii are measured as
    :math:`r = \sqrt{a b} = \sqrt{q} \, a` on the elliptical isodensity
    contours, and the convergence is

    .. math::

        \kappa(r) = \frac{3 - \gamma}{2}
                    \left(\frac{\theta_{\rm E}}{r}\right)^{\gamma - 1}.

    The relation to the :class:`PowerLaw` ``einstein_radius`` (:math:`\theta_{\rm PL}`) is:

    .. math::

        \theta_{\rm E} = \sqrt{q} \left(\frac{2}{1 + q}\right)^{1 / (\gamma - 1)} \theta_{\rm PL},

    which for the isothermal case (:math:`\gamma = 2`) reduces to
    :math:`2 \sqrt{q} / (1 + q) \, \theta_{\rm PL}`.

    Use this profile when parameter values must line up directly with other lens
    modeling codes — its COOLEST ``PEMD`` mapping is an identity
    (``autogalaxy.interop.coolest``) — at the cost of departing from the
    Einstein-radius convention of the rest of **PyAutoGalaxy**.

    Parameters
    ----------
    centre : (float, float)
        (y, x) arc-second coordinates of the profile centre.
    ell_comps : (float, float)
        Ellipticity components (e1, e2) of the elliptical coordinate system.
    einstein_radius : float
        Einstein radius in arcseconds, in the intermediate-axis (COOLEST) convention.
    slope : float
        Logarithmic density slope :math:`\gamma`.
    """

    def einstein_radius_major_from(self, xp=np):
        """
        Convert the stored intermediate-axis ``einstein_radius`` to the major-axis
        convention the power-law formulae are written in (the inverse of the
        relation in the class docstring). Threaded through ``xp`` so the profile
        is JAX-traceable like its parent.
        """
        axis_ratio = self.axis_ratio(xp)
        return (
            self.einstein_radius
            / xp.sqrt(axis_ratio)
            * ((1.0 + axis_ratio) / 2.0) ** (1.0 / (self.slope - 1.0))
        )


class PowerLawSph(PowerLaw):
    r"""Spherical power-law mass profile.

    The spherical limit of :class:`PowerLaw`.  The convergence is:

    .. math::

        \kappa(r) = \frac{3 - \gamma}{2}
                    \left(\frac{\theta_{\rm E}}{r}\right)^{\gamma - 1}

    where :math:`\gamma` is the logarithmic density slope, :math:`\theta_{\rm E}`
    is the Einstein radius, and :math:`r` is the circular projected radius.

    Parameters
    ----------
    centre : (float, float)
        (y, x) arc-second coordinates of the profile centre.
    einstein_radius : float
        Einstein radius in arcseconds.
    slope : float
        Logarithmic density slope :math:`\gamma`; shallower profiles have
        lower values, steeper profiles have higher values.

    References
    ----------
    Tessore & Metcalf (2015), A&A, 580, A79.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
    ):
        """
        Represents a spherical power-law density distribution.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius
            The arc-second Einstein radius.
        slope
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        """

        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            einstein_radius=einstein_radius,
            slope=slope,
        )

    @aa.decorators.to_vector_yx
    @aa.decorators.transform
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        eta = self.radial_grid_from(grid=grid, xp=xp, **kwargs).array
        deflection_r = (
            2.0
            * self.einstein_radius_rescaled(xp)
            * xp.divide(
                xp.power(eta, (3.0 - self.slope)),
                xp.multiply((3.0 - self.slope), eta),
            )
        )

        return self._cartesian_grid_via_radial_from(
            grid=grid, radius=deflection_r, xp=xp
        )
