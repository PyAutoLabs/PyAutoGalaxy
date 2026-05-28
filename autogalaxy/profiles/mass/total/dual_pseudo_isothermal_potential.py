from typing import Tuple
import numpy as np

import autoarray as aa
from autogalaxy.profiles.mass.abstract.abstract import MassProfile


class dPIEPotential(MassProfile):
    r"""Dual pseudo-isothermal elliptical mass profile (dPIE, potential parameterisation).

    A dPIE profile in which ellipticity is applied to the lensing potential rather
    than the mass distribution.  The circularly symmetric convergence takes the
    same functional form as :class:`dPIEMassSph`:

    .. math::

        \kappa(R) = \frac{b_0}{2} \frac{r_s}{r_s - r_a}
        \left(
            \frac{1}{\sqrt{r_a^2 + R^2}}
            - \frac{1}{\sqrt{r_s^2 + R^2}}
        \right)

    but ellipticity enters through a pseudo-elliptical radius
    :math:`R^2 = x^2(1-\epsilon) + y^2(1+\epsilon)` in the deflection, rather
    than through a purely elliptical mass distribution.

    Parameters
    ----------
    centre : (float, float)
        (y, x) arc-second coordinates of the profile centre.
    ell_comps : (float, float)
        Ellipticity components (e1, e2) of the elliptical coordinate system.
    ra : float
        Inner core scale radius in arcseconds.
    rs : float
        Outer truncation scale radius in arcseconds.
    b0 : float
        Lens strength in arcseconds (Einstein radius in the limit
        :math:`r_a \to 0`, :math:`r_s \to \infty`, :math:`q \to 1`).

    References
    ----------
    Eliasdottir et al. (2007), arXiv:0710.5636.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        b0: float = 1.0,
    ):
        """
        The dual Pseudo Isothermal Elliptical Potential (dPIEPotential) with pseudo-ellipticity on potential, based on the
        formulation from Eliasdottir (2007): https://arxiv.org/abs/0710.5636.

        This profile describes a circularly symmetric (non-elliptical) projected mass
        distribution with two scale radii (`ra` and `rs`) and a normalization factor
        `kappa_scale`. Although originally called the dPIEPotential (Elliptical), this version
        lacks ellipticity, so the "E" may be a misnomer.

        The projected surface mass density is given by:

        .. math::

            \\Sigma(R) = \\Sigma_0 (ra * rs) / (rs - ra) *
                          (1 / \\sqrt(ra^2 + R^2) - 1 / \\sqrt(rs^2 + R^2))

        (See Eliasdottir 2007, Eq. A3.)

        In this implementation:
        - `ra` is the core radius in unit of arcseconds.
        - `b0` is the lens strength in unit of arcseconds, when ra->0 & rs->\\infty & q->1, b0 is the Einstein radius.
          `b0` is related to the central velocity dispersion \\sigma_0: b_0 = 4\\pi * \\sigma_0^2 / c^2 * (D_{LS} / D_{S})
          `b0` is not in the Intermediate-Axis-Convention for its r_{em}^2 = x^2 / (1 + \\epsilon)^2 + y^2 / (1 - \\epsilon)^2

        Credit: Jackson O'Donnell for implementing this profile in PyAutoLens.
        Note: To ensure consistency, kappa_scale was replaced with b0, and the corresponding code was adjusted accordingly.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ra
            The inner core scale radius in arcseconds.
        rs
            The outer truncation scale radius in arcseconds.
        b0
            The lens strength in arcseconds.
        """
        super().__init__(centre=centre, ell_comps=ell_comps)

        self.ra = ra
        self.rs = rs
        self.b0 = b0

    def _ellip(self, xp=np):
        ellip = xp.sqrt(self.ell_comps[0] ** 2 + self.ell_comps[1] ** 2)
        MAX_ELLIP = 0.99999
        return xp.min(xp.array([ellip, MAX_ELLIP]))

    def _deflection_angle(self, radii, xp=np):
        """
        For a circularly symmetric dPIEPotential profile, computes the magnitude of the deflection at each radius.
        """
        a, s = self.ra, self.rs
        radii = xp.maximum(radii, 1e-8)
        f = radii / (a + xp.sqrt(a**2 + radii**2)) - radii / (
            s + xp.sqrt(s**2 + radii**2)
        )

        # c.f. Eliasdottir '07 eq. A23
        # magnitude of deflection
        # alpha = self.E0 * (s + a) / s * f
        alpha = self.b0 * s / (s - a) * f
        return alpha

    def _convergence(self, radii, xp=np):

        radsq = radii * radii
        a, s = self.ra, self.rs

        return (
            self.b0
            / 2
            * s
            / (s - a)
            * (1 / xp.sqrt(a**2 + radsq) - 1 / xp.sqrt(s**2 + radsq))
        )

    def convergence_func(self, grid_radius, xp=np):
        return self._convergence(grid_radius, xp=xp)

    @aa.decorators.to_vector_yx
    @aa.decorators.transform(rotate_back=True)
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        ellip = self._ellip(xp)
        grid_radii = xp.sqrt(
            grid.array[:, 1] ** 2 * (1 - ellip) + grid.array[:, 0] ** 2 * (1 + ellip)
        )

        # Compute the deflection magnitude of a *non-elliptical* profile
        alpha_circ = self._deflection_angle(grid_radii, xp)

        # This is in axes aligned to the major/minor axis
        deflection_y = alpha_circ * xp.sqrt(1 + ellip) * (grid.array[:, 0] / grid_radii)
        deflection_x = alpha_circ * xp.sqrt(1 - ellip) * (grid.array[:, 1] / grid_radii)

        return xp.vstack((deflection_y, deflection_x)).T

    @aa.decorators.to_array
    @aa.decorators.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Returns the two dimensional projected convergence on a grid of (y,x) arc-second coordinates.

        The `grid_2d_to_structure` decorator reshapes the ndarrays the convergence is outputted on. See
        *aa.grid_2d_to_structure* for a description of the output.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.
        """
        ellip = self._ellip(xp)
        grid_radii = xp.sqrt(
            grid.array[:, 1] ** 2 * (1 - ellip) + grid.array[:, 0] ** 2 * (1 + ellip)
        )

        # Compute the convergence and deflection of a *circular* profile
        kappa_circ = self._convergence(grid_radii, xp)
        alpha_circ = self._deflection_angle(grid_radii, xp)

        asymm_term = (
            ellip * (1 - ellip) * grid.array[:, 1] ** 2
            - ellip * (1 + ellip) * grid.array[:, 0] ** 2
        ) / grid_radii**2

        # convergence = 1/2 \nabla \alpha = 1/2 \nabla^2 potential
        # The "asymm_term" is asymmetric on x and y, so averages out to
        # zero over all space
        return kappa_circ * (1 - asymm_term) + (alpha_circ / grid_radii) * asymm_term

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        from autogalaxy.profiles.mass.abstract.mge import MGEDecomposer

        radii_min = max(self.ra, 0.001) / 10.0
        radii_max = self.rs * 20.0
        sigmas = xp.exp(xp.linspace(xp.log(radii_min), xp.log(radii_max), 30))
        mge_decomp = MGEDecomposer(mass_profile=self)
        return mge_decomp.potential_2d_via_mge_from(
            grid=grid, xp=xp, sigma_log_list=sigmas,
            ellipticity_convention="major", three_D=False,
        )


class dPIEPotentialSph(dPIEPotential):
    r"""Spherical dual pseudo-isothermal mass profile (dPIE, potential parameterisation).

    The spherical limit of :class:`dPIEPotential`.  The convergence is:

    .. math::

        \kappa(r) = \frac{b_0}{2} \frac{r_s}{r_s - r_a}
        \left(
            \frac{1}{\sqrt{r_a^2 + r^2}}
            - \frac{1}{\sqrt{r_s^2 + r^2}}
        \right)

    where :math:`r` is the circular projected radius.  This profile is
    mathematically identical to :class:`dPIEMassSph`.

    Parameters
    ----------
    centre : (float, float)
        (y, x) arc-second coordinates of the profile centre.
    ra : float
        Inner core scale radius in arcseconds.
    rs : float
        Outer truncation scale radius in arcseconds.
    b0 : float
        Lens strength in arcseconds (Einstein radius in the limit
        :math:`r_a \to 0`, :math:`r_s \to \infty`).

    References
    ----------
    Eliasdottir et al. (2007), arXiv:0710.5636.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        b0: float = 1.0,
    ):
        """
        The dual Pseudo-Isothermal mass profile (dPIEPotential) without ellipticity, based on the
        formulation from Eliasdottir (2007): https://arxiv.org/abs/0710.5636.

        This profile describes a circularly symmetric (non-elliptical) projected mass
        distribution with two scale radii (`ra` and `rs`) and a normalization factor
        `kappa_scale`. Although originally called the dPIEPotential (Elliptical), this version
        lacks ellipticity, so the "E" may be a misnomer.

        The projected surface mass density is given by:

        .. math::

            \\Sigma(R) = \\Sigma_0 (ra * rs) / (rs - ra) *
                          (1 / \\sqrt(ra^2 + R^2) - 1 / \\sqrt(rs^2 + R^2))

        (See Eliasdottir 2007, Eq. A3.)

        In this implementation:
        - `ra` is the core radius in unit of arcseconds.
        - `b0` is the lens strength in unit of arcseconds, when ra->0 & rs->\\infty & q->1, b0 is the Einstein radius.
          `b0` is related to the central velocity dispersion \\sigma_0: b_0 = 4\\pi * \\sigma_0^2 / c^2 * (D_{LS} / D_{S})
          `b0` is not in the Intermediate-Axis-Convention for its r_{em}^2 = x^2 / (1 + \\epsilon)^2 + y^2 / (1 - \\epsilon)^2

        Credit: Jackson O'Donnell for implementing this profile in PyAutoLens.
        Note: This dPIEPotentialSph should be the same with dPIEMassSph for their same mathamatical formulations.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ra
            The inner core scale radius in arcseconds.
        rs
            The outer truncation scale radius in arcseconds.
        b0
            The lens strength in arcseconds.
        """

        super().__init__(centre=centre, ell_comps=(0.0, 0.0))

        self.ra = ra
        self.rs = rs
        self.b0 = b0

    @aa.decorators.to_vector_yx
    @aa.decorators.transform
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        radii = self.radial_grid_from(grid=grid, xp=xp, **kwargs)

        alpha = self._deflection_angle(radii.array, xp)

        # now we decompose the deflection into y/x components
        defl_y = alpha * grid.array[:, 0] / radii.array
        defl_x = alpha * grid.array[:, 1] / radii.array

        return aa.Grid2DIrregular.from_yx_1d(defl_y, defl_x)

    @aa.decorators.to_array
    @aa.decorators.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Returns the two dimensional projected convergence on a grid of (y,x) arc-second coordinates.

        The `grid_2d_to_structure` decorator reshapes the ndarrays the convergence is outputted on. See
        *aa.grid_2d_to_structure* for a description of the output.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.
        """
        # already transformed to center on profile centre so this works
        radsq = grid.array[:, 0] ** 2 + grid.array[:, 1] ** 2

        return self._convergence(xp.sqrt(radsq), xp)

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        from autogalaxy.profiles.mass.abstract.mge import MGEDecomposer

        radii_min = max(self.ra, 0.001) / 10.0
        radii_max = self.rs * 20.0
        sigmas = xp.exp(xp.linspace(xp.log(radii_min), xp.log(radii_max), 30))
        mge_decomp = MGEDecomposer(mass_profile=self)
        return mge_decomp.potential_2d_via_mge_from(
            grid=grid, xp=xp, sigma_log_list=sigmas,
            ellipticity_convention="major", three_D=False,
        )
