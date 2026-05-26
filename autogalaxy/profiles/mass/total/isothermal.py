import numpy as np

from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.total.power_law import PowerLaw


def psi_from(grid, axis_ratio, core_radius, xp=np):
    r"""
    Returns the $\Psi$ term in expressions for the calculation of the deflection of an elliptical isothermal mass
    distribution. This is used in the `Isothermal` and `Chameleon` `MassProfile`'s.

    The expression for Psi is:

    $\Psi = \sqrt(q^2(s^2 + x^2) + y^2)$

    Parameters
    ----------
    grid
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    axis_ratio
            Ratio of profiles ellipse's minor and major axes (b/a)
    core_radius
        The radius of the inner core

    Returns
    -------
    float
        The value of the Psi term.

    """
    return xp.sqrt(
        (axis_ratio**2.0 * (grid.array[:, 1] ** 2.0 + core_radius**2.0))
        + grid.array[:, 0] ** 2.0
        + 1e-16
    )


class Isothermal(PowerLaw):
    r"""Singular isothermal ellipsoid (SIE) mass profile.

    The convergence of the SIE is:

    .. math::

        \kappa(R) = \frac{\theta_{\rm E}}{2R}

    where :math:`\theta_{\rm E}` is the Einstein radius and :math:`R` is the
    elliptical radius defined as :math:`R^2 = q x^2 + y^2 / q` with axis ratio
    :math:`q`.  This is the special case of the power-law profile with slope
    :math:`\gamma = 2`.

    Parameters
    ----------
    centre : (float, float)
        (y, x) arc-second coordinates of the profile centre.
    ell_comps : (float, float)
        Ellipticity components (e1, e2) of the elliptical coordinate system.
    einstein_radius : float
        Einstein radius in arcseconds.

    References
    ----------
    Kormann, Schneider & Bartelmann (1994), A&A, 284, 285.
    Tessore & Metcalf (2015), A&A, 580, A79.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
    ):
        """
        Represents an elliptical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope = 2.0.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        einstein_radius
            The arc-second Einstein radius.
        """

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            einstein_radius=einstein_radius,
            slope=2.0,
        )

    def axis_ratio(self, xp=np):
        axis_ratio = super().axis_ratio(xp=xp)
        return xp.minimum(axis_ratio, 0.99999)

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

        factor = (
            2.0
            * self.einstein_radius_rescaled(xp)
            * self.axis_ratio(xp)
            / xp.sqrt(1 - self.axis_ratio(xp) ** 2)
        )

        psi = psi_from(
            grid=grid, axis_ratio=self.axis_ratio(xp), core_radius=0.0, xp=xp
        )

        deflection_y = xp.arctanh(
            xp.divide(
                xp.multiply(xp.sqrt(1 - self.axis_ratio(xp) ** 2), grid.array[:, 0]),
                psi,
            )
        )
        deflection_x = xp.arctan(
            xp.divide(
                xp.multiply(xp.sqrt(1 - self.axis_ratio(xp) ** 2), grid.array[:, 1]),
                psi,
            )
        )
        return xp.multiply(factor, xp.vstack((deflection_y, deflection_x)).T)

    @aa.decorators.to_vector_yx
    @aa.decorators.transform
    def shear_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        r"""
        Returns the analytic 2D weak-lensing shear vector field :math:`(\gamma_2, \gamma_1)` of the elliptical
        isothermal mass distribution on a grid of ``(y, x)`` arc-second coordinates.

        For an axis-aligned isothermal profile centred on the origin the shear components reduce to:

        .. math::

            \gamma_1 = -\kappa(\theta) \, \frac{x^2 - y^2}{x^2 + y^2}

            \gamma_2 = -2 \, \kappa(\theta) \, \frac{x \, y}{x^2 + y^2}

        where :math:`\kappa(\theta)` is the convergence at the rotated grid coordinate.  After evaluation in the
        profile's reference frame the shear vector field is rotated back into the original frame using the
        ``2 * angle`` rotation appropriate for a spin-2 quantity (the shear transforms as a spin-2 field, so a
        coordinate rotation by ``angle`` rotates the components by ``2 * angle``).

        This analytic path is mathematically equivalent to ``LensCalc.shear_yx_2d_via_hessian_from``, which
        derives the same shear from finite-difference (or JAX) derivatives of ``deflections_yx_2d_from``.  The
        cross-check is exercised in
        ``test_autogalaxy/profiles/mass/total/test_isothermal.py::test__shear_yx_2d_from__matches_via_hessian``.

        Convention
        ----------
        The result is returned as a vector-field with shape ``[total_shear_vectors, 2]`` where:

        - ``[:, 0]`` are the :math:`\gamma_2` values
        - ``[:, 1]`` are the :math:`\gamma_1` values

        i.e. the FIRST column is :math:`\gamma_2` and the SECOND column is :math:`\gamma_1`.  This ordering
        matches the convention used by ``ShearYX2D`` / ``ShearYX2DIrregular`` and
        ``LensCalc.shear_yx_2d_via_hessian_from``.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the shear vectors are computed on.
        xp
            The array module (``numpy`` or ``jax.numpy``).
        """

        convergence = self.convergence_2d_from(grid=grid, xp=xp, **kwargs)

        gamma_2 = (
            -2
            * convergence.array
            * xp.divide(
                grid.array[:, 1] * grid.array[:, 0],
                grid.array[:, 1] ** 2 + grid.array[:, 0] ** 2,
            )
        )
        gamma_1 = -convergence.array * xp.divide(
            grid.array[:, 1] ** 2 - grid.array[:, 0] ** 2,
            grid.array[:, 1] ** 2 + grid.array[:, 0] ** 2,
        )

        shear_field = self.rotated_grid_from_reference_frame_from(
            grid=xp.vstack((gamma_2, gamma_1)).T, xp=xp, angle=self.angle(xp) * 2
        )

        return aa.VectorYX2DIrregular(values=shear_field, grid=grid)

    def convergence_func(self, grid_radius: float, xp=np) -> float:
        return self.einstein_radius_rescaled(xp) / grid_radius.array


class IsothermalSph(Isothermal):
    r"""Singular isothermal sphere (SIS) mass profile.

    The spherical limit of the SIE.  The convergence is:

    .. math::

        \kappa(r) = \frac{\theta_{\rm E}}{2r}

    where :math:`\theta_{\rm E}` is the Einstein radius and :math:`r` is the
    circular projected radius.

    Parameters
    ----------
    centre : (float, float)
        (y, x) arc-second coordinates of the profile centre.
    einstein_radius : float
        Einstein radius in arcseconds.

    References
    ----------
    Kormann, Schneider & Bartelmann (1994), A&A, 284, 285.
    Tessore & Metcalf (2015), A&A, 580, A79.
    """

    def __init__(
        self, centre: Tuple[float, float] = (0.0, 0.0), einstein_radius: float = 1.0
    ):
        """
        Represents a spherical isothermal density distribution, which is equivalent to the spherical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius
            The arc-second Einstein radius.
        """
        super().__init__(
            centre=centre, ell_comps=(0.0, 0.0), einstein_radius=einstein_radius
        )

    def axis_ratio(self, xp=np):
        return 1.0

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the potential on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        eta = self.elliptical_radii_grid_from(grid=grid, xp=xp, **kwargs)
        return 2.0 * self.einstein_radius_rescaled(xp) * eta

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
        return self._cartesian_grid_via_radial_from(
            grid=grid,
            xp=xp,
            radius=xp.full(grid.shape[0], 2.0 * self.einstein_radius_rescaled(xp)),
            **kwargs,
        )
