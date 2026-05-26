import numpy as np

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile

from autogalaxy import convert


class ExternalShear(MassProfile):
    r"""
    Constant external shear field used in strong-lens mass models.

    The external shear represents the line-of-sight contribution to the lensing
    potential from mass external to the primary lens (e.g. nearby group members,
    large-scale structure).  Because this contribution is approximately uniform over
    the small angular extent of a strong-lens system, it is modelled as a constant
    shear with two Cartesian components :math:`(\gamma_1, \gamma_2)`.

    The lensing potential is:

    .. math::

        \psi(\boldsymbol{\theta}) =
            -\tfrac{1}{2} |\gamma| \, r^2 \cos\!\big(2(\varphi - \phi_\gamma)\big)

    The deflection field in a reference frame aligned with the shear direction is:

    .. math::

        \alpha_y = -|\gamma| \, y, \qquad \alpha_x = +|\gamma| \, x

    The convergence vanishes identically:

    .. math::

        \kappa = 0

    The shear magnitude and position angle are:

    .. math::

        |\gamma| = \sqrt{\gamma_1^2 + \gamma_2^2}, \qquad
        \phi_\gamma = \tfrac{1}{2} \mathrm{arctan2}(\gamma_2, \gamma_1)

    References
    ----------
    - Schneider, Ehlers & Falco 1992, *Gravitational Lenses* (Springer)
    """

    def __init__(self, gamma_1: float = 0.0, gamma_2: float = 0.0):
        r"""
        Constant external shear term used in strong-lens mass models.

        ``ExternalShear`` represents the line-of-sight contribution to the lensing potential from mass that is
        not part of the primary lens — typically nearby group/cluster members, large-scale structure, or
        unmodelled satellites.  Because this contribution is approximately uniform across the small angular
        extent of a strong-lens system, it is parameterised as a constant shear with two components,
        :math:`\gamma_1` and :math:`\gamma_2`, in the same convention as
        ``LensCalc.shear_yx_2d_via_hessian_from`` and ``ShearYX2DIrregular``:

        - :math:`\gamma_1` produces stretching along the x/y axes
        - :math:`\gamma_2` produces stretching along the diagonals

        The associated shear *magnitude* and *position angle* (degrees, anticlockwise from the +x axis) are:

        .. math::

            |\gamma| = \sqrt{\gamma_1^2 + \gamma_2^2}, \qquad
            \phi = \tfrac{1}{2} \, \mathrm{arctan2}(\gamma_2, \gamma_1).

        Note that the shear *position angle* lies in the direction of image stretching.  An external mass
        located in the *direction of compression* therefore appears at an angle offset by 90 degrees from
        :math:`\phi`.

        Parameters
        ----------
        gamma_1
            The :math:`\gamma_1` shear component.
        gamma_2
            The :math:`\gamma_2` shear component.
        """

        super().__init__(centre=(0.0, 0.0), ell_comps=(0.0, 0.0))
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2

    def magnitude(self, xp=np):
        r"""Returns the shear magnitude :math:`|\gamma| = \sqrt{\gamma_1^2 + \gamma_2^2}`."""
        return convert.shear_magnitude_from(
            gamma_1=self.gamma_1, gamma_2=self.gamma_2, xp=xp
        )

    def angle(self, xp=np):
        r"""Returns the shear position angle :math:`\phi = \tfrac{1}{2}\,\mathrm{arctan2}(\gamma_2, \gamma_1)`
        in degrees, in the [0, 180) convention used elsewhere in the package."""
        return convert.shear_angle_from(
            gamma_1=self.gamma_1, gamma_2=self.gamma_2, xp=xp
        )

    def convergence_func(self, grid_radius: float) -> float:
        return 0.0

    def average_convergence_of_1_radius(self):
        return 0.0

    @aa.decorators.to_array
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """A pure shear term has zero convergence at every grid point."""
        return xp.zeros(shape=grid.shape[0])

    @aa.decorators.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        r"""
        Returns the lensing potential of the constant external shear, given by:

        .. math::

            \psi(\theta) = -\tfrac{1}{2} |\gamma| \, r^2 \, \cos\!\big(2\,(\varphi - \phi_\gamma)\big)

        where :math:`r, \varphi` are the polar coordinates of ``grid`` and :math:`\phi_\gamma` is the shear
        position angle (offset by ``-90 deg`` to remain consistent with the deflection-angle convention used
        elsewhere in PyAutoLens).
        """
        shear_angle = (
            self.angle(xp) - 90
        )  # offset by -90 deg to match the deflection-angle convention used elsewhere
        phig = xp.deg2rad(shear_angle)
        shear_amp = self.magnitude(xp=xp)
        phicoord = xp.arctan2(grid.array[:, 0], grid.array[:, 1])
        rcoord = xp.sqrt(grid.array[:, 0] ** 2.0 + grid.array[:, 1] ** 2.0)

        return -0.5 * shear_amp * rcoord**2 * xp.cos(2 * (phicoord - phig))

    @aa.decorators.to_vector_yx
    @aa.decorators.transform(rotate_back=True)
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        r"""
        Returns the deflection angles of the constant external shear at each ``(y, x)`` arc-second
        coordinate.

        In the profile's rotated reference frame (where the shear position angle is aligned with the x-axis)
        the deflection reduces to:

        .. math::

            \alpha_y(y, x) = -|\gamma| \, y, \qquad \alpha_x(y, x) = +|\gamma| \, x.

        The ``@transform(rotate_back=True)`` decorator rotates the input grid into this aligned frame and
        rotates the resulting deflection vectors back into the original frame.  Because the shear is a spin-2
        field, the *position angle* :math:`\phi` of the shear is encoded in the rotation of the grid and not
        as an extra factor in the deflection formula above.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        xp
            The array module (``numpy`` or ``jax.numpy``).
        """
        deflection_y = -xp.multiply(self.magnitude(xp=xp), grid.array[:, 0])
        deflection_x = xp.multiply(self.magnitude(xp=xp), grid.array[:, 1])
        return xp.vstack((deflection_y, deflection_x)).T
