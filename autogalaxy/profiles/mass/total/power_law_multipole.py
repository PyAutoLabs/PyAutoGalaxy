import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy import convert

from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.profiles.mass.total import PowerLaw


def radial_and_angle_grid_from(
    grid: aa.type.Grid2DLike, centre: Tuple[float, float] = (0.0, 0.0), xp=np
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts the input grid of Cartesian (y,x) coordinates to their correspond radial and polar grids.

    Parameters
    ----------
    grid
        The grid of (y,x) arc-second coordinates that are converted to radial and polar values.
    centre
        The centre of the multipole profile.

    Returns
    -------
    The radial and polar coordinate grids of the input (y,x) Cartesian grid.
    """
    y, x = grid.array.T

    x_shifted = xp.subtract(x, centre[1])
    y_shifted = xp.subtract(y, centre[0])

    radial_grid = xp.sqrt(x_shifted**2 + y_shifted**2)

    angle_grid = xp.arctan2(y_shifted, x_shifted)

    return radial_grid, angle_grid


class PowerLawMultipole(MassProfile):
    r"""Angular multipole perturbation to a power-law total mass distribution.

    This profile provides only the multipole perturbation; it must be combined
    with a :class:`PowerLaw` profile that shares the same ``einstein_radius`` and
    ``slope`` parameters.  The multipole convergence is:

    .. math::

        \kappa_m(r, \phi) = \frac{1}{2}
            \left(\frac{\theta_{\rm E}}{r}\right)^{\gamma - 1}
            k_m \cos\!\bigl(m(\phi - \phi_m)\bigr)

    where :math:`m` is the multipole order, :math:`\gamma` is the power-law slope,
    :math:`k_m` is the multipole amplitude, and :math:`\phi_m` is the multipole
    orientation angle.  The amplitude and angle are parameterised via ellipticity
    components :math:`(\epsilon_1^{\rm mp},\, \epsilon_2^{\rm mp})`:

    .. math::

        k_m = \sqrt{{\epsilon_1^{\rm mp}}^2 + {\epsilon_2^{\rm mp}}^2}, \qquad
        \phi_m = \frac{1}{m} \arctan\!\frac{\epsilon_2^{\rm mp}}{\epsilon_1^{\rm mp}}

    The pure deflection-only nature of the perturbation means the convergence
    integrates to zero over all angular positions; the net mass contribution is
    therefore zero.

    Parameters
    ----------
    m : int
        Multipole order (e.g. 4 for the quadrupole-like ``m=4`` mode).
    centre : (float, float)
        (y, x) arc-second coordinates of the profile centre.
    einstein_radius : float
        Einstein radius in arcseconds (shared with the base :class:`PowerLaw`).
    slope : float
        Logarithmic density slope :math:`\gamma` (shared with the base :class:`PowerLaw`).
    multipole_comps : (float, float)
        Ellipticity-like components :math:`(\epsilon_1^{\rm mp},\, \epsilon_2^{\rm mp})`
        that encode the multipole amplitude and orientation.

    References
    ----------
    Chu, Hu & Kneib (2013), ApJ, 765, 134.  arXiv:1302.5482
    Evans & Witt (2003), MNRAS, 345, 1351.

    Examples
    --------

    mass = al.mp.PowerLaw(
        centre=(0.0, 0.0),
        ell_comps=(-0.1, 0.2),
        einstein_radius=1.0,
        slope=2.2
    )

    multipole = al.mp.PowerLawMultipole(
        centre=(0.0, 0.0),
        einstein_radius=1.0,
        slope=2.2,
        multipole_comps=(0.3, 0.2)
    )

    galaxy = al.Galaxy(
        redshift=0.5,
        mass=mass,
        multipole=multipole
    )

    grid=al.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.1)

    deflections = galaxy.deflections_yx_2d_from(
        grid=grid
    )
    """

    def __init__(
        self,
        m=4,
        centre: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
        multipole_comps: Tuple[float, float] = (0.0, 0.0),
    ):
        super().__init__(centre=centre, ell_comps=(0.0, 0.0))

        self.m = int(m)

        self.einstein_radius = einstein_radius
        self.slope = slope

        self.multipole_comps = multipole_comps

    def k_m_and_angle_m_from(self, xp=np) -> Tuple[float, float]:
        """
        Return the multipole normalization ``k_m`` and orientation angle ``angle_m``.

        The multipole normalization and angle are computed from the multipole component
        parameters ``(epsilon_1, epsilon_2)`` using
        :func:`convert.multipole_k_m_and_phi_m_from`. The returned angle is converted
        from degrees to radians.

        The numerical backend can be selected via the ``xp`` argument, allowing this
        method to be used with both NumPy and JAX (e.g. inside ``jax.jit``-compiled
        code).

        Parameters
        ----------
        xp
            Numerical backend module, typically ``numpy`` or ``jax.numpy``.

        Returns
        -------
        k_m
            The multipole normalization.
        angle_m
            The multipole orientation angle in radians.
        """
        k_m, angle_m = convert.multipole_k_m_and_phi_m_from(
            multipole_comps=self.multipole_comps, m=self.m, xp=xp
        )
        angle_m *= xp.asarray(np.pi / 180.0)

        return k_m, angle_m

    def get_shape_angle(
        self,
        base_profile: PowerLaw,
    ) -> float:
        """
        The shape angle is the offset between the angle of the ellipse and the angle of the multipole,
        this defines the shape that the multipole takes.

        In the case of the m=4 multipole, angles of 0 indicate pure diskiness, angles +- 45
        indicate pure boxiness.

        Parameters
        ----------
        base_profile
            The base power-law mass profile that is perturbed by the multipole.

        Returns
        -------
        The angle between the ellipse and the multipole, in degrees, between +- 180/m.
        """

        angle = (
            convert.angle_from(base_profile.ell_comps)
            - convert.multipole_k_m_and_phi_m_from(self.multipole_comps, self.m)[1]
        )
        while angle < -180 / self.m:
            angle += 360 / self.m
        while angle > 180 / self.m:
            angle -= 360 / self.m

        return angle

    def jacobian(
        self, a_r: np.ndarray, a_angle: np.ndarray, polar_angle_grid: np.ndarray, xp=np
    ) -> Tuple[np.ndarray, Tuple]:
        """
        The Jacobian transformation from polar to cartesian coordinates.

        Parameters
        ----------
        a_r
            Ask Aris
        a_angle
            Ask Aris
        polar_angle_grid
            The polar angle coordinates of the input (y,x) Cartesian grid of coordinates.
        """
        return (
            a_r * xp.sin(polar_angle_grid) + a_angle * xp.cos(polar_angle_grid),
            a_r * xp.cos(polar_angle_grid) - a_angle * xp.sin(polar_angle_grid),
        )

    @aa.decorators.to_vector_yx
    @aa.decorators.transform
    def deflections_yx_2d_from(
        self, grid: aa.type.Grid1D2DLike, xp=np, **kwargs
    ) -> np.ndarray:
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        For coordinates (0.0, 0.0) the analytic calculation of the deflection angle gives a NaN. Therefore,
        coordinates at (0.0, 0.0) are shifted slightly to (1.0e-8, 1.0e-8).

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        radial_grid, polar_angle_grid = radial_and_angle_grid_from(grid=grid, xp=xp)

        k_m, angle_m = self.k_m_and_angle_m_from(xp=xp)

        a_r = (
            -(
                (3.0 - self.slope)
                * self.einstein_radius ** (self.slope - 1.0)
                * radial_grid ** (2.0 - self.slope)
            )
            / (self.m**2.0 - (3.0 - self.slope) ** 2.0)
            * k_m
            * xp.cos(self.m * (polar_angle_grid - angle_m))
        )

        a_angle = (
            (
                self.m
                * self.einstein_radius ** (self.slope - 1.0)
                * radial_grid ** (2.0 - self.slope)
            )
            / (self.m**2.0 - (3.0 - self.slope) ** 2.0)
            * k_m
            * xp.sin(self.m * (polar_angle_grid - angle_m))
        )

        return xp.stack(
            self.jacobian(
                a_r=a_r, a_angle=a_angle, polar_angle_grid=polar_angle_grid, xp=xp
            ),
            axis=-1,
        )

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def convergence_2d_from(
        self, grid: aa.type.Grid1D2DLike, xp=np, **kwargs
    ) -> np.ndarray:
        """
        Returns the two dimensional projected convergence on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.
        """
        r, angle = radial_and_angle_grid_from(grid=grid, xp=xp)
        k_m, angle_m = self.k_m_and_angle_m_from(xp=xp)

        return (
            1.0
            / 2.0
            * (self.einstein_radius / r) ** (self.slope - 1)
            * k_m
            * xp.cos(self.m * (angle - angle_m))
        )

    @aa.decorators.to_array
    def potential_2d_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ) -> np.ndarray:
        """
        Calculate the potential on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        return xp.zeros(shape=grid.shape[0])
