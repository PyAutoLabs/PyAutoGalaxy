import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy import convert

from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.profiles.mass.total import PowerLaw
from autogalaxy import convert as al_convert
import jax.numpy as jnp


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
    def __init__(
        self,
        m=4,
        centre: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
        multipole_comps: Tuple[float, float] = (0.0, 0.0),
    ):
        r"""
        A multipole extension with multipole order M to the power-law total mass distribution.

        Quantities computed from this profile (e.g. deflections, convergence) are of only the multipole, and not the
        power-law mass distribution itself.

        The typical use case is therefore for the multipoles to be combined with a `PowerLaw` mass profile with the
        same parameters (see example below).

        When combined with a power-law, the functional form of the convergence is:

        .. math::
            \kappa(r, \phi) = \frac{1}{2} \left(\frac{\theta_{\rm E}^{\rm mass}}{r}\right)^{\gamma^{\rm mass} - 1}
            k^{\rm mass}_m \, \cos(m(\phi - \phi^{\rm mass}_m)) \, ,

        Where \\xi are elliptical coordinates calculated according to :class: SphProfile.

        The parameters :math: k^{\rm mass}_m and :math: \phi^{\rm mass}_are parameterized as elliptical components
        :math: (\epsilon_{\rm 1}^{\rm mp}\,\epsilon_{\rm 2}^{\rm mp}), which are given by:

        .. math::
                \phi^{\rm mass}_m = \frac{1}{m} \arctan{\frac{\epsilon_{\rm 2}^{\rm mp}}{\epsilon_{\rm 1}^{\rm mp}}}, \, \,
                k^{\rm mass}_m = \sqrt{{\epsilon_{\rm 1}^{\rm mp}}^2 + {\epsilon_{\rm 2}^{\rm mp}}^2} \, .

        This mass profile is described fully in the following paper: https://arxiv.org/abs/1302.5482

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius
            The arc-second Einstein radius.
        slope
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        multipole_comps
            The first and second ellipticity components of the multipole.

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
        if angle < -180 / self.m:
            angle += 360 / self.m
        elif angle > 180 / self.m:
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

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
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
    @aa.grid_dec.to_array
    @aa.grid_dec.transform
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

    @aa.grid_dec.to_array
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




class PowerLawMultipoleMultiBand(MassProfile):
    def __init__(
        self,
        m=4,
        centre: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
        magnitude: float = 0.0,
        position_angle: float = 0.0,
        shift_centre_0: float = 0.0,
        shift_centre_1: float = 0.0,
        shift_angle: float = 0.0,
        #multipole_comps: Tuple[float, float] = (0.0, 0.0),
    ):
        r"""
        A multipole extension with multipole order M to the power-law total mass distribution.

        Quantities computed from this profile (e.g. deflections, convergence) are of only the multipole, and not the
        power-law mass distribution itself.

        The typical use case is therefore for the multipoles to be combined with a `PowerLaw` mass profile with the
        same parameters (see example below).

        When combined with a power-law, the functional form of the convergence is:

        .. math::
            \kappa(r, \phi) = \frac{1}{2} \left(\frac{\theta_{\rm E}^{\rm mass}}{r}\right)^{\gamma^{\rm mass} - 1}
            k^{\rm mass}_m \, \cos(m(\phi - \phi^{\rm mass}_m)) \, ,

        Where \\xi are elliptical coordinates calculated according to :class: SphProfile.

        The parameters :math: k^{\rm mass}_m and :math: \phi^{\rm mass}_are parameterized as elliptical components
        :math: (\epsilon_{\rm 1}^{\rm mp}\,\epsilon_{\rm 2}^{\rm mp}), which are given by:

        .. math::
                \phi^{\rm mass}_m = \frac{1}{m} \arctan{\frac{\epsilon_{\rm 2}^{\rm mp}}{\epsilon_{\rm 1}^{\rm mp}}}, \, \,
                k^{\rm mass}_m = \sqrt{{\epsilon_{\rm 1}^{\rm mp}}^2 + {\epsilon_{\rm 2}^{\rm mp}}^2} \, .

        This mass profile is described fully in the following paper: https://arxiv.org/abs/1302.5482

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius
            The arc-second Einstein radius.
        slope
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        multipole_comps
            The first and second ellipticity components of the multipole.

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

        centre = (centre[0] + shift_centre_0, centre[1] + shift_centre_1)
        position_angle = position_angle + shift_angle

        super().__init__(centre=centre, ell_comps=(0.0, 0.0))

        self.m = int(m)

        self.einstein_radius = einstein_radius
        self.slope = slope

        self.multipole_comps = al_convert.multipole_comps_from(
            k_m=magnitude, phi_m=position_angle, m=self.m, xp=jnp
        )

        self.shift_centre_0 = shift_centre_0
        self.shift_centre_1 = shift_centre_1
        self.shift_angle = shift_angle

        self.magnitude_value = magnitude
        self.position_angle_value = position_angle

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
        if angle < -180 / self.m:
            angle += 360 / self.m
        elif angle > 180 / self.m:
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

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
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
        # # Qiuhan changed
        # k_m = self.magnitude_value
        # angle_m = self.position_angle_value * xp.asarray(np.pi / 180.0)

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
    @aa.grid_dec.to_array
    @aa.grid_dec.transform
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
        # # Qiuhan changed
        # k_m = self.magnitude_value
        # angle_m = self.position_angle_value * xp.asarray(np.pi / 180.0)

        return (
            1.0
            / 2.0
            * (self.einstein_radius / r) ** (self.slope - 1)
            * k_m
            * xp.cos(self.m * (angle - angle_m))
        )

    @aa.grid_dec.to_array
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
