import numpy as np
from typing import Tuple
from autogalaxy.convert import multipole_comps_from, multipole_k_m_and_phi_m_from


from autogalaxy.ellipse.ellipse.ellipse import Ellipse


class EllipseMultipole:
    def __init__(
        self,
        m=4,
        multipole_comps: Tuple[float, float] = (0.0, 0.0),
    ):
        r"""
        class representing the multipole of an ellispe with, which is used to perform ellipse fitting to
        2D data (e.g. an image).

        The multipole is added to the (y,x) coordinates of an ellipse that are already computed via the `Ellipse` class.

        The addition of the multipole is performed as follows:

        :math: r_m = \sum_{i=1}^{m} \left( a_i \cos(i(\theta - \phi)) + b_i \sin(i(\theta - \phi)) \right)
        :math: y_m = r_m \sin(\theta)
        :math: x_m = r_m \cos(\theta)

        Where:

        m = The order of the multipole.
        r = The radial coordinate of the ellipse perturbed by the multipole.
        \phi = The angle of the ellipse.
        a = The amplitude of the cosine term of the multipole.
        b = The amplitude of the sine term of the multipole.
        y = The y-coordinate of the ellipse perturbed by the multipole.
        x = The x-coordinate of the ellipse perturbed by the multipole.
        """

        self.m = m
        self.multipole_comps = multipole_comps

    def get_shape_angle(
        self,
        ellipse: Ellipse,
        xp=np,
    ) -> float:
        """
        The shape angle is the offset between the angle of the ellipse and the angle of the multipole,
        this defines the shape that the multipole takes.

        In the case of the m=4 multipole, angles of 0 indicate pure diskiness, angles +- 45
        indicate pure boxiness.

        Parameters
        ----------
        ellipse
            The base ellipse profile that is perturbed by the multipole.
        xp
            The array module to use (default: numpy).

        Returns
        -------
        The angle between the ellipse and the multipole, in degrees between +- 180/m.
        The boundary case `angle == period/2.0` exactly maps to `-period/2.0` after wrap.
        """

        angle = (
            ellipse.angle(xp=xp)
            - multipole_k_m_and_phi_m_from(self.multipole_comps, self.m, xp=xp)[1]
        )
        period = 360.0 / self.m
        return xp.mod(angle + period / 2.0, period) - period / 2.0

    def points_perturbed_from(
        self, pixel_scale, points, ellipse: Ellipse, n_i: int = 0, xp=np
    ) -> np.ndarray:
        """
        Returns the (y,x) coordinates of the input points, which are perturbed by the multipole of the ellipse.

        Parameters
        ----------
        pixel_scale
            The pixel scale of the data that the ellipse is fitted to and interpolated over.
        points
            The (y,x) coordinates of the ellipse that are perturbed by the multipole.
        ellipse
            The ellipse that is perturbed by the multipole, which is used to compute the angles of the ellipse.
        xp
            The array module to use (default: numpy).

        Returns
        -------
        The (y,x) coordinates of the input points, which are perturbed by the multipole.
        """
        symmetry = 360 / self.m
        k_orig, phi_orig = multipole_k_m_and_phi_m_from(self.multipole_comps, self.m, xp=xp)
        comps_adjusted = multipole_comps_from(
            k_orig,
            symmetry
            - 2 * phi_orig
            + (symmetry - (ellipse.angle(xp=xp) - phi_orig)),  # Re-align light to match mass
            self.m,
            xp=xp,
        )

        # 1) compute cartesian (polar) angle
        theta = xp.arctan2(points[:, 0], points[:, 1])  # <- true polar angle

        # 2) multipole in that same frame
        delta_theta = self.m * (theta - ellipse.angle_radians(xp=xp))
        radial = comps_adjusted[1] * xp.cos(delta_theta) + comps_adjusted[0] * xp.sin(
            delta_theta
        )

        # 3) perturb along the true radial direction
        x = points[:, 1] + radial * xp.cos(theta)
        y = points[:, 0] + radial * xp.sin(theta)

        return xp.stack((y, x), axis=-1)


class EllipseMultipoleScaled(EllipseMultipole):
    def __init__(
        self,
        m=4,
        scaled_multipole_comps: Tuple[float, float] = (0.0, 0.0),
        major_axis: float = 1.0,
    ):
        r"""
        class representing the multipole of an ellipse, which is used to perform ellipse fitting to
        2D data (e.g. an image). This multipole is fit with its strength held relative to an ellipse with a
        major_axis of 1, allowing for a set of ellipse multipoles to be fit at different major axes but with
        the same scaled strength k/a.

        The scaled_multipole_comps (for all ellipses) are converted to a k value, which is then reset to
        its `true' value for a multipole at the given major axis value, which is then used to perturb an ellipse
        as per the normal `EllipseMultipole' class and below.

        The multipole is added to the (y,x) coordinates of an ellipse that are already computed via the `Ellipse` class.

        The addition of the multipole is performed as follows:

        :math: r_m = \sum_{i=1}^{m} \left( a_i \cos(i(\theta - \phi)) + b_i \sin(i(\theta - \phi)) \right)
        :math: y_m = r_m \sin(\theta)
        :math: x_m = r_m \cos(\theta)

        Where:

        m = The order of the multipole.
        r = The radial coordinate of the ellipse perturbed by the multipole.
        \phi = The angle of the ellipse.
        a = The amplitude of the cosine term of the multipole.
        b = The amplitude of the sine term of the multipole.
        y = The y-coordinate of the ellipse perturbed by the multipole.
        x = The x-coordinate of the ellipse perturbed by the multipole.

        Notes
        -----
        Unlike the previous implementation, the derivation of
        ``specific_multipole_comps`` from ``scaled_multipole_comps`` is now
        performed inside :meth:`points_perturbed_from` rather than at
        ``__init__`` time. This makes the class JAX-traceable: under
        ``jax.jit`` / ``jax.vmap``, ``scaled_multipole_comps`` arrive as
        JAX tracers, and a deferred derivation can thread ``xp`` through to
        the ``convert.py`` helpers. Pre-computing at ``__init__`` time would
        (a) hardcode ``np.*`` (breaks under JAX) and (b) cache stale values
        across ``vmap`` batch elements.
        """

        self.scaled_multipole_comps = scaled_multipole_comps
        self.major_axis = major_axis
        self.m = m

    def get_shape_angle(
        self,
        ellipse: Ellipse,
        xp=np,
    ) -> float:
        """
        The shape angle is the offset between the angle of the ellipse and the angle of the multipole,
        this defines the shape that the multipole takes.

        In the case of the m=4 multipole, angles of 0 indicate pure diskiness, angles +- 45
        indicate pure boxiness.

        This override derives (k_scaled, phi) directly from ``scaled_multipole_comps`` and
        ``major_axis`` rather than from ``self.multipole_comps`` (which is not set on this class).

        Parameters
        ----------
        ellipse
            The base ellipse profile that is perturbed by the multipole.
        xp
            The array module to use (default: numpy).

        Returns
        -------
        The angle between the ellipse and the multipole, in degrees between +- 180/m.
        The boundary case `angle == period/2.0` exactly maps to `-period/2.0` after wrap.
        """

        k_scaled, phi = multipole_k_m_and_phi_m_from(
            self.scaled_multipole_comps, self.m, xp=xp
        )
        angle = ellipse.angle(xp=xp) - phi
        period = 360.0 / self.m
        return xp.mod(angle + period / 2.0, period) - period / 2.0

    def points_perturbed_from(
        self, pixel_scale, points, ellipse: Ellipse, n_i: int = 0, xp=np
    ) -> np.ndarray:
        """
        Returns the (y,x) coordinates of the input points, which are perturbed by the multipole of the ellipse.

        The derivation of ``specific_multipole_comps`` from ``scaled_multipole_comps`` is performed
        here (rather than at ``__init__`` time) so that ``xp`` can be threaded through the
        ``convert.py`` helpers. This makes the method JAX-traceable when ``xp=jnp`` is passed.

        Parameters
        ----------
        pixel_scale
            The pixel scale of the data that the ellipse is fitted to and interpolated over.
        points
            The (y,x) coordinates of the ellipse that are perturbed by the multipole.
        ellipse
            The ellipse that is perturbed by the multipole, which is used to compute the angles of the ellipse.
        xp
            The array module to use (default: numpy).

        Returns
        -------
        The (y,x) coordinates of the input points, which are perturbed by the multipole.
        """
        k_scaled, phi = multipole_k_m_and_phi_m_from(
            multipole_comps=self.scaled_multipole_comps, m=self.m, xp=xp
        )
        k = k_scaled * self.major_axis

        symmetry = 360.0 / self.m
        comps_adjusted = multipole_comps_from(
            k,
            symmetry - 2 * phi + (symmetry - (ellipse.angle(xp=xp) - phi)),
            self.m,
            xp=xp,
        )

        # 1) compute cartesian (polar) angle
        theta = xp.arctan2(points[:, 0], points[:, 1])

        # 2) multipole in that same frame
        delta_theta = self.m * (theta - ellipse.angle_radians(xp=xp))
        radial = comps_adjusted[1] * xp.cos(delta_theta) + comps_adjusted[0] * xp.sin(
            delta_theta
        )

        # 3) perturb along the true radial direction
        x = points[:, 1] + radial * xp.cos(theta)
        y = points[:, 0] + radial * xp.sin(theta)

        return xp.stack(arrays=(y, x), axis=-1)
