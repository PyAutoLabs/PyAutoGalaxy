import numpy as np
from typing import Tuple

from autogalaxy.profiles.geometry_profiles import EllProfile


class Ellipse(EllProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        major_axis: float = 1.0,
    ):
        r"""
        class representing an ellispe, which is used to perform ellipse fitting to 2D data (e.g. an image).

        The elliptical components (`ell_comps`) of this profile are used to define the `axis_ratio` (q)
        and `angle` (\phi) :


        :math: \phi = (180/\pi) * arctan2(e_y / e_x) / 2
        :math: f = sqrt(e_y^2 + e_x^2)
        :math: q = (1 - f) / (1 + f)

        Where:

        e_y = y elliptical component = `ell_comps[0]`
        e_x = x elliptical component = `ell_comps[1]`
        q = axis_ratio (major_axis / minor_axis)

        This means that given an axis-ratio and angle the elliptical components can be computed as:

        :math: f = (1 - q) / (1 + q)
        :math: e_y = f * sin(2*\phi)
        :math: e_x = f * cos(2*\phi)
        """

        super().__init__(centre=centre, ell_comps=ell_comps)
        self.major_axis = major_axis

    @property
    def circular_radius(self) -> float:
        """
        The circumference of the circle that bounds the ellipse, assuming that the `major_axis` is the radius of the circle.
        """
        return 2.0 * np.pi * np.sqrt((2.0 * self.major_axis**2.0) / 2.0)

    @property
    def ellipticity(self) -> float:
        """
        The ellipticity of the ellipse, which is the factor by which the ellipse is offset from a circle.
        """
        return np.sqrt(1 - self.axis_ratio() ** 2.0)

    @property
    def minor_axis(self):
        """
        The minor-axis of the ellipse for a given major-axis, computed as:

        :math: b = a * sqrt(1 - e^2)

        Where:

        a = major-axis
        b = minor-axis
        e = ellipticity

        Parameters
        ----------
        major_axis
            The major-axis of the ellipse.

        Returns
        -------
        The minor-axis of the ellipse.
        """
        return self.major_axis * np.sqrt(1.0 - self.ellipticity**2.0)

    def total_points_from(self, pixel_scale: float) -> int:
        """
        Returns the total number of points on the ellipse based on the resolution of the data that the ellipse is
        fitted to and interpolated over.

        This value is chosen to ensure that the number of points computed matches the number of pixels in the data
        which the ellipse interpolates over. If the ellipse is bigger, the number of points increases in order to
        ensure that the ellipse uses more of the data's pixels.

        To determine the number of pixels the ellipse's circular radius in units of pixels is required. This is
        why `pixel_scale` is an input parameter of this function and other functions in this class.

        For computational efficiency, the maximum number of points is capped at 500, albeit few datasets will have
        more than 500 pixels in their 2D data.

        Parameters
        ----------
        pixel_scale
            The pixel scale of the data that the ellipse is fitted to and interpolated over.

        Returns
        -------
        The total number of points on the ellipse.

        """

        circular_radius_pixels = self.circular_radius / pixel_scale

        return np.min([500, int(np.round(circular_radius_pixels, 1))])

    def angles_from_x0_from(self, pixel_scale: float, n_i: int = 0, xp=np) -> np.ndarray:
        """
        Returns the angles from the x-axis to a discrete number of points ranging from 0.0 to 2.0 * np.pi radians.

        These angles are therefore not linked to the properties of the ellipse, they are just the angles from the
        x-axis to a series of points on a circle.

        They are subtracted by the `angle` of the ellipse to give the angles from the major-axis of the ellipse.

        The final angle, which is 2.0 * np.pi radians, is a repeat of the first angle of zero radians, therefore
        the final angle is removed.

        The number of angles computed is the minimum of 500 and the integer of the circular radius of the ellipse.
        This value is chosen to ensure that the number of angles computed matches the number of pixels in the
        data that the ellipse is being fitted to.

        Parameters
        ----------
        pixel_scale
            The pixel scale of the data that the ellipse is fitted to and interpolated over.
        n_i
            The number of points on the ellipse which hit a masked regions and cannot be computed, where this
            value is used to change the range of angles computed.

        Returns
        -------
        The angles from the x-axis to the points on the circle.
        """
        total_points = self.total_points_from(pixel_scale)

        return xp.linspace(0.0, 2.0 * np.pi, total_points + n_i)[:-1]

    def ellipse_radii_from_major_axis_from(
        self, pixel_scale: float, n_i: int = 0, xp=np
    ) -> np.ndarray:
        """
        Returns the distance from the centre of the ellipse to every point on the ellipse, which are called
        the ellipse radii.

        The order of the ellipse radii is counter-clockwise from the major-axis of the ellipse, which is given
        by the `angle` of the ellipse.

        Parameters
        ----------
        pixel_scale
            The pixel scale of the data that the ellipse is fitted to and interpolated over.
        n_i
            The number of points on the ellipse which hit a masked regions and cannot be computed, where this
            value is used to change the range of angles computed.

        Returns
        -------
        The ellipse radii from the major-axis of the ellipse.
        """

        angles_from_x0 = self.angles_from_x0_from(pixel_scale=pixel_scale, n_i=n_i, xp=xp)

        return xp.divide(
            self.major_axis * self.minor_axis,
            xp.sqrt(
                xp.add(
                    self.major_axis**2.0
                    * xp.sin(angles_from_x0 - self.angle_radians()) ** 2.0,
                    self.minor_axis**2.0
                    * xp.cos(angles_from_x0 - self.angle_radians()) ** 2.0,
                )
            ),
        )

    def x_from_major_axis_from(self, pixel_scale: float, n_i: int = 0, xp=np) -> np.ndarray:
        """
        Returns the x-coordinates of the points on the ellipse, starting from the x-coordinate of the major-axis
        of the ellipse after rotation by its `angle` and moving counter-clockwise.

        Parameters
        ----------
        pixel_scale
            The pixel scale of the data that the ellipse is fitted to and interpolated over.
        n_i
            The number of points on the ellipse which hit a masked regions and cannot be computed, where this
            value is used to change the range of angles computed.

        Returns
        -------
        The x-coordinates of the points on the ellipse.
        """

        angles_from_x0 = self.angles_from_x0_from(pixel_scale=pixel_scale, n_i=n_i, xp=xp)
        ellipse_radii_from_major_axis = self.ellipse_radii_from_major_axis_from(
            pixel_scale=pixel_scale, n_i=n_i, xp=xp
        )

        return ellipse_radii_from_major_axis * xp.cos(angles_from_x0) + self.centre[1]

    def y_from_major_axis_from(self, pixel_scale: float, n_i: int = 0, xp=np) -> np.ndarray:
        """
        Returns the y-coordinates of the points on the ellipse, starting from the y-coordinate of the major-axis
        of the ellipse after rotation by its `angle` and moving counter-clockwise.

        By default, the y-coordinates are multiplied by -1.0 and have the centre subtracted from them. This ensures
        that the convention of the y-axis increasing upwards is followed, meaning that `ell_comps` adopt the
        same definition as used for evaluating light profiles in PyAutoGalaxy.

        Parameters
        ----------
        pixel_scale
            The pixel scale of the data that the ellipse is fitted to and interpolated over.
        n_i
            The number of points on the ellipse which hit a masked regions and cannot be computed, where this
            value is used to change the range of angles computed.

        Returns
        -------
        The y-coordinates of the points on the ellipse.
        """
        angles_from_x0 = self.angles_from_x0_from(pixel_scale=pixel_scale, n_i=n_i, xp=xp)
        ellipse_radii_from_major_axis = self.ellipse_radii_from_major_axis_from(
            pixel_scale=pixel_scale, n_i=n_i, xp=xp
        )

        return (
            -1.0 * (ellipse_radii_from_major_axis * xp.sin(angles_from_x0))
            - self.centre[0]
        )

    def points_from_major_axis_from(
        self,
        pixel_scale: float,
        n_i: int = 0,
        xp=np,
    ) -> np.ndarray:
        """
        Returns the (y,x) coordinates of the points on the ellipse, starting from the major-axis of the ellipse
        and moving counter-clockwise.

        This is the format inputs into the inteprolation functions which match the ellipse to 2D data and enable
        us to determine how well the ellipse represents the data.

        Parameters
        ----------
        pixel_scale
            The pixel scale of the data that the ellipse is fitted to and interpolated over.
        n_i
            The number of points on the ellipse which hit a masked regions and cannot be computed, where this
            value is used to change the range of angles computed.

        Returns
        -------
        The (y,x) coordinates of the points on the ellipse.
        """

        x = self.x_from_major_axis_from(pixel_scale=pixel_scale, n_i=n_i, xp=xp)
        y = self.y_from_major_axis_from(pixel_scale=pixel_scale, n_i=n_i, xp=xp)

        if xp is np:
            idx = np.logical_or(np.isnan(x), np.isnan(y))
            if np.sum(idx) > 0:
                raise NotImplementedError()

        return xp.stack(arrays=(y, x), axis=-1)
