"""
Geometry profiles define the spatial geometry of light and mass profiles, including their centre coordinates and
elliptical orientation.

The `GeometryProfile`, `SphProfile` and `EllProfile` classes provide the base geometric transformations that all
light and mass profiles inherit, including translating a grid to the profile centre and rotating it to the
profile's position angle.
"""

import numpy as np

from typing import Optional, Tuple, Type

import autoarray as aa

from autogalaxy import convert
from autogalaxy.profiles import validate


class GeometryProfile:
    """
    An abstract geometry profile, which describes profiles with y and x centre Cartesian coordinates

    Parameters
    ----------
    centre
        The (y,x) arc-second coordinates of the profile centre.
    """

    def __init__(self, centre: Tuple[float, float] = (0.0, 0.0)):
        self.centre = centre

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return "{}\n{}".format(
            self.__class__.__name__,
            "\n".join(["{}: {}".format(k, v) for k, v in self.__dict__.items()]),
        )

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def has(self, cls: Type) -> bool:
        """
        Returns `True` if any attribute of this profile is an instance of the input class `cls`, else `False`.

        Parameters
        ----------
        cls
            The class type to search for amongst the profile's attributes.

        Returns
        -------
        bool
            `True` if any attribute is an instance of `cls`, else `False`.
        """
        return aa.util.misc.has(values=self.__dict__.values(), cls=cls)

    def transformed_to_reference_frame_grid_from(self, grid, xp=np, **kwargs):
        """
        Transform a grid of (y,x) coordinates to the reference frame of the profile.

        Subclasses override this to perform a translation to the profile centre and, for elliptical profiles,
        a rotation to the profile's position angle.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the original reference frame of the grid.
        """
        raise NotImplemented()

    def transformed_from_reference_frame_grid_from(self, grid, xp=np, **kwargs):
        """
        Transform a grid of (y,x) coordinates from the reference frame of the profile back to the original
        observer reference frame.

        Subclasses override this to reverse the translation to the profile centre and, for elliptical profiles,
        the rotation to the profile's position angle.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the reference frame of the profile.
        """
        raise NotImplemented()


class SphProfile(GeometryProfile):
    """
    A spherical profile, which describes profiles with y and x centre Cartesian coordinates.

    Parameters
    ----------
    centre
        The (y,x) arc-second coordinates of the profile centre.
    """

    def __init__(self, centre: Tuple[float, float] = (0.0, 0.0)):
        super().__init__(centre=centre)

    @aa.decorators.to_array
    def radial_grid_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs) -> np.ndarray:
        """
        Convert a grid of (y, x) coordinates, to their radial distances from the profile
        centre (e.g. :math: r = sqrt(x**2 + y**2)).

        Parameters
        ----------
        grid
            The grid of (y, x) coordinates which are converted to radial distances.
        """
        return xp.sqrt(xp.add(xp.square(grid.array[:, 0]), xp.square(grid.array[:, 1])))

    def angle_to_profile_grid_from(
        self, grid_angles: np.ndarray, xp=np, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a grid of angles, defined in degrees counter-clockwise from the positive x-axis, to a grid of
        angles between the input angles and the profile.

        Parameters
        ----------
        grid_angles
            The angle theta counter-clockwise from the positive x-axis to each coordinate in radians.
        """
        return xp.cos(grid_angles), xp.sin(grid_angles)

    @aa.decorators.to_grid
    def _cartesian_grid_via_radial_from(
        self,
        grid: aa.type.Grid2DLike,
        xp=np,
        radius: Optional[np.ndarray] = None,
        **kwargs,
    ) -> aa.type.Grid2DLike:
        """
        Convert a grid of (y,x) coordinates with their specified radial distances (e.g. :math: r = x**2 + y**2) to
        their original (y,x) Cartesian coordinates.

        Parameters
        ----------
        grid
            The (y, x) coordinates already translated to the reference frame of the profile.
        radius
            The circular radius of each coordinate from the profile center.
        """
        grid_angles = xp.arctan2(grid.array[:, 0], grid.array[:, 1])

        cos_theta, sin_theta = self.angle_to_profile_grid_from(
            grid_angles=grid_angles, xp=xp, **kwargs
        )

        return xp.multiply(radius[:, None], xp.vstack((sin_theta, cos_theta)).T)

    @aa.decorators.to_grid
    def transformed_to_reference_frame_grid_from(self, grid, xp=np, **kwargs):
        """
        Transform a grid of (y,x) coordinates to the reference frame of the profile.

        This performs a translation to the profile's `centre`.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the original reference frame of the grid.
        """
        return xp.subtract(grid.array, xp.array(self.centre))

    @aa.decorators.to_grid
    def transformed_from_reference_frame_grid_from(self, grid, xp=np, **kwargs):
        """
        Transform a grid of (y,x) coordinates from the reference frame of the profile to the original observer
        reference frame.

        This performs a translation from the profile's `centre`.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the reference frame of the profile.
        """
        return xp.add(grid.array, xp.array(self.centre))


class EllProfile(SphProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
    ):
        r"""
        An elliptical profile, which describes the geometry of profiles defined by an ellipse.

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

        For an input (y,x) grid of Cartesian coordinates this is used to compute the elliptical coordinates of a
        profile:

        .. math:: \\xi = q^{0.5} * ((y-y_c^2 + x-x_c^2 / q^2)^{0.5}

        Where:

        y_c = profile y centre = `centre[0]`
        x_c = profile x centre = `centre[1]`

        The majority of elliptical profiles use \\xi to compute their image.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second elliptical components of the profile.
        """
        super().__init__(centre=centre)

        validate.validate_ell_comps(ell_comps=ell_comps)
        self.ell_comps = ell_comps

    def axis_ratio(self, xp=np) -> float:
        """
        The ratio of the minor-axis to major-axis (b/a) of the ellipse defined by profile (0.0 > q > 1.0).
        """
        return convert.axis_ratio_from(ell_comps=self.ell_comps, xp=xp)

    def angle(self, xp=np) -> float:
        """
        The position angle in degrees of the major-axis of the ellipse defined by profile, defined counter clockwise
        from the positive x-axis (0.0 > angle > 180.0).
        """
        return convert.angle_from(ell_comps=self.ell_comps, xp=xp)

    def angle_radians(self, xp=np) -> float:
        """
        The position angle in radians of the major-axis of the ellipse defined by profile, defined counter clockwise
        from the positive x-axis (0.0 > angle > 2pi).
        """
        return xp.radians(self.angle(xp))

    @property
    def _cos_angle(self) -> float:
        return self._cos_and_sin_to_x_axis()[0]

    @property
    def _sin_angle(self) -> float:
        return self._cos_and_sin_to_x_axis()[1]

    def _cos_and_sin_to_x_axis(self, xp=np, **kwargs):
        """
        Determine the sin and cosine of the angle between the profile's ellipse and the positive x-axis,
        counter-clockwise.
        """
        angle_radians = xp.radians(self.angle(xp))
        return xp.cos(angle_radians), xp.sin(angle_radians)

    def angle_to_profile_grid_from(self, grid_angles, xp=np, **kwargs):
        """
        The angle between each angle theta on the grid and the profile, in radians.

        Parameters
        ----------
        grid_angles
            The angle theta counter-clockwise from the positive x-axis to each coordinate in radians.
        """
        theta_coordinate_to_profile = xp.add(grid_angles, -self.angle_radians(xp=xp))
        return xp.cos(theta_coordinate_to_profile), xp.sin(theta_coordinate_to_profile)

    @aa.decorators.to_grid
    def rotated_grid_from_reference_frame_from(
        self, grid, xp=np, angle: Optional[float] = None, **kwargs
    ):
        """
        Rotate a grid of (y,x) coordinates which have been transformed to the elliptical reference frame of a profile
        back to the original unrotated coordinate grid reference frame.

        Note that unlike the method `transformed_from_reference_frame_grid_from` the coordinates are not
        translated back to the profile's original centre.

        This routine is used after computing deflection angles in the reference frame of the profile, so that the
        deflection angles can be re-rotated to the frame of the original coordinates before performing ray-tracing.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the reference frame of an elliptical profile.
        angle
            Manually input an angle which is used instead of the profile's `angle` attribute. This is used in
            certain circumstances where the angle applied is different to the profile's `angle` attribute, for
            example weak lensing rotations which are typically twice that profile's `angle` attribute.
        """

        if angle is None:
            angle = self.angle(xp)

        return aa.util.geometry.transform_grid_2d_from_reference_frame(
            grid_2d=grid, centre=(0.0, 0.0), angle=angle, xp=xp
        )

    @aa.decorators.to_array
    def elliptical_radii_grid_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ) -> np.ndarray:
        """
        Convert a grid of (y,x) coordinates to their elliptical radii values: :math: (x^2 + (y^2/q))^0.5

        Parameters
        ----------
        grid
            The (y, x) coordinates in the reference frame of the elliptical profile.
        """
        return xp.sqrt(
            xp.add(
                xp.square(grid.array[:, 1]),
                xp.square(xp.divide(grid.array[:, 0], self.axis_ratio(xp))),
            )
        )

    @aa.decorators.to_array
    def eccentric_radii_grid_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ) -> np.ndarray:
        """
        Convert a grid of (y,x) coordinates to an eccentric radius: :math: axis_ratio^0.5 (x^2 + (y^2/q))^0.5

        This is used in certain light profiles define their half-light radii as a circular radius.

        If the coordinates have not been transformed to the profile's geometry (e.g. translated to the
        profile `centre`), this is performed automatically.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the reference frame of the elliptical profile.
        """

        grid_radii = self.elliptical_radii_grid_from(grid=grid, xp=xp, **kwargs)

        return xp.multiply(xp.sqrt(self.axis_ratio(xp)), grid_radii.array)

    @aa.decorators.to_grid
    def transformed_to_reference_frame_grid_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ) -> np.ndarray:
        """
        Transform a grid of (y,x) coordinates to the reference frame of the profile.

        This includes a translation to the profile's `centre` and a rotation using its `angle`.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the original reference frame of the grid.
        """
        if self.__class__.__name__.endswith("Sph"):
            return super().transformed_to_reference_frame_grid_from(grid=grid, xp=xp)
        return aa.util.geometry.transform_grid_2d_to_reference_frame(
            grid_2d=grid.array, centre=self.centre, angle=self.angle(xp), xp=xp
        )

    @aa.decorators.to_grid
    def transformed_from_reference_frame_grid_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ) -> aa.type.Grid2DLike:
        """
        Transform a grid of (y,x) coordinates from the reference frame of the profile to the original observer
        reference frame.

        This includes a translation from the profile's `centre` and a rotation using its `angle`.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the reference frame of the profile.
        """
        if self.__class__.__name__.endswith("Sph"):
            return super().transformed_from_reference_frame_grid_from(grid=grid, xp=xp)

        return aa.util.geometry.transform_grid_2d_from_reference_frame(
            grid_2d=grid.array, centre=self.centre, angle=self.angle(xp), xp=xp
        )

    def _eta_u(self, u, coordinates):
        return np.sqrt(
            (
                u
                * (
                    (coordinates[1] ** 2)
                    + (coordinates[0] ** 2 / (1 - (1 - self.axis_ratio(xp) ** 2) * u))
                )
            )
        )
