from typing import Dict, List, Optional, Type, Union

import numpy as np

from autoconf.dictable import instance_as_dict, to_dict

import autoarray as aa
import autofit as af

from autogalaxy import exc
from autogalaxy.operate.image import OperateImageList
from autogalaxy.profiles.geometry_profiles import GeometryProfile
from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.light.linear import LightProfileLinear
from autogalaxy.profiles.light.snr.abstract import LightProfileSNR
from autogalaxy.profiles.mass.abstract.abstract import MassProfile


class Galaxy(af.ModelObject, OperateImageList):
    """
    @DynamicAttrs
    """

    def __init__(self, redshift: float, **kwargs):
        """
        Class representing a galaxy, which is composed of attributes used for fitting galaxies (e.g. light profiles,
        mass profiles, pixelizations, etc.).

        All *has_* methods retun `True` if galaxy has that attribute, `False` if not.

        Parameters
        ----------
        redshift
            The redshift of the galaxy.
        pixelization
            The pixelization of the galaxy used to reconstruct an observed image using an inversion.
        """
        super().__init__()
        self.redshift = redshift

        for name, val in kwargs.items():
            if isinstance(val, list):
                raise exc.GalaxyException(
                    "One or more of the input light / mass profiles has been passed to the Galaxy object"
                    "as a list."
                    ""
                    "The Galaxy object cannot accept a list of light / mass profiles. "
                    ""
                    "Instead, pass these objects as a dictionary, where the key of each dictionary entry is"
                    "the name of the profile and the value is the profile, e.g.:"
                    ""
                    "{bulge : al.lp.Sersic()}"
                    ""
                )

            setattr(self, name, val)

    def __hash__(self):
        return int(self.id)

    def __repr__(self):
        string = "Redshift: {}".format(self.redshift)

        if self.has(cls=LightProfile):
            string += "\nLight Profiles:\n{}".format(
                "\n".join(map(str, self.cls_list_from(cls=LightProfile)))
            )

        if self.has(cls=MassProfile):
            string += "\nMass Profiles:\n{}".format(
                "\n".join(map(str, self.cls_list_from(cls=MassProfile)))
            )

        if self.has(cls=aa.Pixelization):
            string += "\nPixelization:\n{}".format(str(self.pixelization))

        return string

    def __eq__(self, other):
        return self.dict() == other.dict()

    @property
    def profile_dict(self) -> Dict:
        return {
            key: value
            for key, value in self.__dict__.items()
            if isinstance(value, GeometryProfile) or isinstance(value, aa.Pixelization)
        }

    def dict(self) -> Dict:
        d = instance_as_dict(self)
        d["arguments"] = {
            **d.get("arguments", {}),
            **{name: to_dict(profile) for name, profile in self.profile_dict.items()},
        }
        return d

    def cls_list_from(self, cls: Type, cls_filtered: Optional[Type] = None) -> List:
        """
        Returns a list of objects in the galaxy which are an instance of the input `cls`.

        The optional `cls_filtered` input removes classes of an input instance type.

        For example:

        - If the input is `cls=ag.LightProfile`, a list containing all light profiles in the galaxy is returned.

        - If `cls=ag.LightProfile` and `cls_filtered=ag.LightProfileLinear`, a list of all light profiles
          excluding those which are linear light profiles will be returned.

        Parameters
        ----------
        cls
            The type of class that a list of instances of this class in the galaxy are returned for.
        cls_filtered
            A class type which is filtered and removed from the class list.

        Returns
        -------
            The list of objects in the galaxy that inherit from input `cls`.
        """
        return aa.util.misc.cls_list_from(
            values=self.__dict__.values(), cls=cls, cls_filtered=cls_filtered
        )

    def _radial_projected_shape_slim_from(self, grid: aa.type.Grid1D2DLike) -> int:
        """
        To make 1D plots (e.g. `image_1d_from()`) from an input 2D grid, one uses that 2D grid to radially project
        the coordinates across the profile's major-axis.

        This function computes the distance from the profile centre to the edge of this 2D grid.

        Because the centres of the galaxy's light and mass profiles can be offset from one another, thsi means the
        radially grid computed for each profile can have different shapes. Therefore plots using a `Galaxy` object
        use the biggest radial grid.

        If a 1D grid is input it returns the shape of this grid, as the grid itself defines the radial coordinates.

        Parameters
        ----------
        grid
            A 1D or 2D grid from which a 1D plot of the profile is to be created.
        """
        return max(
            [
                profile._radial_projected_shape_slim_from(grid=grid)
                for key, profile in self.profile_dict.items()
            ]
        )

    def grid_radial_from(self, grid, centre, angle):
        if isinstance(grid, aa.Grid1D) or isinstance(grid, aa.Grid2DIrregular):
            return grid

        radial_projected_shape_slim = self._radial_projected_shape_slim_from(grid=grid)

        return grid.grid_2d_radial_projected_from(
            centre=centre, angle=angle + 90, shape_slim=radial_projected_shape_slim
        )

    def image_2d_list_from(
        self, grid: aa.type.Grid2DLike, xp=np, operated_only: Optional[bool] = None
    ) -> List[aa.Array2D]:
        """
        Returns a list of the 2D images of the galaxy's light profiles from a 2D grid of Cartesian (y,x) coordinates.

        If the galaxy has no light profiles, a numpy array of zeros is returned.

        The images output by this function do not include instrument operations, such as PSF convolution (for imaging
        data) or a Fourier transform (for interferometer data).

        Inherited methods in the `autogalaxy.operate.image` package can apply these operations to the images.
        These functions may have the `operated_only` input passed to them, which is why this function includes
        the `operated_only` input.

        If the `operated_only` input is included, the function omits light profiles which are parents of
        the `LightProfileOperated` object, which signifies that the light profile represents emission that has
        already had the instrument operations (e.g. PSF convolution, a Fourier transform) applied to it and therefore
        that operation is not performed again.

        See the `autogalaxy.profiles.light` package for details of how images are computed from a light
        profile.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the image are evaluated.
        operated_only
            By default, the returned list contains all light profile images (irrespective of whether they have been
            operated on or not). If this input is included as a bool, only images which are or are not already
            operated are included in the list, with the images of other light profiles created as a numpy array of
            zeros.
        """
        return [
            light_profile.image_2d_from(grid=grid, xp=xp, operated_only=operated_only)
            for light_profile in self.cls_list_from(
                cls=LightProfile, cls_filtered=LightProfileLinear
            )
        ]

    @aa.grid_dec.to_array
    def image_2d_from(
        self,
        grid: aa.type.Grid2DLike,
        xp=np,
        operated_only: Optional[bool] = None,
    ) -> Union[np.ndarray, aa.Array2D]:
        """
        Returns the 2D image of all galaxy light profiles summed from a 2D grid of Cartesian (y,x) coordinates.

        This function first computes the image of each galaxy, via the function `image_2d_list_from`. The
        images are then summed to give the overall image of the galaxies.

        Refer to the function `image_2d_list_from` for a full description of the calculation and how the `operated_only`
        input is used.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the image are evaluated.
        operated_only
            The returned list from this function contains all light profile images, and they are never operated on
            (e.g. via the imaging PSF). However, inherited methods in the `autogalaxy.operate.image` package can
            apply these operations to the images, which may have the `operated_only` input passed to them. This input
            therefore is used to pass the `operated_only` input to these methods.
        """

        if (
            len(self.cls_list_from(cls=LightProfile, cls_filtered=LightProfileLinear))
            > 0
        ):
            return sum(
                self.image_2d_list_from(grid=grid, xp=xp, operated_only=operated_only)
            )
        return xp.zeros((grid.shape[0],))

    @aa.grid_dec.to_vector_yx
    def deflections_yx_2d_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ) -> np.ndarray:
        """
        Returns the summed 2D deflection angles of the galaxy's mass profiles from a 2D grid of Cartesian (y,x)
        coordinates.

        If the galaxy has no mass profiles, a numpy array of zeros is returned.

        See the `autogalaxy.profiles.mass` package for details of how deflection angles are computed from a
        mass profile.

        The decorator `to_vector_yx` converts the output arrays from ndarrays to a `VectorYX2D` data structure
        using the input `grid`'s attributes.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the deflection angles are evaluated.
        """
        if self.has(cls=MassProfile):
            return sum(
                map(
                    lambda p: p.deflections_yx_2d_from(grid=grid, xp=xp),
                    self.cls_list_from(cls=MassProfile),
                )
            )

        return xp.zeros((grid.shape[0], 2))

    @aa.grid_dec.to_array
    def convergence_2d_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ) -> np.ndarray:
        """
        Returns the summed 2D convergence of the galaxy's mass profiles from a 2D grid of Cartesian (y,x) coordinates.

        If the galaxy has no mass profiles, a numpy array of zeros is returned.

        See the `autogalaxy.profiles.mass` package for details of how convergences are computed from a mass
        profile.

        The decorator `grid_2d_to_structure` converts the output arrays from ndarrays to an `Array2D` data structure
        using the input `grid`'s attributes.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the convergence are evaluated.
        """
        if self.has(cls=MassProfile):
            return sum(
                map(
                    lambda p: p.convergence_2d_from(grid=grid, xp=xp),
                    self.cls_list_from(cls=MassProfile),
                )
            )

        return xp.zeros((grid.shape[0],))

    @aa.grid_dec.to_grid
    def traced_grid_2d_from(
        self, grid: aa.type.Grid2DLike, xp=np
    ) -> aa.type.Grid2DLike:
        """
        Trace an input grid using the galaxy's its deflection angles.
        """
        if isinstance(grid, aa.Grid2D):
            return aa.Grid2D(
                values=grid - self.deflections_yx_2d_from(grid=grid, xp=xp),
                mask=grid.mask,
                over_sample_size=grid.over_sample_size,
                over_sampled=grid.over_sampled
                - self.deflections_yx_2d_from(grid=grid.over_sampled, xp=xp),
                over_sampler=grid.over_sampler,
            )

        return grid - self.deflections_yx_2d_from(grid=grid, xp=xp)

    @aa.grid_dec.to_array
    def potential_2d_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ) -> np.ndarray:
        """
        Returns the summed 2D potential of the galaxy's mass profiles from a 2D grid of Cartesian (y,x) coordinates.

        If the galaxy has no mass profiles, a numpy array of zeros is returned.

        See the `autogalaxy.profiles.mass` package for details of how potentials are computed from a mass
        profile.

        The decorator `grid_2d_to_structure` converts the output arrays from ndarrays to an `Array2D` data structure
        using the input `grid`'s attributes.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the potential are evaluated.
        """
        if self.has(cls=MassProfile):
            return sum(
                map(
                    lambda p: p.potential_2d_from(grid=grid, xp=xp),
                    self.cls_list_from(cls=MassProfile),
                )
            )
        return xp.zeros((grid.shape[0],))

    @property
    def half_light_radius(self):
        return None

    def extract_attribute(self, cls, attr_name):
        """
        Returns an attribute of a class and its children profiles in the galaxy as a `ValueIrregular`
        or `Grid2DIrregular` object.

        For example, if a galaxy has two light profiles and we want the `LightProfile` axis-ratios, the following:

        `galaxy.extract_attribute(cls=LightProfile, name="axis_ratio"`

        would return:

        ArrayIrregular(values=[axis_ratio_0, axis_ratio_1])

        If a galaxy has three mass profiles and we want the `MassProfile` centres, the following:

        `galaxy.extract_attribute(cls=MassProfile, name="centres"`

         would return:

        GridIrregular2D(grid=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1), (centre_y_2, centre_x_2)])

        This is used for visualization, for example plotting the centres of all light profiles colored by their profile.
        """

        def extract(value, name):
            try:
                return getattr(value, name)
            except (AttributeError, IndexError):
                return None

        attributes = [
            extract(value, attr_name)
            for value in self.__dict__.values()
            if isinstance(value, cls)
        ]

        attributes = list(filter(None, attributes))

        if attributes == []:
            return None
        elif isinstance(attributes[0], float):
            return aa.ArrayIrregular(values=attributes)
        elif isinstance(attributes[0], tuple):
            return aa.Grid2DIrregular(values=attributes)

    def luminosity_within_circle_from(self, radius: float):
        """
        Returns the total luminosity of the galaxy's light profiles within a circle of specified radius.

        See `light_profile.luminosity_within_circle` for details of how this is performed.

        Parameters
        ----------
        radius
            The radius of the circle to compute the dimensionless mass within.
        unit_luminosity
            The unit_label the luminosity is returned in {esp, counts}.
        exposure_time
            The exposure time of the observation, which converts luminosity from electrons per second unit_label to counts.
        """
        if self.has(cls=LightProfile):
            return sum(
                map(
                    lambda p: p.luminosity_within_circle_from(radius=radius),
                    self.cls_list_from(
                        cls=LightProfile, cls_filtered=LightProfileLinear
                    ),
                )
            )

    def mass_angular_within_circle_from(self, radius: float):
        """
        Integrate the mass profiles's convergence profile to compute the total mass within a circle of \
        specified radius. This is centred on the mass profile.

        The following unit_label for mass can be specified and output:

        - Dimensionless angular unit_label (default) - 'angular'.
        - Solar masses - 'angular' (multiplies the angular mass by the critical surface mass density).

        Parameters
        ----------
        radius : dim.Length
            The radius of the circle to compute the dimensionless mass within.
        unit_mass
            The unit_label the mass is returned in {angular, angular}.
        critical_surface_density or None
            The critical surface mass density of the strong lens configuration, which converts mass from angulalr \
            unit_label to phsical unit_label (e.g. solar masses).
        """
        if self.has(cls=MassProfile):
            return sum(
                map(
                    lambda p: p.mass_angular_within_circle_from(radius=radius),
                    self.cls_list_from(cls=MassProfile),
                )
            )
        else:
            raise exc.GalaxyException(
                "You cannot perform a mass-based calculation on a galaxy which does not have a mass-profile"
            )

    def set_snr_of_snr_light_profiles(
        self,
        grid: aa.type.Grid2DLike,
        exposure_time: float,
        background_sky_level: float = 0.0,
        psf: Optional[aa.Convolver] = None,
    ):
        """
        Iterate over every galaxy finding all `LightProfileSNR` light profiles and set their `intensity` values to
        values which give their input `signal_to_noise_ratio` value, which is performed as follows:

        - Evaluate the image of each light profile on the input grid.
        - Blur this image with a PSF, if included.
        - Take the value of the brightest pixel.
        - Use an input `exposure_time` and `background_sky` (e.g. from the `SimulatorImaging` object) to determine
          what value of `intensity` gives the desired signal to noise ratio for the image.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the original reference frame of the grid.
        exposure_time
            The exposure time of the simulated imaging.
        background_sky_level
            The level of the background sky of the simulated imaging.
        psf
            The psf of the simulated imaging which can change the S/N of the light profile due to spreading out
            the emission.
        """
        for light_profile in self.cls_list_from(cls=LightProfile):
            if isinstance(light_profile, LightProfileSNR):
                light_profile.set_intensity_from(
                    grid=grid,
                    exposure_time=exposure_time,
                    background_sky_level=background_sky_level,
                    psf=psf,
                )
