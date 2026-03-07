import numpy as np
from typing import Dict, List, Optional, Tuple, Type, Union

from autoconf import conf
import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.profiles.basis import Basis
from autogalaxy.profiles.light.linear import LightProfileLinear
from autogalaxy.operate.image import OperateImageGalaxies


class Galaxies(List, OperateImageGalaxies):
    def __init__(
        self,
        galaxies: List[Galaxy],
    ):
        """
        A collection of galaxies, used to perform operations on the galaxies as a group.

        It is common for a user to have multiple galaxies in a list, for which they may perform operations like
        creating the image of the light profiles or all galaxies or the potential of the mass profiles of all galaxies.

        Many of these calculations are straight forward, for example for the image of all galaxies we simply sum the
        images of each galaxy.

        However, there are more complex operations that can be performed on the galaxies as a group, for example
        computing the blured image of a group of galaxies where some galaxies have operated light profiles (meaning
        that PSF blurring is already applied to them and thus should be skipped) and another subset of galaxies have
        normal light profiles (meaning that PSF blurring should be applied to them).

        This calculation requires a careful set of operations to ensure that the PSF blurring is only applied to the
        subset of galaxies that do not have operated light profiles. This is an example of a calculation that is
        performed by the `Galaxies` class, simplifing the code required to perform this operation.

        Users may find that they often omit the `Galaxies` class and instead perform these operations in a more manual
        way. This is fine, but care must be taken to ensure that the operations are performed correctly.

        Parameters
        ----------
        galaxies
            The list of galaxies whose calculations are performed by this class.
        """

        super().__init__(galaxies)

    @property
    def redshift(self):
        return self[0].redshift

    def image_2d_list_from(
        self, grid: aa.type.Grid2DLike, xp=np, operated_only: Optional[bool] = None
    ) -> List[aa.Array2D]:
        """
        Returns a list of the 2D images for each galaxy from a 2D grid of Cartesian (y,x) coordinates.

        The image of each galaxy is computed by summing the images of all light profiles in that galaxy. If a galaxy
        has no light profiles, a numpy array of zeros is returned.

        For example, if there are 3 galaxies and only the first two have light profiles, the returned list of images
        will be the image of the first two galaxies. The image of the third galaxies will be a numpy array of zeros.

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
            The returned list from this function contains all light profile images, and they are never operated on
            (e.g. via the imaging PSF). However, inherited methods in the `autogalaxy.operate.image` package can
            apply these operations to the images, which may have the `operated_only` input passed to them. This input
            therefore is used to pass the `operated_only` input to these methods.
        """
        return [
            galaxy.image_2d_from(grid=grid, xp=xp, operated_only=operated_only)
            for galaxy in self
        ]

    @aa.grid_dec.to_array
    def image_2d_from(
        self, grid: aa.type.Grid2DLike, xp=np, operated_only: Optional[bool] = None
    ) -> aa.Array2D:
        """
        Returns the 2D image of all galaxies summed from a 2D grid of Cartesian (y,x) coordinates.

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
        return sum(
            self.image_2d_list_from(grid=grid, xp=xp, operated_only=operated_only)
        )

    def galaxy_image_2d_dict_from(
        self,
        grid: aa.type.Grid2DLike,
        xp=np,
        operated_only: Optional[bool] = None,
    ) -> {Galaxy: np.ndarray}:
        """
        Returns a dictionary associating every `Galaxy` object with its corresponding 2D image, using the instance
        of each galaxy as the dictionary keys.

        This object is used for adaptive-features, which use the image of each galaxy in a model-fit in order to
        adapt quantities like a pixelization or regularization scheme to the surface brightness of the galaxies being
        fitted.

        By inheriting from `OperateImageGalaxies` functions which apply operations of this dictionary are accessible,
        for example convolving every image with a PSF or applying a Fourier transform to create a galaxy-visibilities
        dictionary.

        Parameters
        ----------
        grid
            The 2D (y,x) coordinates of the (masked) grid, in its original geometric reference frame.

        Returns
        -------
        A dictionary associated every galaxy with its corresponding 2D image.
        """

        galaxy_image_2d_dict = dict()

        image_2d_list = self.image_2d_list_from(
            grid=grid, xp=xp, operated_only=operated_only
        )

        for galaxy_index, galaxy in enumerate(self):
            galaxy_image_2d_dict[galaxy] = image_2d_list[galaxy_index]

        return galaxy_image_2d_dict

    @aa.grid_dec.to_vector_yx
    def deflections_yx_2d_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ) -> np.ndarray:
        """
        Returns the summed 2D deflections angles of all galaxies from a 2D grid of Cartesian (y,x) coordinates.

        The deflections of each galaxy is computed by summing the deflections of all mass profiles in that galaxy. If a
        galaxy has no mass profiles, a numpy array of zeros is returned.

        This calculation does not account for multi-plane ray-tracing effects, it is simply the sum of the deflections
        of all galaxies. The `Tracer` class in PyAutoLens is required for this.

        For example, if there are 3 galaxies and only the first two have mass profiles, the returned list of deflections
        will be the deflections of the first two galaxies. The deflections of the third galaxies will be a numpy
        array of zeros.

        See the `autogalaxy.profiles.mass` package for details of how deflections are computed from a mass profile.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the deflections are evaluated.
        """
        return sum(map(lambda g: g.deflections_yx_2d_from(grid=grid, xp=xp), self))

    @aa.grid_dec.to_grid
    def traced_grid_2d_from(
        self, grid: aa.type.Grid2DLike, xp=np
    ) -> aa.type.Grid2DLike:
        """
        Trace this plane's grid_stacks to the next plane, using its deflection angles.
        """
        return grid - self.deflections_yx_2d_from(grid=grid, xp=xp)

    @aa.grid_dec.to_array
    def convergence_2d_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ) -> np.ndarray:
        """
        Returns the summed 2D convergence of all galaxies from a 2D grid of Cartesian (y,x) coordinates.

        The convergence of each galaxy is computed by summing the convergence of all mass profiles in that galaxy. If a
        galaxy has no mass profiles, a numpy array of zeros is returned.

        This calculation does not account for multi-plane ray-tracing effects, it is simply the sum of the convergence
        of all galaxies. The `Tracer` class in PyAutoLens is required for this.

        For example, if there are 3 galaxies and only the first two have mass profiles, the returned list of convergence
        will be the convergence of the first two galaxies. The convergence of the third galaxies will be a numpy
        array of zeros.

        See the `autogalaxy.profiles.mass` package for details of how convergence are computed from a mass profile.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the convergence are evaluated.
        """
        return sum(map(lambda g: g.convergence_2d_from(grid=grid, xp=xp), self))

    @aa.grid_dec.to_array
    def potential_2d_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ) -> np.ndarray:
        """
        Returns the summed 2D potential of all galaxies from a 2D grid of Cartesian (y,x) coordinates.

        The potential of each galaxy is computed by summing the potential of all mass profiles in that galaxy. If a
        galaxy has no mass profiles, a numpy array of zeros is returned.

        This calculation does not account for multi-plane ray-tracing effects, it is simply the sum of the potential
        of all galaxies. The `Tracer` class in PyAutoLens is required for this.

        For example, if there are 3 galaxies and only the first two have mass profiles, the returned list of potential
        will be the potential of the first two galaxies. The potential of the third galaxies will be a numpy
        array of zeros.

        See the `autogalaxy.profiles.mass` package for details of how potential are computed from a mass profile.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the potential are evaluated.
        """
        return sum(map(lambda g: g.potential_2d_from(grid=grid, xp=xp), self))

    def has(self, cls: Union[Type, Tuple[Type]]) -> bool:
        """
        Returns a bool specifying whether any of the galaxies has a certain class type.

        For example, for the input `cls=ag.LightProfile`, this function returns True if any galaxy has a
        light profile and false if no galaxy has a light profile.

        This function is used to check for mass profiles and specific types of profiles, like the linear light profile.

        Parameters
        ----------
        cls
            The class type of the galaxy which is checked for in the tracer.

        Returns
        -------
        True if any galaxy in the tracer has the input class type, else False.
        """
        return any(list(map(lambda galaxy: galaxy.has(cls=cls), self)))

    def cls_list_from(self, cls: Type) -> List:
        """
        Returns a list of objects in the galaxies which are an instance of the input `cls`.

        For example:

        - If the input is `cls=ag.LightProfile`, a list containing all light profiles of all galaxies is returned.

        Returns
        -------
            The list of objects in the galaxies that inherit from input `cls`.
        """
        cls_list = []

        for galaxy in self:
            if galaxy.has(cls=cls):
                for cls_galaxy in galaxy.cls_list_from(cls=cls):
                    cls_list.append(cls_galaxy)

        return cls_list

    def galaxies_with_cls_list_from(self, cls: Type) -> List[Galaxy]:
        return list(filter(lambda galaxy: galaxy.has(cls=cls), self))

    def galaxy_has_cls(self, cls: Type) -> bool:
        return any([galaxy.has(cls=cls) for galaxy in self])

    def extract_attribute(self, cls, attr_name):
        """
        Returns an attribute of a class in all galaxies as a `ValueIrregular` or `Grid2DIrregular` object.

        For example, if there is one galaxy with two light profiles and we want its axis-ratios, the following:

        `galaxies.extract_attribute(cls=LightProfile, name="axis_ratio")`

        would return:

        ArrayIrregular(values=[axis_ratio_0, axis_ratio_1])

        If a galaxy has three mass profiles and we want their centres, the following:

        `galaxies.extract_attribute(cls=MassProfile, name="centres")`

        would return:

        GridIrregular2D(grid=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1), (centre_y_2, centre_x_2)])

        This is used for visualization, for example plotting the centres of all mass profiles colored by their profile.
        """

        def extract(value, name):
            try:
                return getattr(value, name)
            except (AttributeError, IndexError):
                return None

        attributes = [
            extract(value, attr_name)
            for galaxy in self
            for value in galaxy.__dict__.values()
            if isinstance(value, cls)
        ]

        if attributes == []:
            return None
        elif isinstance(attributes[0], float):
            return aa.ArrayIrregular(values=attributes)
        elif isinstance(attributes[0], tuple):
            return aa.Grid2DIrregular(values=attributes)

    @property
    def perform_inversion(self) -> bool:
        """
        Returns a bool specifying whether this fit object performs an inversion.

        This is based on whether any of the galaxies have a `Pixelization` or `LightProfileLinear` object, in which
        case an inversion is performed.

        Returns
        -------
            A bool which is True if an inversion is performed.
        """
        if self.has(cls=(aa.Pixelization, LightProfileLinear)):
            return True
        elif self.has(cls=Basis):
            basis_list = self.cls_list_from(cls=Basis)
            for basis in basis_list:
                for light_profile in basis.profile_list:
                    if isinstance(light_profile, LightProfileLinear):
                        return True

        return False

    def plane_image_2d_from(
        self, grid: aa.type.Grid2DLike, zoom_to_brightest: bool = True
    ) -> aa.Array2D:
        return plane_image_from(
            galaxies=self,
            grid=grid.mask.derive_grid.all_false,
            zoom_to_brightest=zoom_to_brightest,
        )


def plane_image_from(
    galaxies: List[Galaxy],
    grid: aa.Grid2D,
    buffer: float = 1.0e-2,
    zoom_to_brightest: bool = True,
) -> aa.Array2D:
    """
    Returns the plane image of a list of galaxies, by summing their individual images.

    For lensing calculations performed by **PyAutoLens**, this function is used to return the unleensed image
    source-plane galaxies.

    By default, an adaptive grid is used to determine the grid that the images of the galaxies are computed on.
    This grid adapts its dimensions to capture the brightest regions of the image, ensuring that visualization of
    the plane-image is focused entirely on where the galaxies are brightest.

    This adaptive grid is based on determining the size of the grid that contains all pixels with an
    input % (typically 99%) of the total flux of the brightest pixel in the image.

    The adaptive grid can be disabled such that the input grid is used to compute the image of the galaxies.

    Parameters
    ----------
    galaxies
        The list of galaxies whose images are summed to compute the plane image.
    grid
        The grid of (y,x) coordinates for which the image of the galaxies is computed on, or from which the adaptive
        grid is derived.
    buffer
        The buffer around the adaptive grid that is used to ensure the image of the galaxies is not cut off.
    zoom_to_brightest
        If True, an adaptive grid is used to compute the image of the galaxies which zooms in on the brightest
        regions of the image. If False, the input grid is used.

    Returns
    -------
    The plane image of the galaxies, which is the sum of their individual images.
    """

    shape = grid.shape_native

    if zoom_to_brightest:
        try:
            image = sum(map(lambda g: g.image_2d_from(grid=grid), galaxies))
            image = image.native

            zoom_percent = conf.instance["visualize"]["general"]["zoom"][
                "plane_percent"
            ]

            fractional_value = np.max(image) * zoom_percent

            fractional_bool = image > fractional_value

            true_indices = np.argwhere(fractional_bool)

            y_max_pix = np.min(true_indices[:, 0])
            y_min_pix = np.max(true_indices[:, 0])
            x_min_pix = np.min(true_indices[:, 1])
            x_max_pix = np.max(true_indices[:, 1])

            grid = grid.native

            extent = (
                grid[0, x_min_pix][1] - buffer,
                grid[0, x_max_pix][1] + buffer,
                grid[y_min_pix, 0][0] - buffer,
                grid[y_max_pix, 0][0] + buffer,
            )

            extent = aa.util.geometry.extent_symmetric_from(extent=extent)

            pixel_scales = (
                float((extent[3] - extent[2]) / shape[0]),
                float((extent[1] - extent[0]) / shape[1]),
            )
            origin = ((extent[3] + extent[2]) / 2.0, (extent[1] + extent[0]) / 2.0)

            grid = aa.Grid2D.uniform(
                shape_native=grid.shape_native,
                pixel_scales=pixel_scales,
                origin=origin,
            )
        except ValueError:
            pass

    image = sum(map(lambda g: g.image_2d_from(grid=grid), galaxies))

    return aa.Array2D.no_mask(
        values=image.native, pixel_scales=grid.pixel_scales, origin=grid.origin
    )
