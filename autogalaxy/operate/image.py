"""
Mixin classes that add image-operation methods (PSF convolution, Fourier transform) to light objects.

The three mixin classes here are arranged in a hierarchy:

- `OperateImage` — methods that operate on a **single** image from a light object
  (e.g. a `LightProfile` or `Galaxy`).
- `OperateImageList` — extends `OperateImage` with methods that operate on a **list** of images,
  one per light profile component.
- `OperateImageGalaxies` — extends `OperateImageList` with methods that operate on a **dict** mapping
  each `Galaxy` to its image, preserving galaxy identity through the pipeline.

All three classes are mixins — they are not instantiated directly but are inherited by light objects
(`LightProfile`, `Galaxy`, `Galaxies`, `Tracer`) to expose a consistent API for blurring and Fourier
transforming images.
"""
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Optional

from autoarray import Array2D

if TYPE_CHECKING:
    from autogalaxy.galaxy.galaxy import Galaxy

import autoarray as aa


class OperateImage:
    """
    Packages methods which operate on the 2D image returned from the `image_2d_from` function of a light object
    (e.g. a `LightProfile`, `Galaxy`).

    The majority of methods apply data operators to the 2D image which perform tasks such as a 2D convolution or
    Fourier transform.

    The methods in `OperateImage` are inherited by light objects to provide a concise API.
    """

    def image_2d_from(
        self, grid: aa.Grid2D, xp=np, operated_only: Optional[bool] = None
    ) -> aa.Array2D:
        """
        Abstract method — return the 2D image of this light object at every coordinate on `grid`.

        Subclasses (e.g. `LightProfile`, `Galaxy`) implement this method to evaluate their surface
        brightness at each (y, x) coordinate.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where the image is evaluated.
        operated_only
            Controls which light profiles contribute. `None` (default) returns all profiles;
            `True` returns only `LightProfileOperated` images; `False` excludes them.
        """
        raise NotImplementedError

    def has(self, cls) -> bool:
        """
        Returns `True` if any attribute of this object is an instance of `cls`, else `False`.

        Parameters
        ----------
        cls
            The class type to search for.
        """
        raise NotImplementedError

    def _blurred_image_2d_from(
        self,
        image_2d: aa.Array2D,
        blurring_image_2d: aa.Array2D,
        psf: aa.Convolver,
        xp=np,
    ) -> aa.Array2D:
        """
        Convolve a 2D image with a PSF using the supplied `Convolver`.

        This is the internal helper used by `blurred_image_2d_from`. It does not handle the
        `LightProfileOperated` logic — the caller must do that separately.

        Parameters
        ----------
        image_2d
            The 2D image of the masked pixels to be convolved.
        blurring_image_2d
            The 2D image of the pixels just outside the mask whose light blurs into the masked region.
        psf
            The PSF convolver that performs the 2D convolution.
        """

        values = psf.convolved_image_from(
            image=image_2d, blurring_image=blurring_image_2d, xp=xp
        )
        return Array2D(values=values, mask=image_2d.mask)

    def blurred_image_2d_from(
        self,
        grid: aa.Grid2D,
        blurring_grid: aa.Grid2D,
        psf: aa.Convolver = None,
        xp=np,
    ) -> aa.Array2D:
        """
        Evaluate the light object's 2D image from a input 2D grid of coordinates and convolve it with a PSF.

        The input 2D grid may be masked, in which case values outside but near the edge of the mask will convolve light
        into the mask. A blurring grid is therefore required, which contains image pixels on the mask edge whose light
        is blurred into the light object's image by the PSF.

        The grid and blurring_grid must be a `Grid2D` objects so the evaluated image can be mapped to a uniform 2D
        array and binned up for convolution. They therefore cannot be `Grid2DIrregular` objects.

        If the PSF is oversampled (`convolve_over_sample_size > 1`), the images are instead evaluated on the
        grids' over-sampled coordinates (in per-pixel sub-block order, without binning) and convolved at the
        fine resolution, with the Convolver binning the result back to image resolution.

        Parameters
        ----------
        grid
            The 2D (y,x) coordinates of the (masked) grid, in its original geometric reference frame.
        psf
            The PSF the light object 2D image is convolved with.
        blurring_grid
            The 2D (y,x) coordinates neighboring the (masked) grid whose light is blurred into the image.
        """
        from autogalaxy.profiles.light.operated import (
            LightProfileOperated,
        )

        if psf.convolve_over_sample_size > 1:

            # grid.over_sampled is a Grid2DIrregular in per-pixel sub-block order, which the
            # over_sample decorator passes through unbinned — the oversampled Convolver's
            # input format.
            image_2d_not_operated = self.image_2d_from(
                grid=grid.over_sampled, xp=xp, operated_only=False
            )
            blurring_image_2d_not_operated = self.image_2d_from(
                grid=blurring_grid.over_sampled, xp=xp, operated_only=False
            )

            blurred_image_2d = psf.convolved_image_from(
                image=image_2d_not_operated,
                blurring_image=blurring_image_2d_not_operated,
                mask=grid.mask,
                xp=xp,
            )

        else:

            image_2d_not_operated = self.image_2d_from(
                grid=grid, xp=xp, operated_only=False
            )
            blurring_image_2d_not_operated = self.image_2d_from(
                grid=blurring_grid, xp=xp, operated_only=False
            )

            blurred_image_2d = self._blurred_image_2d_from(
                image_2d=image_2d_not_operated,
                blurring_image_2d=blurring_image_2d_not_operated,
                psf=psf,
                xp=xp,
            )

        if self.has(cls=LightProfileOperated):
            image_2d_operated = self.image_2d_from(grid=grid, xp=xp, operated_only=True)
            return blurred_image_2d + image_2d_operated

        return blurred_image_2d

    def padded_image_2d_from(self, grid, psf_shape_2d, xp=np):
        """
        Evaluate the light object's 2D image from a input 2D grid of padded coordinates, where this padding is
        sufficient to encapsulate all surrounding pixels that will blur light into the original image given the
        2D shape of the PSF's kernel.

        Convolving an unmasked 2D image with a PSF requires care, because at the edges of the 2D image the light
        profile values will not be evaluated beyond its edge, even though some of its light will be blurred into these
        edges.

        This function creates the padded image, such that the light profile is evaluated beyond the edge. The
        array will still require trimming to remove these additional pixels after convolution is performed.

        Parameters
        ----------
        grid
            The 2D (y,x) coordinates of the (masked) grid, in its original geometric reference frame.
        psf_shape_2d
            The 2D shape of the PSF the light object 2D image is convolved with.
        """
        padded_grid = grid.padded_grid_from(kernel_shape_native=psf_shape_2d)

        return self.image_2d_from(grid=padded_grid, xp=xp)

    def unmasked_blurred_image_2d_from(self, grid, psf):
        """
        Evaluate the light object's 2D image from a input 2D grid of coordinates and convolve it with a PSF, using a
        grid which is not masked.

        Convolving an unmasked 2D image with a PSF requires care, because at the edges of the 2D image the light
        profile values will not be evaluated beyond its edge, even though some of its light will be blurred into these
        edges.

        This function pads the grid first, such that the light profile is evaluated beyond the edge. The function then
        trims the array the image is evaluated on such that it is the original dimensions of the input 2D grid.

        The grid and blurring_grid must be a `Grid2D` objects so the evaluated image can be mapped to a uniform 2D
        array and binned up for convolution. They therefore cannot be `Grid2DIrregular` objects.

        Parameters
        ----------
        grid
            The 2D (y,x) coordinates of the (masked) grid, in its original geometric reference frame.
        psf
            The PSF the light object 2D image is convolved with.
        """
        padded_grid = grid.padded_grid_from(kernel_shape_native=psf.kernel.shape_native)

        padded_image_2d_not_operated = self.image_2d_from(
            grid=padded_grid, operated_only=False
        )

        # Required to make sure right 2D indexes are computed with update mask
        psf.slim_to_native_tuple = None

        padded_image_2d = padded_grid.mask.unmasked_blurred_array_from(
            padded_array=padded_image_2d_not_operated,
            psf=psf,
            image_shape=grid.mask.shape,
        )

        padded_image_2d_operated = self.image_2d_from(
            grid=padded_grid, operated_only=True
        )

        padded_image_2d_operated = padded_grid.mask.trimmed_array_from(
            padded_array=padded_image_2d_operated, image_shape=grid.mask.shape
        )

        return padded_image_2d + padded_image_2d_operated

    def visibilities_from(
        self, grid: aa.Grid2D, transformer: aa.type.Transformer, xp=np
    ) -> aa.Visibilities:
        """
        Evaluate the light object's 2D image from a input 2D grid of coordinates and transform this to an array of
        visibilities using a `autoarray.operators.transformer.Transformer` object and therefore a Fourier Transform.

        The input 2D grid may be masked, in which case values outside the mask are not evaluated. This does not impact
        the Fourier transform.

        The grid must be a `Grid2D` objects for certain Fourier transforms to be valid. It therefore cannot be a
        `Grid2DIrregular` objects.

        If the image is all zeros (e.g. because this light object has no light profiles, for example it is a
        `Galaxy` object with only mass profiles) the Fourier transformed is skipped for efficiency and a `Visibilities`
        object with all zeros is returned.

        Parameters
        ----------
        grid
            The 2D (y,x) coordinates of the (masked) grid, in its original geometric reference frame.
        transformer
            The **PyAutoArray** `Transformer` object describing how the 2D image is Fourier transformed to visiblities
            in the uv-plane.
        """

        from autogalaxy.profiles.light.abstract import LightProfile

        if self.has(cls=LightProfile) or isinstance(self, LightProfile):
            image_2d = self.image_2d_from(grid=grid, xp=xp)
            return transformer.visibilities_from(image=image_2d, xp=xp)

        return aa.Visibilities.zeros(shape_slim=(transformer.uv_wavelengths.shape[0],))


class OperateImageList(OperateImage):
    """
    Packages methods which operate on the list of 2D images returned from the `image_2d_list_from` function of a light
    object which contains multiple light profiles (e.g. a `Galaxy`).

    The majority of methods apply data operators to the list of 2D images which perform tasks such as a 2D convolution
    of Fourier transform.

    The methods in `OperateImageList` are inherited by light objects to provide a concise API.
    """

    def image_2d_list_from(self, grid: aa.Grid2D, operated_only: Optional[bool] = None):
        raise NotImplementedError

    def blurred_image_2d_list_from(
        self,
        grid: aa.Grid2D,
        blurring_grid: aa.Grid2D,
        psf: aa.Convolver,
    ) -> List[aa.Array2D]:
        """
        Evaluate the light object's list of 2D images from a input 2D grid of coordinates and convolve each image with
        a PSF.

        The input 2D grid may be masked, in which case values outside but near the edge of the mask will convolve light
        into the mask. A blurring grid is therefore required, which contains image pixels on the mask edge whose light
        is blurred into the light object's image by the PSF.

        The grid and blurring_grid must be a `Grid2D` objects so the evaluated image can be mapped to a uniform 2D
        array and binned up for convolution. They therefore cannot be `Grid2DIrregular` objects.

        Parameters
        ----------
        grid
            The 2D (y,x) coordinates of the (masked) grid, in its original geometric reference frame.
        psf
            The PSF the light object 2D image is convolved with.
        blurring_grid
            The 2D (y,x) coordinates neighboring the (masked) grid whose light is blurred into the image.
        """

        image_2d_operated_list = self.image_2d_list_from(grid=grid, operated_only=True)

        if psf.convolve_over_sample_size > 1:
            evaluation_grid = grid.over_sampled
            evaluation_blurring_grid = blurring_grid.over_sampled
        else:
            evaluation_grid = grid
            evaluation_blurring_grid = blurring_grid

        image_2d_not_operated_list = self.image_2d_list_from(
            grid=evaluation_grid, operated_only=False
        )
        blurring_image_2d_not_operated_list = self.image_2d_list_from(
            grid=evaluation_blurring_grid, operated_only=False
        )

        blurred_image_2d_list = []

        for i in range(len(image_2d_operated_list)):
            image_2d_not_operated = image_2d_not_operated_list[i]
            blurring_image_2d_not_operated = blurring_image_2d_not_operated_list[i]

            if psf.convolve_over_sample_size > 1:
                blurred_image_2d = psf.convolved_image_from(
                    image=image_2d_not_operated,
                    blurring_image=blurring_image_2d_not_operated,
                    mask=grid.mask,
                )
            else:
                blurred_image_2d = self._blurred_image_2d_from(
                    image_2d=image_2d_not_operated,
                    blurring_image_2d=blurring_image_2d_not_operated,
                    psf=psf,
                )

            image_2d_operated = image_2d_operated_list[i]

            blurred_image_2d_list.append(image_2d_operated + blurred_image_2d)

        return blurred_image_2d_list

    def unmasked_blurred_image_2d_list_from(
        self, grid: aa.Grid2D, psf: aa.Convolver
    ) -> List[aa.Array2D]:
        """
        Evaluate the light object's list of 2D images from a input 2D grid of coordinates and convolve it with a PSF,
        using a grid which is not masked.

        Convolving an unmasked 2D image with a PSF requires care, because at the edges of the 2D image the light
        profile values will not be evaluated beyond its edge, even though some of its light will be blurred into these
        edges.

        This function pads the grid first, such that the light profile is evaluated beyond the edge. The function then
        trims the array the image is evaluated on such that it is the original dimensions of the input 2D grid.

        The grid and blurring_grid must be a `Grid2D` objects so the evaluated image can be mapped to a uniform 2D
        array and binned up for convolution. They therefore cannot be `Grid2DIrregular` objects.

        Parameters
        ----------
        grid
            The 2D (y,x) coordinates of the (masked) grid, in its original geometric reference frame.
        psf
            The PSF the light object 2D image is convolved with.
        """
        padded_grid = grid.padded_grid_from(kernel_shape_native=psf.kernel.shape_native)

        padded_image_2d_list = self.image_2d_list_from(grid=padded_grid)

        unmasked_blurred_image_list = []

        for padded_image_2d in padded_image_2d_list:
            unmasked_blurred_array_2d = padded_grid.mask.unmasked_blurred_array_from(
                padded_array=padded_image_2d, psf=psf, image_shape=grid.mask.shape
            )

            unmasked_blurred_image_list.append(unmasked_blurred_array_2d)

        return unmasked_blurred_image_list

    def visibilities_list_from(
        self,
        grid: aa.Grid2D,
        transformer: aa.type.Transformer,
        xp=np,
    ) -> List[aa.Array2D]:
        """
        Evaluate the light object's list of 2D image from a input 2D grid of coordinates and transform each image to
        arrays of visibilities using a `autoarray.operators.transformer.Transformer` object and therefore a Fourier
        Transform.

        The input 2D grid may be masked, in which case values outside the mask are not evaluated. This does not impact
        the Fourier transform.

        The grid must be a `Grid2D` objects for certain Fourier transforms to be valid. It therefore cannot be a
        `Grid2DIrregular` objects.

        If the image is all zeros (e.g. because this light object has no light profiles, for example it is a
        `Galaxy` object with only mass profiles) the Fourier transformed is skipped for efficiency and a `Visibilities`
        object with all zeros is returned.

        Parameters
        ----------
        grid
            The 2D (y,x) coordinates of the (masked) grid, in its original geometric reference frame.
        transformer
            The **PyAutoArray** `Transformer` object describing how the 2D image is Fourier transformed to visiblities
            in the uv-plane.
        """
        image_2d_list = self.image_2d_list_from(grid=grid)

        visibilities_list = []

        for image_2d in image_2d_list:
            if not xp.any(image_2d.array):
                visibilities = aa.Visibilities.zeros(
                    shape_slim=(transformer.uv_wavelengths.shape[0],)
                )
            else:
                visibilities = transformer.visibilities_from(image=image_2d, xp=xp)

            visibilities_list.append(visibilities)

        return visibilities_list


class OperateImageGalaxies(OperateImageList):
    """
    Packages methods which using a list of galaxies returns a dictionary of their 2D images using the function
    `galaxy_image_2d_dict_from` (e.g. galaxies, a `Tracer` in the library **PyAutoLens**).

    The majority of methods apply data operators to the dictionary of 2D images which perform tasks such as a 2D
    convolution of Fourier transform. This retains the keys of the dictionary to maintain information on the galaxies.

    The methods in `OperateImageGalaxies` are inherited by light objects to provide a concise API.
    """

    def galaxy_image_2d_dict_from(
        self, grid: aa.Grid2D, xp=np, operated_only: Optional[bool] = None
    ) -> Dict[Galaxy, aa.Array2D]:
        raise NotImplementedError

    def galaxy_blurred_image_2d_dict_from(
        self, grid, psf, blurring_grid, xp=np
    ) -> Dict[Galaxy, aa.Array2D]:
        """
        Evaluate the light object's dictionary mapping galaixes to their corresponding 2D images and convolve each
        image with a PSF.

        The input 2D grid may be masked, in which case values outside but near the edge of the mask will convolve light
        into the mask. A blurring grid is therefore required, which contains image pixels on the mask edge whose light
        is blurred into the light object's image by the PSF.

        The grid and blurring_grid must be a `Grid2D` objects so the evaluated image can be mapped to a uniform 2D
        array and binned up for convolution. They therefore cannot be `Grid2DIrregular` objects.

        Parameters
        ----------
        grid
            The 2D (y,x) coordinates of the (masked) grid, in its original geometric reference frame.
        psf
            The PSF the light object 2D image is convolved with.
        blurring_grid
            The 2D (y,x) coordinates neighboring the (masked) grid whose light is blurred into the image.
        """

        if psf.convolve_over_sample_size > 1:
            evaluation_grid = grid.over_sampled
            evaluation_blurring_grid = blurring_grid.over_sampled
        else:
            evaluation_grid = grid
            evaluation_blurring_grid = blurring_grid

        galaxy_image_2d_not_operated_dict = self.galaxy_image_2d_dict_from(
            grid=evaluation_grid, operated_only=False, xp=xp
        )

        galaxy_blurring_image_2d_not_operated_dict = self.galaxy_image_2d_dict_from(
            grid=evaluation_blurring_grid, operated_only=False, xp=xp
        )

        galaxy_image_2d_operated_dict = self.galaxy_image_2d_dict_from(
            grid=grid, operated_only=True, xp=xp
        )

        galaxy_blurred_image_2d_dict = {}

        for galaxy_key in galaxy_image_2d_not_operated_dict.keys():
            image_2d_not_operated = galaxy_image_2d_not_operated_dict[galaxy_key]
            blurring_image_2d_not_operated = galaxy_blurring_image_2d_not_operated_dict[
                galaxy_key
            ]

            blurred_image_2d = psf.convolved_image_from(
                image=image_2d_not_operated,
                blurring_image=blurring_image_2d_not_operated,
                mask=grid.mask if psf.convolve_over_sample_size > 1 else None,
                xp=xp,
            )

            image_2d_operated = galaxy_image_2d_operated_dict[galaxy_key]

            galaxy_blurred_image_2d_dict[galaxy_key] = (
                image_2d_operated + blurred_image_2d
            )

        return galaxy_blurred_image_2d_dict

    def galaxy_visibilities_dict_from(
        self, grid, transformer, xp=np
    ) -> Dict[Galaxy, aa.Visibilities]:
        """
        Evaluate the light object's dictionary mapping galaixes to their corresponding 2D images and transform each
        image to arrays of visibilities using a `autoarray.operators.transformer.Transformer` object and therefore a
        Fourier Transform.

        The input 2D grid may be masked, in which case values outside the mask are not evaluated. This does not impact
        the Fourier transform.

        The grid must be a `Grid2D` objects for certain Fourier transforms to be valid. It therefore cannot be a
        `Grid2DIrregular` objects.

        If the image is all zeros (e.g. because this light object has no light profiles, for example it is a
        `Galaxy` object with only mass profiles) the Fourier transformed is skipped for efficiency and a `Visibilities`
        object with all zeros is returned.

        Parameters
        ----------
        grid
            The 2D (y,x) coordinates of the (masked) grid, in its original geometric reference frame.
        transformer
            The **PyAutoArray** `Transformer` object describing how the 2D image is Fourier transformed to visiblities
            in the uv-plane.
        """

        galaxy_image_2d_dict = self.galaxy_image_2d_dict_from(grid=grid, xp=xp)

        galaxy_visibilities_dict = {}

        for galaxy_key in galaxy_image_2d_dict.keys():
            image_2d = galaxy_image_2d_dict[galaxy_key]

            if not xp.any(image_2d.array):
                visibilities = aa.Visibilities.zeros(
                    shape_slim=(transformer.uv_wavelengths.shape[0],)
                )

            else:
                visibilities = transformer.visibilities_from(image=image_2d)

            galaxy_visibilities_dict[galaxy_key] = visibilities

        return galaxy_visibilities_dict
