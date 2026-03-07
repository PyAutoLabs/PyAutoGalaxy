import numpy as np
from typing import Dict, List, Optional

from autoconf import cached_property

import autoarray as aa

from autogalaxy.abstract_fit import AbstractFitInversion
from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.galaxy.galaxies import Galaxies
from autogalaxy.galaxy.to_inversion import GalaxiesToInversion
from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.light.linear import LightProfileLinear
from autogalaxy.profiles.light.operated.abstract import LightProfileOperated

from autogalaxy import exc


class FitImaging(aa.FitImaging, AbstractFitInversion):
    def __init__(
        self,
        dataset: aa.Imaging,
        galaxies: List[Galaxy],
        dataset_model: Optional[aa.DatasetModel] = None,
        adapt_images: Optional[AdaptImages] = None,
        settings: aa.Settings = None,
        xp=np,
    ):
        """
        Fits an imaging dataset using a list of galaxies.

        The fit performs the following steps:

        1) Compute the sum of all images of galaxy light profiles in the list of galaxies.

        2) Blur this with the imaging PSF to created the `blurred_image`.

        3) Subtract this image from the `data` to create the `profile_subtracted_image`.

        4) If the galaxies list has any linear algebra objects (e.g. linear light profiles, a pixelization / regulariation)
           fit the `profile_subtracted_image` with these objects via an inversion.

        5) Compute the `model_data` as the sum of the `blurred_image` and `reconstructed_data` of the inversion (if
           an inversion is not performed the `model_data` is only the `blurred_image`.

        6) Subtract the `model_data` from the data and compute the residuals, chi-squared and likelihood via the
           noise-map (if an inversion is performed the `log_evidence`, including additional terms describing the linear
           algebra solution, is computed).

        When performing a `model-fit`via an `AnalysisImaging` object the `figure_of_merit` of this object
        is called and returned in the `log_likelihood_function`.

        Parameters
        ----------
        dataset
            The imaging dataset which is fitted by the galaxies.
        galaxies
            The galaxies whose light profile images are used to fit the imaging data.
        dataset_model
            Attributes which allow for parts of a dataset to be treated as a model (e.g. the background sky level).
        adapt_images
            Contains the adapt-images which are used to make a pixelization's mesh and regularization adapt to the
            reconstructed galaxy's morphology.
        settings
            Settings controlling how an inversion is fitted for example which linear algebra formalism is used.
        """

        self.galaxies = Galaxies(galaxies=galaxies)

        super().__init__(
            dataset=dataset,
            dataset_model=dataset_model,
            xp=xp,
        )
        AbstractFitInversion.__init__(
            self=self,
            model_obj=self.galaxies,
            settings=settings,
            xp=xp,
        )

        self.adapt_images = adapt_images
        self.settings = settings or aa.Settings()

    @property
    def blurred_image(self) -> aa.Array2D:
        """
        Returns the image of the light profiles of all galaxies in the fit, convolved with the imaging dataset's PSF.

        If the galaxies do not have any light profiles, the image is computed bypassing the convolution routine
        altogether.
        """

        if len(self.galaxies.cls_list_from(cls=LightProfile)) == len(
            self.galaxies.cls_list_from(cls=LightProfileOperated)
        ):
            return self.galaxies.image_2d_from(
                grid=self.grids.lp,
                xp=self._xp,
            )

        return self.galaxies.blurred_image_2d_from(
            grid=self.grids.lp,
            psf=self.dataset.psf,
            blurring_grid=self.grids.blurring,
            xp=self._xp,
        )

    @property
    def profile_subtracted_image(self) -> aa.Array2D:
        """
        Returns the dataset's image data with all blurred light profile images in the fit subtracted.
        """
        return self.data - self.blurred_image

    @property
    def galaxies_to_inversion(self) -> GalaxiesToInversion:
        dataset = aa.DatasetInterface(
            data=self.profile_subtracted_image,
            noise_map=self.noise_map,
            grids=self.grids,
            psf=self.dataset.psf,
            sparse_operator=self.dataset.sparse_operator,
        )

        return GalaxiesToInversion(
            dataset=dataset,
            galaxies=self.galaxies,
            adapt_images=self.adapt_images,
            settings=self.settings,
            xp=self._xp,
        )

    @cached_property
    def inversion(self) -> Optional[aa.AbstractInversion]:
        """
        If the galaxies have linear objects which are used to fit the data (e.g. a linear light profile / pixelization)
        this function returns a linear inversion, where the flux values of these objects (e.g. the `intensity`
        of linear light profiles) are computed via linear matrix algebra.

        The data passed to this function is the dataset's image with all light profile images of the galaxies subtracted,
        ensuring that the inversion only fits the data with ordinary light profiles subtracted.
        """

        if self.perform_inversion:
            return self.galaxies_to_inversion.inversion

    @property
    def model_data(self) -> aa.Array2D:
        """
        Returns the model-image that is used to fit the data.

        If the galaxies do not have any linear objects and therefore omits an inversion, the model data is the
        sum of all light profile images blurred with the PSF.

        If a inversion is included it is the sum of this image and the inversion's reconstruction of the image.
        """

        if self.perform_inversion:
            return (
                self.blurred_image + self.inversion.mapped_reconstructed_operated_data
            )

        return self.blurred_image

    @property
    def galaxy_image_dict(self) -> Dict[Galaxy, np.ndarray]:
        """
        A dictionary which associates every galaxy in the fit with its image before operation (e.g. no PSF convolution
        or NUFFT performed).

        This image is the image of the sum of:

        - The images of all ordinary light profiles summed before any operation is performed on them.
        - The images of all linear objects (e.g. linear light profiles / pixelizations), where the images are solved
          for first via the inversion.

        This dictionary is used to output to .fits file the galaxy images.
        """

        galaxy_image_2d_dict = self.galaxies.galaxy_image_2d_dict_from(
            grid=self.grids.lp, xp=self._xp
        )

        galaxy_linear_obj_image_dict = self.galaxy_linear_obj_data_dict_from(
            use_operated=False,
        )

        return {**galaxy_image_2d_dict, **galaxy_linear_obj_image_dict}

    @property
    def galaxy_model_image_dict(self) -> Dict[Galaxy, np.ndarray]:
        """
        A dictionary which associates every galaxy in the fit with its `model_image`.

        This image is the image of the sum of:

        - The images of all ordinary light profiles summed and convolved with the imaging data's PSF.
        - The images of all linear objects (e.g. linear light profiles / pixelizations), where the images are solved
          for first via the inversion.

        For modeling, this dictionary is used to set up the `adapt_images` that adapt certain pixelizations to the
        data being fitted.
        """

        galaxy_blurred_image_2d_dict = self.galaxies.galaxy_blurred_image_2d_dict_from(
            grid=self.grids.lp,
            psf=self.dataset.psf,
            blurring_grid=self.grids.blurring,
            xp=self._xp,
        )

        galaxy_linear_obj_image_dict = self.galaxy_linear_obj_data_dict_from(
            use_operated=True,
        )

        return {**galaxy_blurred_image_2d_dict, **galaxy_linear_obj_image_dict}

    @property
    def subtracted_images_of_galaxies_dict(self) -> Dict[Galaxy, aa.Array2D]:
        """
        A dictionary associating every galaxy with its `subtracted_image`.

        A subtracted image of a galaxy is the data where all other galaxy images are subtracted from it, therefore
        showing how a galaxy appears in the data in the absence of all other galaxies.

        This is used to visualize the contribution of each galaxy in the data.
        """

        subtracted_images_of_galaxies_dict = {}

        model_images_of_galaxies_list = self.model_images_of_galaxies_list

        for galaxy_index in range(len(self.galaxies)):
            other_galaxies_model_images = [
                model_image
                for i, model_image in enumerate(model_images_of_galaxies_list)
                if i != galaxy_index
            ]

            subtracted_image = self.data - sum(other_galaxies_model_images)

            subtracted_images_of_galaxies_dict[self.galaxies[galaxy_index]] = (
                subtracted_image
            )

        return subtracted_images_of_galaxies_dict

    @property
    def model_images_of_galaxies_list(self) -> List:
        """
        A list of the model images of each galaxy.
        """
        return list(self.galaxy_model_image_dict.values())

    @property
    def subtracted_images_of_galaxies_list(self) -> List[aa.Array2D]:
        """
        A list of the subtracted image of every galaxy.

        A subtracted image of a galaxy is the data where all other galaxy images are subtracted from it, therefore
        showing how a galaxy appears in the data in the absence of all other galaxies.

        This is used to visualize the contribution of each galaxy in the data.
        """
        return list(self.subtracted_images_of_galaxies_dict.values())

    @property
    def unmasked_blurred_image(self) -> aa.Array2D:
        """
        The blurred image of the overall fit that would be evaluated without a mask being used.

        Linear objects are tied to the mask defined to used to perform the fit, therefore their unmasked blurred
        image cannot be computed.
        """
        if self.galaxies.has(cls=LightProfileLinear):
            exc.raise_linear_light_profile_in_unmasked()

        return self.galaxies.unmasked_blurred_image_2d_from(
            grid=self.grids.lp, psf=self.dataset.psf
        )

    @property
    def unmasked_blurred_image_of_galaxies_list(self) -> List[aa.Array2D]:
        """
        The blurred image of every galaxy in the fit, that would be evaluated without a mask being used.

        Linear objects are tied to the mask defined to used to perform the fit, therefore their unmasked blurred
        image cannot be computed.
        """
        if self.galaxies.has(cls=LightProfileLinear):
            exc.raise_linear_light_profile_in_unmasked()

        return self.galaxies.unmasked_blurred_image_2d_list_from(
            grid=self.grids.lp, psf=self.dataset.psf
        )

    @property
    def galaxies_linear_light_profiles_to_light_profiles(self) -> List[Galaxy]:
        """
        The galaxy list where all linear light profiles have been converted to ordinary light profiles, where their
        `intensity` values are set to the values inferred by this fit.

        This is typically used for visualization, because linear light profiles cannot be used in `LightProfilePlotter`
        or `GalaxyPlotter` objects.
        """
        return self.model_obj_linear_light_profiles_to_light_profiles
