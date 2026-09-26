"""
Fit an interferometer (ALMA/JVLA uv-plane) dataset with a model consisting of one or more galaxies.

`FitInterferometer` mirrors `FitImaging` but works in the uv-plane:

1. Compute the sum of all galaxy light profile images.
2. Fourier-transform that image to ``profile_visibilities`` using the dataset's transformer.
3. Subtract from the observed visibilities to create ``profile_subtracted_visibilities``.
4. If linear light profiles or a pixelization are present, fit the residual visibilities via a
   linear inversion.
5. Combine the profile visibilities and inversion reconstruction into ``model_data``.
6. Compute residuals, chi-squared, and log-likelihood (or log-evidence when an inversion is used).
"""
import functools
import numpy as np
from typing import Dict, List, Optional

from autonerves import cached_property

import autoarray as aa

from autogalaxy.abstract_fit import AbstractFitInversion
from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.galaxy.galaxies import Galaxies
from autogalaxy.galaxy.to_inversion import GalaxiesToInversion
from autogalaxy.profiles.basis import Basis
from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.light.linear import LightProfileLinear


def _has_light_profile_non_linear(galaxies: List[Galaxy]) -> bool:
    """
    Returns whether any galaxy has an ordinary (non-linear) light profile, whose visibilities are subtracted
    from the data before an inversion.

    A `Basis` is itself a (non-linear) `LightProfile` class, so its contents are checked: a basis made only of
    linear light profiles (e.g. an MGE) contributes nothing to the subtracted visibilities.
    """
    for galaxy in galaxies:
        for light_profile in galaxy.cls_list_from(
            cls=LightProfile, cls_filtered=LightProfileLinear
        ):
            if not isinstance(light_profile, Basis):
                return True

            if any(
                not isinstance(profile, LightProfileLinear)
                for profile in light_profile.light_profile_list
            ):
                return True

    return False


def sparse_dirty_image_from(
    dataset: aa.Interferometer,
    galaxies: List[Galaxy],
    visibilities: aa.Visibilities,
    noise_map: aa.VisibilitiesNoiseMap,
    xp=np,
) -> Optional[np.ndarray]:
    """
    Returns the dirty image the sparse (w-tilde) inversion of an interferometer fit must use to form its data
    vector, or `None` if the dirty image cached on the dataset's `sparse_operator` is already the correct one.

    The `sparse_operator` caches the noise-weighted dirty image `Re(Fᴴ W d)` of the dataset's visibilities,
    as computed by `Interferometer.apply_sparse_operator`. A fit with ordinary (non-linear) light profiles
    inverts the `profile_subtracted_visibilities` instead, so the cached image would give a data vector
    inconsistent with the data the chi-squared is computed from. This function computes the dirty image of
    the input (profile-subtracted) visibilities with the same weighting (one adjoint NUFFT / DFT).

    Whether this is needed is decided structurally (is there a sparse operator, do any galaxies have an
    ordinary light profile), so the branch is fixed at trace time and is safe under `jax.jit`. Fits whose
    light is entirely linear return `None` and pay no extra cost.

    Parameters
    ----------
    dataset
        The interferometer dataset being fitted, whose `sparse_operator` and `transformer` are used.
    galaxies
        The galaxies of the fit, checked for ordinary light profiles.
    visibilities
        The visibilities fitted by the inversion (the profile-subtracted visibilities).
    noise_map
        The noise-map of the visibilities, whose real and imaginary parts weight the dirty image.
    xp
        The array module (`numpy` or `jax.numpy`).
    """
    if dataset.sparse_operator is None:
        return None

    if not _has_light_profile_non_linear(galaxies=galaxies):
        return None

    visibilities = getattr(visibilities, "array", visibilities)
    noise_map = getattr(noise_map, "array", noise_map)

    return dataset.transformer.image_from(
        visibilities=aa.Visibilities(
            visibilities=visibilities.real * noise_map.real**-2.0
            + 1j * visibilities.imag * noise_map.imag**-2.0
        ),
        xp=xp,
    ).array


class FitInterferometer(aa.FitInterferometer, AbstractFitInversion):
    def __init__(
        self,
        dataset: aa.Interferometer,
        galaxies: List[Galaxy],
        dataset_model: Optional[aa.DatasetModel] = None,
        adapt_images: Optional[AdaptImages] = None,
        settings: aa.Settings = None,
        xp=np,
    ):
        """
        Fits an interferometer dataset using a list of galaxies.

        The fit performs the following steps:

        1) Compute the sum of all images of galaxy light profiles.

        2) Fourier transform this image with the transformer object and `uv_wavelengths` to create
           the `profile_visibilities`.

        3) Subtract these visibilities from the `data` to create the `profile_subtracted_visibilities`.

        4) If the galaxies have any linear algebra objects (e.g. linear light profiles, a pixelization / regulariation)
           fit the `profile_subtracted_visibilities` with these objects via an inversion.

        5) Compute the `model_data` as the sum of the `profile_visibilities` and `reconstructed_data` of the inversion
           (if an inversion is not performed the `model_data` is only the `profile_visibilities`.

        6) Subtract the `model_data` from the data and compute the residuals, chi-squared and likelihood via the
           noise-map (if an inversion is performed the `log_evidence`, including addition terms describing the linear
           algebra solution, is computed).

        When performing a model-fit` via ` AnalysisInterferometer` object the `figure_of_merit` of
        this object is called and returned in the `log_likelihood_function`.

        Parameters
        ----------
        dataset
            The interfometer dataset which is fitted by the galaxies.
        galaxies
            The galaxies whose light profile images are used to fit the interferometer data.
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
            dataset=dataset, dataset_model=dataset_model, use_mask_in_fit=False, xp=xp
        )
        AbstractFitInversion.__init__(
            self=self,
            model_obj=self.galaxies,
            settings=settings,
            xp=xp,
        )

        self.adapt_images = adapt_images
        self.settings = settings or aa.Settings()

    @functools.cached_property
    def profile_visibilities(self) -> aa.Visibilities:
        """
        Returns the visibilities of every light profile of every galaxy, which are computed by performing
        a Fourier transform to the sum of light profile images.
        """
        return self.galaxies.visibilities_from(
            grid=self.grids.lp, transformer=self.dataset.transformer, xp=self._xp
        )

    @functools.cached_property
    def profile_subtracted_visibilities(self) -> aa.Visibilities:
        """
        Returns the interferometer dataset's visibilities with all transformed light profile images subtracted.
        """
        return self.data - self.profile_visibilities

    @property
    def galaxies_to_inversion(self) -> GalaxiesToInversion:
        dataset = aa.DatasetInterface(
            data=self.profile_subtracted_visibilities,
            noise_map=self.noise_map,
            grids=self.grids,
            transformer=self.dataset.transformer,
            sparse_operator=self.dataset.sparse_operator,
            sparse_dirty_image=sparse_dirty_image_from(
                dataset=self.dataset,
                galaxies=self.galaxies,
                visibilities=self.profile_subtracted_visibilities,
                noise_map=self.noise_map,
                xp=self._xp,
            ),
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

        The data passed to this function is the dataset's visibilities with all light profile visibilities subtracted.
        """
        if self.perform_inversion:
            return self.galaxies_to_inversion.inversion

    @functools.cached_property
    def model_data(self) -> aa.Visibilities:
        """
        Returns the model data that is used to fit the data.

        If the galaxies do not have any linear objects and therefore omits an inversion, the model data is the
        sum of all light profile images Fourier transformed to visibilities.

        If a inversion is included it is the sum of these visibilities and the inversion's reconstructed visibilities.
        """

        if self.perform_inversion:
            return (
                self.profile_visibilities
                + self.inversion.mapped_reconstructed_operated_data
            )

        return self.profile_visibilities

    @functools.cached_property
    def galaxy_image_dict(self) -> Dict[Galaxy, np.ndarray]:
        """
        A dictionary which associates every galaxy with its `image`.

        This image is the sum of:

        - The images of all ordinary light profiles summed.
        - The images of all linear objects (e.g. linear light profiles / pixelizations), where the images are solved
          for first via the inversion.

        For modeling, this dictionary is used to set up the `adapt_images` that adapt certain pixelizations to the
        data being fitted.
        """
        galaxy_image_dict = self.galaxies.galaxy_image_2d_dict_from(
            grid=self.grids.lp, xp=self._xp
        )

        galaxy_linear_obj_image_dict = self.galaxy_linear_obj_data_dict_from(
            use_operated=False
        )

        return {**galaxy_image_dict, **galaxy_linear_obj_image_dict}

    @functools.cached_property
    def galaxy_model_visibilities_dict(self) -> Dict[Galaxy, np.ndarray]:
        """
        A dictionary which associates every galaxy with its model visibilities.

        These visibilities are the sum of:

        - The visibilities of all ordinary light profiles summed and Fourier transformed to visibilities space.
        - The visibilities of all linear objects (e.g. linear light profiles / pixelizations), where the visibilities
          are solved for first via the inversion.

        For modeling, this dictionary is used to set up the `adapt_visibilities` that adapt certain pixelizations to the
        data being fitted.
        """
        galaxy_model_visibilities_dict = self.galaxies.galaxy_visibilities_dict_from(
            grid=self.grids.lp, transformer=self.dataset.transformer, xp=self._xp
        )

        galaxy_linear_obj_data_dict = self.galaxy_linear_obj_data_dict_from(
            use_operated=True
        )

        return {**galaxy_model_visibilities_dict, **galaxy_linear_obj_data_dict}

    @functools.cached_property
    def model_visibilities_of_galaxies_list(self) -> List:
        """
        A list of the model visibilities of each galaxy.
        """
        return list(self.galaxy_model_visibilities_dict.values())

    @property
    def galaxies_linear_light_profiles_to_light_profiles(self) -> List[Galaxy]:
        """
        The galaxies where all linear light profiles have been converted to ordinary light profiles, where their
        `intensity` values are set to the values inferred by this fit.

        This is typically used for visualization, because linear light profiles cannot be used in `LightProfile`
        or `Galaxy` objects.
        """
        return self.model_obj_linear_light_profiles_to_light_profiles
