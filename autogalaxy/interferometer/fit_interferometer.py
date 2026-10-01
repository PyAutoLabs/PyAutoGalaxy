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

import copy
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


def sparse_profile_terms_from(
    dataset: aa.Interferometer,
    galaxies: List[Galaxy],
    image: aa.Array2D,
    xp=np,
):
    """
    Returns the sparse dirty image and chi-squared data term of the profile-subtracted visibilities
    `d - F i_p` of an interferometer fit, as the tuple `(sparse_dirty_image, data_term)`, computed from the
    dataset's `sparse_operator` without forming the profile visibilities `F i_p`:

        sparse_dirty_image = d~ - W~ i_p
        data_term          = sum(|d|^2 / sigma^2) - 2 i_p^T d~ + i_p^T W~ i_p  =  sum(|d - F i_p|^2 / sigma^2)

    Both reuse one product `W~ i_p` (see `aa.util.inversion_interferometer.sparse_profile_terms_from`). The
    data term is `None` if the operator carries no cached `data_term`.

    Returns `(None, None)` when the dataset has no `sparse_operator` or no galaxy has an ordinary
    (non-linear) light profile -- nothing is subtracted, so the operator's cached dirty image and data term are
    already the correct ones. These checks are structural, so the branch is fixed at trace time and is safe
    under `jax.jit`; the returned data term is an `xp` scalar traced with `image`.

    Parameters
    ----------
    dataset
        The interferometer dataset being fitted, whose `sparse_operator` is used.
    galaxies
        The galaxies of the fit, checked for ordinary light profiles.
    image
        The image `i_p` of the ordinary light profiles on the fit's `grids.lp`, which is exactly the image that
        is Fourier transformed to the fit's `profile_visibilities`.
    xp
        The array module (`numpy` or `jax.numpy`).
    """
    if dataset.sparse_operator is None:
        return None, None

    if not _has_light_profile_non_linear(galaxies=galaxies):
        return None, None

    _, sparse_dirty_image, data_term = (
        aa.util.inversion_interferometer.sparse_profile_terms_from(
            sparse_operator=dataset.sparse_operator,
            image=image,
            extent_index_for_masked_pixel=dataset.real_space_mask.extent_index_for_masked_pixel,
            xp=xp,
        )
    )

    return sparse_dirty_image, data_term


def sparse_dirty_image_from(
    dataset: aa.Interferometer,
    galaxies: List[Galaxy],
    image: aa.Array2D,
    xp=np,
) -> Optional[np.ndarray]:
    """
    Returns the dirty image the sparse (w-tilde) inversion of an interferometer fit must use to form its data
    vector, or `None` if the dirty image cached on the dataset's `sparse_operator` is already the correct one.

    The `sparse_operator` caches the noise-weighted dirty image `d~ = Re(Fᴴ W d)` of the dataset's visibilities
    `d`, as computed by `Interferometer.apply_sparse_operator`. A fit with ordinary (non-linear) light profiles
    inverts the profile-subtracted visibilities `d - F i_p` instead, where `i_p` is the image of those light
    profiles, so the cached image would give a data vector inconsistent with the data the chi-squared is
    computed from. By linearity, the dirty image of the profile-subtracted visibilities is:

        Re(Fᴴ W (d - F i_p)) = d~ - W~ i_p

    where `W~ = Re(Fᴴ W F)` is the operator the sparse inversion already uses for its curvature matrix. Applying
    `W~` is one FFT convolution on the real-space grid, far cheaper than an adjoint NUFFT / DFT over every
    visibility, and it relies on no assumption the sparse curvature matrix does not already make (equal real and
    imaginary noise, enforced by `Interferometer.apply_sparse_operator`).

    Whether this is needed is decided structurally (is there a sparse operator, do any galaxies have an
    ordinary light profile), so the branch is fixed at trace time and is safe under `jax.jit`. Fits whose
    light is entirely linear return `None` and pay no extra cost.

    Parameters
    ----------
    dataset
        The interferometer dataset being fitted, whose `sparse_operator` is used.
    galaxies
        The galaxies of the fit, checked for ordinary light profiles.
    image
        The image `i_p` of the ordinary light profiles on the fit's `grids.lp`, which is exactly the image that
        is Fourier transformed to the fit's `profile_visibilities`.
    xp
        The array module (`numpy` or `jax.numpy`).
    """
    return sparse_profile_terms_from(
        dataset=dataset, galaxies=galaxies, image=image, xp=xp
    )[0]


def uses_precomputed_data_term_from(
    dataset: aa.Interferometer,
    galaxies: List[Galaxy],
    data: aa.Visibilities,
    noise_map: aa.VisibilitiesNoiseMap,
) -> bool:
    """
    Returns whether the inversion of an interferometer fit can read the data term `d^T N^-1 d` of its
    `fast_chi_squared` from the scalar cached on the dataset's `sparse_operator`, instead of being passed the
    visibilities and reducing over them on every likelihood call.

    This holds when:

    - The dataset has a `sparse_operator` carrying a precomputed `data_term` (built by
      `Interferometer.apply_sparse_operator`, `Interferometer.apply_sparse_operator_from_chunks` or, for an
      array-free dataset, `Interferometer.from_stream` / `from_sparse_terms`).
    - The data and noise-map fitted are the dataset's own (not ones a subclass has modified or scaled), since the
      cached scalar was computed from them.
    - Either no galaxy has an ordinary (non-linear) light profile, so nothing is subtracted from the visibilities
      before the inversion and the data it fits are exactly the raw visibilities the operator was built from;
      or the dataset is array-free (`dataset.data is None`). In the latter case the light profiles'
      visibilities cannot be formed, and the fit instead passes the inversion the data term of the
      profile-subtracted visibilities, `data_term - 2 i_p^T d~ + i_p^T W~ i_p`, computed alongside the
      subtracted dirty image `d~ - W~ i_p` (`sparse_profile_terms_from`), which `fast_chi_squared` reads in
      preference to the operator's unsubtracted scalar.

    An in-memory dataset with ordinary light profiles keeps passing the profile-subtracted visibilities
    themselves (the array path), so its likelihood is unchanged bit-for-bit.

    When it holds, the fit passes `data=None` to the inversion's `DatasetInterface`, so the likelihood path
    allocates and reduces over no visibility-sized array. Every check is structural (object identity / `None` /
    profile types), so the branch is fixed at trace time and is safe under `jax.jit`.

    Parameters
    ----------
    dataset
        The interferometer dataset being fitted.
    galaxies
        The galaxies of the fit, checked for ordinary light profiles.
    data
        The visibilities the fit is fitting (`fit.data`).
    noise_map
        The noise-map the fit uses (`fit.noise_map`).
    """
    sparse_operator = getattr(dataset, "sparse_operator", None)

    if getattr(sparse_operator, "data_term", None) is None:
        return False

    if data is not dataset.data or noise_map is not dataset.noise_map:
        return False

    if dataset.data is None:
        return True

    return not _has_light_profile_non_linear(galaxies=galaxies)


def _require_transformer(fit, quantity: str):
    """
    Raise a typed `aa.exc.DatasetException` when `fit`'s dataset is array-free (built by
    `Interferometer.from_stream` / `from_sparse_terms`, so it has no transformer) and `quantity`, which is formed
    in visibility space, is requested.
    """
    if fit.dataset.transformer is None:
        raise aa.exc.DatasetException(
            f"This FitInterferometer's dataset is array-free (built by from_stream / "
            f"from_sparse_terms) and has no transformer, so `{quantity}` (model visibilities) "
            f"cannot be formed. Use `model_image_natural` (the real-space model image) or "
            f"`dirty_model_image_natural` (its natural-weighted dirty image) instead, or the "
            f"in-memory constructor if you need the visibilities."
        )


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
    def profile_image(self) -> aa.Array2D:
        """
        Returns the summed image of every ordinary (non-linear) light profile of every galaxy, which is Fourier
        transformed to the `profile_visibilities`.
        """
        return self.galaxies.image_2d_from(grid=self.grids.lp, xp=self._xp)

    @functools.cached_property
    def profile_visibilities(self) -> Optional[aa.Visibilities]:
        """
        Returns the visibilities of every light profile of every galaxy, which are computed by performing
        a Fourier transform to the sum of light profile images.

        If the galaxies have no ordinary (non-linear) light profile (e.g. their light is entirely an MGE of
        linear Gaussians), the image is all zeros and the Fourier transform is skipped. This is decided
        structurally, so it is safe under `jax.jit`.

        On an array-free dataset (built by `Interferometer.from_stream` / `from_sparse_terms`, which has no
        `uv_wavelengths` and so no transformer) there are no visibilities to compute and this returns `None`.
        The likelihood never needs them there: ordinary light profiles enter through their real-space
        `profile_image` and the data-term identity (`sparse_profile_terms_from`, `sparse_chi_squared`).
        """
        if self.dataset.transformer is None:
            return None

        if _has_light_profile_non_linear(galaxies=self.galaxies):
            return self.dataset.transformer.visibilities_from(
                image=self.profile_image, xp=self._xp
            )

        return aa.Visibilities.zeros(
            shape_slim=(self.dataset.transformer.uv_wavelengths.shape[0],)
        )

    @functools.cached_property
    def profile_subtracted_visibilities(self) -> Optional[aa.Visibilities]:
        """
        Returns the interferometer dataset's visibilities with all transformed light profile images subtracted.

        On an array-free dataset there are no visibilities, so this is `None` (and `profile_visibilities` is not
        evaluated). The inversion of such a fit instead receives the profile-subtracted dirty image and data term
        (see `galaxies_to_inversion`).
        """
        if self.data is None:
            return None

        return self.data - self.profile_visibilities

    @property
    def sparse_chi_squared(self):
        """
        The chi-squared of this fit computed from the dataset's `sparse_operator` without any visibility-sized
        array, which `chi_squared` (and so `log_likelihood` and, without an inversion, `figure_of_merit`) returns
        on an array-free dataset (see `aa.FitInterferometer.sparse_chi_squared`).

        - With an inversion it is the inversion's `fast_chi_squared`, whose data term is that of the
          profile-subtracted visibilities, so it equals `sum(|d - F i_p - F s|^2 / sigma^2)` up to the
          `s^T (eps I) s` the curvature matrix's `no_regularization_add_to_curvature_diag_value` adds for
          unregularized linear objects (the chi-squared convention `log_evidence` already uses; ~1e-7 relative
          on `log_likelihood` versus the dense residual-map chi-squared for e.g. an MGE).
        - Without one the model visibilities are only `F i_p`, the transform of the ordinary light profiles'
          `profile_image`, and it is `data_term - 2 i_p^T d~ + i_p^T W~ i_p` (`sparse_profile_terms_from`); with
          no ordinary light either the model is zero and it is the operator's cached `data_term`.

        `None` when the dataset has no `sparse_operator`. Every branch is structural, so it is safe under
        `jax.jit`, where the value is traced with the light profiles' parameters.
        """
        sparse_operator = self.dataset.sparse_operator

        if sparse_operator is None:
            return None

        if self.perform_inversion:
            return self.inversion.fast_chi_squared

        _, data_term = sparse_profile_terms_from(
            dataset=self.dataset,
            galaxies=self.galaxies,
            image=self.profile_image,
            xp=self._xp,
        )

        if data_term is None:
            return getattr(sparse_operator, "data_term", None)

        return data_term

    @property
    def _uses_precomputed_data_term(self) -> bool:
        """
        Whether this fit's inversion reads its data term from the scalar cached on the dataset's
        `sparse_operator` (see `uses_precomputed_data_term_from`), in which case `galaxies_to_inversion` passes
        `data=None` and the likelihood never evaluates `profile_visibilities` or
        `profile_subtracted_visibilities`.
        """
        return uses_precomputed_data_term_from(
            dataset=self.dataset,
            galaxies=self.galaxies,
            data=self.data,
            noise_map=self.noise_map,
        )

    @property
    def galaxies_to_inversion(self) -> GalaxiesToInversion:
        """
        Returns the object which builds this fit's inversion from its galaxies' linear objects.

        The inversion fits the `profile_subtracted_visibilities`, except when `_uses_precomputed_data_term`, where
        `data=None` is passed and the sparse inversion touches no visibility-sized array:

        - With no ordinary light profile nothing is subtracted, so the inversion takes its data vector from the
          operator's cached dirty image and the data term of its `fast_chi_squared` from the operator's cached
          scalar.
        - On an array-free dataset with ordinary light profiles, it is passed the dirty image `d~ - W~ i_p` and
          data term `data_term - 2 i_p^T d~ + i_p^T W~ i_p` of the profile-subtracted visibilities, both formed
          from one `W~ i_p` product (`sparse_profile_terms_from`), so the profile visibilities `F i_p` are never
          formed.

        On an in-memory sparse dataset with ordinary light profiles the subtracted dirty image is still supplied
        (the data vector must use it) alongside the subtracted visibilities. Where the visibilities exist they
        remain available to outputs via `fit.data`.
        """
        sparse_dirty_image, data_term = sparse_profile_terms_from(
            dataset=self.dataset,
            galaxies=self.galaxies,
            image=self.profile_image,
            xp=self._xp,
        )

        if self._uses_precomputed_data_term:
            data = None
        else:
            data = self.profile_subtracted_visibilities
            data_term = None

        dataset = aa.DatasetInterface(
            data=data,
            noise_map=self.noise_map,
            grids=self.grids,
            transformer=self.dataset.transformer,
            sparse_operator=self.dataset.sparse_operator,
            sparse_dirty_image=sparse_dirty_image,
            data_term=data_term,
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

    @property
    def inversion_with_data(self) -> Optional[aa.AbstractInversion]:
        """
        The fit's `inversion`, guaranteed to carry the visibilities it fitted as its dataset's `data`, for
        output quantities that read them (e.g. `data_subtracted_dict`, plotted by `subplot_of_mapper`).

        On the sparse path with no ordinary light profile (`_uses_precomputed_data_term`) the likelihood's
        inversion is built with `data=None`, so that it touches no visibility-sized array. Nothing was subtracted
        from the visibilities in that case, so the data it fitted are `fit.data`: this returns a shallow copy of
        the inversion (solved first, so it shares the reconstruction and every other cached quantity) whose dataset interface carries
        `fit.data`. In every other case it returns `inversion` itself.

        On an array-free dataset (built by `Interferometer.from_stream` / `from_sparse_terms`) `fit.data` is
        `None`, so there are no visibilities to carry and `inversion` itself is returned; output quantities
        that read the visibilities are unavailable on such a fit.
        """
        inversion = self.inversion

        if inversion is None or inversion.dataset.data is not None or self.data is None:
            return inversion

        # Solve first, so the copy shares the reconstruction (and everything it cached) rather than repeating it.
        inversion.reconstruction

        dataset = copy.copy(inversion.dataset)
        dataset.data = self.data

        inversion_with_data = copy.copy(inversion)
        inversion_with_data.dataset = dataset

        return inversion_with_data

    @functools.cached_property
    def model_data(self) -> aa.Visibilities:
        """
        Returns the model data that is used to fit the data.

        If the galaxies do not have any linear objects and therefore omits an inversion, the model data is the
        sum of all light profile images Fourier transformed to visibilities.

        If a inversion is included it is the sum of these visibilities and the inversion's reconstructed visibilities.

        On an array-free dataset (built by `Interferometer.from_stream` / `from_sparse_terms`) there is no
        transformer to form model visibilities with, so this raises an `aa.exc.DatasetException`; the real-space
        `model_image_natural` and its natural dirty image `dirty_model_image_natural` describe the model there.
        """
        _require_transformer(fit=self, quantity="model_data")

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

    @property
    def model_image_natural(self) -> aa.Array2D:
        """
        The real-space model image `m` of the fit, on the dataset's `real_space_mask`: the image of every
        ordinary (non-linear) light profile (`profile_image`) plus, when the fit has an inversion, the solved
        linear objects' reconstruction (`inversion.mapped_reconstructed_data`, linear light profiles and
        pixelizations) -- the real-space image whose visibilities are `model_data`.

        It is built from these two terms rather than from `galaxy_image_dict`, whose entry for a galaxy with
        both ordinary and linear light holds only the linear reconstruction.

        It needs neither visibilities nor a transformer, so it is available on an array-free dataset (built by
        `Interferometer.from_stream` / `from_sparse_terms`), where it is the image the natural-weighted dirty
        model image `dirty_model_image_natural` is formed from.
        """
        image = np.asarray(
            getattr(self.profile_image, "array", self.profile_image), dtype=np.float64
        )

        if self.inversion is not None:
            reconstruction = self.inversion.mapped_reconstructed_data
            image = image + np.asarray(
                getattr(reconstruction, "array", reconstruction), dtype=np.float64
            )

        return aa.Array2D(
            values=image,
            mask=self.dataset.real_space_mask,
        )

    @property
    def dirty_model_image_natural(self) -> aa.Array2D:
        """
        The naturally weighted, normalised dirty image of the model visibilities, `W~ m / sum(w)`, formed from
        `model_image_natural` with the dataset's `sparse_operator` (see
        `autoarray.fit.fit_interferometer.dirty_model_image_natural_from`).

        It is the model counterpart of the dataset's `dirty_image_natural` and needs no visibilities, so it is
        how a fit on an array-free dataset is visualized. It is available on any dataset carrying a
        `sparse_operator` (array-free, or in-memory after `apply_sparse_operator()`); otherwise it raises an
        `aa.exc.DatasetException`.
        """
        return aa.fit.fit_interferometer.dirty_model_image_natural_from(
            dataset=self.dataset, image=self.model_image_natural
        )

    @property
    def dirty_residual_map_natural(self) -> aa.Array2D:
        """
        The naturally weighted dirty residual map, `dirty_image_natural - dirty_model_image_natural`, which is
        `Re(F^H W (d - F m)) / sum(w)`: the natural dirty image of the visibility residuals, computed without
        them.
        """
        return aa.Array2D(
            values=np.asarray(self.dataset.dirty_image_natural.array)
            - np.asarray(self.dirty_model_image_natural.array),
            mask=self.dataset.real_space_mask,
        )

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

        On an array-free dataset there is no transformer and this raises an `aa.exc.DatasetException`.
        """
        _require_transformer(fit=self, quantity="galaxy_model_visibilities_dict")

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
