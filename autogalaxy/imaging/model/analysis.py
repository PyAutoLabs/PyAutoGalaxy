"""
`AnalysisImaging` — the **PyAutoFit** `Analysis` class for fitting galaxy models to CCD imaging data.

This module provides `AnalysisImaging`, which implements `log_likelihood_function` by:

1. Extracting galaxies from the model instance.
2. Constructing a `FitImaging` object using those galaxies and the stored `Imaging` dataset.
3. Returning the `figure_of_merit` of the fit (log-likelihood or log-evidence).

It also handles:

- Adapt images: per-galaxy model images from a previous search that drive adaptive pixelizations.
- Visualization: automatic figure generation during and after the model-fit.
- Results: wrapping fit outputs into a `ResultImaging` object for downstream use.
"""
import numpy as np
from typing import Optional

import autofit as af
import autoarray as aa

from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages
from autogalaxy.analysis.analysis.dataset import AnalysisDataset
from autogalaxy.cosmology.model import LensingCosmology
from autogalaxy.imaging.model.result import ResultImaging
from autogalaxy.imaging.model.visualizer import VisualizerImaging
from autogalaxy.imaging.fit_imaging import FitImaging


_FIT_IMAGING_PYTREES_REGISTERED = False


class AnalysisImaging(AnalysisDataset):
    Result = ResultImaging
    Visualizer = VisualizerImaging

    def __init__(
        self,
        dataset: aa.Imaging,
        adapt_images: Optional[AdaptImages] = None,
        cosmology: LensingCosmology = None,
        settings: aa.Settings = None,
        title_prefix: str = None,
        use_jax: bool = True,
        **kwargs,
    ):
        """
        Fits a galaxy model to an imaging dataset via a non-linear search.

        The `Analysis` class defines the `log_likelihood_function` which fits the model to the dataset and returns the
        log likelihood value defining how well the model fitted the data.

        It handles many other tasks, such as visualization, outputting results to hard-disk and storing results in
        a format that can be loaded after the model-fit is complete.

        This class is used for model-fits which fit galaxies to an imaging dataset.

        This class stores the settings used to perform the model-fit for certain components of the model (e.g. a
        pixelization or inversion), the Cosmology used for the analysis and adapt images used for certain model
        classes.

        Parameters
        ----------
        dataset
            The `Imaging` dataset that the model is fitted to.
        adapt_images
            The adapt-model image and galaxies images of a previous result in a model-fitting pipeline, which are
            used by certain classes for adapting the analysis to the properties of the dataset.
        cosmology
            The Cosmology assumed for this analysis.
        settings
            Settings controlling how an inversion is fitted for example which linear algebra formalism is used.
        title_prefix
            A string that is added before the title of all figures output by visualization, for example to
            put the name of the dataset and galaxy in the title.
        """
        super().__init__(
            dataset=dataset,
            adapt_images=adapt_images,
            cosmology=cosmology,
            settings=settings,
            title_prefix=title_prefix,
            use_jax=use_jax,
            **kwargs,
        )

    @property
    def imaging(self):
        return self.dataset

    @property
    def LATENT_KEYS(self):
        from autogalaxy.imaging.model.latent import latent_keys_enabled
        return latent_keys_enabled()

    def log_likelihood_function(self, instance: af.ModelInstance) -> float:
        """
        Given an instance of the model, where the model parameters are set via a non-linear search, fit the model
        instance to the imaging dataset.

        This function returns a log likelihood which is used by the non-linear search to guide the model-fit.

        For this analysis class, this function performs the following steps:

        1) If the analysis has a adapt image, associated the model galaxy images of this dataset to the galaxies in
           the model instance.

        2) Extract attributes which model aspects of the data reductions, like the scaling the background sky
           and background noise.

        3) Extracts all galaxies from the model instance.

        4) Use the galaxies and other attributes to create a `FitImaging` object, which performs steps such as creating
           model images of every galaxy, blurring them with the imaging dataset's PSF and computing residuals,
           a chi-squared statistic and the log likelihood.

        Certain models will fail to fit the dataset and raise an exception. For example if an `Inversion` is used, the
        linear algebra calculation may be invalid and raise an Exception. In such circumstances the model is discarded
        and its likelihood value is passed to the non-linear search in a way that it ignores it (for example, using a
        value of -1.0e99).

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        float
            The log likelihood indicating how well this model instance fitted the imaging data.
        """
        return self.fit_from(instance=instance).figure_of_merit

    def fit_from(self, instance: af.ModelInstance) -> FitImaging:
        """
        Given a model instance create a `FitImaging` object.

        This function is used in the `log_likelihood_function` to fit the model to the imaging data and compute the
        log likelihood.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        FitImaging
            The fit of the galaxies to the imaging dataset, which includes the log likelihood.
        """

        if self._use_jax:
            self._register_fit_imaging_pytrees()

        galaxies = self.galaxies_via_instance_from(
            instance=instance,
        )

        dataset_model = self.dataset_model_via_instance_from(instance=instance)

        adapt_images = self.adapt_images_via_instance_from(
            instance=instance,
            dataset_model=dataset_model,
            galaxies=galaxies,
            xp=self._xp,
        )

        return FitImaging(
            dataset=self.dataset,
            galaxies=galaxies,
            dataset_model=dataset_model,
            adapt_images=adapt_images,
            settings=self.settings,
            xp=self._xp,
        )

    def compute_latent_variables(self, parameters, model):
        """
        Compute the catalogue of latent variables enabled in
        ``config/latent.yaml`` for the given parameter vector.

        Returns a tuple positionally aligned with :attr:`LATENT_KEYS` —
        PyAutoFit zips it with the keys at
        ``autofit/non_linear/analysis/analysis.py:285`` and stacks per
        sample for the JIT batch path at lines 223-234.

        Raises ``NotImplementedError`` when no latents are enabled so
        PyAutoFit's outer ``except NotImplementedError`` short-circuits
        the latent pipeline cleanly (no empty ``latent.csv`` written).
        """
        from autogalaxy.imaging.model.latent import LATENT_FUNCTIONS

        keys = self.LATENT_KEYS
        if not keys:
            raise NotImplementedError

        xp = self._xp
        instance = model.instance_from_vector(vector=parameters)
        fit = self.fit_from(instance=instance)
        magzero = self.kwargs.get("magzero", None)
        context = {"fit": fit, "magzero": magzero, "xp": xp}
        return tuple(LATENT_FUNCTIONS[k](**context) for k in keys)

    @staticmethod
    def _register_fit_imaging_pytrees() -> None:
        """Register every type reachable from a ``FitImaging`` return value
        so ``jax.jit(fit_from)`` can flatten its output.

        ``dataset``, ``adapt_images`` and ``settings`` are constants per
        analysis — ride as aux so JAX does not recurse into them. Everything
        else (``galaxies``, ``dataset_model`` and the autoarray wrappers they
        carry) is dynamic per fit.

        Idempotent — guarded by the module-level
        ``_FIT_IMAGING_PYTREES_REGISTERED`` flag so repeated calls from each
        ``fit_from`` invocation are cheap. ``DatasetModel`` may already be
        registered by ``autofit.jax.pytrees.register_model`` (its
        ``_REGISTERED_INSTANCE_CLASSES`` set is independent of autoarray's
        ``_pytree_registered_classes``); cross-populate so
        ``register_instance_pytree`` short-circuits instead of asking JAX to
        register it twice. Mirrors the defense in
        ``autogalaxy/ellipse/model/analysis.py``.
        """
        global _FIT_IMAGING_PYTREES_REGISTERED
        if _FIT_IMAGING_PYTREES_REGISTERED:
            return

        from autoarray.abstract_ndarray import (
            register_instance_pytree,
            _pytree_registered_classes,
        )
        from autoarray.dataset.dataset_model import DatasetModel
        from autogalaxy.analysis.jax_pytrees import register_galaxies_pytree

        try:
            from autofit.jax.pytrees import (
                _REGISTERED_INSTANCE_CLASSES as _af_registered,
            )
        except ImportError:
            _af_registered = set()

        if DatasetModel in _af_registered:
            _pytree_registered_classes.add(DatasetModel)

        register_instance_pytree(
            FitImaging,
            no_flatten=("dataset", "adapt_images", "settings"),
        )
        register_instance_pytree(DatasetModel)
        register_galaxies_pytree()

        _FIT_IMAGING_PYTREES_REGISTERED = True

    def save_attributes(self, paths: af.DirectoryPaths):
        """
        Before the non-linear search begins, this routine saves attributes of the `Analysis` object to the `files`
        folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

        For this analysis, it uses the `AnalysisDataset` object's method to output the following:

        - The settings associated with the inversion.
        - The settings associated with the pixelization.
        - The Cosmology.
        - The adapt image's model image and galaxy images, as `adapt_images.fits`, if used.

        The following .fits files are also output via the plotter interface:

        - The mask applied to the dataset, in the `PrimaryHDU` of `dataset.fits`.
        - The imaging dataset as `dataset.fits` (data / noise-map / psf / over sampler / etc.).

        It is common for these attributes to be loaded by many of the template aggregator functions given in the
        `aggregator` modules. For example, when using the database tools to perform a fit, the default behaviour is for
        the dataset, settings and other attributes necessary to perform the fit to be loaded via the pickle files
        output by this function.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        """
        super().save_attributes(paths=paths)
