import logging
import numpy as np
from typing import Optional

from autoconf.dictable import to_dict

import autofit as af
import autoarray as aa

from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages
from autogalaxy.analysis.analysis.dataset import AnalysisDataset
from autogalaxy.cosmology.model import LensingCosmology
from autogalaxy.interferometer.model.result import ResultInterferometer
from autogalaxy.interferometer.fit_interferometer import FitInterferometer
from autogalaxy.interferometer.model.visualizer import VisualizerInterferometer

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisInterferometer(AnalysisDataset):
    Result = ResultInterferometer
    Visualizer = VisualizerInterferometer

    def __init__(
        self,
        dataset: aa.Interferometer,
        adapt_images: Optional[AdaptImages] = None,
        cosmology: LensingCosmology = None,
        settings: aa.Settings = None,
        title_prefix: str = None,
        use_jax: bool = True,
    ):
        """
        Fits a galaxy model to an interferometer dataset via a non-linear search.

        The `Analysis` class defines the `log_likelihood_function` which fits the model to the dataset and returns the
        log likelihood value defining how well the model fitted the data.

        It handles many other tasks, such as visualization, outputting results to hard-disk and storing results in
        a format that can be loaded after the model-fit is complete.

        This Analysis class is used for all model-fits which fit galaxies to an interferometer dataset.

        This class stores the settings used to perform the model-fit for certain components of the model (e.g. a
        pixelization or inversion), the Cosmology used for the analysis and adapt images used for certain model
        classes.

        Parameters
        ----------
        dataset
            The interferometer dataset that the model is fitted too.
        adapt_images
            The adapt-model image and galaxies images of a previous result in a model-fitting pipeline, which are
            used by certain classes for adapting the analysis to the properties of the dataset.
        cosmology
            The Cosmology assumed for this analysis.
        settings
            Settings controlling how an inversion is fitted, for example which linear algebra formalism is used.
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
        )

    @property
    def interferometer(self):
        return self.dataset

    def log_likelihood_function(self, instance: af.ModelInstance) -> float:
        """
        Given an instance of the model, where the model parameters are set via a non-linear search, fit the model
        instance to the interferometer dataset.

        This function returns a log likelihood which is used by the non-linear search to guide the model-fit.

        For this analysis class, this function performs the following steps:

        1) If the analysis has a adapt image, associated the model galaxy images of this dataset to the galaxies in
           the model instance.

        2) Extract attributes which model aspects of the data reductions, like scaling the background background noise.

        3) Extracts all galaxies from the model instance.

        4) Use the galaxies and other attributes to create a `FitInterferometer` object, which performs steps such as
           creating model images of every galaxy, transforming them to the uv-plane via a Fourier transform
           and computing residuals, a chi-squared statistic and the log likelihood.

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
            The log likelihood indicating how well this model instance fitted the interferometer data.
        """
        return self.fit_from(instance=instance).figure_of_merit

    def fit_from(self, instance: af.ModelInstance) -> FitInterferometer:
        """
        Given a model instance create a `FitInterferometer` object.

        This function is used in the `log_likelihood_function` to fit the model to the interferometer data and compute
        the log likelihood.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        FitInterferometer
            The fit of the galaxies to the interferometer dataset, which includes the log likelihood.
        """
        galaxies = self.galaxies_via_instance_from(
            instance=instance,
        )

        adapt_images = self.adapt_images_via_instance_from(instance=instance)

        return FitInterferometer(
            dataset=self.dataset,
            galaxies=galaxies,
            adapt_images=adapt_images,
            settings=self.settings,
            xp=self._xp,
        )

    def save_attributes(self, paths: af.DirectoryPaths):
        """
        Before the model-fit begins, this routine saves attributes of the `Analysis` object to the `files` folder
        such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

        It outputs the following attributes of the dataset:



        For this analysis, it uses the `AnalysisDataset` object's method to output the following:

        - The settings associated with the inversion.
        - The settings associated with the pixelization.
        - The Cosmology.
        - The adapt image's model image and galaxy images, as `adapt_images.fits`, if used.

        The following .fits files are also output via the plotter interface:

        - The real space mask applied to the dataset, in the `PrimaryHDU` of `dataset.fits`.
        - The interferometer dataset as `dataset.fits` (data / noise-map / uv_wavelengths).

        It is common for these attributes to be loaded by many of the template aggregator functions given in the
        `aggregator` modules. For example, when using the database tools to perform a fit, the default behaviour is for
        the dataset, settings and other attributes necessary to perform the fit to be loaded via the pickle files
        output by this function.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored, visualization,
            and the pickled objects used by the aggregator output by this function.
        """
        super().save_attributes(paths=paths)

        paths.save_json(
            "transformer_class",
            to_dict(self.dataset.transformer.__class__),
        )
