import logging
import numpy as np
from typing import Optional, Union

from autoconf.dictable import to_dict, output_to_json

import autofit as af
import autoarray as aa

from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages
from autogalaxy.cosmology.model import LensingCosmology
from autogalaxy.analysis.analysis.analysis import Analysis
from autogalaxy.analysis.result import ResultDataset

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisDataset(Analysis):
    def __init__(
        self,
        dataset: Union[aa.Imaging, aa.Interferometer],
        adapt_images: Optional[AdaptImages] = None,
        cosmology: LensingCosmology = None,
        settings: aa.Settings = None,
        title_prefix: str = None,
        use_jax: bool = True,
        **kwargs,
    ):
        """
        Abstract Analysis class for all model-fits which fit galaxies to a dataset, like imaging or interferometer data.

        This class stores the settings used to perform the model-fit for certain components of the model (e.g. a
        pixelization or inversion), the Cosmology used for the analysis and adapt images used for certain model
        classes.

        Parameters
        ----------
        dataset
            The dataset that is the model is fitted too.
        adapt_images
            The adapt-model image and galaxies images of a previous result in a model-fitting pipeline, which are
            used by certain classes for adapting the analysis to the properties of the dataset.
        cosmology
            The Cosmology assumed for this analysis.
        settings
            Settings controlling how an inversion is fitted during the model-fit, for example which linear algebra
            formalism is used.
        title_prefix
            A string that is added before the title of all figures output by visualization, for example to
            put the name of the dataset and galaxy in the title.
        """
        super().__init__(
            cosmology=cosmology,
            use_jax=use_jax,
            **kwargs,
        )

        self.dataset = dataset
        self.adapt_images = adapt_images

        self.settings = settings or aa.Settings()

        self.title_prefix = title_prefix

    def modify_before_fit(self, paths: af.DirectoryPaths, model: af.Collection):
        """
        This function is called immediately before the non-linear search begins and performs final tasks and checks
        before it begins.

        This function checks that the adapt-dataset is consistent with previous adapt-datasets if the model-fit is
        being resumed from a previous run, and it visualizes objects which do not change throughout the model fit
        like the dataset.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """

        # Ensures preloaded in memory so repeated calls are fast

        self.dataset.grids.lp
        self.dataset.grids.lp.over_sampled
        self.dataset.grids.lp.over_sampler

        self.dataset.grids.pixelization
        self.dataset.grids.pixelization.over_sampled
        self.dataset.grids.pixelization.over_sampler

        self.dataset.grids.blurring

        if self.dataset.grids.blurring is not None:

            self.dataset.grids.blurring.over_sampled
            self.dataset.grids.blurring.over_sampler

        self.dataset.grids.border_relocator

        super().modify_before_fit(paths=paths, model=model)

        return self

    def save_attributes(self, paths: af.DirectoryPaths):
        """
        Before the model-fit via the non-linear search begins, this routine saves attributes of the `Analysis` object
        to the `files` folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator
        tools.

        For this analysis the following are output:

        - The settings associated with the inversion.
        - The settings associated with the pixelization.
        - The Cosmology.

        The following .fits files are also output via the plotter interface:

        - The mask applied to the dataset, in the `PrimaryHDU` of `dataset.fits`.
        - The dataset (data / noise-map / over sampler / etc.).
        - The adapt image's model image and galaxy images, if used.

        It is common for these attributes to be loaded by many of the template aggregator functions given in the
        `aggregator` modules. For example, when using the database tools to reperform a fit, this will by default
        load the dataset, settings and other attributes necessary to perform a fit using the attributes output by
        this function.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization, and the pickled objects used by the aggregator output by this function.
        """
        paths.save_json(
            name="settings",
            object_dict=to_dict(self.settings),
        )
        paths.save_json(
            name="cosmology",
            object_dict=to_dict(self.cosmology),
        )

    def save_results(self, paths: af.DirectoryPaths, result: ResultDataset):
        """
        At the end of a model-fit, this routine saves attributes of the `Analysis` object to the `files`
        folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

        For this analysis it outputs the following:

        - The maximum log likelihood galaxies of the fit.

        Parameters
        ----------
        paths
            The paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        result
            The result of a model fit, including the non-linear search, samples and maximum likelihood tracer.
        """
        try:
            output_to_json(
                obj=result.max_log_likelihood_galaxies,
                file_path=paths._files_path / "galaxies.json",
            )
        except AttributeError:
            pass

    def adapt_images_via_instance_from(self, instance: af.ModelInstance) -> AdaptImages:
        try:
            return self.adapt_images.updated_via_instance_from(instance=instance)
        except AttributeError:
            pass
