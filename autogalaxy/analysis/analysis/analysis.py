import logging
import numpy as np
from typing import List, Optional

import autofit as af
import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.galaxy.galaxies import Galaxies
from autogalaxy.cosmology.model import LensingCosmology

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class Analysis(af.Analysis):
    def __init__(
        self,
        cosmology: LensingCosmology = None,
        use_jax: bool = True,
        **kwargs,
    ):
        """
        Fits a model to a dataset via a non-linear search.

        This abstract Analysis class for all model-fits which fit galaxies, but does not perform a model-fit by
        itself (and is therefore only inherited from).

        This class stores the Cosmology used for the analysis and adapt images used for certain model classes.

        Parameters
        ----------
        cosmology
            The Cosmology assumed for this analysis.
        """

        from autogalaxy.cosmology.model import Planck15

        self.cosmology = cosmology or Planck15()

        super().__init__(use_jax=use_jax, **kwargs)

    def galaxies_via_instance_from(
        self,
        instance: af.ModelInstance,
    ) -> List[Galaxy]:
        """
        Create a list of galaxies from a model instance, which is used to fit the dataset.

        The instance may only contain galaxies, in which case this function is redundant. However, if extra galaxies
        are included, the instance will contain both galaxies and extra galaxies, and they should be added to create
        the single list of galaxies used to fit the dataset.

        Parameters
        ----------
        instance
            An instance of the model that is fitted to the data by this analysis (whose parameters may have been set
            via a non-linear search).

        Returns
        -------
        A list of galaxies that is used to then fit the dataset.
        """
        if hasattr(instance, "extra_galaxies"):
            if getattr(instance, "extra_galaxies", None) is not None:
                return Galaxies(
                    galaxies=instance.galaxies + instance.extra_galaxies,
                )

        return Galaxies(galaxies=instance.galaxies)

    def dataset_model_via_instance_from(
        self, instance: af.ModelInstance
    ) -> aa.DatasetModel:
        """
        Create a dataset model from a model instance, which is used to fit the dataset.

        Parameters
        ----------
        instance
            An instance of the model that is fitted to the data by this analysis (whose parameters may have been set
            via a non-linear search).

        Returns
        -------
        A dataset_model that is used to then fit the dataset.
        """
        if hasattr(instance, "dataset_model"):
            return instance.dataset_model

    def make_result(
        self,
        samples_summary: af.SamplesSummary,
        paths: af.AbstractPaths,
        samples: Optional[af.SamplesPDF] = None,
        search_internal: Optional[object] = None,
        analysis: Optional[af.Analysis] = None,
    ) -> af.Result:
        """
        After the non-linear search is complete create its `Result`, which includes:

        - The samples of the non-linear search (E.g. MCMC chains, nested sampling samples) which are used to compute
          the maximum likelihood model, posteriors and other properties.

        - The model used to fit the data, which uses the samples to create specific instances of the model (e.g.
          an instance of the maximum log likelihood model).

        - The non-linear search used to perform the model fit.

        The `ResultImaging` object contains a number of methods which use the above objects to create the max
        log likelihood galaxies `FitImaging`, adapt-galaxy images,etc.

        Parameters
        ----------
        samples
            A PyAutoFit object which contains the samples of the non-linear search, for example the chains of an MCMC
            run of samples of the nested sampler.
        search
            The non-linear search used to perform this model-fit.

        Returns
        -------
        ResultImaging
            The result of fitting the model to the imaging dataset, via a non-linear search.
        """
        return self.Result(
            samples_summary=samples_summary,
            paths=paths,
            samples=samples,
            search_internal=search_internal,
            analysis=self,
        )

    def perform_quick_update(self, paths, instance):
        """
        Perform a quick visualization update during non-linear search fitting.

        This method is called intermittently while the sampler is running to produce
        the `subplot+fit` plots of the current maximum-likelihood model fit. The intent
        is to provide fast feedback (without waiting for the full run to complete) so that
        users can monitor whether the fit is behaving sensibly.

        The plot appears both in a matplotlib window (if running locally) and is also saved to the
        `output` folder of the output path.

        Parameters
        ----------
        paths : af.DirectoryPaths
            Object describing the output folder structure where visualization files
            should be written.
        instance : model instance
            The current maximum-likelihood instance of the model, used to generate
            the visualization plots.
        """

        self.Visualizer().visualize(
            analysis=self,
            paths=paths,
            instance=instance,
            during_analysis=True,
            quick_update=True,
        )
