"""
`AnalysisEllipse` — the **PyAutoFit** `Analysis` class for fitting isophotal ellipse models to imaging data.

This module provides `AnalysisEllipse`, which implements `log_likelihood_function` by:

1. Extracting the ellipse (and optional multipoles) from the model instance.
2. Constructing a `FitEllipse` object.
3. Returning the `figure_of_merit` of the fit.

Unlike `AnalysisImaging`, this class does not use PSF convolution or linear inversions. It directly fits
the isophotal structure of the image via interpolation along the ellipse perimeter.
"""
import logging
import numpy as np
from typing import List, Optional

import autofit as af
import autoarray as aa

from autogalaxy.ellipse.fit_ellipse import FitEllipse
from autogalaxy.ellipse.model.result import ResultEllipse
from autogalaxy.ellipse.model.visualizer import VisualizerEllipse

from autogalaxy import exc

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisEllipse(af.Analysis):
    Result = ResultEllipse
    Visualizer = VisualizerEllipse

    def __init__(
        self,
        dataset: aa.Imaging,
        title_prefix: str = None,
        use_jax: bool = False,
        **kwargs,
    ):
        """
        Fits a model made of ellipses to an imaging dataset via a non-linear search.

        The `Analysis` class defines the `log_likelihood_function` which fits the model to the dataset and returns the
        log likelihood value defining how well the model fitted the data.

        It handles many other tasks, such as visualization, outputting results to hard-disk and storing results in
        a format that can be loaded after the model-fit is complete.

        This class is used for model-fits which fit ellipses to an imaging dataset.

        Parameters
        ----------
        dataset
            The `Imaging` dataset that the model containing ellipses is fitted to.
        title_prefix
            A string that is added before the title of all figures output by visualization, for example to
            put the name of the dataset and galaxy in the title.
        """
        self.dataset = dataset
        self.title_prefix = title_prefix

        super().__init__(use_jax=use_jax, **kwargs)

        if use_jax:
            self._register_fit_ellipse_pytrees()

    @staticmethod
    def _register_fit_ellipse_pytrees() -> None:
        """Register every type reachable from a ``FitEllipse`` return value so
        ``jax.jit`` can flatten its output.

        ``dataset`` is per-analysis-constant — ride as aux so JAX does not
        recurse into it. ``ellipse`` and ``multipole_list`` carry the
        traced model parameters (``centre``, ``ell_comps``, ``major_axis``,
        ``multipole_comps``).
        """
        from autoarray.abstract_ndarray import register_instance_pytree
        from autogalaxy.ellipse.ellipse.ellipse import Ellipse
        from autogalaxy.ellipse.ellipse.ellipse_multipole import EllipseMultipole

        register_instance_pytree(Ellipse)
        # ``m`` is the multipole order (integer discriminator like 4, 6 — not
        # fittable). The traced parameters live in ``multipole_comps``.
        register_instance_pytree(EllipseMultipole, no_flatten=("m",))
        register_instance_pytree(FitEllipse, no_flatten=("dataset",))

    def log_likelihood_function(self, instance: af.ModelInstance) -> float:
        """
        Given an instance of the model, where the model parameters are set via a non-linear search, fit the model
        instance to the imaging dataset.

        This function returns a log likelihood which is used by the non-linear search to guide the model-fit.

        For this analysis class, this function performs the following steps:

        1) Extract all ellipses from the model instance.

        2) Use the ellipses to create a list of `FitEllipse` objects, which fits each ellipse to the data and noise-map
        via interpolation and subtracts these values from their mean values in order to quantify how well the ellipse
        traces around the data.

        Certain models will fail to fit the dataset and raise an exception. For example the ellipse parameters may be
        ill defined and raise an Exception. In such circumstances the model is discarded and its likelihood value is
        passed to the non-linear search in a way that it ignores it (for example, using a value of -1.0e99).

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
        fit_list = self.fit_list_from(instance=instance)
        return sum(fit.log_likelihood for fit in fit_list)

    def fit_list_from(self, instance: af.ModelInstance) -> List[FitEllipse]:
        """
        Given a model instance create a list of `FitEllipse` objects.

        This function unpacks the `instance`, specifically the `ellipses` and (in input) the `multipoles` and uses
        them to create a list of `FitEllipse` objects that are used to fit the model to the imaging data.

        This function is used in the `log_likelihood_function` to fit the model containing ellipses to the imaging data
        and compute the log likelihood.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        The fit of the ellipses to the imaging dataset, which includes the log likelihood.
        """
        fit_list = []

        for i in range(len(instance.ellipses)):
            ellipse = instance.ellipses[i]

            try:
                multipole_list = instance.multipoles[i]
            except AttributeError:
                multipole_list = None

            fit = FitEllipse(
                dataset=self.dataset, ellipse=ellipse, multipole_list=multipole_list
            )

            fit_list.append(fit)

        return fit_list

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

        The `ResultEllipse` object contains a number of methods which use the above objects to create the max
        log likelihood galaxies `FitEllipse`, etc.

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
            The result of fitting the ellipse model to the imaging dataset, via a non-linear search.
        """
        return self.Result(
            samples_summary=samples_summary,
            paths=paths,
            samples=samples,
            search_internal=search_internal,
            analysis=self,
        )

    def save_attributes(self, paths: af.DirectoryPaths):
        """
         Before the non-linear search begins, this routine saves attributes of the `Analysis` object to the `files`
         folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

         For this analysis, it uses the `AnalysisDataset` object's method to output the following:

         - The imaging dataset (data / noise-map / etc.).
         - The mask applied to the dataset.
         - The Cosmology.

         This function also outputs attributes specific to an imaging dataset:

        - Its mask.

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
        pass
