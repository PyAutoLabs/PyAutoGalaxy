import numpy as np

from autoconf.dictable import to_dict

import autofit as af

from autogalaxy.analysis.analysis.analysis import Analysis
from autogalaxy.cosmology.model import LensingCosmology
from autogalaxy.quantity.dataset_quantity import DatasetQuantity
from autogalaxy.quantity.model.result import ResultQuantity
from autogalaxy.quantity.model.visualizer import VisualizerQuantity
from autogalaxy.quantity.fit_quantity import FitQuantity


class AnalysisQuantity(Analysis):
    Result = ResultQuantity
    Visualizer = VisualizerQuantity

    def __init__(
        self,
        dataset: DatasetQuantity,
        func_str: str,
        cosmology: LensingCosmology = None,
        title_prefix: str = None,
        use_jax: bool = True,
        **kwargs,
    ):
        """
        Fits a galaxy model to a quantity dataset via a non-linear search.

        The `Analysis` class defines the `log_likelihood_function` which fits the model to the dataset and returns the
        log likelihood value defining how well the model fitted the data.

        It handles many other tasks, such as visualization, outputting results to hard-disk and storing results in
        a format that can be loaded after the model-fit is complete.

        This class is used for model-fits which fit derived quantity of galaxies, for example their
        convergence, potential or deflection angles, to another model for that quantity. For example, one could find
        the `PowerLaw` mass profile model that best fits the deflection angles of an `NFW` mass profile.

        The `func_str` input defines what quantity is fitted, it corresponds to the function of the model galaxy
        objects that is called to create the model quantity. For example, if `func_str="convergence_2d_from"`, the
        convergence is computed from each model galaxy.

        This class stores the settings used to perform the model-fit for certain components of the model (e.g. the
        Cosmology used for the analysis).

        Parameters
        ----------
        dataset
            The `DatasetQuantity` dataset that the model is fitted too.
        func_str
            A string giving the name of the method of the input galaxy used to compute the quantity that fits
            the dataset.
        cosmology
            The Cosmology assumed for this analysis.
        title_prefix
            A string that is added before the title of all figures output by visualization, for example to
            put the name of the dataset and galaxy in the title.
        """
        super().__init__(cosmology=cosmology, use_jax=use_jax, **kwargs)

        self.dataset = dataset
        self.func_str = func_str
        self.title_prefix = title_prefix

        if use_jax:
            self._register_fit_quantity_pytrees()

    @staticmethod
    def _register_fit_quantity_pytrees() -> None:
        """Register every type reachable from a ``FitQuantity`` return value
        so ``jax.jit`` can flatten its output.

        ``dataset``, ``func_str``, ``use_mask_in_fit`` are per-analysis-
        constant — ride as aux so JAX does not recurse into them.
        ``light_mass_obj`` (a ``Galaxies``) and ``model_data_manual``
        carry the traced model arrays and ride as pytree children.
        """
        from autoarray.abstract_ndarray import register_instance_pytree
        from autogalaxy.analysis.jax_pytrees import register_galaxies_pytree

        register_instance_pytree(
            FitQuantity,
            no_flatten=("dataset", "func_str", "use_mask_in_fit"),
        )
        register_galaxies_pytree()

    def log_likelihood_function(self, instance: af.ModelInstance) -> float:
        """
        Given an instance of the model, where the model parameters are set via a non-linear search, fit the model
        instance to the quantity's dataset.

        This function returns a log likelihood which is used by the non-linear search to guide the model-fit.

        For this analysis class, this function performs the following steps:

        1) Use the input quantity of the analysis to determine the function that is passed to `FitQuantity`, which
           generates the quantity from the model which is compared to data.

        2) Use this function to create a `FitQuantity` object, which performs steps such as creating the `model_data`
           of the quantity and computing residuals, a chi-squared statistic and the log likelihood.

        Certain models will fail to fit the dataset and raise an exception. For example if extreme values of the model
        create numerical infinities. In such circumstances the model is discarded and its likelihood value is passed to
        the non-linear search in a way that it ignores it (for example, using a value of -1.0e99).

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
        return self.fit_quantity_for_instance(instance=instance).figure_of_merit

    def fit_quantity_for_instance(self, instance: af.ModelInstance) -> FitQuantity:
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
        FitQuantity
            The fit of the galaxies to the imaging dataset, which includes the log likelihood.
        """

        galaxies = self.galaxies_via_instance_from(instance=instance)

        return FitQuantity(
            dataset=self.dataset, light_mass_obj=galaxies, func_str=self.func_str
        )

    def fit_from(self, instance: af.ModelInstance) -> FitQuantity:
        """
        Standard-name alias for :meth:`fit_quantity_for_instance`.

        Exposing ``fit_from`` lets the autofit base ``Analysis.fit_for_visualization``
        helper dispatch through the same JIT-cached wrapper as imaging /
        interferometer (it calls ``self.fit_from`` unconditionally). Without
        this method, ``use_jax_for_visualization=True`` on ``AnalysisQuantity``
        would be a silent no-op — see the Phase 0c shipped notes in
        ``PyAutoPrompt/complete.md``.
        """
        return self.fit_quantity_for_instance(instance=instance)

    def save_attributes(self, paths: af.DirectoryPaths):
        """
        Before the non-linear search begins, this routine saves attributes of the `Analysis` object to the `files`
        folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

        For this analysis, it uses the `AnalysisDataset` object's method to output the following:

        - The settings associated with the dataset.
        - The Cosmology.

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

        paths.save_json(
            name="cosmology",
            object_dict=to_dict(self.cosmology),
        )
