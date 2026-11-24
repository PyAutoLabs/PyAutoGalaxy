import logging
import numpy as np
from typing import Optional, Union

from astropy.io.fits import HDUList

from autonerves.dictable import to_dict, output_to_json

import autofit as af
import autoarray as aa

from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages
from autogalaxy.cosmology.model import LensingCosmology
from autogalaxy.analysis.analysis.analysis import Analysis
from autogalaxy.analysis.analysis.setup import AnalysisSetupMixin
from autogalaxy.analysis.result import ResultDataset
from autogalaxy.analysis import setup_util

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisDataset(AnalysisSetupMixin, Analysis):
    # Lensing latents (e.g. effective Einstein radius via
    # `LensCalc.einstein_radius_jit_from`) call into `jax_zero_contour.ZeroSolver`,
    # which upstream documents as incompatible with `jax.vmap` (uses
    # `jax.lax.cond` / `jax.lax.while_loop` for early termination). Override
    # the PyAutoFit default from "vmap" to per-sample "jit" so the JIT cache
    # is reused across samples without invoking vmap on the inner zero-contour
    # primitives.
    LATENT_BATCH_MODE = "jit"

    def __init__(
        self,
        dataset: Union[aa.Imaging, aa.Interferometer],
        adapt_images: Optional[AdaptImages] = None,
        cosmology: LensingCosmology = None,
        settings: aa.Settings = None,
        title_prefix: str = None,
        use_jax: bool = True,
        setup=None,
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
        use_jax
            Whether the likelihood is evaluated with JAX. It is honoured as given when `setup` is `None`; a
            resolved `setup` sets the backend instead.
        setup
            The compute setup of the likelihood (backend, device, operator, precision, transformer): `None`,
            `"auto"`, an `aa.Setup` or a setup_id string such as `"numba_cpu_sparse_fp64"`. It is resolved in
            `modify_before_fit`, because the rule key needs the model (pixelization mesh or MGE):

            - `None` (the default): nothing changes; one log line reports what `"auto"` would choose.
            - `"auto"`: `aa.Setup.auto` chooses the designated setup for the likelihood and hardware
              (`PYAUTO_SETUP=<setup_id>` overrides it).
            - An `aa.Setup` or setup_id string: that setup is used as given.

            A resolved setup sets the analysis backend, is applied to the dataset (`dataset.apply_setup`),
            checked for an operator / backend mismatch, pins the thread pools, is logged as a banner line and
            is written to `files/setup.json`.
        """
        super().__init__(
            cosmology=cosmology,
            use_jax=use_jax,
            **kwargs,
        )

        self._init_setup(setup)

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

        # The setup is resolved first: applying it may rebuild the dataset, so the grids below are
        # preloaded on the dataset the likelihood will use.

        self.resolve_setup(model=model)

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

    @staticmethod
    def _setup_environment():
        """
        The detected hardware (`autonerves.environment.Environment`), or `None` if detection fails
        (kept for backwards compatibility; see `setup_util.setup_environment`).
        """
        return setup_util.setup_environment()

    def setup_key_and_note_from(self, model: af.Collection):
        """
        The `(dataset_type, variant)` rule key of `aa.Setup.auto` for this analysis's dataset and the
        model (see `setup_util.setup_key_and_note_from`), plus a note when the nearest rule was used.
        """
        return setup_util.setup_key_and_note_from(dataset=self.dataset, model=model)

    def _apply_resolved_setup(self, setup: aa.Setup):
        """
        Apply the resolved setup to the dataset (`dataset.apply_setup`), which attaches the operator it
        names, then check the operator against the setup and the resolved backend with the inversion
        factory's check, so a mismatch raises an `InversionException` here rather than rejecting every
        sample. Called by `resolve_setup` (`AnalysisSetupMixin`) after `_use_jax` is set.
        """
        self.dataset = self.dataset.apply_setup(setup)

        setup_util.check_setup_consistency(dataset=self.dataset, use_jax=self._use_jax)

    @staticmethod
    def _save_dataset_fits(paths: af.AbstractPaths, hdu_list: HDUList) -> None:
        """Save the dataset for directory and database aggregators."""
        dataset_path = paths.image_path / "dataset.fits"
        if not dataset_path.exists():
            hdu_list.writeto(dataset_path, overwrite=True)

        if isinstance(paths, af.DatabasePaths) and not any(
            fits.name == "dataset" for fits in paths.fit.fits
        ):
            # PyAutoFit reloads database HDU data as big-endian float64.
            for hdu in hdu_list:
                if hdu.data is not None:
                    hdu.data = hdu.data.astype(">f8", copy=False)
            paths.save_fits(name="dataset", fits=hdu_list)

    def save_attributes(self, paths: af.DirectoryPaths):
        """
        Before the model-fit via the non-linear search begins, this routine saves attributes of the `Analysis` object
        to the `files` folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator
        tools.

        For this analysis the following are output:

        - The settings associated with the inversion.
        - The settings associated with the pixelization.
        - The Cosmology.
        - The resolved compute setup (`setup.json`), when a `setup` was given.

        The dataset itself is output as `dataset.fits` to the `image` folder by the `save_attributes` method of
        each dataset-specific subclass (e.g. `AnalysisImaging`), with the mask in its `PrimaryHDU`.
        Database paths also register the dataset FITS with the search's database fit.

        The following .fits files are also output via the plotter interface:

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

        self.save_setup(paths=paths)

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
        except (AttributeError, af.exc.SamplesException, af.exc.FitException) as e:
            # Building the galaxies requires materializing the maximum log likelihood
            # sample as a model instance, which the model may reject (e.g. `ell_comps`
            # outside the unit disk). Writing an extra output file must never kill a
            # completed fit before `paths.completed()` is called (PyAutoFit #1535), so
            # the failure is logged and the fit finishes without `galaxies.json`.
            logger.warning(
                f"The maximum log likelihood galaxies could not be written to "
                f"galaxies.json, the model-fit is otherwise unaffected:\n{e}"
            )

    def adapt_images_via_instance_from(
        self,
        instance: af.ModelInstance,
        dataset_model: Optional[aa.DatasetModel] = None,
        galaxies=None,
        xp=np,
    ) -> AdaptImages:
        try:
            return self.adapt_images.updated_via_instance_from(
                instance=instance,
                dataset_model=dataset_model,
                galaxies=galaxies,
                xp=xp,
            )
        except AttributeError:
            pass
