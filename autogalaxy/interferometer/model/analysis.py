"""
`AnalysisInterferometer` — the **PyAutoFit** `Analysis` class for fitting galaxy models to interferometer data.

This module provides `AnalysisInterferometer`, which implements `log_likelihood_function` by:

1. Extracting galaxies from the model instance.
2. Constructing a `FitInterferometer` object from those galaxies and the stored `Interferometer` dataset.
3. Returning the `figure_of_merit` of the fit (log-likelihood or log-evidence).

It also handles adapt images, visualization, and result wrapping into `ResultInterferometer`.
"""

import logging
import numpy as np
from typing import Optional

from autonerves.dictable import to_dict
from autonerves.fitsable import hdu_list_for_output_from

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


_FIT_INTERFEROMETER_PYTREES_REGISTERED = False


# The FITS header keys (at most 8 characters, so no `HIERARCH` cards are needed) under which
# `interferometer_hdu_list_from` stores the scalar fields and recorded provenance of the
# `SparseTerms` of an array-free dataset, and from which the aggregator reloads them.
SPARSE_TERMS_HEADER_KEYS = {
    "sum_weights": "SUMW",
    "data_term": "DATATERM",
    "noise_normalization": "NOISENRM",
    "n_vis": "NVIS",
    "eps": "EPS",
    "transformer_class_name": "TRNSFRMR",
    "phase_centre_y": "PHCENTY",
    "phase_centre_x": "PHCENTX",
    "oversample": "OVERSAMP",
    "oversample_pad": "OVERSPAD",
}

# The order of the float64 1-D array in the `SPARSE_TERMS_SCALARS` HDU, the lossless store of the numeric
# `SparseTerms` scalars. A FITS header card holds at most 20 characters of value, so astropy truncates
# exponent-form float64 values (e.g. `1.2345678901234567e+20` is written as `1.23456789012345E+20`); the
# header cards under `SPARSE_TERMS_HEADER_KEYS` are kept as human-readable copies only. `eps`, the
# `(y, x)` phase centre (arcsec) and the `oversample` factor / pad of the fine grids are `NaN` when not
# recorded (the oversampling is recorded only when the fine grids themselves are written, so a reloaded
# `SparseTerms` never claims an `oversample` without its grids). Entries are only ever appended, so a
# file written before an entry existed has a shorter array and the loader treats the missing entries as
# not recorded.
SPARSE_TERMS_SCALARS_ORDER = (
    "sum_weights",
    "data_term",
    "noise_normalization",
    "n_vis",
    "eps",
    "phase_centre_y",
    "phase_centre_x",
    "oversample",
    "oversample_pad",
)


def _primary_beam_native_from(dataset: aa.Interferometer) -> Optional[np.ndarray]:
    """
    Returns the primary beam of `dataset` as a float64 array of the native shape of its real-space mask
    (masked pixels 0), as written to the `PRIMARY_BEAM` HDU of `dataset.fits`, or `None` if the dataset
    has no primary beam.
    """
    if dataset.primary_beam_slim is None:
        return None

    return np.asarray(
        aa.Array2D(
            values=dataset.primary_beam_slim, mask=dataset.real_space_mask
        ).native.array,
        dtype="float64",
    )


def interferometer_hdu_list_from(
    dataset: aa.Interferometer, include_fine_grids: bool = False
):
    """
    Returns the `HDUList` written to `dataset.fits` by `AnalysisInterferometer.save_attributes`, from which
    the aggregator (`autogalaxy.aggregator.interferometer.interferometer._interferometer_from`) reloads the
    dataset.

    An in-memory dataset writes the real-space mask (in the `PrimaryHDU`), the visibilities, the noise-map and
    the uv-wavelengths, with EXTNAMEs `MASK`, `DATA`, `NOISE_MAP` and `UV_WAVELENGTHS`.

    An array-free dataset (built by `Interferometer.from_stream` / `from_sparse_terms`) has no visibility arrays,
    so it instead writes its `SparseTerms`: the real-space mask (in the `PrimaryHDU`), the NUFFT precision
    operator `W~` and the native noise-weighted dirty image and dirty beam, with EXTNAMEs `MASK`,
    `NUFFT_PRECISION_OPERATOR`, `DIRTY_IMAGE` and `DIRTY_BEAM`. The scalar terms and recorded provenance are
    stored losslessly as a float64 1-D array in the `SPARSE_TERMS_SCALARS` HDU (order
    `SPARSE_TERMS_SCALARS_ORDER`), and copied for readability to the header under `SPARSE_TERMS_HEADER_KEYS`
    (`EPS` / `TRNSFRMR` / `PHCENTY` / `PHCENTX` only when recorded; `TRNSFRMR` is stored only in the header,
    and the phase centre the visibilities were re-centred on is stored as its `(y, x)` arcsec pair); the mask
    shape, pixel scales and origin come from the mask itself. The file size is set by the real-space grid, not
    the number of visibilities.

    Terms accumulated with `oversample=q` (`Interferometer.from_stream(..., oversample=q)`) also carry the
    fine precision-operator and dirty-image grids. These are written -- as HDUs `PRECISION_OPERATOR_FINE`
    and `DIRTY_IMAGE_FINE` after `SPARSE_TERMS_SCALARS`, with `oversample` / `oversample_pad` in the scalars
    and the `OVERSAMP` / `OVERSPAD` cards -- **only** when `include_fine_grids=True`. They are not written
    by default because `dataset.fits` is written into every search's output by `save_attributes` and the
    grids grow as `q^2`: ~840 MB per file at a 400 x 400 grid and `q = 8`. Without them the file is the
    same as for terms without fine grids (the oversampling scalars are `NaN`), and reloads with no fine
    grids; rebuild them from the visibilities, or write this HDU list yourself with
    `include_fine_grids=True`, when they are needed.

    A dataset with a primary beam (`Interferometer(primary_beam=...)`) also writes it, in both layouts, as a
    trailing HDU with EXTNAME `PRIMARY_BEAM`: a float64 array of the native shape of the real-space mask with
    masked pixels 0. It is the last HDU, so the positional indices of the legacy in-memory layout (0-3) are
    unchanged, and a dataset without a primary beam writes no such HDU (the file is identical to one written
    before the beam was persisted). The aggregator detects it by name and passes it back as `primary_beam=`,
    so a fit of the reloaded dataset reproduces the beamed likelihood.

    Parameters
    ----------
    dataset
        The interferometer dataset being fitted.
    include_fine_grids
        If `True` and the dataset's `SparseTerms` carry `oversample` fine grids, write them too.
    """
    mask = dataset.real_space_mask

    primary_beam_native = _primary_beam_native_from(dataset)

    if primary_beam_native is None:
        primary_beam_values = []
        primary_beam_ext_names = []
    else:
        primary_beam_values = [primary_beam_native]
        primary_beam_ext_names = ["primary_beam"]

    if not dataset.is_array_free:
        return hdu_list_for_output_from(
            values_list=[
                mask.astype("float"),
                dataset.data.in_array,
                dataset.noise_map.in_array,
                dataset.uv_wavelengths,
                *primary_beam_values,
            ],
            ext_name_list=[
                "mask",
                "data",
                "noise_map",
                "uv_wavelengths",
                *primary_beam_ext_names,
            ],
            header_dict=mask.header_dict,
        )

    terms = dataset.sparse_terms

    if terms is None:
        raise aa.exc.DatasetException(
            "An array-free Interferometer must carry the `SparseTerms` it was built from (`sparse_terms`) "
            "for its dataset to be saved; build it via `Interferometer.from_stream` / `from_sparse_terms`."
        )

    header_dict = {
        **mask.header_dict,
        SPARSE_TERMS_HEADER_KEYS["sum_weights"]: float(terms.sum_weights),
        SPARSE_TERMS_HEADER_KEYS["data_term"]: float(terms.data_term),
        SPARSE_TERMS_HEADER_KEYS["noise_normalization"]: float(
            terms.noise_normalization
        ),
        SPARSE_TERMS_HEADER_KEYS["n_vis"]: int(terms.n_vis),
    }

    if terms.eps is not None:
        header_dict[SPARSE_TERMS_HEADER_KEYS["eps"]] = float(terms.eps)

    if terms.transformer_class_name is not None:
        header_dict[SPARSE_TERMS_HEADER_KEYS["transformer_class_name"]] = str(
            terms.transformer_class_name
        )

    if terms.phase_centre is not None:
        header_dict[SPARSE_TERMS_HEADER_KEYS["phase_centre_y"]] = float(
            terms.phase_centre[0]
        )
        header_dict[SPARSE_TERMS_HEADER_KEYS["phase_centre_x"]] = float(
            terms.phase_centre[1]
        )

    phase_centre = (
        (np.nan, np.nan) if terms.phase_centre is None else terms.phase_centre
    )

    write_fine_grids = include_fine_grids and (
        terms.precision_operator_fine is not None and terms.dirty_image_fine is not None
    )

    if write_fine_grids:
        header_dict[SPARSE_TERMS_HEADER_KEYS["oversample"]] = int(terms.oversample)
        header_dict[SPARSE_TERMS_HEADER_KEYS["oversample_pad"]] = float(
            terms.oversample_pad
        )
        oversample = (float(terms.oversample), float(terms.oversample_pad))
        fine_values = [
            np.asarray(terms.precision_operator_fine, dtype="float64"),
            np.asarray(terms.dirty_image_fine, dtype="float64"),
        ]
        fine_ext_names = ["precision_operator_fine", "dirty_image_fine"]
    else:
        oversample = (np.nan, np.nan)
        fine_values = []
        fine_ext_names = []

    return hdu_list_for_output_from(
        values_list=[
            mask.astype("float"),
            np.asarray(terms.nufft_precision_operator, dtype="float64"),
            np.asarray(terms.dirty_image_native, dtype="float64"),
            np.asarray(terms.dirty_beam_native, dtype="float64"),
            np.array(
                [
                    float(terms.sum_weights),
                    float(terms.data_term),
                    float(terms.noise_normalization),
                    float(terms.n_vis),
                    np.nan if terms.eps is None else float(terms.eps),
                    float(phase_centre[0]),
                    float(phase_centre[1]),
                    float(oversample[0]),
                    float(oversample[1]),
                ],
                dtype="float64",
            ),
            *fine_values,
            *primary_beam_values,
        ],
        ext_name_list=[
            "mask",
            "nufft_precision_operator",
            "dirty_image",
            "dirty_beam",
            "sparse_terms_scalars",
            *fine_ext_names,
            *primary_beam_ext_names,
        ],
        header_dict=header_dict,
    )


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
        setup=None,
        **kwargs,
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
        use_jax
            Whether the likelihood is evaluated with JAX; honoured as given when `setup` is `None`.
        setup
            The compute setup of the likelihood: `None` (default, nothing changes), `"auto"` (the designated
            setup for the likelihood and hardware, chosen at fit start from the model), an `aa.Setup` or a
            setup_id string. See `AnalysisDataset`.
        """
        super().__init__(
            dataset=dataset,
            adapt_images=adapt_images,
            cosmology=cosmology,
            settings=settings,
            title_prefix=title_prefix,
            use_jax=use_jax,
            setup=setup,
            **kwargs,
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

        if self._use_jax:
            self._register_fit_interferometer_pytrees()

        galaxies = self.galaxies_via_instance_from(
            instance=instance,
        )

        adapt_images = self.adapt_images_via_instance_from(
            instance=instance, galaxies=galaxies
        )

        return FitInterferometer(
            dataset=self.dataset,
            galaxies=galaxies,
            adapt_images=adapt_images,
            settings=self.settings,
            xp=self._xp,
        )

    @staticmethod
    def _register_fit_interferometer_pytrees() -> None:
        """Register every type reachable from a ``FitInterferometer`` return
        value so ``jax.jit(fit_from)`` can flatten its output.

        ``dataset``, ``adapt_images`` and ``settings`` are constants per
        analysis — ride as aux so JAX does not recurse into them. Everything
        else (``galaxies`` and the autoarray wrappers it carries) is dynamic
        per fit.

        Idempotent — guarded by the module-level
        ``_FIT_INTERFEROMETER_PYTREES_REGISTERED`` flag. See
        ``autogalaxy/imaging/model/analysis.py`` for the cross-registration
        rationale.
        """
        global _FIT_INTERFEROMETER_PYTREES_REGISTERED
        if _FIT_INTERFEROMETER_PYTREES_REGISTERED:
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
            FitInterferometer,
            no_flatten=("dataset", "adapt_images", "settings"),
        )
        register_instance_pytree(DatasetModel)
        register_galaxies_pytree()

        _FIT_INTERFEROMETER_PYTREES_REGISTERED = True

    def save_attributes(self, paths: af.AbstractPaths):
        """
        Before the model-fit begins, this routine saves attributes of the `Analysis` object to the `files` folder
        such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

        It outputs the following attributes of the dataset:



        For this analysis, it uses the `AnalysisDataset` object's method to output the following:

        - The settings associated with the inversion.
        - The settings associated with the pixelization.
        - The Cosmology.
        - The adapt image's model image and galaxy images, as `adapt_images.fits`, if used.

        It also outputs, to the `image` folder, the file `dataset.fits`, which contains:

        - The real space mask applied to the dataset, in the `PrimaryHDU`.
        - The interferometer dataset (data / noise-map / uv_wavelengths), or for an array-free dataset
          its `SparseTerms` (see `interferometer_hdu_list_from`).
        - The primary beam on the real-space mask (native, masked pixels 0), as a trailing `PRIMARY_BEAM`
          HDU, if the dataset has one, so the aggregator-rebuilt dataset and fit carry it.

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

        hdu_list = interferometer_hdu_list_from(dataset=self.dataset)

        self._save_dataset_fits(paths=paths, hdu_list=hdu_list)

        # An array-free dataset has no transformer; its transformer class name is recorded in
        # the `dataset.fits` header instead.
        if self.dataset.transformer is not None:
            paths.save_json(
                "transformer_class",
                to_dict(self.dataset.transformer.__class__),
            )
