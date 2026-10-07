"""
Aggregator interface for loading ``Interferometer`` datasets from model-fit results.

After an interferometer model-fit the dataset (complex visibilities, noise-map,
uv-wavelengths, real-space mask, and transformer class) is written to a FITS file in
the output directory or SQLite database.  This module reconstructs a fully-configured
``aa.Interferometer`` object from those stored files.

Two public objects are provided:

- ``_interferometer_from`` — a free function that accepts a single ``PyAutoFit`` ``Fit``
  entry and returns a list of ``aa.Interferometer`` datasets (one per summed
  ``Analysis``).
- ``InterferometerAgg`` — a class wrapping an ``Aggregator`` that returns a lazy
  generator of ``Interferometer`` datasets, avoiding loading all results into memory at
  once.
"""

from __future__ import annotations
from functools import partial
from typing import List

import numpy as np

import autofit as af
import autoarray as aa

from autogalaxy.aggregator import agg_util
from autogalaxy.interferometer.model.analysis import (
    SPARSE_TERMS_HEADER_KEYS,
    SPARSE_TERMS_SCALARS_ORDER,
)

# The positions of the HDUs of the legacy in-memory `dataset.fits` layout (mask / data /
# noise_map / uv_wavelengths), used when a file predates EXTNAMEs being read by name.
_LEGACY_HDU_INDEX = {"mask": 0, "data": 1, "noise_map": 2, "uv_wavelengths": 3}


def _hdu(hdu_list, name: str):
    """
    Returns the HDU of `hdu_list` whose EXTNAME is `name` (case-insensitive), falling back to its position
    in the legacy in-memory `dataset.fits` layout (`_LEGACY_HDU_INDEX`) if no HDU has that EXTNAME.
    """
    try:
        return hdu_list[name.upper()]
    except KeyError:
        return hdu_list[_LEGACY_HDU_INDEX[name]]


def _has_hdu(hdu_list, name: str) -> bool:
    """
    Returns whether `hdu_list` has an HDU whose EXTNAME is `name` (case-insensitive).
    """
    try:
        hdu_list[name.upper()]
    except KeyError:
        return False

    return True


def _sparse_terms_from(hdu_list, real_space_mask: aa.Mask2D) -> aa.SparseTerms:
    """
    Returns the `SparseTerms` of an array-free interferometer dataset from the `dataset.fits` HDU list written
    by `autogalaxy.interferometer.model.analysis.interferometer_hdu_list_from`.

    The arrays are read from the HDUs named `NUFFT_PRECISION_OPERATOR`, `DIRTY_IMAGE` and `DIRTY_BEAM`, the
    numeric scalars losslessly from the `SPARSE_TERMS_SCALARS` HDU (order `SPARSE_TERMS_SCALARS_ORDER`),
    falling back to the primary header cards (`SPARSE_TERMS_HEADER_KEYS`, which truncate exponent-form
    float64 values) for files written before that HDU existed, the transformer class name from the header,
    and the mask provenance (shape, pixel scales, origin) from the reloaded real-space mask.

    The `(y, x)` phase centre is reloaded from the same scalars / header cards; a file written before it was
    persisted (a shorter `SPARSE_TERMS_SCALARS` array and no `PHCENTY` / `PHCENTX` cards) reloads it as `None`
    (not recorded), as does a `NaN` entry.

    The `oversample` fine grids are read from the `PRECISION_OPERATOR_FINE` / `DIRTY_IMAGE_FINE` HDUs, with
    `oversample` / `oversample_pad` from the scalars, when the file has them (they are written only on
    request, see `interferometer_hdu_list_from(include_fine_grids=True)`). A file without them -- written
    without opting in, or before they existed -- reloads all four as `None`.
    """
    header = hdu_list[0].header

    if _has_hdu(hdu_list, "sparse_terms_scalars"):
        scalars = dict(
            zip(
                SPARSE_TERMS_SCALARS_ORDER,
                np.asarray(_hdu(hdu_list, "sparse_terms_scalars").data, dtype="float64"),
            )
        )

        def header_value(field):
            if field in scalars:
                value = scalars[field]
                return None if np.isnan(value) else value
            # Not in the array: a file written before this entry was appended, so the header card (if
            # any) is the only record.
            return header.get(SPARSE_TERMS_HEADER_KEYS[field])

    else:

        def header_value(field):
            return header.get(SPARSE_TERMS_HEADER_KEYS[field])

    eps = header_value("eps")
    transformer_class_name = header_value("transformer_class_name")
    phase_centre_y = header_value("phase_centre_y")
    phase_centre_x = header_value("phase_centre_x")

    phase_centre = (
        None
        if phase_centre_y is None or phase_centre_x is None
        else (float(phase_centre_y), float(phase_centre_x))
    )

    oversample = header_value("oversample")
    oversample_pad = header_value("oversample_pad")

    if (
        _has_hdu(hdu_list, "precision_operator_fine")
        and _has_hdu(hdu_list, "dirty_image_fine")
        and oversample is not None
    ):
        fine_grids = dict(
            precision_operator_fine=np.asarray(
                _hdu(hdu_list, "precision_operator_fine").data, dtype="float64"
            ),
            dirty_image_fine=np.asarray(
                _hdu(hdu_list, "dirty_image_fine").data, dtype="float64"
            ),
            oversample=int(oversample),
            oversample_pad=None if oversample_pad is None else float(oversample_pad),
        )
    else:
        fine_grids = {}

    return aa.SparseTerms(
        nufft_precision_operator=np.asarray(
            _hdu(hdu_list, "nufft_precision_operator").data, dtype="float64"
        ),
        dirty_image_native=np.asarray(
            _hdu(hdu_list, "dirty_image").data, dtype="float64"
        ),
        dirty_beam_native=np.asarray(_hdu(hdu_list, "dirty_beam").data, dtype="float64"),
        sum_weights=float(header_value("sum_weights")),
        data_term=float(header_value("data_term")),
        noise_normalization=float(header_value("noise_normalization")),
        n_vis=int(header_value("n_vis")),
        shape_native=tuple(real_space_mask.shape_native),
        pixel_scales=tuple(float(value) for value in real_space_mask.pixel_scales),
        origin=tuple(float(value) for value in real_space_mask.origin),
        eps=None if eps is None else float(eps),
        transformer_class_name=(
            None if transformer_class_name is None else str(transformer_class_name)
        ),
        phase_centre=phase_centre,
        **fine_grids,
    )


def _interferometer_from(
    fit: af.Fit,
) -> List[aa.Interferometer]:
    """
    Returns a list of `Interferometer` objects from a `PyAutoFit` loaded directory `Fit` or sqlite database `Fit` object.

    The results of a model-fit can be loaded from hard-disk or stored in a sqlite database, including the following
    attributes of the fit:

    - The interferometer visibilities data as a .fits file (`dataset.fits[hdu=1]`).
    - The visibilities noise-map as a .fits file (`dataset.fits[hdu=2]`).
    - The uv wavelengths as a .fits file (`dataset/uv_wavelengths.fits`).
    - The real space mask defining the grid of the interferometer for the FFT (`dataset/real_space_mask.fits`).
    - The settings of the `Interferometer` data structure used in the fit (`dataset/settings.json`).

    A fit of an array-free dataset (built by `Interferometer.from_stream` / `from_sparse_terms`) stores its
    `SparseTerms` in `dataset.fits` instead of the visibilities (an HDU named `NUFFT_PRECISION_OPERATOR` is
    present). It is rebuilt via `Interferometer.from_sparse_terms`, so the returned dataset is array-free with
    its sparse operator re-attached, and a fit of it reproduces the original `log_evidence`.

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a `Interferometer` object, including having its
    settings updated to the values used by the model-fit.

    If multiple `Interferometer` objects were fitted simultaneously via analysis summing, the `fit.child_values()`
    method is instead used to load lists of the data, noise-map, PSF and mask and combine them into a list of
    `Interferometer` objects.

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry which has been loaded from
        an output directory or from an sqlite database..
    """

    fit_list = [fit] if not fit.children else fit.children

    dataset_list = []

    for fit in fit_list:
        real_space_mask, header = agg_util.mask_header_from(fit=fit)

        hdu_list = fit.value(name="dataset")

        if _has_hdu(hdu_list, "nufft_precision_operator"):
            dataset_list.append(
                aa.Interferometer.from_sparse_terms(
                    _sparse_terms_from(hdu_list, real_space_mask=real_space_mask),
                    real_space_mask=real_space_mask,
                )
            )
            continue

        data = aa.Visibilities(
            visibilities=_hdu(hdu_list, "data").data.astype("float")
        )
        noise_map = aa.VisibilitiesNoiseMap(
            _hdu(hdu_list, "noise_map").data.astype("float")
        )
        uv_wavelengths = _hdu(hdu_list, "uv_wavelengths").data

        transformer_class = fit.value(name="transformer_class")

        dataset = aa.Interferometer(
            data=data,
            noise_map=noise_map,
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            transformer_class=transformer_class,
        )

        dataset_list.append(dataset)

    return dataset_list


class InterferometerAgg:
    def __init__(self, aggregator: af.Aggregator):
        """
        Interfaces with an `PyAutoFit` aggregator object to create instances of `Interferometer` objects from the results
        of a model-fit.

        The results of a model-fit can be loaded from hard-disk or stored in a sqlite database, including the following
        attributes of the fit:

        - The interferometer visibilities data as a .fits file (`dataset.fits[hdu=1]`).
        - The visibilities noise-map as a .fits file (`dataset.fits[hdu=2]`).
        - The uv wavelengths as a .fits file (`dataset/uv_wavelengths.fits`).
        - The real space mask defining the grid of the interferometer for the FFT (`dataset/real_space_mask.fits`).
        - The settings of the `Interferometer` data structure used in the fit (`dataset/settings.json`).

    A fit of an array-free dataset (built by `Interferometer.from_stream` / `from_sparse_terms`) stores its
    `SparseTerms` in `dataset.fits` instead of the visibilities (an HDU named `NUFFT_PRECISION_OPERATOR` is
    present). It is rebuilt via `Interferometer.from_sparse_terms`, so the returned dataset is array-free with
    its sparse operator re-attached, and a fit of it reproduces the original `log_evidence`.

        The `aggregator` contains the path to each of these files, and they can be loaded individually. This class
        can load them all at once and create an `Interferometer` object via the `_interferometer_from` method.

        This class's methods returns generators which create the instances of the `Interferometer` objects. This ensures
        that large sets of results can be efficiently loaded from the hard-disk and do not require storing all
        `Interferometer` instances in the memory at once.

        For example, if the `aggregator` contains 3 model-fits, this class can be used to create a generator which
        creates instances of the corresponding 3 `Interferometer` objects.

        If multiple `Interferometer` objects were fitted simultaneously via analysis summing, the `fit.child_values()`
        method is instead used to load lists of the data, noise-map, PSF and mask and combine them into a list of
        `Interferometer` objects.

        This can be done manually, but this object provides a more concise API.

        Parameters
        ----------
        aggregator
            A `PyAutoFit` aggregator object which can load the results of model-fits.
        """
        self.aggregator = aggregator

    def dataset_gen_from(
        self,
    ) -> List[aa.Interferometer]:
        """
        Returns a generator of `Interferometer` objects from an input aggregator.

        See `__init__` for a description of how the `Interferometer` objects are created by this method.
        """
        func = partial(
            _interferometer_from,
        )

        return self.aggregator.map(func=func)
