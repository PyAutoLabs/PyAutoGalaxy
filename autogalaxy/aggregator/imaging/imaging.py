"""
Aggregator interface for loading ``Imaging`` datasets from model-fit results.

After a model-fit the imaging dataset (data array, noise-map, PSF, mask, and
over-sampling settings) is written to a FITS file in the output directory or SQLite
database.  This module reconstructs a fully-masked ``aa.Imaging`` object from those
stored files.

Two public objects are provided:

- ``_imaging_from`` — a free function that accepts a single ``PyAutoFit`` ``Fit`` entry
  and returns a list of ``aa.Imaging`` datasets (one per summed ``Analysis``).
- ``ImagingAgg`` — a class wrapping an ``Aggregator`` that returns a lazy generator of
  ``Imaging`` datasets, avoiding loading all results into memory at once.
"""
from functools import partial
from typing import List

from autonerves.fitsable import ndarray_via_hdu_from

import autofit as af
import autoarray as aa

from autogalaxy.aggregator import agg_util


def _imaging_from(
    fit: af.Fit,
) -> List[aa.Imaging]:
    """
    Returns a list of `Imaging` objects from a `PyAutoFit` loaded directory `Fit` or sqlite database `Fit` object.

    The results of a model-fit can be loaded from hard-disk or stored in a sqlite database, including the following
    attributes of the fit:

    - The mask used to mask the `Imaging` data structure in the fit (`dataset.fits[hdu=0]`).
    - The imaging data as a .fits file (`dataset.fits[hdu=1]`).
    - The noise-map as a .fits file (`dataset.fits[hdu=2]`).
    - The point spread function as a .fits file (`dataset.fits[hdu=3]`).
    - The settings of the `Imaging` data structure used in the fit (`dataset/settings.json`).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a `Imaging` object, has the mask applied to the
    `Imaging` data structure and its settings updated to the values used by the model-fit.

    If multiple `Imaging` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of the data, noise-map, PSF and mask and combine them into a list of
    `Imaging` objects.

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry which has been loaded from
        an output directory or from an sqlite database..
    """

    fit_list = [fit] if not fit.children else fit.children

    dataset_list = []

    for fit in fit_list:
        mask, header = agg_util.mask_header_from(fit=fit)

        def values_from(hdu: int, cls):
            return cls.no_mask(
                values=ndarray_via_hdu_from(fit.value(name="dataset")[hdu]),
                pixel_scales=mask.pixel_scales,
                header=header,
                origin=mask.origin,
            )

        data = values_from(hdu=1, cls=aa.Array2D)
        noise_map = values_from(hdu=2, cls=aa.Array2D)

        try:
            kernel = values_from(hdu=3, cls=aa.Array2D)
            psf = aa.Convolver(kernel=kernel)
        except (TypeError, IndexError):
            kernel = None
            psf = None

        dataset = aa.Imaging(
            data=data,
            noise_map=noise_map,
            psf=psf,
            check_noise_map=False,
        )

        dataset = dataset.apply_mask(mask=mask)

        try:
            over_sample_size_lp = values_from(hdu=4, cls=aa.Array2D).native
            over_sample_size_lp = over_sample_size_lp.apply_mask(mask=mask)
        except (TypeError, IndexError):
            over_sample_size_lp = 1

        try:
            over_sample_size_pixelization = values_from(hdu=5, cls=aa.Array2D).native
            over_sample_size_pixelization = over_sample_size_pixelization.apply_mask(
                mask=mask
            )
        except (TypeError, IndexError):
            over_sample_size_pixelization = 1

        dataset = dataset.apply_over_sampling(
            over_sample_size_lp=over_sample_size_lp,
            over_sample_size_pixelization=over_sample_size_pixelization,
        )

        dataset_list.append(dataset)

    return dataset_list


class ImagingAgg:
    def __init__(self, aggregator: af.Aggregator):
        """
            Interfaces with an `PyAutoFit` aggregator object to create instances of `Imaging` objects from the results
            of a model-fit.

            The results of a model-fit can be loaded from hard-disk or stored in a sqlite database, including the following
        attributes of the fit:

            - The imaging data as a .fits file (`dataset.fits[hdu=1]`).
            - The noise-map as a .fits file (`dataset.fits[hdu=2]`).
            - The point spread function as a .fits file (`dataset.fits[hdu=3]`).
            - The settings of the `Imaging` data structure used in the fit (`dataset/settings.json`).
            - The mask used to mask the `Imaging` data structure in the fit (`dataset.fits[hdu=0]`).

            The `aggregator` contains the path to each of these files, and they can be loaded individually. This class
            can load them all at once and create an `Imaging` object via the `_imaging_from` method.

            This class's methods returns generators which create the instances of the `Imaging` objects. This ensures
            that large sets of results can be efficiently loaded from the hard-disk and do not require storing all
            `Imaging` instances in the memory at once.

            For example, if the `aggregator` contains 3 model-fits, this class can be used to create a generator which
            creates instances of the corresponding 3 `Imaging` objects.

            If multiple `Imaging` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
            is instead used to load lists of the data, noise-map, PSF and mask and combine them into a list of
            `Imaging` objects.

            This can be done manually, but this object provides a more concise API.

            Parameters
            ----------
            aggregator
                A `PyAutoFit` aggregator object which can load the results of model-fits.
        """
        self.aggregator = aggregator

    def dataset_gen_from(
        self,
    ) -> List[aa.Imaging]:
        """
        Returns a generator of `Imaging` objects from an input aggregator.

        See `__init__` for a description of how the `Imaging` objects are created by this method.
        """

        func = partial(_imaging_from)

        return self.aggregator.map(func=func)
