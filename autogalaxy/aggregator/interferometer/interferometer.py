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
from functools import partial
from typing import List

import autofit as af
import autoarray as aa

from autogalaxy.aggregator import agg_util


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

        data = aa.Visibilities(
            visibilities=fit.value(name="dataset")[1].data.astype("float")
        )
        noise_map = aa.VisibilitiesNoiseMap(
            fit.value(name="dataset")[2].data.astype("float")
        )
        uv_wavelengths = fit.value(name="dataset")[3].data

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
