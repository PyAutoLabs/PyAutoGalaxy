"""
Aggregator interface for loading ``FitInterferometer`` objects from model-fit results.

This module combines all the pieces that are needed to re-create a ``FitInterferometer``
object offline — the interferometer dataset, galaxy model, dataset model, and adapt
images — and assembles them into a ``FitInterferometer`` instance identical to the one
produced during the original non-linear search.

Two public objects are provided:

- ``_fit_interferometer_from`` — a free function that accepts a single ``PyAutoFit``
  ``Fit`` entry and returns a list of ``FitInterferometer`` objects (one per summed
  ``Analysis``).
- ``FitInterferometerAgg`` — a ``PyAutoFit`` ``AggBase`` subclass wrapping an
  ``Aggregator`` that exposes a generator of ``FitInterferometer`` objects, enabling
  memory-efficient iteration over large result sets.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List

if TYPE_CHECKING:
    from autogalaxy.interferometer.fit_interferometer import FitInterferometer

import autofit as af
import autoarray as aa

from autogalaxy.aggregator import agg_util
from autogalaxy.aggregator.interferometer.interferometer import _interferometer_from
from autogalaxy.aggregator.dataset_model import _dataset_model_from
from autogalaxy.aggregator.galaxies import _galaxies_from


def _fit_interferometer_from(
    fit: af.Fit,
    instance: Optional[af.ModelInstance] = None,
    settings: aa.Settings = None,
) -> List[FitInterferometer]:
    """
    Returns a list of `FitInterferometer` objects from a `PyAutoFit` loaded directory `Fit` or sqlite database `Fit` object.

    The results of a model-fit can be loaded from hard-disk or stored in a sqlite database, including the following
    attributes of the fit:

    - The interferometer data, noise-map, uv-wavelengths and settings as .fits files (e.g. `dataset/data.fits`).
    - The real space mask defining the grid of the interferometer for the FFT (`dataset/real_space_mask.fits`).
    - The settings of inversions used by the fit (`dataset/settings.json`).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a `FitInterferometer` object for a given non-linear
    search sample (e.g. the maximum likelihood model). This includes associating adapt images with their respective
    galaxies.

    If multiple `FitInterferometer` objects were fitted simultaneously via analysis summing, the `fit.child_values()`
    method is instead used to load lists of the data, noise-map, PSF and mask and combine them into a list of
    `FitInterferometer` objects.

    The settings of an inversion can be overwritten by inputting a `settings` object, for
    example if you want to use a grid with a different inversion solver.

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry which has been loaded from
        an output directory or from an sqlite database..
    instance
        A manual instance that overwrites the max log likelihood instance in fit (e.g. for drawing the instance
        randomly from the PDF).
    settings
        Optionally overwrite the `Settings` of the `Inversion` object that is created from the fit.
    """
    from autogalaxy.interferometer.fit_interferometer import FitInterferometer

    dataset_list = _interferometer_from(
        fit=fit,
    )

    galaxies_list = _galaxies_from(fit=fit, instance=instance)

    dataset_model_list = _dataset_model_from(fit=fit, instance=instance)

    adapt_images_list = agg_util.adapt_images_from(fit=fit)

    settings = settings or fit.value(name="settings")

    fit_dataset_list = []

    for dataset, galaxies, dataset_model, adapt_images in zip(
        dataset_list,
        galaxies_list,
        dataset_model_list,
        adapt_images_list,
    ):

        fit_dataset_list.append(
            FitInterferometer(
                dataset=dataset,
                galaxies=galaxies,
                dataset_model=dataset_model,
                adapt_images=adapt_images,
                settings=settings,
            )
        )

    return fit_dataset_list


class FitInterferometerAgg(af.AggBase):
    def __init__(
        self,
        aggregator: af.Aggregator,
        settings: Optional[aa.Settings] = None,
    ):
        """
        Interfaces with an `PyAutoFit` aggregator object to create instances of `FitInterferometer` objects from the
        results of a model-fit.

        The results of a model-fit can be loaded from hard-disk or stored in a sqlite database, including the following
        attributes of the fit:

        - The interferometer data, noise-map, uv-wavelengths and settings as .fits files (e.g. `dataset/data.fits`).
        - The real space mask defining the grid of the interferometer for the FFT (`dataset/real_space_mask.fits`).
        - The settings of inversions used by the fit (`dataset/settings.json`).

        The `aggregator` contains the path to each of these files, and they can be loaded individually. This class
        can load them all at once and create an `FitInterferometer` object via the `_fit_interferometer_from` method.

        This class's methods returns generators which create the instances of the `FitInterferometer` objects. This ensures
        that large sets of results can be efficiently loaded from the hard-disk and do not require storing all
        `FitInterferometer` instances in the memory at once.

        For example, if the `aggregator` contains 3 model-fits, this class can be used to create a generator which
        creates instances of the corresponding 3 `FitInterferometer` objects.

        If multiple `Imaging` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
        is instead used to load lists of the data, noise-map, PSF and mask and combine them into a list of
        `FitImaging` objects.

        This can be done manually, but this object provides a more concise API.

        Parameters
        ----------
        aggregator
            A `PyAutoFit` aggregator object which can load the results of model-fits.
        settings
            Optionally overwrite the `Settings` of the `Inversion` object that is created from the fit.

        Parameters
        ----------
        aggregator
            A `PyAutoFit` aggregator object which can load the results of model-fits.
        settings
            Optionally overwrite the `Settings` of the `Inversion` object that is created from the fit.
        """
        super().__init__(aggregator=aggregator)

        self.settings = settings

    def object_via_gen_from(
        self, fit, instance: Optional[af.ModelInstance] = None
    ) -> List[FitInterferometer]:
        """
        Returns a generator of `FitInterferometer` objects from an input aggregator.

        See `__init__` for a description of how the `FitInterferometer` objects are created by this method.

        Parameters
        ----------
        fit
            A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry which has been loaded from
            an output directory or from an sqlite database..
        instance
            A manual instance that overwrites the max log likelihood instance in fit (e.g. for drawing the instance
            randomly from the PDF).
        """
        return _fit_interferometer_from(
            fit=fit,
            instance=instance,
            settings=self.settings,
        )
