"""
Aggregator interface for loading ``FitEllipse`` objects from ellipse model-fit results.

This module reconstructs ``FitEllipse`` instances from the imaging dataset, fitted
ellipses, and (optionally) multipole perturbations stored in an output directory or
SQLite database, allowing full offline inspection of isophote fits without re-running the
non-linear search.

Two public objects are provided:

- ``_fit_ellipse_from`` — a free function that accepts a single ``PyAutoFit`` ``Fit``
  entry and returns a list of ``FitEllipse`` objects (one per ellipse in the fit).
- ``FitEllipseAgg`` — a ``PyAutoFit`` ``AggBase`` subclass wrapping an ``Aggregator``
  that exposes a generator of ``FitEllipse`` object lists, one entry per stored
  model-fit.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List

if TYPE_CHECKING:
    from autogalaxy.ellipse.fit_ellipse import FitEllipse

import autofit as af

from autogalaxy.aggregator.imaging.imaging import _imaging_from
from autogalaxy.aggregator.ellipse.ellipses import _ellipses_from
from autogalaxy.aggregator.ellipse.multipoles import _multipoles_from


def _fit_ellipse_from(
    fit: af.Fit,
    instance: Optional[af.ModelInstance] = None,
) -> List[List[FitEllipse]]:
    """
    Returns a list of `FitEllipse` objects from a `PyAutoFit` loaded directory `Fit` or sqlite database `Fit` object.

    The results of a model-fit can be loaded from hard-disk or stored in a sqlite database, including the following
    attributes of the fit:

    - The imaging data, noise-map, PSF and settings as .fits files (e.g. `dataset/data.fits`).
    - The mask used to mask the `Imaging` data structure in the fit (`dataset.fits[hdu=0]`).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a `FitEllipse` object for a given non-linear search sample
    (e.g. the maximum likelihood model). This includes associating adapt images with their respective galaxies.

    If multiple `FitEllipse` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of the data, noise-map, PSF and mask and combine them into a list of
    `FitEllipse` objects.

    The settings of an inversion can be overwritten by inputting a `settings` object, for example
    if you want to use a grid with a different inversion solver.

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry which has been loaded from
        an output directory or from an sqlite database..
    instance
        A manual instance that overwrites the max log likelihood instance in fit (e.g. for drawing the instance
        randomly from the PDF).
    """

    from autogalaxy.ellipse.fit_ellipse import FitEllipse

    dataset_list = _imaging_from(fit=fit)

    ellipse_list_list = _ellipses_from(fit=fit, instance=instance)

    try:
        multipole_list_list = _multipoles_from(fit=fit, instance=instance)
    except AttributeError:
        multipole_list_list = [[None for i in ellipse_list_list[0]]]

    fit_dataset_list = []

    for dataset, ellipse_list, multipole_lists in zip(
        dataset_list, ellipse_list_list, multipole_list_list
    ):
        for ellipse, multipole_list in zip(ellipse_list, multipole_lists):
            fit_dataset_list.append(
                FitEllipse(
                    dataset=dataset,
                    ellipse=ellipse,
                    multipole_list=multipole_list,
                )
            )

    return [fit_dataset_list]


class FitEllipseAgg(af.AggBase):
    def __init__(
        self,
        aggregator: af.Aggregator,
    ):
        """
            Interfaces with an `PyAutoFit` aggregator object to create instances of `FitEllipse` objects from the results
            of a model-fit.

            The results of a model-fit can be loaded from hard-disk or stored in a sqlite database, including the following
        attributes of the fit:

            - The imaging data, noise-map, PSF and settings as .fits files (e.g. `dataset/data.fits`).
            - The mask used to mask the `Imaging` data structure in the fit (`dataset.fits[hdu=0]`).

            The `aggregator` contains the path to each of these files, and they can be loaded individually. This class
            can load them all at once and create an `FitEllipse` object via the `_fit_ellipse_from` method.

            This class's methods returns generators which create the instances of the `FitEllipse` objects. This ensures
            that large sets of results can be efficiently loaded from the hard-disk and do not require storing all
            `FitEllipse` instances in the memory at once.

            For example, if the `aggregator` contains 3 model-fits, this class can be used to create a generator which
            creates instances of the corresponding 3 `FitEllipse` objects.

            If multiple `Ellipse` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
            is instead used to load lists of the data, noise-map, PSF and mask and combine them into a list of
            `FitEllipse` objects.

            This can be done manually, but this object provides a more concise API.

            Parameters
            ----------
            aggregator
                A `PyAutoFit` aggregator object which can load the results of model-fits.
        """
        super().__init__(aggregator=aggregator)

    def object_via_gen_from(
        self, fit, instance: Optional[af.ModelInstance] = None
    ) -> List[List[FitEllipse]]:
        """
        Returns a generator of `FitEllipse` objects from an input aggregator.

        See `__init__` for a description of how the `FitEllipse` objects are created by this method.

        Parameters
        ----------
        fit
            A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry which has been loaded from
        an output directory or from an sqlite database..
        instance
            A manual instance that overwrites the max log likelihood instance in fit (e.g. for drawing the instance
            randomly from the PDF).
        """
        return _fit_ellipse_from(
            fit=fit,
            instance=instance,
        )
