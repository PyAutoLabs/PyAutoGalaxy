"""
Aggregator interface for loading ``EllipseMultipole`` objects from ellipse model-fit results.

``EllipseMultipole`` objects represent higher-order harmonic perturbations (e.g. m=3, m=4
boxy/disky distortions) on top of the base ellipse isophotes.  This module reconstructs
the nested list of multipole objects from a stored ``PyAutoFit`` ``Fit`` entry so that
multipole-enhanced isophote fits can be inspected offline.

Two public objects are provided:

- ``_multipoles_from`` — a free function that accepts a single ``PyAutoFit`` ``Fit``
  entry and returns the nested list of ``EllipseMultipole`` objects for the requested
  model instance.
- ``MultipolesAgg`` — a ``PyAutoFit`` ``AggBase`` subclass wrapping an ``Aggregator``
  that exposes a generator of multipole lists, one per stored model-fit.
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional, List

if TYPE_CHECKING:
    from autogalaxy.ellipse.ellipse.ellipse_multipole import EllipseMultipole

import autofit as af


logger = logging.getLogger(__name__)


def _multipoles_from(
    fit: af.Fit, instance: af.ModelInstance
) -> List[List[List[EllipseMultipole]]]:
    """
    Returns a list of `EllipseMultipole` objects from a `PyAutoFit` loaded directory `Fit` or sqlite database `Fit` object.

    The results of a model-fit can be loaded from hard-disk or stored in a sqlite database, including the following
    attributes of the fit:

    - The model and its best fit parameters (e.g. `model.json`).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a list of `EllipseMultipole` object for a given non-linear search
    sample (e.g. the maximum likelihood model).

    If multiple `EllipseMultipole` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of multipoles. This is necessary if each analysis has different multipoles (e.g. certain
    parameters vary across each dataset and `Analysis` object).

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry which has been loaded from
        an output directory or from an sqlite database..
    instance
        A manual instance that overwrites the max log likelihood instance in fit (e.g. for drawing the instance
        randomly from the PDF).
    """

    if instance is not None:
        multipoles = instance.multipoles
    else:
        multipoles = fit.instance.multipoles

    if fit.children is not None:
        if len(fit.children) > 0:
            logger.info(
                """
                Using database for a fit with multiple summed Analysis objects.

                EllipseMultipole objects do not fully support this yet (e.g. variables across Analysis objects may not be correct)
                so proceed with caution!
                """
            )

            return [multipoles] * len(fit.children)

    return [multipoles]


class MultipolesAgg(af.AggBase):
    """
    Interfaces with an `PyAutoFit` aggregator object to create instances of `EllipseMultipole` objects from the results
    of a model-fit.

    The results of a model-fit can be loaded from hard-disk or stored in a sqlite database, including the following
    attributes of the fit:

    - The model and its best fit parameters (e.g. `model.json`).

    The `aggregator` contains the path to each of these files, and they can be loaded individually. This class
    can load them all at once and create lists of `EllipseMultipole` objects via the `_multipoles_from` method.

    This class's methods returns generators which create the instances of the `EllipseMultipole` objects. This ensures
    that large sets of results can be efficiently loaded from the hard-disk and do not require storing all
    `EllipseMultipole` instances in the memory at once.

    For example, if the `aggregator` contains 3 model-fits, this class can be used to create a generator which
    creates instances of the corresponding 3 `EllipseMultipole` objects.

    If multiple `EllipseMultipole` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of multipoles. This is necessary if each analysis has different multipoles (e.g. certain
    parameters vary across each dataset and `Analysis` object).

    This can be done manually, but this object provides a more concise API.

    Parameters
    ----------
    aggregator
        A `PyAutoFit` aggregator object which can load the results of model-fits.
    """

    def object_via_gen_from(
        self, fit, instance: Optional[af.ModelInstance] = None
    ) -> List[List[List[EllipseMultipole]]]:
        """
        Returns a generator of `EllipseMultipole` objects from an input aggregator.

        See `__init__` for a description of how the `EllipseMultipole` objects are created by this method.

        Parameters
        ----------
        fit
            A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry which has been loaded from
        an output directory or from an sqlite database..
        instance
            A manual instance that overwrites the max log likelihood instance in fit (e.g. for drawing the instance
            randomly from the PDF).
        """
        return _multipoles_from(fit=fit, instance=instance)
