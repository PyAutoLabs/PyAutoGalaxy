"""
Aggregator interface for loading ``Ellipse`` objects from ellipse model-fit results.

After an ellipse model-fit the best-fit ellipse parameters are stored in the output
directory or SQLite database as part of the model JSON file.  This module reconstructs
the list of ``Ellipse`` instances so that downstream analysis (e.g. re-plotting
isophotes) can be performed without re-running the fit.

Two public objects are provided:

- ``_ellipses_from`` — a free function that accepts a single ``PyAutoFit`` ``Fit`` entry
  and returns the list of ``Ellipse`` objects for the requested model instance.
- ``EllipsesAgg`` — a ``PyAutoFit`` ``AggBase`` subclass wrapping an ``Aggregator`` that
  exposes a generator of ellipse lists, one per stored model-fit.
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Optional, List

if TYPE_CHECKING:
    from autogalaxy.ellipse.ellipse import Ellipse

import autofit as af


logger = logging.getLogger(__name__)


def _ellipses_from(fit: af.Fit, instance: af.ModelInstance) -> List[List[Ellipse]]:
    """
    Returns a list of `Ellipse` objects from a `PyAutoFit` loaded directory `Fit` or sqlite database `Fit` object.

    The results of a model-fit can be loaded from hard-disk or stored in a sqlite database, including the following
    attributes of the fit:

    - The model and its best fit parameters (e.g. `model.json`).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a list of `Ellipse` object for a given non-linear search
    sample (e.g. the maximum likelihood model).

    If multiple `Ellipse` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of ellipses. This is necessary if each analysis has different ellipses (e.g. certain
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
        ellipses = instance.ellipses
    else:
        ellipses = fit.instance.ellipses

    if fit.children is not None:
        if len(fit.children) > 0:
            logger.info(
                """
                Using database for a fit with multiple summed Analysis objects.

                Ellipse objects do not fully support this yet (e.g. variables across Analysis objects may not be correct)
                so proceed with caution!
                """
            )

            return [ellipses] * len(fit.children)

    return [ellipses]


class EllipsesAgg(af.AggBase):
    """
    Interfaces with an `PyAutoFit` aggregator object to create instances of `Ellipse` objects from the results
    of a model-fit.

    The results of a model-fit can be loaded from hard-disk or stored in a sqlite database, including the following
    attributes of the fit:

    - The model and its best fit parameters (e.g. `model.json`).

    The `aggregator` contains the path to each of these files, and they can be loaded individually. This class
    can load them all at once and create lists of `Ellipse` objects via the `_ellipses_from` method.

    This class's methods returns generators which create the instances of the `Ellipse` objects. This ensures
    that large sets of results can be efficiently loaded from the hard-disk and do not require storing all
    `Ellipse` instances in the memory at once.

    For example, if the `aggregator` contains 3 model-fits, this class can be used to create a generator which
    creates instances of the corresponding 3 `Ellipse` objects.

    If multiple `Ellipse` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of ellipses. This is necessary if each analysis has different ellipses (e.g. certain
    parameters vary across each dataset and `Analysis` object).

    This can be done manually, but this object provides a more concise API.

    Parameters
    ----------
    aggregator
        A `PyAutoFit` aggregator object which can load the results of model-fits.
    """

    def object_via_gen_from(
        self, fit, instance: Optional[af.ModelInstance] = None
    ) -> List[List[Ellipse]]:
        """
        Returns a generator of `Ellipse` objects from an input aggregator.

        See `__init__` for a description of how the `Ellipse` objects are created by this method.

        Parameters
        ----------
        fit
            A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry which has been loaded from
        an output directory or from an sqlite database..
        instance
            A manual instance that overwrites the max log likelihood instance in fit (e.g. for drawing the instance
            randomly from the PDF).
        """
        return _ellipses_from(fit=fit, instance=instance)
