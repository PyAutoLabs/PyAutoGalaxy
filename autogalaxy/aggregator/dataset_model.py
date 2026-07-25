"""
Aggregator interface for loading ``DatasetModel`` objects from model-fit results.

A ``DatasetModel`` holds dataset-level nuisance parameters (e.g. an overall background
offset or a noise scaling factor) that form part of the fitted model.  This module
provides the helper function ``_dataset_model_from`` which reconstructs these objects from
a stored ``PyAutoFit`` ``Fit`` entry so that the full fit can be replicated during
offline analysis.

When no ``dataset_model`` is present in the model (the common case), ``None`` is returned
for that entry so that downstream ``Fit*`` constructors receive a well-typed argument.
"""
from __future__ import annotations
import logging
from typing import List

import autofit as af
import autoarray as aa

from autogalaxy.aggregator import agg_util

logger = logging.getLogger(__name__)


def _dataset_model_from(
    fit: af.Fit, instance: af.ModelInstance
) -> List[aa.DatasetModel]:
    """
    Returns a `DatasetModel` object from a `PyAutoFit` loaded directory `Fit` or sqlite database `Fit` object.

    The results of a model-fit can be loaded from hard-disk or stored in a sqlite database, including the following
    attributes of the fit:

    - The model and its best fit parameters (e.g. `model.json`).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a `DatasetModel` object for a given non-linear search
    sample (e.g. the maximum likelihood model).

    If multiple `DatasetModel` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of dataset models. This is necessary if each analysis has different dataset
    models (e.g. certain parameters vary across each dataset and `Analysis` object).

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry which has been loaded from
        an output directory or from an sqlite database..
    instance
        A manual instance that overwrites the max log likelihood instance in fit (e.g. for drawing the instance
        randomly from the PDF).
    """

    instance_list = agg_util.instance_list_from(fit=fit, instance=instance)

    dataset_model_list = []

    for instance in instance_list:

        try:
            dataset_model = instance.dataset_model
        except AttributeError:
            dataset_model = None

        dataset_model_list.append(dataset_model)

    return dataset_model_list
