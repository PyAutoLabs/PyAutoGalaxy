========
Modeling
========

Analysis
========

The ``Analysis`` objects define the ``log_likelihood_function`` of how a galaxy model is fitted to a dataset.

It acts as an interface between the data, model and the non-linear search.

.. currentmodule:: autogalaxy

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   AnalysisImaging
   AnalysisInterferometer
   AnalysisEllipse

Non-linear Searches
-------------------

A non-linear search is an algorithm which fits a model to data. The searches are provided by
**PyAutoFit**, and every one can be used with **PyAutoGalaxy**: see PyAutoFit's
:ref:`capability matrix <autofit:search_capability_matrix>`, which lists every search, what it
supports (JAX, gradients, evidence, posterior kind, install) and links to its API reference.

.. currentmodule:: autofit

Priors
------

The priors of parameters of every component of a mdoel, which is fitted to data, are customized using ``Prior`` objects.

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   UniformPrior
   GaussianPrior
   LogUniformPrior
   LogGaussianPrior

Adapt
-----

.. currentmodule:: autogalaxy

.. autosummary::
   :toctree: generated/

   AdaptImages