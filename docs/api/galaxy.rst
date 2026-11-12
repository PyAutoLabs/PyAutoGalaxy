.. _galaxy:

Galaxies
========

Build a galaxy from named components such as a bulge, disk or pixelization. Use ``Galaxies`` to evaluate several galaxies together, for example when their light overlaps in an image.

.. _id1:

Galaxies and collections
------------------------

``Galaxy`` holds components at a specified redshift. Component names such as ``bulge``, ``disk`` and ``mass`` let you identify their contributions. ``Galaxies`` evaluates a collection, combining its light or mass quantities as appropriate.

.. currentmodule:: autogalaxy

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Galaxy
   Galaxies

PyAutoGalaxy models the galaxy light directly. Multi-plane gravitational lensing is provided by **PyAutoLens** and its ``Tracer`` class.

Build a galaxy
--------------

.. code-block:: python

    import autogalaxy as ag

    galaxy = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.Sersic(intensity=1.0, effective_radius=0.6),
        disk=ag.lp.Exponential(intensity=0.2, effective_radius=1.5),
    )
    galaxies = ag.Galaxies(galaxies=[galaxy])

This constructs a galaxy with fixed profile values; it does not fit data. The `galaxy guide <https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/guides/galaxies.ipynb>`_ shows how to evaluate profiles and combine galaxies.

External mass fields
--------------------

``MassField`` holds mass components at their own redshift without assigning them galaxy light. Use it for environmental contributions such as external shear, a mass sheet or an external potential. It also accepts other mass profiles when they represent the environment.

A ``Galaxies`` collection can include a mass field: it contributes to aggregate mass calculations and contributes no light. This is useful for mass calculations, rather than as a component of an ordinary galaxy-light fit. In PyAutoLens, ``Tracer(fields=...)`` includes these fields in multi-plane ray tracing.

Attaching external mass components directly to a ``Galaxy`` remains supported.

.. currentmodule:: autogalaxy

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   MassField

Fixed and fitted redshifts
--------------------------

For a known redshift, pass a number directly to ``Galaxy(redshift=...)``. When the redshift itself is to be inferred, ``Redshift`` provides a model component whose value can be assigned a prior.

.. code-block:: python

    import autofit as af

    redshift = af.Model(ag.Redshift)
    redshift.redshift = af.UniformPrior(lower_limit=0.8, upper_limit=1.2)
    galaxy_model = af.Model(ag.Galaxy, redshift=redshift)

The prior limits are illustrative, not a default recommendation. Add the light or mass components required by your model before fitting.

.. currentmodule:: autogalaxy

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Redshift
