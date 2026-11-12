Plotting
========

Use ``autogalaxy.plot`` to inspect datasets, models and fits. Functions produce individual figures or diagnostic subplots; pass plotting options directly to the function you call.

For a runnable introduction and customization examples, open the `plotting guide <https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/guides/plot/start_here.ipynb>`_.

Plot an image
-------------

Given an existing ``Array2D`` called ``image``:

.. code-block:: python

    import autogalaxy.plot as aplt

    aplt.plot_array(array=image, title="Image", colormap="gray")

.. _plotters-aplt:

Choose a plot
-------------

Choose the group that matches the object or result you want to inspect. The linked references give each function’s arguments.

.. currentmodule:: autogalaxy.plot

Basic plots
~~~~~~~~~~~

Display an array or a grid of coordinates.

.. autosummary::
   :toctree: _autosummary

    plot_array
    plot_grid

Galaxy and Light and Mass Profile subplots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inspect profile components, combined galaxy images, basis profiles and adapt images.

.. autosummary::
   :toctree: _autosummary

    subplot_galaxy_light_profiles
    subplot_galaxy_mass_profiles
    subplot_basis_image
    subplot_galaxies
    subplot_galaxy_images
    subplot_adapt_images

Imaging Fit subplots
~~~~~~~~~~~~~~~~~~~~

Compare the image data and fitted model, and inspect residuals and other fit diagnostics.

.. autosummary::
   :toctree: _autosummary

    subplot_fit_imaging
    subplot_fit_imaging_of_galaxy

Interferometer Fit subplots
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inspect visibility-space fit diagnostics alongside real-space and dirty-image views.

.. autosummary::
   :toctree: _autosummary

    subplot_fit_interferometer
    subplot_fit_dirty_images
    subplot_fit_real_space

Ellipse Fit subplots
~~~~~~~~~~~~~~~~~~~~

Inspect fitted isophotes and the errors associated with the ellipse fits.

.. autosummary::
   :toctree: _autosummary

    subplot_fit_ellipse
    subplot_ellipse_errors

.. _non-linear-search-plot-functions-aplt:

Search results
--------------

These functions visualize samples and search progress. They are implemented by **PyAutoFit** and re-exported through ``aplt``.

.. currentmodule:: autofit.plot

.. autosummary::
   :toctree: _autosummary

   corner_cornerpy
   corner_anesthetic
   subplot_parameters
   log_likelihood_vs_iteration

.. _plot-customization-aplt:

Customize a plot
----------------

Options depend on the function. For ``plot_array``, you can set:

- ``title``: the figure title.
- ``colormap``: a Matplotlib colormap, such as ``"gray"`` or ``"viridis"``.
- ``use_log10``: request logarithmic colour scaling. Use a linear or suitable signed scale when inspecting quantities that can be negative, such as residuals.
- ``output_path``, ``output_filename`` and ``output_format``: the output directory, filename and format.

Subplot functions may use different options. For example, ``subplot_fit_ellipse`` accepts ``title_prefix`` rather than ``title`` and does not expose ``output_filename``. Check the linked signature before transferring options between functions.

Defaults such as figure size, fonts and colours are configured through the workspace’s ``config/visualize`` YAML files. See `Configuration <https://pyautogalaxy.readthedocs.io/en/latest/general/configs.html>`_ for the configuration workflow.

.. _figure-output-aplt:

Save figures
------------

For ``plot_array``, save the existing image to ``output/image.png`` with:

.. code-block:: python

    aplt.plot_array(
        array=image,
        output_path="output",
        output_filename="image",
        output_format="png",
    )

The functions above use the output keywords documented in their signatures. ``Output`` is a lower-level **PyAutoArray** helper; it is not a replacement argument for these public plotting functions. Its reference remains available for code that uses it directly.

.. currentmodule:: autoarray.plot

.. autosummary::
   :toctree: _autosummary

   Output
