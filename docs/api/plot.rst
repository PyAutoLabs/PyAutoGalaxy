========
Plotting
========

**PyAutoGalaxy** custom visualization library.

Step-by-step Juypter notebook guides illustrating all objects listed on this page are
provided on the `autogalaxy_workspace: plot tutorials <https://github.com/PyAutoLabs/autogalaxy_workspace/tree/main/notebooks/guides/plot>`_ and
it is strongly recommended you use those to learn plot customization.

**Examples / Tutorials:**

- `autogalaxy_workspace: plot tutorials <https://github.com/PyAutoLabs/autogalaxy_workspace/tree/main/notebooks/guides/plot>`_

Plotters [aplt]
---------------

Create figures and subplots showing quantities of standard **PyAutoGalaxy** objects.

.. currentmodule:: autogalaxy.plot

**Basic Plot Functions:**

.. autosummary::
   :toctree: _autosummary

    plot_array
    plot_grid

**Galaxy and Light / Mass Profile Subplots:**

.. autosummary::
   :toctree: _autosummary

    subplot_galaxy_light_profiles
    subplot_galaxy_mass_profiles
    subplot_basis_image
    subplot_galaxies
    subplot_galaxy_images
    subplot_adapt_images

**Imaging Fit Subplots:**

.. autosummary::
   :toctree: _autosummary

    subplot_fit_imaging
    subplot_fit_imaging_of_galaxy

**Interferometer Fit Subplots:**

.. autosummary::
   :toctree: _autosummary

    subplot_fit_interferometer
    subplot_fit_dirty_images
    subplot_fit_real_space

**Ellipse Fit Subplots:**

.. autosummary::
   :toctree: _autosummary

    subplot_fit_ellipse
    subplot_ellipse_errors

Non-linear Search Plot Functions [aplt]
---------------------------------------

Module-level functions for visualizing non-linear search results.

.. currentmodule:: autofit.plot

.. autosummary::
   :toctree: _autosummary

   corner_cornerpy
   corner_anesthetic
   subplot_parameters
   log_likelihood_vs_iteration
