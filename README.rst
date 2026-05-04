PyAutoGalaxy: Open-Source Multi Wavelength Galaxy Structure & Morphology
========================================================================

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.5.1.4/start_here.ipynb

.. image:: https://readthedocs.org/projects/pyautogalaxy/badge/?version=latest
   :target: https://pyautogalaxy.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/Jammy2211/PyAutoGalaxy/actions/workflows/main.yml/badge.svg
   :target: https://github.com/Jammy2211/PyAutoGalaxy/actions

.. image:: https://github.com/Jammy2211/PyAutoBuild/actions/workflows/release.yml/badge.svg
   :target: https://github.com/Jammy2211/PyAutoBuild/actions

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

.. image:: https://joss.theoj.org/papers/10.21105/joss.04475/status.svg
   :target: https://doi.org/10.21105/joss.04475

.. image:: https://pyopensci.org/badges/peer-reviewed.svg
   :target: https://github.com/pyOpenSci/software-submission/issues/235
   :alt: pyOpenSci Peer-Reviewed

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7546914.svg
   :target: https://doi.org/10.5281/zenodo.7546914
   :alt: Zenodo DOI

.. image:: https://www.repostatus.org/badges/latest/active.svg
    :target: https://www.repostatus.org/#active
    :alt: Project Status: Active

.. image:: https://img.shields.io/pypi/pyversions/autogalaxy
    :target: https://pypi.org/project/autogalaxy/
    :alt: Python Versions

.. image:: https://img.shields.io/pypi/v/autogalaxy.svg
    :target: https://pypi.org/project/autogalaxy/
    :alt: PyPI Version

`Installation Guide <https://pyautogalaxy.readthedocs.io/en/latest/installation/overview.html>`_ |
`readthedocs <https://pyautogalaxy.readthedocs.io/en/latest/index.html>`_ |
`Introduction on Colab <https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.5.1.4/start_here.ipynb>`_ |
`HowToGalaxy <https://pyautogalaxy.readthedocs.io/en/latest/howtogalaxy/howtogalaxy.html>`_

**PyAutoGalaxy** is software for analysing the morphologies and structures of galaxies:

.. image:: https://github.com/Jammy2211/PyAutoGalaxy/blob/main/paper/hstcombined.png?raw=true
        :target: https://github.com/Jammy2211/PyAutoGalaxy/blob/main/paper/hstcombined.png

**PyAutoGalaxy** also fits interferometer data from observatories such as ALMA:

.. image:: https://github.com/Jammy2211/PyAutoGalaxy/blob/main/paper/almacombined.png?raw=true
        :target: https://github.com/Jammy2211/PyAutoGalaxy/blob/main/paper/almacombined.png

Getting Started
---------------

The following links are useful for new starters:

- `The PyAutoGalaxy readthedocs <https://pyautogalaxy.readthedocs.io/en/latest>`_, which includes `an overview of PyAutoGalaxy's core features <https://pyautogalaxy.readthedocs.io/en/latest/overview/overview_1_start_here.html>`_, `a new user starting guide <https://pyautogalaxy.readthedocs.io/en/latest/overview/overview_2_new_user_guide.html>`_ and `an installation guide <https://pyautogalaxy.readthedocs.io/en/latest/installation/overview.html>`_.

- `The introduction Jupyter Notebook on Google Colab <https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.5.1.4/start_here.ipynb>`_, where you can try **PyAutoGalaxy** in a web browser (without installation).

- `The autogalaxy_workspace GitHub repository <https://github.com/PyAutoLabs/autogalaxy_workspace>`_: example scripts covering every **PyAutoGalaxy** use case.

- `The HowToGalaxy GitHub repository <https://github.com/PyAutoLabs/HowToGalaxy>`_: a Jupyter notebook lecture series teaching galaxy modeling from the ground up.

Core Aims
---------

**PyAutoGalaxy** has three core aims:

- **Big Data**: Scaling automated Sérsic fitting to extremely large datasets, *accelerated with JAX on GPUs and using tools like an SQL database to **build a scalable scientific workflow***.

- **Model Complexity**: Fitting complex galaxy morphology models (e.g. Multi Gaussian Expansion, Shapelets, Ellipse Fitting, Irregular Meshes) that go beyond just simple Sérsic fitting.

- **Data Variety**: Support for many data types (e.g. CCD imaging, interferometry, multi-band imaging) which can be fitted independently or simultaneously.

A complete overview of the software's aims is provided in our `Journal of Open Source Software paper <https://joss.theoj.org/papers/10.21105/joss.04475>`_.

Community & Support
-------------------

Support for **PyAutoGalaxy** is available via our Slack workspace, where the community shares updates, discusses
galaxy modeling and analysis, and helps troubleshoot problems.

Slack is invitation-only. If you’d like to join, please send an email requesting an invite.

For installation issues, bug reports, or feature requests, please raise an issue on the `GitHub issues page <https://github.com/Jammy2211/PyAutoGalaxy/issues>`_.

HowToGalaxy
-----------

For users less familiar with galaxy analysis, Bayesian inference, and scientific analysis, you may wish to read through
the **HowToGalaxy** lectures. These introduce the basic principles of galaxy modeling and Bayesian inference, with
the material pitched at undergraduate level and above.

A complete overview of the lectures `is provided on the HowToGalaxy readthedocs page <https://pyautogalaxy.readthedocs.io/en/latest/howtogalaxy/howtogalaxy.html>`_, and the notebooks themselves live in the `PyAutoLabs/HowToGalaxy <https://github.com/PyAutoLabs/HowToGalaxy>`_ repository.

Citations
---------

Information on how to cite **PyAutoGalaxy** in publications can be found `on the citations page <https://github.com/Jammy2211/PyAutoGalaxy/blob/main/CITATIONS.rst>`_.

Contributing
------------

Information on how to contribute to **PyAutoGalaxy** can be found `on the contributing page <https://github.com/Jammy2211/PyAutoGalaxy/blob/main/CONTRIBUTING.md>`_.

Hands on support for contributions is available via our Slack workspace, again please email to request an invite.