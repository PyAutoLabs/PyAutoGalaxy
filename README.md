# PyAutoGalaxy: Open-Source Multi Wavelength Galaxy Structure & Morphology

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.7.3.648/notebooks/imaging/start_here.ipynb)
[![Documentation Status](https://readthedocs.org/projects/pyautogalaxy/badge/?version=latest)](https://pyautogalaxy.readthedocs.io/en/latest/?badge=latest)
[![Tests](https://github.com/PyAutoLabs/PyAutoGalaxy/actions/workflows/main.yml/badge.svg)](https://github.com/PyAutoLabs/PyAutoGalaxy/actions)
[![Build](https://github.com/Jammy2211/PyAutoBuild/actions/workflows/release.yml/badge.svg)](https://github.com/Jammy2211/PyAutoBuild/actions)
[![Code Style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![JOSS](https://joss.theoj.org/papers/10.21105/joss.04475/status.svg)](https://doi.org/10.21105/joss.04475)
[![pyOpenSci Peer-Reviewed](https://pyopensci.org/badges/peer-reviewed.svg)](https://github.com/pyOpenSci/software-submission/issues/235)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7546914.svg)](https://doi.org/10.5281/zenodo.7546914)
[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Python Versions](https://img.shields.io/pypi/pyversions/autogalaxy)](https://pypi.org/project/autogalaxy/)
[![PyPI Version](https://img.shields.io/pypi/v/autogalaxy.svg)](https://pypi.org/project/autogalaxy/)

[Installation Guide](https://pyautogalaxy.readthedocs.io/en/latest/installation/overview.html) |
[readthedocs](https://pyautogalaxy.readthedocs.io/en/latest/index.html) |
[Introduction on Colab](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.7.3.648/notebooks/imaging/start_here.ipynb) |
[HowToGalaxy](https://pyautogalaxy.readthedocs.io/en/latest/howtogalaxy/howtogalaxy.html)

**PyAutoGalaxy** is software for analysing the morphologies and structures of galaxies:

[![HST Combined](https://github.com/PyAutoLabs/PyAutoGalaxy/blob/main/paper/hstcombined.png?raw=true)](https://github.com/PyAutoLabs/PyAutoGalaxy/blob/main/paper/hstcombined.png)

**PyAutoGalaxy** also fits interferometer data from observatories such as ALMA:

[![ALMA Combined](https://github.com/PyAutoLabs/PyAutoGalaxy/blob/main/paper/almacombined.png?raw=true)](https://github.com/PyAutoLabs/PyAutoGalaxy/blob/main/paper/almacombined.png)

## Getting Started

The following links are useful for new starters:

- [The PyAutoGalaxy readthedocs](https://pyautogalaxy.readthedocs.io/en/latest), which includes [an overview of PyAutoGalaxy's core features](https://pyautogalaxy.readthedocs.io/en/latest/overview/overview_1_start_here.html), [a new user starting guide](https://pyautogalaxy.readthedocs.io/en/latest/overview/overview_2_new_user_guide.html) and [an installation guide](https://pyautogalaxy.readthedocs.io/en/latest/installation/overview.html).
- [The introduction Jupyter Notebook on Google Colab](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.7.3.648/notebooks/imaging/start_here.ipynb), where you can try **PyAutoGalaxy** in a web browser (without installation).
- [The autogalaxy_workspace GitHub repository](https://github.com/PyAutoLabs/autogalaxy_workspace): example scripts covering every **PyAutoGalaxy** use case.
- [The HowToGalaxy GitHub repository](https://github.com/PyAutoLabs/HowToGalaxy): a Jupyter notebook lecture series teaching galaxy modeling from the ground up.

## Core Aims

**PyAutoGalaxy** has three core aims:

- **Big Data**: Scaling automated Sérsic fitting to extremely large datasets, *accelerated with JAX on GPUs and using tools like an SQL database to **build a scalable scientific workflow***.
- **Model Complexity**: Fitting complex galaxy morphology models (e.g. Multi Gaussian Expansion, Shapelets, Ellipse Fitting, Irregular Meshes) that go beyond just simple Sérsic fitting.
- **Data Variety**: Support for many data types (e.g. CCD imaging, interferometry, multi-band imaging) which can be fitted independently or simultaneously.

A complete overview of the software's aims is provided in our [Journal of Open Source Software paper](https://joss.theoj.org/papers/10.21105/joss.04475).

## Community & Support

Support for **PyAutoGalaxy** is available via our Slack workspace, where the community shares updates, discusses
galaxy modeling and analysis, and helps troubleshoot problems.

Slack is invitation-only. If you'd like to join, please send an email requesting an invite.

For installation issues, bug reports, or feature requests, please raise an issue on the [GitHub issues page](https://github.com/PyAutoLabs/PyAutoGalaxy/issues).

## HowToGalaxy

For users less familiar with galaxy analysis, Bayesian inference, and scientific analysis, you may wish to read through
the **HowToGalaxy** lectures. These introduce the basic principles of galaxy modeling and Bayesian inference, with
the material pitched at undergraduate level and above.

A complete overview of the lectures [is provided on the HowToGalaxy readthedocs page](https://pyautogalaxy.readthedocs.io/en/latest/howtogalaxy/howtogalaxy.html), and the notebooks themselves live in the [PyAutoLabs/HowToGalaxy](https://github.com/PyAutoLabs/HowToGalaxy) repository.

## Citations

Information on how to cite **PyAutoGalaxy** in publications can be found [on the citations page](https://github.com/PyAutoLabs/PyAutoGalaxy/blob/main/CITATIONS.md).

## Contributing

Information on how to contribute to **PyAutoGalaxy** can be found [on the contributing page](https://github.com/PyAutoLabs/PyAutoGalaxy/blob/main/CONTRIBUTING.md).

Hands on support for contributions is available via our Slack workspace, again please email to request an invite.
