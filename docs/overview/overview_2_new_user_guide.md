(overview-2-new-user-guide)=

# New User Guide

**PyAutoGalaxy** can analyse galaxies for different types of data (e.g. CCD imaging and interferometer observations).
Depending on the data you use, the analysis you perform may differ significantly.

The autogalaxy_workspace contains a suite of example Jupyter Notebooks, organised by dataset type. To help you find
the most appropriate starting point, answer one simple question:

## What Dataset Type?

You now need to decide what type of data you are interested in:

- **CDD Imaging**: For image data from telescopes like Hubble and James Webb, go to [imaging/start_here.ipynb](https://github.com/Jammy2211/autogalaxy_workspace/blob/release/notebooks/imaging/start_here.ipynb).
- **Interferometer**: For radio / sub-mm interferometer from instruments like ALMA, go to [interferometer/start_here.ipynb](https://github.com/Jammy2211/autogalaxy_workspace/blob/release/notebooks/interferometer/start_here.ipynb).
- **Multi-Band Imaging**: For galaxies observed in multiple wavebands go to [multi_band//start_here.ipynb](hhttps://github.com/Jammy2211/autogalaxy_workspace/blob/release/notebooks/point_source/start_here.ipynb).

## Google Colab

You can also open and run each notebook directly in Google Colab, which provides a free cloud computing
environment with all the required dependencies already installed.

This is a great way to get started quickly without needing to install **PyAutoGalaxy** on your own machine,
so you can check it is the right software for you before going through the installation process:

- [imaging/start_here.ipynb](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.5.1.4/notebooks/imaging/start_here.ipynb):
  Galaxy modeling with CCD imaging (e.g. Hubble, James Webb, ground-based telescopes).
- [interferometer/start_here.ipynb](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.5.1.4/notebooks/interferometer/start_here.ipynb):
  Galaxy modeling with interferometer data (e.g. ALMA), fitting directly in the uv-plane.
- [multi_band/start_here.ipynb](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.5.1.4/notebooks/multi/start_here.ipynb):
  Multi-band galaxy modeling to study colour gradients and wavelength-dependent structure.

## Still Unsure?

Each notebook is short and self-contained, and can be completed and adapted quickly to your particular task.
Therefore, if you're unsure exactly which scale of lensing applies to you, or quite what data you want to use, you
should just read through a few different notebooks and go from there.

## HowToGalaxy

For experienced scientists, the run through above will have been a breeze. Concepts surrounding galaxy structure and
morphology were already familiar and the statistical techniques used for fitting and modeling already understood.

For those less familiar with these concepts (e.g. undergraduate students, new PhD students or interested members of the
public), things may have been less clear and a slower more detailed explanation of each concept would be beneficial.

The **HowToGalaxy** Jupyter Notebook lectures provide exactly this. They are a 3+ chapter guide which thoroughly
takes you through the core concepts of galaxy light profiles, teaches you the principles of the statistical
techniques used in modeling and ultimately will allow you to undertake scientific research like a professional astronomer.

If this sounds like it suits you, checkout the [HowToGalaxy](https://github.com/PyAutoLabs/HowToGalaxy) repository now.

## Wrap Up

After completing this guide, you should be able to use **PyAutoGalaxy** for your science research.

The biggest decisions you'll need to make are what features and functionality your specific science case requires,
which the next readthedocs page gives an overview of to help you decide.
