(overview-2-new-user-guide)=

# New User Guide

## PyAutoGalaxy AI Assistant

The [**PyAutoGalaxy AI Assistant**](https://github.com/PyAutoLabs/autogalaxy_assistant) supports conversation agents such as ChatGPT and coding agents such as Claude Code and Codex. You can get started simply by asking it a question about galaxy structure or describing the task you would like to perform with **PyAutoGalaxy**. See the assistant for its full scope and instructions.

## Human-Readable Guide

**PyAutoGalaxy** can analyse galaxies for different types of data (e.g. CCD imaging and interferometer observations)
and across a range of system scales (a single galaxy, blended multi-galaxy systems, and cluster fields).
Depending on the data you use and the scale of your system, the analysis you perform may differ significantly.

The autogalaxy_workspace contains a suite of example Jupyter Notebooks, organised by dataset type and system scale.
To help you find the most appropriate starting point, answer two simple questions:

## What Scale System?

How many galaxies must be modeled together? There are three scales, which form a ladder (mirroring the lensing
regime ladder of **PyAutoLens**'s `autolens_workspace`):

- **Single Galaxy**: One galaxy dominates the image; any neighbours are contaminants to mask out. This is the
  standard starting point — go to the question below called "What Dataset Type?".
- **Multi Galaxy**: Two or more galaxies of comparable brightness whose light blends together (interacting pairs,
  close projected pairs, compact multiples) — each gets its own free light model, fitted simultaneously. Go to the
  [multi_galaxy/start_here.ipynb](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_galaxy/start_here.ipynb) notebook.
- **Cluster**: A brightest cluster galaxy plus tens-to-hundreds of member galaxies loaded from a CSV catalogue,
  whose photometry pins the faint members while only shared normalizations stay free. Go to the
  [cluster/start_here.ipynb](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/cluster/start_here.ipynb) notebook.

A note for lensing users coming from **PyAutoLens**: the two doc trees mirror each other, with one deliberate
divergence at the cluster rung — **PyAutoGalaxy's cluster workflow models the foreground galaxies' light (that is
its entire subject), whereas PyAutoLens's cluster workflow does not model lens light at all** (it fits
point-source multiple-image positions of the lensed background sources).

## What Dataset Type?

You now need to decide what type of data you are interested in:

- **CDD Imaging**: For image data from telescopes like Hubble and James Webb, go to [imaging/start_here.ipynb](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/start_here.ipynb).
- **Interferometer**: For radio / sub-mm interferometer from instruments like ALMA, go to [interferometer/start_here.ipynb](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/start_here.ipynb).
- **Multi-Band Imaging**: For galaxies observed in multiple wavebands go to [multi_dataset/start_here.ipynb](https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/start_here.ipynb).

## Google Colab

You can also open and run each notebook directly in Google Colab, which provides a free cloud computing
environment with all the required dependencies already installed.

This is a great way to get started quickly without needing to install **PyAutoGalaxy** on your own machine,
so you can check it is the right software for you before going through the installation process:

- [imaging/start_here.ipynb](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.8.4.1/notebooks/imaging/start_here.ipynb):
  Galaxy modeling with CCD imaging (e.g. Hubble, James Webb, ground-based telescopes).
- [interferometer/start_here.ipynb](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.8.4.1/notebooks/interferometer/start_here.ipynb):
  Galaxy modeling with interferometer data (e.g. ALMA), fitting directly in the uv-plane.
- [multi_band/start_here.ipynb](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.8.4.1/notebooks/multi_dataset/start_here.ipynb):
  Multi-band galaxy modeling to study colour gradients and wavelength-dependent structure.
- [multi_galaxy/start_here.ipynb](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.8.4.1/notebooks/multi_galaxy/start_here.ipynb):
  Blended multi-galaxy systems — one free light model per galaxy, fitted simultaneously.
- [cluster/start_here.ipynb](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.8.4.1/notebooks/cluster/start_here.ipynb):
  Cluster fields — a BCG plus a catalogue-driven member population.

## Still Unsure?

Each notebook is short and self-contained, and can be completed and adapted quickly to your particular task.
Therefore, if you're unsure exactly which system scale applies to you, or quite what data you want to use, you
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
