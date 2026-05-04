---
title: "`PyAutoGalaxy`: Open-Source Multiwavelength Galaxy Structure & Morphology"
tags:
  - astronomy
  - galaxy formation and evolution
  - galaxy morphology
  - Python
  - interferometry
authors:
  - name: James. W. Nightingale
    orcid: 0000-0002-8987-7401
    affiliation: 1
  - name: Aristeidis Amvrosiadis
    orcid: 0000-0002-4465-1564
    affiliation: 1 
  - name: Richard G. Hayes
    affiliation: 1
  - name: Qiuhan He
    orcid: 0000-0003-3672-9365
    affiliation: 1
  - name: Amy Etherington
    affiliation: 1 
  - name: XiaoYue Cao
    affiliation: 2               
  - name: Shaun Cole
    orcid: 0000-0002-5954-7903
    affiliation: 1
  - name: Jonathan Frawley
    orcid: 0000-0002-9437-7399
    affiliation: 3    
  - name: Carlos S. Frenk
    orcid: 0000-0002-2338-716X
    affiliation: 1  
  - name: Sam Lange
    affiliation: 1   
  - name: Ran Li
    orcid: 0000-0003-3899-0612
    affiliation: 2
  - name: Richard J. Massey
    orcid:  0000-0002-6085-3780
    affiliation: 1
  - name: Mattia Negrello
    orcid: 0000-0002-7925-7663
    affiliation: 4
  - name: Andrew Robertson
    orcid: 0000-0002-0086-0524
    affiliation: 1
affiliations:
  - name: Institute for Computational Cosmology, Stockton Rd, Durham DH1 3LE
    index: 1
  - name: National Astronomical Observatories, Chinese Academy of Sciences, 20A Datun Road, Chaoyang District, Beijing 100012, China
    index: 2
  - name: Advanced Research Computing, Durham University, Durham DH1 3LE
    index: 3
  - name: School of Physics and Astronomy, Cardiff University, The Parade, Cardiff CF24 3AA, UK
    index: 4

date: 12 May 2022
codeRepository: https://github.com/Jammy2211/PyAutoGalaxy
license: MIT
bibliography: paper.bib
---

# Summary

Nearly a century ago, Edwin Hubble famously classified galaxies into three distinct groups: ellipticals, spirals and 
irregulars [@Hubble1926]. Today, by analysing millions of galaxies with advanced image processing techniques Astronomers have 
expanded on this picture and revealed the rich diversity of galaxy morphology in both the nearby and distant 
Universe [@Kormendy2015a; @Vulcani2014; @VanDerWel2012]. `PyAutoGalaxy` is an open-source Python 3.8 - 3.11 package 
for analysing the morphologies and structures of large multiwavelength galaxy samples, with core features including 
fully automated Bayesian model-fitting of galaxy two-dimensional surface brightness profiles, support for dataset and 
interferometer datasets and comprehensive tools for simulating galaxy images. The software places a focus 
on big data analysis, including support for hierarchical models that simultaneously fit thousands of galaxies, 
massively parallel model-fitting and an SQLite3 database that allows large suites of modeling results to be loaded, 
queried and analysed. Accompanying `PyAutoGalaxy` is the [autogalaxy workspace](https://github.com/PyAutoLabs/autogalaxy_workspace),
which includes example scripts and galaxy datasets covering every use case. The [`HowToGalaxy`](https://github.com/PyAutoLabs/HowToGalaxy)
repository provides a separate Jupyter notebook lecture series which introduces non-experts to galaxy morphology studies using `PyAutoGalaxy`. Readers can try `PyAutoGalaxy` right now by going 
to [the introduction Jupyter notebook on Colab](https://colab.research.google.com/github/PyAutoLabs/autogalaxy_workspace/blob/2026.5.1.4/start_here.ipynb) or 
checkout the [readthedocs](https://pyautogalaxy.readthedocs.io/en/latest/) for a complete overview of `PyAutoGalaxy`'s 
features.

# Background

Galaxy morphology studies aim to understand the different luminous structures that galaxies are composed 
of [@Lackner2012; @Oh2017]. Using large CCD imaging datasets of galaxies observed at ultraviolet, optical and near-infrared wavelengths from 
instruments like the Hubble Space Telescope (HST), Astronomers have uncovered the plentiful structures that make up 
a galaxy, such as bars, bulges, disks and rings [@Graham2013; @Hodge2019] and revealed that evolving galaxies transition from disk-like structures 
to bulge-like elliptical galaxies [@Coenda2017]. At sub-mm and radio wavelengths interferometer datasets from instruments like the 
Atacama Large Millimeter Array (ALMA) have revealed the integral role that dust plays in forming galaxies in the 
distant Universe [@Blain2002; @Casey2014], early in their lifetimes. Studies typically represent a galaxy's light using analytic functions such as 
the Sersic profile [@Sersic1968], which quantifies the global appearance of most galaxies into one of three groups: (i) bulge-like 
structures which follow a de Vaucouleurs profile [@Dev1948]; (ii) disk-like structures which follow an exponential profile 
or; (iii) irregular morphologies which are difficult to quantify with symmetric and smooth analytic profiles. Galaxies 
are often composed of many sub-components which may be a combination of these different structures [@Nightingale2019].

![Hubble Space Telescope imaging of a spiral galaxy (left column), a parametric fit to its smooth bulge and disk components (middle column) and a non-parametric fit to its asymmetric and irregular structures like its spiral arms (right column).\label{figure:example1}](hstcombined.png)

![Atacama Large Millimeter Array interferometer visibilities data of a high redshift galaxy (left column), its dirty image created in real space via a Fourier transform (middle column) and a parametric fit to its smooth bulge and disk components which is performed directly on the visibility data (right column).\label{figure:example2}](almacombined.png)

Figure 1 shows example `PyAutoGalaxy` models of two galaxies taken with two different datasets. The top row shows
a structural decomposition of HST imaging of a galaxy, where `PyAutoGalaxy` has decomposed the galaxy into two distinct
parametric components, a bulge and disk, whilst simultaneously using a non-parametric model to represent the galaxy's 
irregular and asymmetric spiral arms. Instrumental effects like diffraction from the telescope optics are fully accounted 
for. Figure 2 shows a `PyAutoGalaxy` fit to ALMA interferometry, where the model galaxy's light is fitted directly in 
the complex uv-plane and Fourier transformed to real space for visualization.
 
# Statement of Need

In the next decade, wide-field surveys such as Euclid, the Vera Rubin Observatory and Square Kilometer Array are 
poised to observe images of _billions_ of galaxies. Analysing these extremely large galaxy datasets demands 
advanced Bayesian model-fitting techniques which can scale up in a fully automated manner. Equally, the James Webb 
Space Telescope, thirty-meter class ground telescopes and Square Kilometer Array radio interferometer will observe 
galaxies at an unprecedented resolution and level of detail. This demands more flexible modeling techniques that can 
accurately represent the complex irregular structures such high-resolution observations reveal. `PyAutoGalaxy` 
aims to meet both these needs, by interfacing galaxy model-fitting with the probabilistic programming language `PyAutoFit` to 
provide Bayesian fitting tools suited to big data analysis alongside image processing tools that represent irregular 
galaxy structures using non-parametric models.

# Software API and Features

At the heart of the `PyAutoGalaxy` API is the `Galaxy` object, which groups together one or more `LightProfile` objects
at an input redshift. Passing these objects a `Grid2D` returns an image of the galaxy(s), which can subsequently
be passed through `Operator` objects to apply a 2D convolution or Fast Fourier Transform and thereby compare 
the `Galaxy`'s image to an imaging or interferometer dataset. The `inversion` package contains non-parametric 
models which fit a galaxy's light using Bayesian linear matrix algebra. These were originally developed to 
reconstruct the source galaxies of strong gravitational lenses in `PyAutoGalaxy`'s child project `PyAutoLens` [@Nightingale2015; @pyautolens]. 
`PyAutoGalaxy` includes a comprehensive visualization library for the analysis of both direct imaging and interferometer 
datasets and tools for preprocessing data to formats suitable for galaxy model-fitting. The `astropy` cosmology module 
handles unit conversions and calculations are optimized using the packages `NumPy` [@numpy], `numba` [@numba],
and `PyNUFFT` [@pynufft].

To perform model-fitting, `PyAutoGalaxy` adopts the probabilistic programming  
language `PyAutoFit` (https://github.com/rhayes777/PyAutoFit). `PyAutoFit` allows users to compose a 
model from `LightProfile` and `Galaxy` objects, customize the model parameterization and fit it to data via a 
non-linear search, for example, `dynesty` [@dynesty], `emcee` [@emcee] or `PySwarms` [@pyswarms]. By composing a model with 
`Pixelization` objects, the galaxy's light is reconstructed using a non-parametric rectangular 
grid that accounts for irregular galaxy morphologies. Multiple images of the same galaxy can be fitted simultaneously,
using models whose parameters vary across wavelengths.

`PyAutoFit`'s graphical modeling framework allows one to fit a hierarchical model to images of thousands of galaxies 
simultaneously. Using a technique called expectation propagation [@Vehtari2020], this fits each galaxy dataset 
one by one and combines the results of every fit into a global model using a self-consistent Bayesian framework. 
Automated fitting of complex galaxy models is possible using `PyAutoFit`'s search chaining, which breaks the fitting of 
a galaxy into a chained sequence of non-linear searches. These fits pass information gained about simpler models 
fitted by earlier searches to subsequent searches, which fit progressively more complex models. By granularizing the 
model-fitting procedure, automated pipelines that fit complex galaxy models without human intervention can be carefully 
crafted, with example pipelines found on the [autogalaxy workspace](https://github.com/Jammy2211/autogalaxy_workspace). 
To ensure the analysis and interpretation of fits to large galaxy datasets is feasible, `PyAutoFit`'s database tools 
write modeling results to a relational database which can be queried from a storage drive to a Python script or Jupyter 
notebook. This uses memory-light `Python` generators, ensuring it is practical for results containing hundreds of thousands of galaxies.

# Workspace and HowToGalaxy Tutorials

`PyAutoGalaxy` is distributed with the [autogalaxy workspace](https://github.com/PyAutoLabs/autogalaxy_workspace), which
contains example scripts for modeling and simulating galaxies and tutorials on how to preprocess imaging and
interferometer datasets before a `PyAutoGalaxy` analysis. The [`HowToGalaxy`](https://github.com/PyAutoLabs/HowToGalaxy)
tutorials — a standalone repository separate from the workspace — are a four-chapter lecture series composed of over
20 Jupyter notebooks aimed at non-experts, introducing them to galaxy morphology analysis, Bayesian inference and
teaching them how to use `PyAutoGalaxy` for scientific study. The lectures are available on
[Colab](https://colab.research.google.com/github/PyAutoLabs/HowToGalaxy/blob/2026.5.1.4/notebooks/chapter_1_introduction/tutorial_1_grids_and_galaxies.ipynb)
and may therefore be taken without a local `PyAutoGalaxy` installation.

# Software Citations

`PyAutoGalaxy` is written in Python 3.8 - 3.11 and uses the following software packages:

- `Astropy` [@astropy1; @astropy2]
- `COLOSSUS` [@colossus]
- `corner.py` [@corner]
- `dynesty` [@dynesty]
- `emcee` [@emcee]
- `Matplotlib` [@matplotlib]
- `numba` [@numba]
- `NumPy` [@numpy]
- `PyAutoFit` [@pyautofit]
- `PyNUFFT` [@pynufft]
- `pyprojroot` (https://github.com/chendaniely/pyprojroot)
- `PySwarms` [@pyswarms]
- `scikit-image` [@scikit-image]
- `scikit-learn` [@scikit-learn]
- `Scipy` [@scipy]

# Related Software

- `PyAutoLens` [@Nightingale2018; @pyautolens]
- `galfit` https://users.obs.carnegiescience.edu/peng/work/galfit/galfit.html [@Peng2002]
- `GaLight` https://github.com/sibirrer/lenstronomy [@Ding2021]
- `GIM2D` http://www.astro.uvic.ca/~simard/GIM2D/
- `imfit` https://github.com/perwin/imfit
- `megamorph` https://www.nottingham.ac.uk/astronomy/megamorph/ [@Haussler2013]
- `ProFit` https://github.com/ICRAR/ProFit [@Robotham2017]
- `SourceXtractor++` https://github.com/astrorama/SourceXtractorPlusPlus

# Acknowledgments

JWN and RJM are supported by the UK Space Agency, through grant ST/V001582/1, and by InnovateUK through grant TS/V002856/1. 
RGH is supported by STFC Opportunities grant ST/T002565/1.
AA, QH, CSF and SMC are supported by ERC Advanced Investigator grant, DMIDAS [GA 786910] and also by the STFCConsolidated 
Grant for Astronomy at Durham [grant numbers ST/F001166/1, ST/I00162X/1,ST/P000541/1].
AE and SL are supported by STFC via grants ST/R504725/1 and ST/T506047/1. 
RJM is supported by a Royal Society University Research Fellowship.
AR is supported by the ERC Horizon2020 project ‘EWC’ (award AMD-776247-6).
MN has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Sklodowska-Curie grant agreement no. 707601.
This work used the DiRAC@Durham facility managed by the Institute for Computational Cosmology on behalf of the STFC DiRAC HPC Facility (www.dirac.ac.uk). The equipment was funded by BEIS capital funding via STFC capital grants ST/K00042X/1, ST/P002293/1, ST/R002371/1 and ST/S002502/1, Durham University and STFC operations grant ST/R000832/1. DiRAC is part of the National e-Infrastructure.

# References
