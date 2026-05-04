(overview-3-features)=

# Features

This page provides an overview of the advanced features of **PyAutoGalaxy**.

Firstly, brief one sentence descriptions of each feature are given, with more detailed descriptions below including
links to the relevant workspace examples.

**Interferometry**: Modeling of interferometer data (e.g. ALMA, LOFAR) directly in the uv-plane.

**Multi-Wavelength**: Simultaneous analysis of imaging and / or interferometer datasets observed at different wavelengths.

**Ellipse Fitting**: Fitting ellipses to determine a galaxy's ellipticity, position angle and centre.

**Multi Gaussian Expansion (MGE)**: Decomposing a galaxy into hundreds of Gaussians, capturing more complex structures than simple light profiles.

**Shapelets**: Decomposing a galaxy into a set of shapelet orthogonal basis functions, capturing more complex structures than simple light profiles.

**Sky Background**: Including the background sky in the model to ensure robust fits to the outskirts of galaxies.

**Operated Light Profiles**: Assuming a light profile has already been convolved with the PSF, for when the PSF is a significant effect.

**Pixelizations**: Reconstructing a galaxy's on a mesh of pixels, to capture extremely irregular structures like spiral arms.

## Interferometry

Modeling interferometer data from submillimeter (e.g. ALMA) and radio (e.g. LOFAR) observatories:

```{image} https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/paper/almacombined.png
:alt: Alternative text
:width: 600
```

Visibilities data is fitted directly in the uv-plane, circumventing issues that arise when fitting a dirty image
such as correlated noise. This uses the non-uniform fast fourier transform algorithm
\[PyNUFFT\](<https://github.com/jyhmiinlin/pynufft>) to efficiently map the galaxy model images to the uv-plane.

Checkout the `autogalaxy_workspace/*/interferometer` package to get started.

## Multi-Wavelength

Modeling imaging datasets observed at different wavelengths (e.g. HST F814W and F150W) simultaneously or simultaneously
analysing imaging and interferometer data:

```{image} https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/docs/overview/images/overview_3/g_image.png
:alt: Alternative text
:width: 400
```

```{image} https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/docs/overview/images/overview_3/r_image.png
:alt: Alternative text
:width: 400
```

The appearance of the galaxy changes as a function of wavelength, therefore multi-wavelength analysis means we can learn
more about the different components in a galaxy (e.g a redder bulge and bluer disk) or when imaging and interferometer
data are combined, we can compare the emission from stars and dust.

Checkout the `autogalaxy_workspace/*/multi` package to get started, however combining datasets is a more advanced
feature and it is recommended you first get to grips with the core API.

## Ellipse Fitting

Ellipse fitting is a technique which fits many ellipses to a galaxy's emission to determine its ellipticity, position
angle and centre, without assuming a parametric form for its light (e.g. a Sersic profile):

```{image} https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/docs/overview/images/overview_3/ellipse.png
:alt: Alternative text
:width: 600
```

This provides complementary information to parametric light profile fitting, for example giving insights on whether
the ellipticity and position angle are constant with radius or if the galaxy's emission is lopsided.

There are also multipole moment extensions to ellipse fitting, which determine higher order deviations from elliptical
symmetry providing even more information on the galaxy's structure.

The following paper describes the technique in detail: <https://arxiv.org/html/2407.12983v1>

Checkout `autogalaxy_workspace/notebooks/features/ellipse_fitting.ipynb` to learn how to use ellipse fitting.

## Multi Gaussian Expansion (MGE)

An MGE decomposes the light of a galaxy into tens or hundreds of two dimensional Gaussians:

```{image} https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/docs/overview/images/overview_3/mge.png
:alt: Alternative text
:width: 600
```

In the image above, 30 Gaussians are shown, where their sizes go from below the pixel scale (in order to resolve
point emission) to beyond the size of the galaxy (to capture its extended emission).

Scientific Applications include capturing departures from elliptical symmetry in the light of galaxies, providing a
flexible model to deblend the emission of point sources (e.g. quasars) from the emission of their host galaxy and
deprojecting the light of a galaxy from 2D to 3D.

Checkout `autogalaxy_workspace/notebooks/features/multi_gaussian_expansion.ipynb` to learn how to use an MGE.

## Shapelets

Shapelets are a set of orthogonal basis functions that can be combined the represent galaxy structures:

```{image} https://raw.githubusercontent.com/Jammy2211/PyAutoLens/main/docs/overview/images/overview_3/shapelets.png
:alt: Alternative text
:width: 600
```

Scientific Applications include capturing symmetric structures in a galaxy which are more complex than a Sersic profile,
irregular and asymmetric structures in a galaxy like spiral arms and providing a flexible model to deblend the emission
of point sources (e.g. quasars) from the emission of their host galaxy.

Checkout `autogalaxy_workspace/notebooks/features/shapelets.ipynb` to learn how to use shapelets.

## Sky Background

When an image of a galaxy is observed, the background sky contributes light to the image and adds noise:

For detailed studies of the outskirts of galaxies (e.g. stellar halos, faint extended disks), the sky background must be
accounted for in the model to ensure robust and accurate fits.

Checkout `autogalaxy_workspace/notebooks/features/sky_background.ipynb` to learn how to use include the sky
background in your model.

## Operated Light Profiles

An operated light profile is one where it is assumed to already be convolved with the PSF of the data, with the
`Moffat` and `Gaussian` profiles common choices:

They are used for certain scientific applications where the PSF convolution is known to be a significant effect and
the knowledge of the PSF allows for detailed modeling abd deblending of the galaxy's light.

Checkout `autogalaxy_workspace/notebooks/features/operated_light_profiles.ipynb` to learn how to use operated profiles.

## Pixelizations

A pixelization reconstructs a galaxy's light on a mesh of pixels, for example a rectangular mesh, Delaunay
triangulation or Voronoi grid.

These models are highly flexible and can capture complex structures in a galaxy's light that parametric models
like a Sersic profile cannot, for example spiral arms or asymmetric merging features.

The image below shows a non parametric of a galaxy observed in the Hubble Ultra Deep Field. Its bulge and disk are
fitted accurately using light profiles, whereas its asymmetric and irregular spiral arm features are accurately
captured using a rectangular mesh:

```{image} https://raw.githubusercontent.com/Jammy2211/PyAutoGalaxy/main/paper/hstcombined.png
:alt: Alternative text
:width: 600
```

Checkout `autogalaxy_workspace/notebooks/features/pixelizations.ipynb` to learn how to use a pixelization, however
this is a more advanced feature and it is recommended you first get to grips with the core API.

## Other

- Automated pipelines / database tools.
- Graphical models.
