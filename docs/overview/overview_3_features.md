(overview-3-features)=

# Features

<span id="choose-a-feature-by-dataset"></span><span id="imaging"></span><span id="interferometer"></span><span id="multiple-datasets"></span><span id="blended-galaxies"></span>

Explore the modelling tools you can add to your analysis. Each introduction explains the scientific use cases and links to worked examples.

For your first dataset and starting notebook, use the [New User Guide](https://autogalaxy.readthedocs.io/en/latest/overview/overview_2_new_user_guide.html).

<table class="feature-gallery"><tbody><tr><td width="25%"><a href="#multi-gaussian-expansion-mge"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/docs/overview/images/overview_3/mge_illustration.png" alt="Multi Gaussian Expansion (MGE) illustration" width="220"><br>MGE</a></td><td width="25%"><a href="#pixelizations"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/docs/overview/images/overview_3/pixelization_illustration.png" alt="Pixelizations illustration" width="220"><br>Pixelizations</a></td><td width="25%"><a href="#shapelets"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/docs/overview/images/overview_3/shapelets_illustration.png" alt="Shapelets illustration" width="220"><br>Shapelets</a></td><td width="25%"><a href="#ellipse-fitting"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/docs/overview/images/overview_3/ellipse.png" alt="Ellipse Fitting illustration" width="220"><br>Ellipse Fitting</a></td></tr></tbody></table>

*Conceptual illustrations, not measured data or guaranteed reconstructions.*

<details class="feature-more-images"><summary>More feature illustrations</summary><table class="feature-gallery"><tbody><tr><td width="25%"><a href="#linear-light-profiles"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/docs/overview/images/overview_3/linear_light_profiles.png" alt="Linear Light Profiles illustration" width="220"><br>Linear Light Profiles</a></td><td width="25%"><a href="#extra-galaxies"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/docs/overview/images/overview_2/galaxy_systems.png" alt="Extra Galaxies illustration" width="220"><br>Extra Galaxies</a></td><td width="25%"><a href="#point-source"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/docs/overview/images/overview_3/point_source.png" alt="Point Source illustration" width="220"><br>Point Source</a></td><td width="25%"><a href="#operated-light-profiles"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/docs/overview/images/overview_3/operated_light_profile.png" alt="Operated Light Profiles illustration" width="220"><br>Operated Light Profiles</a></td></tr><tr><td width="25%"><a href="#sky-background"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/docs/overview/images/overview_3/sky_background.png" alt="Sky Background illustration" width="220"><br>Sky Background</a></td><td width="25%"><a href="#interferometry"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/docs/overview/images/overview_2/galaxy_interferometry.png" alt="Interferometry illustration" width="220"><br>Interferometry</a></td><td width="25%"><a href="#multi-wavelength"><img src="https://raw.githubusercontent.com/PyAutoLabs/PyAutoGalaxy/main/docs/overview/images/overview_2/galaxy_multi_wavelength.png" alt="Multi-Wavelength illustration" width="220"><br>Multi-Wavelength</a></td></tr></tbody></table></details>

## Measure galaxy structure

<div id="multi-gaussian-expansion-mge" class="feature-entry"><h3>Multi Gaussian Expansion (MGE)</h3><p>Describe central and extended galaxy light with a sum of Gaussians. An MGE captures changes in shape with radius while solving component brightnesses together, providing a flexible alternative to a small set of light profiles.</p>
<p><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/multi_gaussian_expansion/start_here.ipynb">Explore multi gaussian expansion (mge)</a>.</p>
</div>

<div id="pixelizations" class="feature-entry"><h3>Pixelizations</h3><p>Reconstruct clumpy or asymmetric galaxy light on a mesh when a few analytic profiles are too restrictive. The data, mesh and regularization determine which structures can be recovered.</p>
<p><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/pixelization/start_here.ipynb">Start with pixelization</a>.</p>
</div>

<div id="shapelets" class="feature-entry"><h3>Shapelets</h3><p>Study asymmetric morphology and structured residuals using a sum of spatial basis functions. Shapelet order, scale and geometry control the flexibility; assess the combined brightness distribution rather than treating individual basis modes as physical components.</p>
<p><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/shapelets/start_here.ipynb">Explore shapelets</a>.</p>
</div>

<div id="ellipse-fitting" class="feature-entry"><h3>Ellipse Fitting</h3><p>Measure how a galaxy’s ellipticity, position angle and centre change with radius without imposing a radial light-profile law. Add multipoles to study lopsided or non-elliptical isophotes. This dedicated PyAutoGalaxy capability fits observed light; the examples do not convolve a light model with the PSF.</p>
<p><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/ellipse/fit.ipynb">Explore ellipse fitting</a>.</p>
<p>Continue to <a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/ellipse/modeling.ipynb">modeling</a> and <a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/ellipse/multipoles.ipynb">multipoles</a>.</p>
</div>

## Separate light components

<div id="linear-light-profiles" class="feature-entry"><h3>Linear Light Profiles</h3><p>Fit galaxy geometry with fewer sampled parameters by solving component brightnesses together at each likelihood evaluation. This is especially useful for bulge–disc decomposition and models with many overlapping light components.</p>
<p><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/linear_light_profiles/start_here.ipynb">Explore linear light profiles</a>.</p>
</div>

<div id="extra-galaxies" class="feature-entry"><h3>Extra Galaxies</h3><p>Measure a target galaxy in a crowded field by fitting neighbouring light alongside it. Separating overlapping emission helps protect the inferred size, shape and flux of the target.</p>
<p><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/extra_galaxies/start_here.ipynb">Explore extra galaxies</a>.</p>
</div>

<div id="point-source" class="feature-entry"><h3>Point Source</h3><p>Simulate an unresolved nucleus inside an extended host galaxy. This example shows how compact and extended light combine after the instrumental response; it is a simulator, not a point-source fitting workflow.</p>
<p><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/point_source/start_here.ipynb">Explore point source</a>.</p>
</div>

<div id="operated-light-profiles" class="feature-entry"><h3>Operated Light Profiles</h3><p>Separate compact nuclear emission from an extended host with a component that already includes the imaging PSF response. For interferometer data, the compact sky model is still transformed into visibilities.</p>
<p><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/operated_light_profile/start_here.ipynb">Explore operated light profiles</a>.</p>
</div>

<div id="sky-background" class="feature-entry"><h3>Sky Background</h3><p>Measure faint galaxy outskirts while fitting an uncertain sky level alongside the galaxy light. The example fits a constant background; estimating its mean does not remove photon noise.</p>
<p><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/sky_background/start_here.ipynb">Explore sky background</a>.</p>
</div>

## Combine observations and samples

<div id="interferometry" class="feature-entry"><h3>Interferometry</h3><p>Extend visibility modelling with MGE, shapelets or pixelized galaxy light to investigate complex emission.</p>
<p><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/tree/main/notebooks/interferometer/features">Interferometer features</a>.</p>
</div>

<div id="multi-wavelength" class="feature-entry"><h3>Multi-Wavelength</h3><p>Compare galaxy structure across bands and distinguish components such as a red bulge and blue disc, or stellar and dust emission. Fit datasets together with shared or wavelength-dependent parameters while preserving their individual responses and noise.</p>
<p><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/start_here.ipynb">Explore multi-wavelength</a>.</p>
</div>

<div id="graphical-models" class="feature-entry"><h3>Graphical Models</h3><p>Infer population trends by fitting galaxy datasets with shared parameters. Hierarchical models describe distributions of properties such as size or Sersic index, helping study how galaxy structure varies with mass or redshift.</p>
<p>See the <a href="https://github.com/PyAutoLabs/autofit_workspace/tree/main/notebooks/features">PyAutoFit graphical-model examples</a> for the shared-parameter API.</p>
</div>

<section class="directory" id="directory"><h2>All workspace links</h2><p class="minor">Choose your dataset to browse every feature introduction.</p><details><summary>Imaging · 8 introductions</summary><ul><li><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/extra_galaxies/start_here.ipynb">Extra Galaxies</a></li><li><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/linear_light_profiles/start_here.ipynb">Linear Light Profiles</a></li><li><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/multi_gaussian_expansion/start_here.ipynb">Multi Gaussian Expansion</a></li><li><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/operated_light_profile/start_here.ipynb">Operated Light Profile</a></li><li><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/pixelization/start_here.ipynb">Pixelization</a></li><li><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/point_source/start_here.ipynb">Point Source</a></li><li><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/shapelets/start_here.ipynb">Shapelets</a></li><li><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/imaging/features/sky_background/start_here.ipynb">Sky Background</a></li></ul></details><details><summary>Interferometer · 6 introductions</summary><ul><li><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/extra_galaxies/start_here.ipynb">Extra Galaxies</a></li><li><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/linear_light_profiles/start_here.ipynb">Linear Light Profiles</a></li><li><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/multi_gaussian_expansion/start_here.ipynb">Multi Gaussian Expansion</a></li><li><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/operated_light_profile/start_here.ipynb">Operated Light Profile</a></li><li><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/pixelization/start_here.ipynb">Pixelization</a></li><li><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/interferometer/features/shapelets/start_here.ipynb">Shapelets</a></li></ul></details><details><summary>Multi dataset · 6 introductions</summary><ul><li><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/dataset_offsets/start_here.ipynb">Dataset Offsets</a></li><li><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/imaging_and_interferometer/start_here.ipynb">Imaging And Interferometer</a></li><li><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/one_by_one/start_here.ipynb">One By One</a></li><li><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/pixelization/start_here.ipynb">Pixelization</a></li><li><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/same_wavelength/start_here.ipynb">Same Wavelength</a></li><li><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_dataset/features/wavelength_dependence/start_here.ipynb">Wavelength Dependence</a></li></ul></details><details><summary>Multi galaxy · 1 introductions</summary><ul><li><a href="https://github.com/PyAutoLabs/autogalaxy_workspace/blob/main/notebooks/multi_galaxy/features/extra_galaxies/start_here.ipynb">Extra Galaxies</a></li></ul></details></section>