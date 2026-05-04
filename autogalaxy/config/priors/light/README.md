The `light` folder contains configuration files for the default priors assumed for light profiles in **PyAutoGalaxy** (e.g. `Sersic`, `Gaussian`).

# Folders

- `standard`: Configs for standard light profiles (specified via `ag.lp.`), which include an `intensity` parameter to control their overall amount of emission.
- `linear`: Configs for linear light profiles (specified via `ag.lp_linear.`),, where the `intensity` parameter is implicitly solved for via linear algebra.
- `shapelets`: Configs for shapelets (specified via `ag.lp_linear.`), which decomposed the light of a galaxy via shapelet basis functions.
- `operated`: Configs for operated light profiles (specified via `ag.lp_operated.`), where the light profile represents the already PSF convolved emission.
- `linear_operated`: Configs for linear operated light profiles (specified via `ag.lp_linear_operated.`), which behave like operated light profiles but with a linearly solved for `intensity`.
