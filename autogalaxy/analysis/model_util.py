import numpy as np
from typing import Optional, Tuple

import autofit as af


def mge_model_from(
    mask_radius: float,
    total_gaussians: int = 30,
    gaussian_per_basis: int = 1,
    centre_prior_is_uniform: bool = True,
    centre: Tuple[float, float] = (0.0, 0.0),
    centre_fixed: Optional[Tuple[float, float]] = None,
    use_spherical: bool = False,
) -> af.Collection:
    """
    Construct a Multi-Gaussian Expansion (MGE) for the lens or source galaxy light

    This model is designed as a "start here" configuration for lens modeling:

    - The lens and source light are represented by a Basis object composed of many
      Gaussian light profiles with fixed logarithmically spaced widths (`sigma`).
    - All Gaussians within each basis share common centres and ellipticity
      components, reducing degeneracy while retaining flexibility.

    - Users can combine with a lens mass model of their choiuce.

    The resulting model provides a good balance of speed, flexibility, and accuracy
    for fitting most galaxy-scale strong lenses.

    This code is mostly to make the API simple for new users, hiding the technical
    details of setting up an MGE. More advanced users may wish to customize the
    model further.

    Parameters
    ----------
    mask_radius
        The outer radius (in arcseconds) of the circular mask applied to the data.
        This determines the maximum Gaussian width (`sigma`) used in the lens MGE.
    lens_total_gaussians
        Total number of Gaussian light profiles used in the lens MGE basis.
    source_total_gaussians
        Total number of Gaussian light profiles used in the source MGE basis.
    lens_gaussian_per_basis
        Number of separate Gaussian bases to include for the lens light profile.
        Each basis has `lens_total_gaussians` components.
    source_gaussian_per_basis
        Number of separate Gaussian bases to include for the source light profile.
        Each basis has `source_total_gaussians` components.

    Returns
    -------
    model : af.Collection
        An `autofit.Collection` containing:
        - A lens galaxy at redshift 0.5, with:
          * bulge light profile: MGE basis of Gaussians
          * mass profile: Isothermal ellipsoid
          * external shear
        - A source galaxy at redshift 1.0, with:
          * bulge light profile: MGE basis of Gaussians

    Notes
    -----
    - Lens light Gaussians have widths (sigma) logarithmically spaced between 0.01"
      and the mask radius.
    - Source light Gaussians have widths logarithmically spaced between 0.01" and 1.0".
    - Gaussian centres are free parameters but tied across all components in each
      basis to reduce dimensionality.
    - This function is a convenience utility: it hides the technical setup of MGE
      composition and provides a ready-to-use lens model for quick experimentation.
    """

    from autogalaxy.profiles.light.linear import Gaussian, GaussianSph
    from autogalaxy.profiles.basis import Basis

    # The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".
    log10_sigma_list = np.linspace(-4, np.log10(mask_radius), total_gaussians)

    # By defining the centre here, it creates two free parameters that are assigned below to all Gaussians.

    if centre_fixed is not None:
        centre_0 = centre[0]
        centre_1 = centre[1]
    elif centre_prior_is_uniform:
        centre_0 = af.UniformPrior(
            lower_limit=centre[0] - 0.1, upper_limit=centre[0] + 0.1
        )
        centre_1 = af.UniformPrior(
            lower_limit=centre[1] - 0.1, upper_limit=centre[1] + 0.1
        )
    else:
        centre_0 = af.GaussianPrior(mean=centre[0], sigma=0.3)
        centre_1 = af.GaussianPrior(mean=centre[1], sigma=0.3)

    if use_spherical:
        model_cls = GaussianSph
    else:
        model_cls = Gaussian

    bulge_gaussian_list = []

    for j in range(gaussian_per_basis):
        # A list of Gaussian model components whose parameters are customized belows.

        gaussian_list = af.Collection(
            af.Model(model_cls) for _ in range(total_gaussians)
        )

        # Iterate over every Gaussian and customize its parameters.

        for i, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = centre_0  # All Gaussians have same y centre.
            gaussian.centre.centre_1 = centre_1  # All Gaussians have same x centre.
            if not use_spherical:
                gaussian.ell_comps = gaussian_list[
                    0
                ].ell_comps  # All Gaussians have same elliptical components.
            gaussian.sigma = (
                10 ** log10_sigma_list[i]
            )  # All Gaussian sigmas are fixed to values above.

        bulge_gaussian_list += gaussian_list

    # The Basis object groups many light profiles together into a single model component.

    return af.Model(
        Basis,
        profile_list=bulge_gaussian_list,
    )



def mge_model_V2_from(
    mask_radius: float,
    total_gaussians: int = 30,
    gaussian_per_basis: int = 1,
    centre_prior_is_uniform: bool = True,
    centre_uniform_range: float = 0.1,
    centre: Tuple[float, float] = (0.0, 0.0),
    centre_fixed: Optional[Tuple[float, float]] = None,
    use_spherical: bool = False,
    ell_comps_range: Optional[float] = None,
) -> af.Collection:
    """
    Construct a Multi-Gaussian Expansion (MGE) for the lens or source galaxy light

    This model is designed as a "start here" configuration for lens modeling:

    - The lens and source light are represented by a Basis object composed of many
      Gaussian light profiles with fixed logarithmically spaced widths (`sigma`).
    - All Gaussians within each basis share common centres and ellipticity
      components, reducing degeneracy while retaining flexibility.

    - Users can combine with a lens mass model of their choiuce.

    The resulting model provides a good balance of speed, flexibility, and accuracy
    for fitting most galaxy-scale strong lenses.

    This code is mostly to make the API simple for new users, hiding the technical
    details of setting up an MGE. More advanced users may wish to customize the
    model further.

    Parameters
    ----------
    mask_radius
        The outer radius (in arcseconds) of the circular mask applied to the data.
        This determines the maximum Gaussian width (`sigma`) used in the lens MGE.
    lens_total_gaussians
        Total number of Gaussian light profiles used in the lens MGE basis.
    source_total_gaussians
        Total number of Gaussian light profiles used in the source MGE basis.
    lens_gaussian_per_basis
        Number of separate Gaussian bases to include for the lens light profile.
        Each basis has `lens_total_gaussians` components.
    source_gaussian_per_basis
        Number of separate Gaussian bases to include for the source light profile.
        Each basis has `source_total_gaussians` components.

    Returns
    -------
    model : af.Collection
        An `autofit.Collection` containing:
        - A lens galaxy at redshift 0.5, with:
          * bulge light profile: MGE basis of Gaussians
          * mass profile: Isothermal ellipsoid
          * external shear
        - A source galaxy at redshift 1.0, with:
          * bulge light profile: MGE basis of Gaussians

    Notes
    -----
    - Lens light Gaussians have widths (sigma) logarithmically spaced between 0.01"
      and the mask radius.
    - Source light Gaussians have widths logarithmically spaced between 0.01" and 1.0".
    - Gaussian centres are free parameters but tied across all components in each
      basis to reduce dimensionality.
    - This function is a convenience utility: it hides the technical setup of MGE
      composition and provides a ready-to-use lens model for quick experimentation.
    """

    from autogalaxy.profiles.light.linear import Gaussian, GaussianSph
    from autogalaxy.profiles.basis import Basis

    # The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius, 3.0".
    log10_sigma_list = np.linspace(-2, np.log10(mask_radius), total_gaussians)

    # By defining the centre here, it creates two free parameters that are assigned below to all Gaussians.

    if centre_fixed is not None:
        centre_0 = centre[0]
        centre_1 = centre[1]
    elif centre_prior_is_uniform:
        centre_0 = af.UniformPrior(
            lower_limit=centre[0] - centre_uniform_range, upper_limit=centre[0] + centre_uniform_range
        )
        centre_1 = af.UniformPrior(
            lower_limit=centre[1] - centre_uniform_range, upper_limit=centre[1] + centre_uniform_range
        )
    else:
        centre_0 = af.GaussianPrior(mean=centre[0], sigma=0.3)
        centre_1 = af.GaussianPrior(mean=centre[1], sigma=0.3)

    if use_spherical:
        model_cls = GaussianSph
    else:
        model_cls = Gaussian

    bulge_gaussian_list = []

    for j in range(gaussian_per_basis):
        # A list of Gaussian model components whose parameters are customized belows.

        gaussian_list = af.Collection(
            af.Model(model_cls) for _ in range(total_gaussians)
        )

        # Iterate over every Gaussian and customize its parameters.

        for i, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = centre_0  # All Gaussians have same y centre.
            gaussian.centre.centre_1 = centre_1  # All Gaussians have same x centre.
            if not use_spherical:
                if ell_comps_range is not None:
                    if i == 0:
                        gaussian.ell_comps.ell_comps_0 = af.UniformPrior(
                            lower_limit=-ell_comps_range, upper_limit=ell_comps_range
                        )
                        gaussian.ell_comps.ell_comps_1 = af.UniformPrior(
                            lower_limit=-ell_comps_range, upper_limit=ell_comps_range
                        )
                    else:
                        gaussian.ell_comps = gaussian_list[0].ell_comps  # All Gaussians have same elliptical components.
            gaussian.sigma = (
                10 ** log10_sigma_list[i]
            )  # All Gaussian sigmas are fixed to values above.

        bulge_gaussian_list += gaussian_list

    # The Basis object groups many light profiles together into a single model component.

    return af.Model(
        Basis,
        profile_list=bulge_gaussian_list,
    )

def mge_point_model_from(
    pixel_scales: float,
    total_gaussians: int = 10,
    centre: Tuple[float, float] = (0.0, 0.0),
) -> af.Model:
    """
    Construct a Multi-Gaussian Expansion (MGE) model for a compact or unresolved
    point-like component (e.g. a nuclear starburst, AGN, or unresolved bulge).

    The model is composed of ``total_gaussians`` linear Gaussians whose sigma values
    are logarithmically spaced between 0.01 arcseconds and twice the pixel scale.
    All Gaussians share the same centre and ellipticity components, keeping the
    parameter count low while capturing a realistic PSF-convolved point source.

    Parameters
    ----------
    pixel_scales
        The pixel scale of the image in arcseconds per pixel.  The maximum Gaussian
        width is set to ``2 * pixel_scales`` so that the model is compact relative to
        the resolution of the data.
    total_gaussians
        Number of Gaussian components in the basis.
    centre
        (y, x) centre of the point source in arc-seconds.  A ±0.1 arcsecond uniform
        prior is placed on each coordinate.

    Returns
    -------
    af.Model
        An ``autofit.Model`` wrapping a ``Basis`` of linear Gaussians.
    """

    from autogalaxy.profiles.light.linear import Gaussian
    from autogalaxy.profiles.basis import Basis

    if total_gaussians < 1:
        raise ValueError(
            f"mge_point_model_from requires total_gaussians >= 1, got {total_gaussians}."
        )

    if pixel_scales <= 0:
        raise ValueError(
            f"mge_point_model_from requires pixel_scales > 0, got {pixel_scales}."
        )

    # Sigma values are logarithmically spaced between 0.01 arcsec (10**-2)
    # and twice the pixel scale, with a floor to avoid taking log10 of
    # very small or non-positive values.
    min_log10_sigma = -2.0  # corresponds to 0.01 arcsec
    max_sigma = max(2.0 * pixel_scales, 10**min_log10_sigma)
    max_log10_sigma = np.log10(max_sigma)

    log10_sigma_list = np.linspace(min_log10_sigma, max_log10_sigma, total_gaussians)
    centre_0 = af.UniformPrior(lower_limit=centre[0] - 0.1, upper_limit=centre[0] + 0.1)
    centre_1 = af.UniformPrior(lower_limit=centre[1] - 0.1, upper_limit=centre[1] + 0.1)

    gaussian_list = af.Collection(af.Model(Gaussian) for _ in range(total_gaussians))

    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0
        gaussian.centre.centre_1 = centre_1
        gaussian.ell_comps = gaussian_list[0].ell_comps
        gaussian.sigma = 10 ** log10_sigma_list[i]

    return af.Model(Basis, profile_list=gaussian_list)


def hilbert_pixels_from_pixel_scale(pixel_scale: float) -> int:
    """
    Return the number of Hilbert-curve pixels appropriate for a given image pixel scale.

    The Hilbert pixel count controls the resolution of the Hilbert-curve ordering used
    in adaptive source-plane pixelizations. Finer pixel scales resolve smaller angular
    features and therefore benefit from a higher Hilbert resolution.

    Parameters
    ----------
    pixel_scale
        The pixel scale of the image in arcseconds per pixel.

    Returns
    -------
    int
        The recommended number of Hilbert pixels.
    """
    if not np.isfinite(pixel_scale) or pixel_scale <= 0:
        raise ValueError(
            f"hilbert_pixels_from_pixel_scale requires pixel_scale to be finite and > 0, got {pixel_scale}."
        )

    if pixel_scale > 0.06:
        return 1000
    elif pixel_scale > 0.04:
        return 1250
    elif pixel_scale >= 0.03:
        return 1500
    else:
        return 1750



def simulator_start_here_model_from():

    from autogalaxy.profiles.light.snr import Sersic
    from autogalaxy.galaxy.galaxy import Galaxy

    bulge = af.Model(Sersic)

    bulge.centre = (0.0, 0.0)
    bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
    )
    bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=0.0, sigma=0.2, lower_limit=-1.0, upper_limit=1.0
    )
    bulge.signal_to_noise_ratio = af.UniformPrior(lower_limit=20.0, upper_limit=60.0)
    bulge.effective_radius = af.UniformPrior(lower_limit=1.0, upper_limit=5.0)
    bulge.sersic_index = af.TruncatedGaussianPrior(
        mean=4.0, sigma=0.5, lower_limit=0.8, upper_limit=5.0
    )

    galaxy = af.Model(Galaxy, redshift=0.5, bulge=bulge)

    return galaxy
