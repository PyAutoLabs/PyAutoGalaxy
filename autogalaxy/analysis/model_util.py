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
    centre_per_basis: bool = False,
    centre_sigma: float = 0.3,
    ell_comps_prior_is_uniform: bool = False,
    ell_comps_uniform_width: float = 0.2,
    ell_comps_sigma : float = 0.3,
    use_spherical: bool = False,
) -> af.Collection:
    """
    Construct a Multi-Gaussian Expansion (MGE) for the lens or source galaxy light.

    This model is designed as a "start here" configuration for lens modeling:

    - The lens and source light are represented by a Basis object composed of many
      Gaussian light profiles with fixed logarithmically spaced widths (`sigma`).
    - All Gaussians within each basis share common centres and ellipticity
      components, reducing degeneracy while retaining flexibility.
    - Users can combine with a lens mass model of their choice.

    When ``gaussian_per_basis > 1``, each basis receives **independent** ellipticity
    components (``ell_comps``), allowing the model to represent twisting or varying
    isophotes across different radial scales. Centres are **shared** across bases by
    default (the common case: one luminosity centre, complex isophotal shape). Set
    ``centre_per_basis=True`` to give each basis its own centre priors.

    Expected free-parameter counts (elliptical, ``use_spherical=False``):

    - ``gaussian_per_basis=1`` : 2 centre + 2 ell_comps = 4
    - ``gaussian_per_basis=K`` (shared centre) : 2 + 2K
    - ``gaussian_per_basis=K, centre_per_basis=True`` : 2K + 2K = 4K

    Spherical (``use_spherical=True``): no ell_comps, only centres.

    - Shared centre: 2.  Per-basis centre: 2K.

    Parameters
    ----------
    mask_radius
        The outer radius (in arcseconds) of the circular mask applied to the data.
        This determines the maximum Gaussian width (`sigma`) used in the MGE.
    total_gaussians
        Total number of Gaussian light profiles used in each basis.
    gaussian_per_basis
        Number of separate Gaussian bases. Each basis has ``total_gaussians``
        components sharing the same centre and ellipticity. Multiple bases allow
        independent ellipticity (and optionally centre) per radial scale group.
    centre_prior_is_uniform
        If True (default), centre priors are ``UniformPrior(±0.1)`` around
        ``centre``. If False, ``GaussianPrior`` with ``centre_sigma``.
    centre
        (y, x) centre in arcseconds used as the mean/midpoint for centre priors.
    centre_fixed
        If not None, fix all Gaussian centres to this (y, x) value instead of
        making them free parameters. Overrides ``centre_per_basis``.
    centre_per_basis
        If True, each basis gets independently drawn centre priors. If False
        (default), all bases share the same centre. Ignored when ``centre_fixed``
        is set.
    centre_sigma
        Sigma for ``GaussianPrior`` centre priors (used when
        ``centre_prior_is_uniform=False``).
    ell_comps_prior_is_uniform
        If True, ell_comps priors are ``UniformPrior``. If False (default),
        ``TruncatedGaussianPrior``.
    ell_comps_uniform_width
        Half-width for uniform ell_comps priors.
    ell_comps_sigma
        Sigma for truncated-Gaussian ell_comps priors.
    use_spherical
        If True, use ``GaussianSph`` (no ell_comps). If False (default), use
        ``Gaussian`` with ellipticity.

    Returns
    -------
    af.Model
        An ``autofit.Model`` wrapping a ``Basis`` of linear Gaussians.
    """

    import os

    if os.environ.get("PYAUTO_SMALL_DATASETS") == "1":
        total_gaussians = 2
        gaussian_per_basis = 1

    from autogalaxy.profiles.light.linear import Gaussian, GaussianSph
    from autogalaxy.profiles.basis import Basis

    # The sigma values of the Gaussians will be fixed to values spanning 0.01 to the mask radius.
    log10_sigma_list = np.linspace(-4, np.log10(mask_radius), total_gaussians)

    if use_spherical:
        model_cls = GaussianSph
    else:
        model_cls = Gaussian

    def _make_centre_priors():
        if centre_fixed is not None:
            return centre[0], centre[1]
        elif centre_prior_is_uniform:
            return (
                af.UniformPrior(
                    lower_limit=centre[0] - 0.1, upper_limit=centre[0] + 0.1
                ),
                af.UniformPrior(
                    lower_limit=centre[1] - 0.1, upper_limit=centre[1] + 0.1
                ),
            )
        else:
            return (
                af.GaussianPrior(mean=centre[0], sigma=centre_sigma),
                af.GaussianPrior(mean=centre[1], sigma=centre_sigma),
            )

    def _make_ell_comps_priors():
        if ell_comps_prior_is_uniform:
            return (
                af.UniformPrior(lower_limit=-ell_comps_uniform_width, upper_limit=ell_comps_uniform_width),
                af.UniformPrior(lower_limit=-ell_comps_uniform_width, upper_limit=ell_comps_uniform_width),
            )
        else:
            return (
                af.TruncatedGaussianPrior(mean=0.0, sigma=ell_comps_sigma, lower_limit=-1.0, upper_limit=1.0),
                af.TruncatedGaussianPrior(mean=0.0, sigma=ell_comps_sigma, lower_limit=-1.0, upper_limit=1.0),
            )

    # Shared centre priors (used when centre_per_basis=False).
    if not centre_per_basis or centre_fixed is not None:
        shared_centre_0, shared_centre_1 = _make_centre_priors()

    bulge_gaussian_list = []

    for j in range(gaussian_per_basis):

        # Per-basis centre priors when requested.
        if centre_per_basis and centre_fixed is None:
            centre_0, centre_1 = _make_centre_priors()
        else:
            centre_0, centre_1 = shared_centre_0, shared_centre_1

        # Per-basis ell_comps priors (always independent across bases).
        if not use_spherical:
            ell_comps_0, ell_comps_1 = _make_ell_comps_priors()

        gaussian_list = af.Collection(
            af.Model(model_cls) for _ in range(total_gaussians)
        )

        for i, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = centre_0
            gaussian.centre.centre_1 = centre_1
            if not use_spherical:
                gaussian.ell_comps.ell_comps_0 = ell_comps_0
                gaussian.ell_comps.ell_comps_1 = ell_comps_1
            gaussian.sigma = 10 ** log10_sigma_list[i]

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


SIMULATOR_RANDOM_GALAXY_SUMMARY = (
    "Each simulated galaxy draws a fresh bulge from: "
    "signal-to-noise ratio in [20, 60], "
    "effective radius in [1.0, 5.0] arcsec, "
    "sersic index in [3.5, 4.5], "
    "ell_comps each ~ Normal(0, 0.2) clipped to [-1, 1]."
)


def random_galaxy_for_simulation_from(
    rng: Optional[np.random.Generator] = None,
) -> "Galaxy":
    """
    Sample a random ``Galaxy`` instance with an SNR-normalised Sersic bulge.

    This helper is for **synthetic data generation only** — it draws each
    parameter directly from a numpy ``Generator`` and returns a concrete
    ``Galaxy`` instance whose bulge is an ``lp_snr.Sersic`` with a target
    signal-to-noise ratio. The SNR profile internally back-computes the
    matching ``intensity`` from the simulator's noise level.

    Do **not** use the returned ``Galaxy`` as a fitting model — SNR is a
    property of the data, not a parameter you fit. For fitting, build a
    regular ``af.Model(ag.lp.Sersic)`` with an explicit ``intensity`` prior.

    Parameters
    ----------
    rng
        Optional ``numpy.random.Generator`` for reproducible sampling. If
        ``None`` (default) a fresh ``default_rng()`` is created on each call,
        so each call returns a different galaxy.

    Returns
    -------
    Galaxy
        A ``Galaxy`` at redshift 0.5 with a single ``lp_snr.Sersic`` bulge.
    """
    from autogalaxy.profiles.light.snr import Sersic
    from autogalaxy.galaxy.galaxy import Galaxy

    rng = rng if rng is not None else np.random.default_rng()

    def _clipped_ell_comp() -> float:
        return float(np.clip(rng.normal(0.0, 0.2), -1.0, 1.0))

    bulge = Sersic(
        centre=(0.0, 0.0),
        ell_comps=(_clipped_ell_comp(), _clipped_ell_comp()),
        effective_radius=float(rng.uniform(1.0, 5.0)),
        sersic_index=float(rng.uniform(3.5, 4.5)),
        signal_to_noise_ratio=float(rng.uniform(20.0, 60.0)),
    )

    return Galaxy(redshift=0.5, bulge=bulge)
