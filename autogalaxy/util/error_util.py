"""
Utility functions for computing posterior error estimates on 1D profiles and 2D ellipses.

These helpers operate on lists of sampled quantities (e.g. 1D light-profile intensity
curves or 2D isophote coordinate arrays drawn from a ``PyAutoFit`` posterior PDF) and
return the median value together with a lower/upper confidence interval computed from
user-specified quantiles.

Key functions
-------------
- ``value_median_and_error_region_via_quantile`` — scalar median + credible interval.
- ``profile_1d_median_and_error_region_via_quantile`` — per-radial-bin median + interval
  for 1D profiles (e.g. a convergence or surface-brightness radial profile).
- ``quantile_profile_1d`` — low-level per-bin quantile computation, adapted from
  ``corner.py``.
- ``ellipse_median_and_error_region_via_quantile`` — Cartesian median + interval for 2D
  isophote coordinate arrays.
- ``ellipse_median_and_error_region_in_polar`` — same but computed in polar coordinates
  so that the confidence region is expressed radially rather than in (y, x), which is
  more robust for highly elongated isophotes.
"""
import numpy as np

from autofit.non_linear.samples.pdf import quantile


def value_median_and_error_region_via_quantile(value_list, low_limit):
    median_profile_1d = quantile(x=value_list, q=0.5)
    lower_profile_1d = quantile(x=value_list, q=low_limit)
    upper_profile_1d = quantile(x=value_list, q=1 - low_limit)

    return median_profile_1d, [lower_profile_1d, upper_profile_1d]


def profile_1d_median_and_error_region_via_quantile(profile_1d_list, low_limit):
    median_profile_1d = quantile_profile_1d(profile_1d_list=profile_1d_list, q=0.5)
    lower_profile_1d = quantile_profile_1d(profile_1d_list=profile_1d_list, q=low_limit)
    upper_profile_1d = quantile_profile_1d(
        profile_1d_list=profile_1d_list, q=1 - low_limit
    )

    return median_profile_1d, [lower_profile_1d, upper_profile_1d]


def quantile_profile_1d(profile_1d_list, q, weights=None):
    """
    This function is adapted from from corner.py

    It compute sample quantiles specifically for 1D profiles of light and mass profiles. This is so that their
    errors can be plotted on 1D plots, via random draws from a PDF.

    The input to the function is a list of 1D profiles, where each entry in this list contains a list or numpy array
    of the 1D profile values. The quantile method is used on each list of radial coordinates to compute the 1D profile
    at a given input confidence (e.g. `q=0.5` gives the median 1D profile).

    Note
    ----
    When ``weight_list`` is ``None``, this method simply calls numpy's percentile
    function with the values of ``q`` multiplied by 100.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.
    q : array_like[nquantiles,]
       The list of quantiles to compute. These should all be in the range
       ``[0, 1]``.
    weights : Optional[array_like[nsamples,]]
        An optional weight corresponding to each sample. These

    Returns
    -------
    quantiles : array_like[nquantiles,]
        The sample quantiles computed at ``q``.

    Raises
    ------
    ValueError
        For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch between ``x`` and ``weight_list``.
    """

    radial_quantile = np.zeros(shape=profile_1d_list[0].shape[0])

    for radial_index in range(profile_1d_list[0].shape[0]):
        radial_list = [profile_1d[radial_index] for profile_1d in profile_1d_list]

        radial_quantile[radial_index] = quantile(x=radial_list, q=q, weights=weights)[0]

    return radial_quantile


def quantile_ellipse(ellipse_list, q):
    """
    Compute the per-point quantile ellipse for a list of 2D ellipses.
    """
    stacked = np.stack(ellipse_list, axis=0)  # shape: (n_samples, n_points, 2)
    return np.quantile(stacked, q=q, axis=0)  # shape: (n_points, 2)


def ellipse_median_and_error_region_via_quantile(ellipse_list, low_limit):
    """
    Compute the median and confidence bounds for an ellipse shape.
    """
    median_ellipse = quantile_ellipse(ellipse_list, q=0.5)
    lower_ellipse = quantile_ellipse(ellipse_list, q=low_limit)
    upper_ellipse = quantile_ellipse(ellipse_list, q=1.0 - low_limit)

    return median_ellipse, [lower_ellipse, upper_ellipse]


def cartesian_to_polar(yx, center=(0.0, 0.0)):
    y, x = yx[..., 0], yx[..., 1]
    y0, x0 = center
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    theta = np.arctan2(y - y0, x - x0)
    # Wrap to [0, 2π]
    theta = np.mod(theta, 2 * np.pi)
    return r, theta


def polar_to_cartesian(r, theta, center=(0.0, 0.0)):
    y0, x0 = center
    x = r * np.cos(theta) + x0
    y = r * np.sin(theta) + y0
    return np.stack([y, x], axis=-1)


def ellipse_median_and_error_region_in_polar(
    ellipse_list, low_limit, center=(0.0, 0.0)
):
    """
    Computes median and error ellipses in (y,x) space using polar quantiles.

    Parameters
    ----------
    ellipse_list : list of arrays, each of shape (n_points, 2)
        A list of (y, x) coordinate ellipses.
    sigma : float
        The sigma confidence level (e.g. 1.0 for 68%).
    center : tuple of float
        The center (y0, x0) for the polar projection.

    Returns
    -------
    (median, [lower, upper]) : tuple of np.ndarray
        Each array has shape (n_points, 2), giving the (y, x) coordinates
        of the median and error ellipses.
    """

    # Convert all ellipses to polar (r, θ), shape: (n_ellipses, n_points)
    r_list = []
    theta_ref = None

    for ellipse in ellipse_list:

        r, theta = cartesian_to_polar(ellipse, center=center)
        r_list.append(r)

        if theta_ref is None:
            theta_ref = theta  # Assume all ellipses use same angle ordering

    r_array = np.stack(r_list, axis=0)  # shape: (n_ellipses, n_points)

    median_r = np.quantile(r_array, q=0.5, axis=0)
    lower_r = np.quantile(r_array, q=low_limit, axis=0)
    upper_r = np.quantile(r_array, q=1 - low_limit, axis=0)

    # Convert back to (y, x)
    median_yx = polar_to_cartesian(median_r, theta_ref, center=center)
    lower_yx = polar_to_cartesian(lower_r, theta_ref, center=center)
    upper_yx = polar_to_cartesian(upper_r, theta_ref, center=center)

    return median_yx, [lower_yx, upper_yx]
