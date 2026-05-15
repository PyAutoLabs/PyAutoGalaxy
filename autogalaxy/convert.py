import numpy as np
from typing import Tuple


def ell_comps_from(axis_ratio: float, angle: float, xp=np) -> Tuple[float, float]:
    """
    Returns the elliptical components e1 and e2 of a light or mass profile from an input angle in degrees and axis
    ratio.

    The elliptical components of a light or mass profile are given by:

    elliptical_component_y = ell_comps[0] = (1-axis_ratio)/(1+axis_ratio) * sin(2 * angle)
    elliptical_component_x = ell_comps[1] = (1-axis_ratio)/(1+axis_ratio) * cos(2 * angle)

    Which are the values this function returns.

    Parameters
    ----------
    axis_ratio
        Ratio of light profiles ellipse's minor and major axes (b/a).
    angle
        Rotation angle of light profile counter-clockwise from positive x-axis.
    """
    angle *= xp.pi / 180.0
    fac = (1 - axis_ratio) / (1 + axis_ratio)
    ellip_y = fac * xp.sin(2 * angle)
    ellip_x = fac * xp.cos(2 * angle)
    return (ellip_y, ellip_x)


def axis_ratio_and_angle_from(
    ell_comps: Tuple[float, float], xp=np
) -> Tuple[float, float]:
    """
    Returns the axis-ratio and position angle in degrees (-45 < angle < 135.0) from input elliptical components e1
    and e2 of a light or mass profile.

    The elliptical components of a light or mass profile are given by:

    elliptical_component_y = ell_comps[0] = (1-axis_ratio)/(1+axis_ratio) * sin(2 * angle)
    elliptical_component_x = ell_comps[1] = (1-axis_ratio)/(1+axis_ratio) * cos(2 * angle)

    The axis-ratio and angle are therefore given by:

    axis_ratio = (1 - fac) / (1 + fac)
    angle = 0.5 * arctan(ell_comps[0] / ell_comps[1])

    where `fac = sqrt(ell_comps[1] ** 2 + ell_comps[0] ** 2).

    This function returns the axis-ratio and angle in degrees.

    An additional check is performed which requires the angle is between -45 and 135 degrees. This ensures that
    for certain values of `ell_comps` the angle does not jump from one boundary to another (e.g. without this check
    certain values of `ell_comps` return -1.0 degrees and others 179.0 degrees). This ensures that when error
    estimates are computed from samples of a lens model via marginalization, the calculation is not biased by the
    angle jumping between these two values.

    Parameters
    ----------
    ell_comps
        The elliptical components of the light or mass profile which are converted to an angle.
    """
    angle = 0.5 * xp.arctan2(
        ell_comps[0],
        xp.where((ell_comps[0] == 0) & (ell_comps[1] == 0), 1.0, ell_comps[1]),
    )
    angle *= 180.0 / xp.pi

    angle = xp.where(angle < -45, angle + 180, angle)

    fac = xp.sqrt(ell_comps[1] ** 2 + ell_comps[0] ** 2)
    if xp.__name__.startswith("jax"):
        import jax

        fac = jax.lax.min(fac, 0.999)
    else:  # NumPy
        fac = np.minimum(fac, 0.999)

    axis_ratio = (1 - fac) / (1 + fac)
    return axis_ratio, angle


def axis_ratio_from(ell_comps: Tuple[float, float], xp=np):
    """
    Returns the axis-ratio from input elliptical components e1 and e2 of a light or mass profile.

    The elliptical components of a light or mass profile are given by:

    elliptical_component_y = ell_comps[0] = (1-axis_ratio)/(1+axis_ratio) * sin(2 * angle)
    elliptical_component_x = ell_comps[1] = (1-axis_ratio)/(1+axis_ratio) * cos(2 * angle)

    The axis-ratio is therefore given by:

    axis_ratio = (1 - fac) / (1 + fac)
    angle = 0.5 * arctan(ell_comps[0] / ell_comps[1])

    where `fac = sqrt(ell_comps[1] ** 2 + ell_comps[0] ** 2).

    This function returns the axis-ratio.

    Parameters
    ----------
    ell_comps
        The elliptical components of the light or mass profile which are converted to an angle.
    """
    axis_ratio, angle = axis_ratio_and_angle_from(ell_comps=ell_comps, xp=xp)
    return axis_ratio


def angle_from(ell_comps: Tuple[float, float], xp=np) -> float:
    """
    Returns the position angle in degrees (-45 < angle < 135.0) from input elliptical components e1 and e2
    of a light or mass profile.

    The elliptical components of a light or mass profile are given by:

    elliptical_component_y = ell_comps[0] = (1-q)/(1+q) * sin(2 * angle)
    elliptical_component_x = ell_comps[1] = (1-q)/(1+q) * cos(2 * angle)

    axis_ratio = (1 - fac) / (1 + fac)
    angle = 0.5 * arctan(ell_comps[0] / ell_comps[1])

    where `fac = sqrt(ell_comps[1] ** 2 + ell_comps[0] ** 2).

    This function returns the angle in degrees.

    An additional check is performed which requires the angle is between -45 and 135 degrees. This ensures that
    for certain values of `ell_comps` the angle does not jump from one boundary to another (e.g. without this check
    certain values of `ell_comps` return -1.0 degrees and others 179.0 degrees).

    This ensures that when error estimates are computed from samples of a lens model via marginalization, the
    calculation is not biased by the angle jumping between these two values.

    Parameters
    ----------
    ell_comps
        The elliptical components of the light or mass profile which are converted to an angle.
    """
    axis_ratio, angle = axis_ratio_and_angle_from(ell_comps=ell_comps, xp=xp)
    return angle


def shear_gamma_1_2_from(magnitude: float, angle: float, xp=np) -> Tuple[float, float]:
    """
    Returns the shear gamma 1 and gamma 2 values an input shear magnitude and angle in degrees.

    The gamma 1 and gamma 2 components of a shear are given by:

    gamma_1 = magnitude * xp.cos(2 * angle * xp.pi / 180.0)
    gamma_2 = magnitude * xp.sin(2 * angle * xp.pi / 180.0)

    Which are the values this function returns.

    Converting from gamma 1 and gamma 2 to magnitude and angle is given by:

    magnitude = xp.sqrt(gamma_1**2 + gamma_2**2)
    angle = xp.arctan2(gamma_2, gamma_1) / 2 * 180.0 / xp.pi

    Parameters
    ----------
    axis_ratio
        Ratio of light profiles ellipse's minor and major axes (b/a).
    angle
        Rotation angle of light profile counter-clockwise from positive x-axis.
    """
    gamma_1 = magnitude * xp.cos(2 * angle * xp.pi / 180.0)
    gamma_2 = magnitude * xp.sin(2 * angle * xp.pi / 180.0)
    return (gamma_1, gamma_2)


def shear_magnitude_and_angle_from(
    gamma_1: float, gamma_2: float, xp=np
) -> Tuple[float, float]:
    """
    Returns the shear magnitude and angle in degrees from input shear gamma 1 and gamma 2 values.

    The gamma 1 and gamma 2 components of a shear are given by:

    gamma_1 = magnitude * xp.cos(2 * angle * xp.pi / 180.0)
    gamma_2 = magnitude * xp.sin(2 * angle * xp.pi / 180.0)

    Converting from gamma 1 and gamma 2 to magnitude and angle is given by:

    magnitude = xp.sqrt(gamma_1**2 + gamma_2**2)
    angle = xp.arctan2(gamma_2, gamma_1) / 2 * 180.0 / xp.pi

    Which are the values this function returns.

    Additional checks are performed which requires the angle is between -45 and 135 degrees. This ensures that
    for certain values of `gamma_1` and `gamma_2` the angle does not jump from one boundary to another (e.g. without
    this check certain values of `gamma_1` and `gamma_2` return -1.0 degrees and others 179.0 degrees).

    This ensures that when error estimates are computed from samples of a lens model via marginalization, the
    calculation is not biased by the angle jumping between these two values.

    Parameters
    ----------
    gamma_1
        The gamma 1 component of the shear.
    gamma_2
        The gamma 2 component of the shear.
    """
    angle = (
        0.5
        * xp.arctan2(gamma_2, xp.where((gamma_1 == 0) & (gamma_2 == 0), 1.0, gamma_1))
        * 180.0
        / xp.pi
    )
    magnitude = xp.sqrt(gamma_1**2 + gamma_2**2)

    angle = xp.where(angle < 0, angle + 180.0, angle)
    angle = xp.where(
        (xp.abs(angle - 90.0) > 45.0) & (angle > 90.0), angle - 180.0, angle
    )

    return magnitude, angle


def shear_magnitude_from(gamma_1: float, gamma_2: float, xp=np) -> float:
    """
    Returns the shear magnitude and angle in degrees from input shear gamma 1 and gamma 2 values.

    The gamma 1 and gamma 2 components of a shear are given by:

    gamma_1 = magnitude * xp.cos(2 * angle * xp.pi / 180.0)
    gamma_2 = magnitude * xp.sin(2 * angle * xp.pi / 180.0)

    Converting from gamma 1 and gamma 2 to magnitude and angle is given by:

    magnitude = xp.sqrt(gamma_1**2 + gamma_2**2)
    angle = xp.arctan2(gamma_2, gamma_1) / 2 * 180.0 / xp.pi

    The magnitude value is what this function returns.

    Parameters
    ----------
    gamma_1
        The gamma 1 component of the shear.
    gamma_2
        The gamma 2 component of the shear.
    """
    magnitude, angle = shear_magnitude_and_angle_from(
        gamma_1=gamma_1, gamma_2=gamma_2, xp=xp
    )
    return magnitude


def shear_angle_from(gamma_1: float, gamma_2: float, xp=np) -> float:
    """
    Returns the shear magnitude and angle in degrees from input shear gamma 1 and gamma 2 values.

    The gamma 1 and gamma 2 components of a shear are given by:

    gamma_1 = magnitude * xp.cos(2 * angle * xp.pi / 180.0)
    gamma_2 = magnitude * xp.sin(2 * angle * xp.pi / 180.0)

    Converting from gamma 1 and gamma 2 to magnitude and angle is given by:

    magnitude = xp.sqrt(gamma_1**2 + gamma_2**2)
    angle = xp.arctan2(gamma_2, gamma_1) / 2 * 180.0 / xp.pi

    The angle value is what this function returns.

    Parameters
    ----------
    gamma_1
        The gamma 1 component of the shear.
    gamma_2
        The gamma 2 component of the shear.
    """
    magnitude, angle = shear_magnitude_and_angle_from(
        gamma_1=gamma_1, gamma_2=gamma_2, xp=xp
    )
    return angle


def multipole_k_m_and_phi_m_from(
    multipole_comps: Tuple[float, float], m: int, xp=np
) -> Tuple[float, float]:
    """
    Returns the multipole normalization value `k_m` and angle `phi` from the multipole component parameters.

    The normalization and angle are given by:

    .. math::
        \phi^{\rm mass}_m = \frac{1}{m} \arctan{\frac{\epsilon_{\rm 2}^{\rm mp}}{\epsilon_{\rm 1}^{\rm mp}}}, \, \,
        k^{\rm mass}_m = \sqrt{{\epsilon_{\rm 1}^{\rm mp}}^2 + {\epsilon_{\rm 2}^{\rm mp}}^2} \, .

    The conversion depends on the multipole order `m`, to ensure that all possible rotationally symmetric
    multiple mass profiles are available in the conversion for multiple components spanning -inf to inf.

    Additional checks are performed which requires the angle `phi_m` is between -45 and 135 degrees. This ensures that
    for certain multipole component values the angle does not jump from one boundary to another (e.g. without
    this check certain values of `gamma_1` and `gamma_2` return -1.0 degrees and others 179.0 degrees).

    This ensures that when error estimates are computed from samples of a lens model via marginalization, the
    calculation is not biased by the angle jumping between these two values.

    Parameters
    ----------
    multipole_comps
        The first and second components of the multipole.

    Returns
    -------
    The normalization and angle parameters of the multipole.
    """
    phi_m = (
        xp.arctan2(
            multipole_comps[0],
            xp.where(
                (multipole_comps[0] == 0) & (multipole_comps[1] == 0),
                1.0,
                multipole_comps[1],
            ),
        )
        * 180.0
        / xp.pi
        / float(m)
    )

    k_m = xp.sqrt(multipole_comps[1] ** 2 + multipole_comps[0] ** 2)

    phi_m = xp.where(phi_m < -90.0 / m, phi_m + 360.0 / m, phi_m)

    return k_m, phi_m


def multipole_comps_from(
    k_m: float, phi_m: float, m: int, xp=np
) -> Tuple[float, float]:
    """
    Returns the multipole component parameters from their normalization value `k_m` and angle `phi`.

    .. math::
        \phi^{\rm mass}_m = \frac{1}{m} \arctan{\frac{\epsilon_{\rm 2}^{\rm mp}}{\epsilon_{\rm 1}^{\rm mp}}}, \, \,
        k^{\rm mass}_m = \sqrt{{\epsilon_{\rm 1}^{\rm mp}}^2 + {\epsilon_{\rm 2}^{\rm mp}}^2} \, .

    The conversion depends on the multipole order `m`, to ensure that all possible rotationally symmetric
    multiple mass profiles are available in the conversion for multiple components spanning -inf to inf.

    Parameters
    ----------
    k_m
        The magnitude of the multipole.
    phi_m
        The angle of the multipole.

    Returns
    -------
    The multipole component parameters.
    """
    # Degrees -> radians conversion
    deg_to_rad = xp.asarray(np.pi / 180.0)

    angle = phi_m * float(m) * deg_to_rad

    multipole_comp_0 = k_m * xp.sin(angle)
    multipole_comp_1 = k_m * xp.cos(angle)

    return multipole_comp_0, multipole_comp_1


def rotate_centre(centre_0, centre_1, ref_centre_0, ref_centre_1, angle, xp=np):
    """
    Rotate a centre around a reference centre by an angle in degrees.

    Coordinates are in (y, x) order: `centre_0` is y and `centre_1` is x.
    A positive angle rotates counter-clockwise.
    """
    angle_radians = angle * xp.pi / 180.0

    y_shift = centre_0 - ref_centre_0
    x_shift = centre_1 - ref_centre_1

    centre_0_rotated = (
        ref_centre_0
        + x_shift * xp.sin(angle_radians)
        + y_shift * xp.cos(angle_radians)
    )
    centre_1_rotated = (
        ref_centre_1
        + x_shift * xp.cos(angle_radians)
        - y_shift * xp.sin(angle_radians)
    )

    return centre_0_rotated, centre_1_rotated

