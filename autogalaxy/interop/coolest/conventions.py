from typing import Tuple

import numpy as np

from autogalaxy import convert


def phi_coolest_from(angle: float) -> float:
    """
    Convert a PyAutoGalaxy position angle to a COOLEST position angle.

    PyAutoGalaxy position angles are measured counter-clockwise from the
    positive x axis; COOLEST position angles are measured counter-clockwise
    from the positive y axis ("East-of-North") and lie in (-90, +90].

    Parameters
    ----------
    angle
        Position angle in degrees, counter-clockwise from the positive x axis.

    Returns
    -------
    Position angle in degrees, counter-clockwise from the positive y axis,
    normalized to (-90, +90].
    """
    phi = float(angle) - 90.0
    while phi <= -90.0:
        phi += 180.0
    while phi > 90.0:
        phi -= 180.0
    return phi


def angle_from_phi_coolest(phi: float) -> float:
    """
    Convert a COOLEST position angle (counter-clockwise from the positive y
    axis) to a PyAutoGalaxy position angle (counter-clockwise from the
    positive x axis).

    Parameters
    ----------
    phi
        Position angle in degrees, counter-clockwise from the positive y axis.
    """
    return float(phi) + 90.0


def q_phi_from_ell_comps(ell_comps: Tuple[float, float]) -> Tuple[float, float]:
    """
    Convert PyAutoGalaxy elliptical components (e1, e2) to the COOLEST axis
    ratio ``q`` and position angle ``phi`` (East-of-North, degrees).

    Parameters
    ----------
    ell_comps
        The elliptical components (e1, e2) of a light or mass profile.
    """
    axis_ratio, angle = convert.axis_ratio_and_angle_from(ell_comps=ell_comps)
    return float(axis_ratio), phi_coolest_from(angle=float(angle))


def ell_comps_from_q_phi(q: float, phi: float) -> Tuple[float, float]:
    """
    Convert a COOLEST axis ratio ``q`` and position angle ``phi``
    (East-of-North, degrees) to PyAutoGalaxy elliptical components (e1, e2).

    Parameters
    ----------
    q
        The axis ratio (b/a) of the ellipse.
    phi
        Position angle in degrees, counter-clockwise from the positive y axis.
    """
    ell_comps = convert.ell_comps_from(
        axis_ratio=q, angle=angle_from_phi_coolest(phi=phi)
    )
    return float(ell_comps[0]), float(ell_comps[1])


def center_from_centre(centre: Tuple[float, float]) -> Tuple[float, float]:
    """
    Convert a PyAutoGalaxy (y, x) profile centre to a COOLEST
    (center_x, center_y) position. Both are arcseconds with x increasing to
    the right and y increasing upwards, so only the ordering changes.

    Parameters
    ----------
    centre
        The (y, x) arc-second coordinates of the profile centre.
    """
    return float(centre[1]), float(centre[0])


def centre_from_center(center_x: float, center_y: float) -> Tuple[float, float]:
    """
    Convert a COOLEST (center_x, center_y) position to a PyAutoGalaxy (y, x)
    profile centre.
    """
    return float(center_y), float(center_x)


def radius_intermediate_from(radius_major: float, q: float) -> float:
    """
    Convert a major-axis radius to the COOLEST intermediate-axis radius
    r = sqrt(a * b) = sqrt(q) * a of the same elliptical contour.

    Parameters
    ----------
    radius_major
        The radius measured along the ellipse's major axis.
    q
        The axis ratio (b/a) of the ellipse.
    """
    return float(radius_major) * np.sqrt(q)


def radius_major_from(radius_intermediate: float, q: float) -> float:
    """
    Convert a COOLEST intermediate-axis radius r = sqrt(a * b) to the
    major-axis radius of the same elliptical contour.

    Parameters
    ----------
    radius_intermediate
        The intermediate-axis radius sqrt(a * b) of the elliptical contour.
    q
        The axis ratio (b/a) of the ellipse.
    """
    return float(radius_intermediate) / np.sqrt(q)
