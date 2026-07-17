"""
Bidirectional parameter conversion between PyAutoGalaxy mass profiles and the
COOLEST standard's mass profiles.

Every converter is dict-in / dict-out on plain floats — no ``coolest``
import — see ``autogalaxy.interop.coolest.__init__`` for the convention
summary. Each mapping is defined by equating the profile's *convergence*
in the two parameterisations at every point, so a profile exported to
COOLEST and re-imported is numerically identical.

Einstein radius conventions
---------------------------

The PyAutoGalaxy power-law convergence is (``power_law_core.py``):

    kappa(xi) = (3 - gamma) / (1 + q) * (theta_ag / xi)^(gamma - 1)

where ``xi = sqrt(x^2 + (y/q)^2)`` is the *major-axis* radius of the
elliptical contour through (x, y) in the rotated profile frame.

The COOLEST PEMD convergence (following lenstronomy / Tessore & Metcalf 2015)
is:

    kappa(r) = (3 - gamma) / 2 * (theta_cool / r)^(gamma - 1)

where ``r = sqrt(q x^2 + y^2 / q) = sqrt(q) * xi`` is the *intermediate-axis*
radius. Equating the two at every point gives:

    theta_cool = sqrt(q) * (2 / (1 + q))^(1 / (gamma - 1)) * theta_ag

which for the isothermal case (gamma = 2) reduces to the familiar
``theta_cool = 2 sqrt(q) / (1 + q) * theta_ag``.
"""

from typing import Callable, Dict, Optional

import numpy as np

from autogalaxy import convert
from autogalaxy import exc
from autogalaxy.interop.coolest import conventions
from autogalaxy.profiles.mass.dark.nfw import NFW
from autogalaxy.profiles.mass.dark.nfw import NFWSph
from autogalaxy.profiles.mass.sheets.external_shear import ExternalShear
from autogalaxy.profiles.mass.sheets.mass_sheet import MassSheet
from autogalaxy.profiles.mass.total.isothermal import Isothermal
from autogalaxy.profiles.mass.total.isothermal import IsothermalSph
from autogalaxy.profiles.mass.total.power_law import PowerLaw
from autogalaxy.profiles.mass.total.power_law import PowerLawSph


def einstein_radius_coolest_from(
    einstein_radius: float, axis_ratio: float, slope: float = 2.0
) -> float:
    """
    Convert a PyAutoGalaxy power-law Einstein radius to the COOLEST
    (intermediate-axis) Einstein radius — see the module docstring for the
    derivation.

    Parameters
    ----------
    einstein_radius
        The PyAutoGalaxy ``einstein_radius`` parameter of the profile.
    axis_ratio
        The axis ratio q = b/a of the profile.
    slope
        The logarithmic density slope gamma of the power-law.
    """
    return (
        einstein_radius
        * np.sqrt(axis_ratio)
        * (2.0 / (1.0 + axis_ratio)) ** (1.0 / (slope - 1.0))
    )


def einstein_radius_ag_from(
    theta_E: float, axis_ratio: float, slope: float = 2.0
) -> float:
    """
    Convert a COOLEST (intermediate-axis) Einstein radius ``theta_E`` to the
    PyAutoGalaxy power-law ``einstein_radius`` parameter — the inverse of
    ``einstein_radius_coolest_from``.
    """
    return (
        theta_E
        / np.sqrt(axis_ratio)
        * ((1.0 + axis_ratio) / 2.0) ** (1.0 / (slope - 1.0))
    )


def _sie_dict_from(profile, **kwargs) -> Dict:
    q, phi = conventions.q_phi_from_ell_comps(ell_comps=profile.ell_comps)
    center_x, center_y = conventions.center_from_centre(centre=profile.centre)
    return {
        "type": "SIE",
        "parameters": {
            "theta_E": float(
                einstein_radius_coolest_from(
                    einstein_radius=profile.einstein_radius,
                    axis_ratio=q,
                    slope=2.0,
                )
            ),
            "q": q,
            "phi": phi,
            "center_x": center_x,
            "center_y": center_y,
        },
    }


def _isothermal_from(parameters: Dict) -> Isothermal:
    q = parameters["q"]
    centre = conventions.centre_from_center(
        center_x=parameters["center_x"], center_y=parameters["center_y"]
    )
    einstein_radius = float(
        einstein_radius_ag_from(theta_E=parameters["theta_E"], axis_ratio=q, slope=2.0)
    )
    # An exactly-round COOLEST profile maps to the spherical class — the
    # elliptical Isothermal clips its axis ratio to 0.99999 for the stability
    # of its analytic deflections, so it is not numerically exact at q = 1.
    if q == 1.0:
        return IsothermalSph(centre=centre, einstein_radius=einstein_radius)
    return Isothermal(
        centre=centre,
        ell_comps=conventions.ell_comps_from_q_phi(q=q, phi=parameters["phi"]),
        einstein_radius=einstein_radius,
    )


def _pemd_dict_from(profile, **kwargs) -> Dict:
    q, phi = conventions.q_phi_from_ell_comps(ell_comps=profile.ell_comps)
    center_x, center_y = conventions.center_from_centre(centre=profile.centre)
    return {
        "type": "PEMD",
        "parameters": {
            "gamma": float(profile.slope),
            "theta_E": float(
                einstein_radius_coolest_from(
                    einstein_radius=profile.einstein_radius,
                    axis_ratio=q,
                    slope=profile.slope,
                )
            ),
            "q": q,
            "phi": phi,
            "center_x": center_x,
            "center_y": center_y,
        },
    }


def _power_law_from(parameters: Dict) -> PowerLaw:
    q = parameters["q"]
    slope = parameters["gamma"]
    centre = conventions.centre_from_center(
        center_x=parameters["center_x"], center_y=parameters["center_y"]
    )
    einstein_radius = float(
        einstein_radius_ag_from(
            theta_E=parameters["theta_E"], axis_ratio=q, slope=slope
        )
    )
    if q == 1.0:
        return PowerLawSph(
            centre=centre, einstein_radius=einstein_radius, slope=slope
        )
    return PowerLaw(
        centre=centre,
        ell_comps=conventions.ell_comps_from_q_phi(q=q, phi=parameters["phi"]),
        einstein_radius=einstein_radius,
        slope=slope,
    )


def _nfw_dict_from(profile, sigma_crit: Optional[float] = None, **kwargs) -> Dict:
    """
    The PyAutoGalaxy NFW convergence is ``kappa(xi) = 2 kappa_s g(xi / r_s)``
    with ``xi`` the major-axis elliptical radius and ``kappa_s`` dimensionless;
    COOLEST parameterises the NFW by its scale radius ``r_s`` (intermediate
    axis) and characteristic density ``rho_c``, with
    ``kappa_s = rho_c * r_s / Sigma_crit``. The conversion therefore requires
    the critical surface mass density ``Sigma_crit`` of the lens
    configuration, which depends on redshifts and cosmology and must be
    supplied by the caller (PyAutoLens computes it from the lens model).

    ``rho_c`` is returned in units of ``[sigma_crit] / arcsec`` — supply
    ``sigma_crit`` in mass per arcsec**2 to obtain mass per arcsec**3.
    """
    if sigma_crit is None:
        raise exc.ProfileException(
            "Converting an NFW profile to COOLEST requires sigma_crit, the "
            "critical surface mass density of the lens configuration, because "
            "COOLEST parameterises the NFW by a physical density rho_c whereas "
            "the PyAutoGalaxy NFW kappa_s is dimensionless. Compute sigma_crit "
            "from the lens and source redshifts and a cosmology (see "
            "autolens.interop.coolest, which does this automatically)."
        )
    q, phi = conventions.q_phi_from_ell_comps(ell_comps=profile.ell_comps)
    center_x, center_y = conventions.center_from_centre(centre=profile.centre)
    r_s = conventions.radius_intermediate_from(
        radius_major=profile.scale_radius, q=q
    )
    return {
        "type": "NFW",
        "parameters": {
            "r_s": r_s,
            "rho_c": float(profile.kappa_s * sigma_crit / r_s),
            "q": q,
            "phi": phi,
            "center_x": center_x,
            "center_y": center_y,
        },
    }


def _nfw_from(parameters: Dict, sigma_crit: Optional[float] = None) -> NFW:
    if sigma_crit is None:
        raise exc.ProfileException(
            "Converting a COOLEST NFW profile to PyAutoGalaxy requires "
            "sigma_crit, the critical surface mass density of the lens "
            "configuration — see autogalaxy.interop.coolest.mass._nfw_dict_from."
        )
    q = parameters["q"]
    r_s = parameters["r_s"]
    centre = conventions.centre_from_center(
        center_x=parameters["center_x"], center_y=parameters["center_y"]
    )
    kappa_s = float(parameters["rho_c"] * r_s / sigma_crit)
    scale_radius = float(conventions.radius_major_from(radius_intermediate=r_s, q=q))
    if q == 1.0:
        return NFWSph(centre=centre, kappa_s=kappa_s, scale_radius=scale_radius)
    return NFW(
        centre=centre,
        ell_comps=conventions.ell_comps_from_q_phi(q=q, phi=parameters["phi"]),
        kappa_s=kappa_s,
        scale_radius=scale_radius,
    )


def _external_shear_dict_from(profile, **kwargs) -> Dict:
    magnitude, angle = convert.shear_magnitude_and_angle_from(
        gamma_1=profile.gamma_1, gamma_2=profile.gamma_2
    )
    return {
        "type": "ExternalShear",
        "parameters": {
            "gamma_ext": float(magnitude),
            "phi_ext": conventions.phi_coolest_from(angle=float(angle)),
        },
    }


def _external_shear_from(parameters: Dict) -> ExternalShear:
    gamma_1, gamma_2 = convert.shear_gamma_1_2_from(
        magnitude=parameters["gamma_ext"],
        angle=conventions.angle_from_phi_coolest(phi=parameters["phi_ext"]),
    )
    return ExternalShear(gamma_1=float(gamma_1), gamma_2=float(gamma_2))


def _convergence_sheet_dict_from(profile, **kwargs) -> Dict:
    if tuple(profile.centre) != (0.0, 0.0):
        raise exc.ProfileException(
            "The COOLEST ConvergenceSheet is defined with its origin fixed at "
            f"(0, 0), but this MassSheet has centre={profile.centre}. Only "
            "sheets centred on the origin can be converted."
        )
    return {
        "type": "ConvergenceSheet",
        "parameters": {"kappa_s": float(profile.kappa)},
    }


def _mass_sheet_from(parameters: Dict) -> MassSheet:
    return MassSheet(centre=(0.0, 0.0), kappa=float(parameters["kappa_s"]))


_TO_COOLEST: Dict[type, Callable] = {
    Isothermal: _sie_dict_from,
    IsothermalSph: _sie_dict_from,
    PowerLaw: _pemd_dict_from,
    PowerLawSph: _pemd_dict_from,
    NFW: _nfw_dict_from,
    NFWSph: _nfw_dict_from,
    ExternalShear: _external_shear_dict_from,
    MassSheet: _convergence_sheet_dict_from,
}

_FROM_COOLEST: Dict[str, Callable] = {
    "SIE": _isothermal_from,
    "PEMD": _power_law_from,
    "ExternalShear": _external_shear_from,
    "ConvergenceSheet": _mass_sheet_from,
}


def coolest_dict_from_mass(profile, sigma_crit: Optional[float] = None) -> Dict:
    """
    Convert a PyAutoGalaxy mass profile to its COOLEST representation, a dict
    ``{"type": <COOLEST profile name>, "parameters": {<name>: <float>}}`` with
    all parameters in COOLEST conventions.

    Parameters
    ----------
    profile
        The mass profile to convert. Supported: ``Isothermal`` /
        ``IsothermalSph`` (-> SIE), ``PowerLaw`` / ``PowerLawSph`` (-> PEMD),
        ``NFW`` / ``NFWSph`` (-> NFW), ``ExternalShear``, ``MassSheet``
        (-> ConvergenceSheet).
    sigma_crit
        The critical surface mass density of the lens configuration, required
        only for NFW profiles (COOLEST uses a physical density normalization).
    """
    try:
        to_coolest = _TO_COOLEST[type(profile)]
    except KeyError:
        raise exc.ProfileException(
            f"The mass profile {type(profile).__name__} has no COOLEST "
            f"converter. Supported profiles: "
            f"{sorted(cls.__name__ for cls in _TO_COOLEST)}."
        )
    return to_coolest(profile, sigma_crit=sigma_crit)


def mass_profile_from(
    profile_type: str, parameters: Dict, sigma_crit: Optional[float] = None
):
    """
    Build a PyAutoGalaxy mass profile from a COOLEST profile type name and its
    parameter dict (in COOLEST conventions).

    Parameters
    ----------
    profile_type
        The COOLEST profile name, e.g. "SIE", "PEMD", "NFW", "ExternalShear",
        "ConvergenceSheet".
    parameters
        The COOLEST parameters of the profile, e.g. point-estimate values read
        from a COOLEST template file.
    sigma_crit
        The critical surface mass density of the lens configuration, required
        only for NFW profiles.
    """
    if profile_type == "NFW":
        return _nfw_from(parameters=parameters, sigma_crit=sigma_crit)
    try:
        from_coolest = _FROM_COOLEST[profile_type]
    except KeyError:
        raise exc.ProfileException(
            f"The COOLEST mass profile '{profile_type}' has no PyAutoGalaxy "
            f"converter. Supported: {sorted(_FROM_COOLEST) + ['NFW']}."
        )
    return from_coolest(parameters)
