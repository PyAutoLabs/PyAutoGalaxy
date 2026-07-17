"""
Bidirectional parameter conversion between PyAutoGalaxy light profiles and the
COOLEST standard's light profiles.

The PyAutoGalaxy ``Sersic`` evaluates its image on *eccentric* radii
``sqrt(q) * sqrt(x^2 + (y/q)^2)`` (``geometry_profiles.py``), which is exactly
the COOLEST intermediate-axis radius ``sqrt(q x^2 + y^2 / q)``, and its
``intensity`` is the amplitude at ``effective_radius``. The Sersic mapping is
therefore an identity on (I_eff, theta_eff, n) — only the centre ordering and
the ellipticity / position-angle conventions change.
"""

from typing import Callable, Dict

from autogalaxy import exc
from autogalaxy.interop.coolest import conventions
from autogalaxy.profiles.light.standard.sersic import Sersic
from autogalaxy.profiles.light.standard.sersic import SersicSph


def _sersic_dict_from(profile, **kwargs) -> Dict:
    q, phi = conventions.q_phi_from_ell_comps(ell_comps=profile.ell_comps)
    center_x, center_y = conventions.center_from_centre(centre=profile.centre)
    return {
        "type": "Sersic",
        "parameters": {
            "I_eff": float(profile.intensity),
            "theta_eff": float(profile.effective_radius),
            "n": float(profile.sersic_index),
            "q": q,
            "phi": phi,
            "center_x": center_x,
            "center_y": center_y,
        },
    }


def _sersic_from(parameters: Dict) -> Sersic:
    q = parameters["q"]
    centre = conventions.centre_from_center(
        center_x=parameters["center_x"], center_y=parameters["center_y"]
    )
    if q == 1.0:
        return SersicSph(
            centre=centre,
            intensity=float(parameters["I_eff"]),
            effective_radius=float(parameters["theta_eff"]),
            sersic_index=float(parameters["n"]),
        )
    return Sersic(
        centre=centre,
        ell_comps=conventions.ell_comps_from_q_phi(q=q, phi=parameters["phi"]),
        intensity=float(parameters["I_eff"]),
        effective_radius=float(parameters["theta_eff"]),
        sersic_index=float(parameters["n"]),
    )


_TO_COOLEST: Dict[type, Callable] = {
    Sersic: _sersic_dict_from,
    SersicSph: _sersic_dict_from,
}

_FROM_COOLEST: Dict[str, Callable] = {
    "Sersic": _sersic_from,
}


def coolest_dict_from_light(profile) -> Dict:
    """
    Convert a PyAutoGalaxy light profile to its COOLEST representation, a dict
    ``{"type": <COOLEST profile name>, "parameters": {<name>: <float>}}`` with
    all parameters in COOLEST conventions.

    Parameters
    ----------
    profile
        The light profile to convert. Supported: ``Sersic`` / ``SersicSph``.
    """
    try:
        to_coolest = _TO_COOLEST[type(profile)]
    except KeyError:
        raise exc.ProfileException(
            f"The light profile {type(profile).__name__} has no COOLEST "
            f"converter. Supported profiles: "
            f"{sorted(cls.__name__ for cls in _TO_COOLEST)}."
        )
    return to_coolest(profile)


def light_profile_from(profile_type: str, parameters: Dict):
    """
    Build a PyAutoGalaxy light profile from a COOLEST profile type name and
    its parameter dict (in COOLEST conventions).

    Parameters
    ----------
    profile_type
        The COOLEST profile name, e.g. "Sersic".
    parameters
        The COOLEST parameters of the profile, e.g. point-estimate values read
        from a COOLEST template file.
    """
    try:
        from_coolest = _FROM_COOLEST[profile_type]
    except KeyError:
        raise exc.ProfileException(
            f"The COOLEST light profile '{profile_type}' has no PyAutoGalaxy "
            f"converter. Supported: {sorted(_FROM_COOLEST)}."
        )
    return from_coolest(parameters)
