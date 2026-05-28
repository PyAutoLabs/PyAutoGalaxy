"""
Latent variables for galaxy imaging fits.

Provides flux-to-magnitude / magnitude-to-flux helpers, a flat registry of
named latent computations, and a yaml-driven enable/disable mechanism. The
registry is consumed by `AnalysisImaging.compute_latent_variables`, which
returns latent values as a tuple positionally aligned with `LATENT_KEYS`
(PyAutoFit zips the tuple with the keys at
`autofit/non_linear/analysis/analysis.py:285`).

User-level enable/disable: each key in `autogalaxy/config/latent.yaml` maps
to a bool. Disabled keys are dropped from `LATENT_KEYS` and not computed.
"""
import logging
from typing import Callable, Dict, List, Optional

import numpy as np

from autoconf import conf

logger = logging.getLogger(__name__)

# Latent names that have already emitted a "magzero missing" warning in this
# process. Used by ``_maybe_magzero_warn`` to deduplicate the message across
# the many fit evaluations a single search performs.
_MAGZERO_WARNED: set = set()


def _maybe_magzero_warn(magzero, name) -> bool:
    """
    Return True when ``magzero`` is missing (and emit a one-time-per-process
    warning for ``name``); False otherwise.

    Callers that get True must early-return ``xp.nan`` — the µJy conversion
    is meaningless without a zero-point, but a search-killing raise here
    would discard otherwise-converged fits.
    """
    if magzero is None:
        if name not in _MAGZERO_WARNED:
            logger.warning(
                "magzero not set on Analysis; '%s' latent will be NaN. "
                "Pass magzero=<value> to AnalysisImaging to enable it, "
                "or disable in config/latent.yaml to silence this warning.",
                name,
            )
            _MAGZERO_WARNED.add(name)
        return True
    return False


def ab_mag_via_flux_from(flux, magzero, xp=np):
    """
    Convert a linear flux to an AB magnitude given the image's magnitude
    zero-point.

    Parameters
    ----------
    flux
        The integrated flux in the image units used by the fit.
    magzero
        The magnitude zero-point of the image, such that
        ``mag = -2.5 * log10(flux) + magzero``.
    xp
        The numerical module (``numpy`` or ``jax.numpy``).
    """
    return -2.5 * xp.log10(flux) + magzero


def flux_mujy_via_ab_mag_from(ab_mag, xp=np):
    """
    Convert an AB magnitude to a flux in microjanskies.

    The AB system defines an AB magnitude of 23.9 as 1 microjansky.
    """
    return 10 ** ((23.9 - ab_mag) / 2.5)


def total_galaxy_0_flux(fit, magzero=None, xp=np):
    """
    Total integrated flux of the first galaxy in the fit, in the raw image
    units the fit was performed in (the sum of the model image pixels).

    Requires no instrument inputs — ``magzero`` is accepted for uniform
    dispatcher context but ignored. See ``autolens_workspace`` and
    ``autogalaxy_workspace`` flux guides for how to convert this to a
    microjansky flux using a user-supplied ``magzero``.

    Returns NaN when the first galaxy has no light profile (which raises
    inside ``fit.galaxy_image_dict``).
    """
    try:
        image = fit.galaxy_image_dict[fit.galaxies[0]]
    except (AttributeError, KeyError, IndexError):
        return xp.nan
    return xp.sum(image.array)


def total_galaxy_0_flux_mujy(fit, magzero, xp=np):
    """
    Total integrated flux of the first galaxy in the fit, converted to
    microjanskies via the image's magnitude zero-point.

    Returns NaN — with a one-time-per-process warning — when ``magzero``
    is missing, rather than raising. The µJy conversion is meaningless
    without a zero-point, but a hard raise during post-fit latent
    computation would discard the result of an otherwise-converged
    multi-hour search. Users who want this column populated should pass
    ``magzero=<value>`` to ``AnalysisImaging`` (Euclid pipeline pattern)
    or use :func:`total_galaxy_0_flux` and convert in post.

    Also returns NaN when the first galaxy has no light profile.
    """
    if _maybe_magzero_warn(magzero, "total_galaxy_0_flux_mujy"):
        return xp.nan

    try:
        image = fit.galaxy_image_dict[fit.galaxies[0]]
    except (AttributeError, KeyError, IndexError):
        return xp.nan

    total_flux = xp.sum(image.array)
    ab_mag = ab_mag_via_flux_from(flux=total_flux, magzero=magzero, xp=xp)
    return flux_mujy_via_ab_mag_from(ab_mag=ab_mag, xp=xp)


LATENT_FUNCTIONS: Dict[str, Callable] = {
    "total_galaxy_0_flux": total_galaxy_0_flux,
    "total_galaxy_0_flux_mujy": total_galaxy_0_flux_mujy,
}


def latent_keys_enabled(yaml_config: Optional[Dict[str, bool]] = None) -> List[str]:
    """
    Return the ordered list of enabled latent keys.

    Reads ``conf.instance["latent"]`` (a flat ``key: bool`` dict from
    ``autogalaxy/config/latent.yaml``) unless ``yaml_config`` is passed
    explicitly — tests pass a literal dict to avoid pushing a temporary
    config directory.

    Unknown keys (present in the yaml but not in :data:`LATENT_FUNCTIONS`)
    are dropped with a logger warning rather than raising — this lets the
    yaml carry forward-compat entries for latents that ship in later
    releases.
    """
    if yaml_config is None:
        yaml_config = dict(conf.instance["latent"])

    enabled: List[str] = []
    for key, on in yaml_config.items():
        if not on:
            continue
        if key not in LATENT_FUNCTIONS:
            logger.warning(
                "latent.yaml lists '%s' but no such latent is registered; "
                "dropping. Known latents: %s",
                key,
                sorted(LATENT_FUNCTIONS),
            )
            continue
        enabled.append(key)
    return enabled
