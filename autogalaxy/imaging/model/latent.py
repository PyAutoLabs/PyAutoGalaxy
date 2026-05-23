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


def total_galaxy_0_flux_mujy(fit, magzero, xp=np):
    """
    Total integrated flux of the first galaxy in the fit, converted to
    microjanskies via the image's magnitude zero-point.

    Returns NaN when the first galaxy has no light profile (which raises
    AttributeError inside ``fit.galaxy_image_dict``). Raises if ``magzero``
    is missing — there is no sensible default for a photometric calibration.
    """
    if magzero is None:
        raise ValueError(
            "magzero must be passed to AnalysisImaging via kwargs to compute "
            "the 'total_galaxy_0_flux_mujy' latent. Disable it in "
            "config/latent.yaml or pass magzero=<value> when constructing "
            "the Analysis."
        )

    try:
        image = fit.galaxy_image_dict[fit.galaxies[0]]
    except (AttributeError, KeyError, IndexError):
        return xp.nan

    total_flux = xp.sum(image.array)
    ab_mag = ab_mag_via_flux_from(flux=total_flux, magzero=magzero, xp=xp)
    return flux_mujy_via_ab_mag_from(ab_mag=ab_mag, xp=xp)


LATENT_FUNCTIONS: Dict[str, Callable] = {
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
