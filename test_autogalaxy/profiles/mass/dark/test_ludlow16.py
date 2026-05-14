"""
Numpy-path unit tests for the JAX-native Ludlow16 concentration that
replaces the colossus pure_callback formerly in ``mcr_util.py``.

JAX-path coverage lives in ``autolens_workspace_test/scripts/
jax_likelihood_functions/imaging/subhalo.py`` (Scenarios C and D),
which is the regression check for the production fit pipeline.
"""

import numpy as np
import pytest

from autogalaxy.profiles.mass.dark import mcr_util
from autogalaxy.profiles.mass.dark.ludlow16 import (
    PLANCK15_COSMOLOGY,
    ludlow16_concentration,
)


# Skip the colossus cross-check tests if colossus is not installed —
# it is an optional test/dev dependency (was a hard runtime dep before
# this PR; now lives under [test] / [dev] extras in pyproject.toml).
colossus = pytest.importorskip("colossus")
from colossus.cosmology import cosmology as col_cosmology  # noqa: E402
from colossus.halo.concentration import concentration as col_concentration  # noqa: E402


def _colossus_c200c(M_Msun_per_h, z):
    col_cosmology.setCosmology("planck15")
    return float(col_concentration(M_Msun_per_h, "200c", z, model="ludlow16"))


@pytest.mark.parametrize(
    "log_M_per_h, z",
    [
        (10.5, 0.3),
        (11.5, 0.5),
        (12.5, 1.0),
        (13.5, 1.5),
        (12.0, 0.1),
        (12.0, 2.5),
    ],
)
def test_ludlow16_concentration_matches_colossus(log_M_per_h, z):
    """JAX-native (numpy backend) c200c agrees with colossus to ~1e-3."""
    M = 10.0 ** log_M_per_h
    c_col = _colossus_c200c(M, z)
    c_new = float(ludlow16_concentration(M, z, **PLANCK15_COSMOLOGY))
    assert c_new == pytest.approx(c_col, rel=2.0e-3)


@pytest.mark.parametrize(
    "mass_at_200, redshift_object, redshift_source",
    [
        (1.0e12, 0.3, 1.5),
        (1.0e11, 0.5, 2.0),
        (1.0e13, 0.5, 1.0),
    ],
)
def test_kappa_s_and_scale_radius_for_ludlow_matches_colossus_chain(
    mass_at_200, redshift_object, redshift_source
):
    """End-to-end NFW parameters via the rewritten helper match the
    historical colossus-callback values to better than 0.2%."""
    # Reference: run colossus directly and follow the same numpy algebra
    # the helper does, to get a baseline kappa_s / scale_radius that
    # matches what the pre-Phase-2 code returned.
    h = PLANCK15_COSMOLOGY["h"]
    c_col = _colossus_c200c(mass_at_200 * h, redshift_object)

    from autogalaxy.cosmology.model import Planck15

    cosmology = Planck15()
    rho = float(cosmology.critical_density(redshift_object, xp=np))
    sigma_c = float(
        cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
            redshift_0=redshift_object, redshift_1=redshift_source, xp=np,
        )
    )
    kpc_per_arcsec = float(
        cosmology.kpc_per_arcsec_from(redshift=redshift_object, xp=np)
    )

    radius_at_200 = (mass_at_200 / (200.0 * rho * (4.0 * np.pi / 3.0))) ** (1.0 / 3.0)
    de_c = (
        200.0
        / 3.0
        * (c_col ** 3 / (np.log(1.0 + c_col) - c_col / (1.0 + c_col)))
    )
    scale_radius_kpc = radius_at_200 / c_col
    kappa_s_ref = rho * de_c * scale_radius_kpc / sigma_c
    scale_radius_ref = scale_radius_kpc / kpc_per_arcsec
    radius_at_200_ref = radius_at_200

    # Now call the rewritten helper (which goes through the JAX-native
    # numpy path instead of the colossus callback).
    kappa_s, scale_radius, radius_at_200 = mcr_util.kappa_s_and_scale_radius_for_ludlow(
        mass_at_200=mass_at_200,
        scatter_sigma=0.0,
        redshift_object=redshift_object,
        redshift_source=redshift_source,
    )

    assert float(kappa_s) == pytest.approx(kappa_s_ref, rel=2.0e-3)
    assert float(scale_radius) == pytest.approx(scale_radius_ref, rel=2.0e-3)
    assert float(radius_at_200) == pytest.approx(radius_at_200_ref, rel=1.0e-12)


def test_ludlow16_concentration_array_input():
    """Function accepts plain numpy scalar inputs and returns a finite scalar."""
    out = ludlow16_concentration(1.0e12, 0.5, **PLANCK15_COSMOLOGY)
    assert np.isfinite(float(out))
    assert float(out) > 1.0
    assert float(out) < 100.0
