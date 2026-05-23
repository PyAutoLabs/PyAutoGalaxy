import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import autofit as af
import autogalaxy as ag
from autogalaxy.imaging.model.latent import (
    LATENT_FUNCTIONS,
    ab_mag_via_flux_from,
    flux_mujy_via_ab_mag_from,
    latent_keys_enabled,
    total_galaxy_0_flux_mujy,
)


def test_ab_mag_round_trip_to_microjansky():
    flux = 100.0
    magzero = 25.0
    ab_mag = ab_mag_via_flux_from(flux=flux, magzero=magzero)
    muJy = flux_mujy_via_ab_mag_from(ab_mag=ab_mag)

    expected_ab_mag = -2.5 * np.log10(flux) + magzero
    expected_muJy = 10 ** ((23.9 - expected_ab_mag) / 2.5)

    assert ab_mag == pytest.approx(expected_ab_mag)
    assert muJy == pytest.approx(expected_muJy)


def test_total_galaxy_0_flux_mujy_against_known_image():
    image = SimpleNamespace(array=np.array([1.0, 2.0, 3.0, 4.0]))
    galaxy = object()
    fit = SimpleNamespace(
        galaxies=[galaxy],
        galaxy_image_dict={galaxy: image},
    )

    result = total_galaxy_0_flux_mujy(fit=fit, magzero=25.0)

    expected = flux_mujy_via_ab_mag_from(
        ab_mag=ab_mag_via_flux_from(flux=10.0, magzero=25.0)
    )
    assert result == pytest.approx(expected)


def test_total_galaxy_0_flux_mujy_returns_nan_when_no_light_profile():
    fit = MagicMock()
    fit.galaxy_image_dict.__getitem__.side_effect = KeyError("no light")
    fit.galaxies = [object()]

    result = total_galaxy_0_flux_mujy(fit=fit, magzero=25.0)

    assert np.isnan(result)


def test_total_galaxy_0_flux_mujy_missing_magzero_raises():
    fit = MagicMock()

    with pytest.raises(ValueError, match="magzero"):
        total_galaxy_0_flux_mujy(fit=fit, magzero=None)


def test_latent_keys_enabled_filters_disabled():
    enabled = latent_keys_enabled(yaml_config={"total_galaxy_0_flux_mujy": False})
    assert enabled == []


def test_latent_keys_enabled_preserves_yaml_order():
    # Insert an unknown second key so we can assert ordering across two keys
    # without depending on a future second registered latent.
    yaml_config = {
        "total_galaxy_0_flux_mujy": True,
        "future_latent_zzz": True,
    }
    enabled = latent_keys_enabled(yaml_config=yaml_config)

    # Unknown keys drop; the known key stays in its yaml-insertion position.
    assert enabled == ["total_galaxy_0_flux_mujy"]


def test_latent_keys_enabled_drops_unknown_with_warning(caplog):
    caplog.set_level(logging.WARNING)
    enabled = latent_keys_enabled(
        yaml_config={"never_registered_latent": True, "total_galaxy_0_flux_mujy": True}
    )

    assert enabled == ["total_galaxy_0_flux_mujy"]
    assert any("never_registered_latent" in rec.message for rec in caplog.records)


def test_analysis_imaging_compute_latent_variables_aligns_with_latent_keys(
    masked_imaging_7x7,
):
    galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=0.1))
    model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

    analysis = ag.AnalysisImaging(
        dataset=masked_imaging_7x7, use_jax=False, magzero=25.0
    )

    parameters = np.array(model.physical_values_from_prior_medians)
    values = analysis.compute_latent_variables(parameters=parameters, model=model)

    assert isinstance(values, tuple)
    assert len(values) == len(analysis.LATENT_KEYS)
    assert analysis.LATENT_KEYS == ["total_galaxy_0_flux_mujy"]
    assert np.isfinite(values[0])


def test_analysis_imaging_compute_latent_variables_raises_when_empty(monkeypatch):
    # When no latents are enabled, autofit's `except NotImplementedError`
    # at autofit/non_linear/analysis/analysis.py:304 short-circuits the
    # latent pipeline. We match that contract by raising explicitly.
    monkeypatch.setattr(
        ag.AnalysisImaging,
        "LATENT_KEYS",
        property(lambda self: []),
    )
    analysis = ag.AnalysisImaging(dataset=MagicMock(), use_jax=False)

    with pytest.raises(NotImplementedError):
        analysis.compute_latent_variables(parameters=np.array([]), model=MagicMock())


def test_analysis_imaging_latent_keys_property_reads_config():
    # The autouse fixture in test_autogalaxy/conftest.py pushes the test
    # config dir whose latent.yaml enables total_galaxy_0_flux_mujy.
    dataset = MagicMock()
    analysis = ag.AnalysisImaging(dataset=dataset, use_jax=False)

    assert analysis.LATENT_KEYS == ["total_galaxy_0_flux_mujy"]
    assert set(analysis.LATENT_KEYS).issubset(LATENT_FUNCTIONS.keys())
