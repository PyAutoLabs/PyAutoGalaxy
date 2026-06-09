import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import autofit as af
import autogalaxy as ag
from autogalaxy.imaging.model import latent as _latent_module
from autogalaxy.imaging.model.latent import (
    LATENT_FUNCTIONS,
    LatentGalaxy,
    ab_mag_via_flux_from,
    flux_mujy_via_ab_mag_from,
    latent_keys_enabled,
    total_galaxy_0_flux,
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


def test_total_galaxy_0_flux_mujy_missing_magzero_returns_nan_and_warns(caplog):
    _latent_module._MAGZERO_WARNED.discard("total_galaxy_0_flux_mujy")
    caplog.set_level(logging.WARNING)

    image = SimpleNamespace(array=np.array([1.0, 2.0, 3.0]))
    galaxy = object()
    fit = SimpleNamespace(galaxies=[galaxy], galaxy_image_dict={galaxy: image})

    result = total_galaxy_0_flux_mujy(fit=fit, magzero=None)

    assert np.isnan(result)
    assert any(
        "magzero" in rec.message and "total_galaxy_0_flux_mujy" in rec.message
        for rec in caplog.records
    )


def test_total_galaxy_0_flux_against_known_image():
    image = SimpleNamespace(array=np.array([1.0, 2.0, 3.0, 4.0]))
    galaxy = object()
    fit = SimpleNamespace(galaxies=[galaxy], galaxy_image_dict={galaxy: image})

    # Raw sum — no AB-mag conversion. `magzero` is accepted but ignored.
    assert total_galaxy_0_flux(fit=fit, magzero=None) == pytest.approx(10.0)
    assert total_galaxy_0_flux(fit=fit, magzero=25.0) == pytest.approx(10.0)


def test_total_galaxy_0_flux_returns_nan_when_no_light_profile():
    fit = MagicMock()
    fit.galaxy_image_dict.__getitem__.side_effect = KeyError("no light")
    fit.galaxies = [object()]

    assert np.isnan(total_galaxy_0_flux(fit=fit))


def test_maybe_magzero_warn_logs_only_once_per_name(caplog):
    _latent_module._MAGZERO_WARNED.discard("total_galaxy_0_flux_mujy")
    caplog.set_level(logging.WARNING)

    image = SimpleNamespace(array=np.array([1.0]))
    galaxy = object()
    fit = SimpleNamespace(galaxies=[galaxy], galaxy_image_dict={galaxy: image})

    for _ in range(3):
        total_galaxy_0_flux_mujy(fit=fit, magzero=None)

    matching = [
        r for r in caplog.records if "total_galaxy_0_flux_mujy" in r.message
    ]
    assert len(matching) == 1


def test_latent_keys_enabled_filters_disabled():
    enabled = latent_keys_enabled(
        yaml_config={
            "total_galaxy_0_flux": False,
            "total_galaxy_0_flux_mujy": False,
        }
    )
    assert enabled == []


def test_latent_keys_enabled_preserves_yaml_order():
    yaml_config = {
        "total_galaxy_0_flux_mujy": True,
        "total_galaxy_0_flux": True,
    }
    enabled = latent_keys_enabled(yaml_config=yaml_config)

    assert enabled == ["total_galaxy_0_flux_mujy", "total_galaxy_0_flux"]


def test_latent_keys_enabled_drops_unknown_with_warning(caplog):
    caplog.set_level(logging.WARNING)
    enabled = latent_keys_enabled(
        yaml_config={"never_registered_latent": True, "total_galaxy_0_flux_mujy": True}
    )

    assert enabled == ["total_galaxy_0_flux_mujy"]
    assert any("never_registered_latent" in rec.message for rec in caplog.records)


def test_latent_galaxy_variables_aligns_with_keys(
    masked_imaging_7x7,
):
    galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=0.1))
    model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

    analysis = ag.AnalysisImaging(
        dataset=masked_imaging_7x7, use_jax=False, magzero=25.0
    )

    parameters = np.array(model.physical_values_from_prior_medians)
    values = LatentGalaxy.variables(analysis, parameters=parameters, model=model)
    keys = LatentGalaxy.keys(analysis)

    assert isinstance(values, tuple)
    assert len(values) == len(keys)
    # test_autogalaxy/config/latent.yaml enables both keys, raw flux first.
    assert keys == ["total_galaxy_0_flux", "total_galaxy_0_flux_mujy"]
    assert all(np.isfinite(v) for v in values)


def test_latent_galaxy_variables_raises_when_empty(monkeypatch):
    # When no latents are enabled, autofit's `except NotImplementedError`
    # short-circuits the latent pipeline. LatentGalaxy.variables matches that
    # contract by raising explicitly.
    monkeypatch.setattr(_latent_module, "latent_keys_enabled", lambda *a, **k: [])
    analysis = ag.AnalysisImaging(dataset=MagicMock(), use_jax=False)

    with pytest.raises(NotImplementedError):
        LatentGalaxy.variables(analysis, parameters=np.array([]), model=MagicMock())


def test_analysis_imaging_declares_latent_galaxy_and_keys_read_config():
    # The autouse fixture in test_autogalaxy/conftest.py pushes the test
    # config dir whose latent.yaml enables both keys.
    dataset = MagicMock()
    analysis = ag.AnalysisImaging(dataset=dataset, use_jax=False)

    assert ag.AnalysisImaging.Latent is LatentGalaxy
    keys = ag.AnalysisImaging.Latent.keys(analysis)
    assert keys == ["total_galaxy_0_flux", "total_galaxy_0_flux_mujy"]
    assert set(keys).issubset(LATENT_FUNCTIONS.keys())
