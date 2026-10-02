"""Physical field invariants for capped LensCalc evaluation grids."""

import numpy as np
import pytest

import autoarray as aa
from autonerves import conf
from autogalaxy.operate.lens_calc import evaluation_grid


@evaluation_grid
def capture_grid(_, grid, pixel_scale):
    return grid, pixel_scale


def set_cap(monkeypatch, cap):
    monkeypatch.setitem(conf.instance["general"]["grid"], "max_evaluation_grid_size", cap)


def test_cluster_field_is_preserved_at_the_cap(monkeypatch):
    set_cap(monkeypatch, 1000)
    grid = aa.Grid2D.uniform(
        shape_native=(120, 120), pixel_scales=0.5, respect_small_datasets=False
    )
    result, spacing = capture_grid(None, grid, pixel_scale=0.05)
    assert result.shape_native == (1000, 1000)
    assert spacing == pytest.approx(0.06)
    assert result.pixel_scales == pytest.approx((0.06, 0.06))
    assert result.geometry.shape_native_scaled == pytest.approx((60.0, 60.0))
    assert result.origin == (0.0, 0.0)


@pytest.mark.parametrize("shape", [(4, 5), (5, 4), (8, 11), (11, 8)])
def test_both_axes_bounded_without_cropping_effective_zoom(monkeypatch, shape):
    set_cap(monkeypatch, 20)
    grid = aa.Grid2D.uniform(
        shape_native=shape, pixel_scales=1.0, respect_small_datasets=False
    )
    zoom = aa.Zoom2D(mask=grid.mask)
    wanted = np.array(zoom.shape_native) * grid.pixel_scale
    result, spacing = capture_grid(None, grid, pixel_scale=0.2)
    actual = np.array(result.geometry.shape_native_scaled)
    assert max(result.shape_native) == 20
    assert np.all(actual >= wanted - 1e-12)
    assert np.all(actual - wanted < spacing + 1e-12)
    assert result.origin == zoom.offset_scaled
    assert result.pixel_scales == pytest.approx((spacing, spacing))
    assert max(actual) == pytest.approx(max(wanted))


def test_masked_shifted_zoom_keeps_effective_centre_when_capped(monkeypatch):
    mask_values = np.ones((9, 11), dtype=bool)
    mask_values[1:6, 5:9] = False
    mask = aa.Mask2D(mask=mask_values, pixel_scales=0.3, origin=(1.2, -0.7))
    grid = aa.Grid2D.from_mask(mask=mask)
    zoom = aa.Zoom2D(mask=mask)
    set_cap(monkeypatch, 100)
    uncapped, _ = capture_grid(None, grid, pixel_scale=0.03)
    set_cap(monkeypatch, 20)
    capped, spacing = capture_grid(None, grid, pixel_scale=0.03)
    assert capped.origin == uncapped.origin == zoom.offset_scaled
    assert capped.origin != (0.0, 0.0)
    actual = np.array(capped.geometry.shape_native_scaled)
    wanted = np.array(zoom.shape_native) * grid.pixel_scale
    assert max(capped.shape_native) == 20
    assert np.all(actual >= wanted - 1e-12)
    assert np.all(actual - wanted < spacing + 1e-12)


def test_below_cap_keeps_requested_spacing_and_existing_rounding(monkeypatch):
    set_cap(monkeypatch, 1000)
    grid = aa.Grid2D.uniform(
        shape_native=(5, 4), pixel_scales=0.3, respect_small_datasets=False
    )
    result, spacing = capture_grid(None, grid, pixel_scale=0.2)
    assert result.shape_native == (7, 5)
    assert spacing == 0.2
    assert result.pixel_scales == (0.2, 0.2)
    assert result.is_evaluation_grid


def test_existing_evaluation_grid_is_passed_through(monkeypatch):
    set_cap(monkeypatch, 2)
    grid = aa.Grid2D.uniform(
        shape_native=(5, 5), pixel_scales=0.3, respect_small_datasets=False
    )
    grid.is_evaluation_grid = True
    result, spacing = capture_grid(None, grid, pixel_scale=0.05)
    assert result is grid
    assert spacing == 0.05
