import numpy as np
import pytest

import autoarray as aa
import autogalaxy as ag
from autogalaxy.profiles.mass.input.gaussian_random_field import (
    gaussian_random_field_from,
)


def cleaned_circular_mask(shape=(16, 16), pixel_scale=0.5, radius=3.4):
    mask = aa.Mask2D.circular(
        shape_native=shape, pixel_scales=pixel_scale, radius=radius
    )
    cleaned, _ = aa.util.derivative.cleaned_mask_from(np.asarray(mask))
    return aa.Mask2D(mask=cleaned, pixel_scales=pixel_scale)


def test__realization_is_reproducible_and_zero_mean():
    field_a = gaussian_random_field_from(
        shape_native=(32, 32),
        pixel_scale=0.5,
        power_amplitude=1.0,
        power_slope=2.0,
        seed=3,
    )
    field_b = gaussian_random_field_from(
        shape_native=(32, 32),
        pixel_scale=0.5,
        power_amplitude=1.0,
        power_slope=2.0,
        seed=3,
    )
    field_c = gaussian_random_field_from(
        shape_native=(32, 32),
        pixel_scale=0.5,
        power_amplitude=1.0,
        power_slope=2.0,
        seed=4,
    )

    assert field_a == pytest.approx(field_b, abs=0.0)
    assert not np.allclose(field_a, field_c)
    assert field_a.mean() == pytest.approx(0.0, abs=1.0e-10)


def test__amplitude_scales_field():
    field_1 = gaussian_random_field_from(
        shape_native=(32, 32),
        pixel_scale=0.5,
        power_amplitude=1.0,
        power_slope=2.0,
        seed=3,
    )
    field_4 = gaussian_random_field_from(
        shape_native=(32, 32),
        pixel_scale=0.5,
        power_amplitude=4.0,
        power_slope=2.0,
        seed=3,
    )

    assert field_4 == pytest.approx(2.0 * field_1, abs=1.0e-10)


def test__profile_potential_matches_realization_on_unmasked_pixels():
    mask = cleaned_circular_mask()
    grid = aa.Grid2D.from_mask(mask=mask)

    profile = ag.mp.GaussianRandomField(
        mask=mask, power_amplitude=1.0, power_slope=1.0, seed=2
    )

    potential = np.asarray(profile.potential_2d_from(grid=grid))
    expected = profile.lensing_potential_native[~np.asarray(mask)]

    assert potential == pytest.approx(expected, abs=1.0e-10)

    deflections = np.asarray(profile.deflections_yx_2d_from(grid=grid))
    assert deflections.shape == (expected.shape[0], 2)
    assert np.isfinite(deflections).all()
