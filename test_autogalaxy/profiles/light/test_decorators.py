import numpy as np
import pytest
from typing import Optional

import autoarray as aa
import autogalaxy as ag

from autogalaxy.profiles.light.decorators import (
    check_operated_only,
)


class MockLightProfile(ag.LightProfile):
    @check_operated_only
    def image_2d_from(
        self, grid: aa.type.Grid2DLike, xp=np, operated_only: Optional[bool] = None
    ):
        return np.ones(shape=(3, 3))


class MockLightProfileOperated(ag.lp_operated.LightProfileOperated):
    @check_operated_only
    def image_2d_from(
        self, grid: aa.type.Grid2DLike, xp=np, operated_only: Optional[bool] = None
    ):
        return np.ones(shape=(3, 3))


def test__image_2d_from__standard_profile__operated_only_none__returns_image():
    grid = ag.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)

    lp = ag.lp.Gaussian()

    lp_image_2d = lp.image_2d_from(grid=grid)
    image_2d = lp.image_2d_from(grid=grid)

    assert image_2d == pytest.approx(lp_image_2d.array, 1.0e-4)


def test__image_2d_from__standard_profile__operated_only_true__returns_zeros():
    grid = ag.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)

    lp = ag.lp.Gaussian()

    image_2d = lp.image_2d_from(grid=grid, operated_only=True)

    assert image_2d == pytest.approx(np.zeros(shape=(9,)), 1.0e-4)


def test__image_2d_from__standard_profile__operated_only_false__returns_image():
    grid = ag.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)

    lp = ag.lp.Gaussian()

    lp_image_2d = lp.image_2d_from(grid=grid)
    image_2d = lp.image_2d_from(grid=grid, operated_only=False)

    assert image_2d == pytest.approx(lp_image_2d.array, 1.0e-4)


def test__image_2d_from__operated_profile__operated_only_none__returns_image():
    grid = ag.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)

    lp_standard = ag.lp.Gaussian()
    lp_image_2d = lp_standard.image_2d_from(grid=grid)

    lp = ag.lp_operated.Gaussian()
    image_2d = lp.image_2d_from(grid=grid)

    assert image_2d == pytest.approx(lp_image_2d.array, 1.0e-4)


def test__image_2d_from__operated_profile__operated_only_true__returns_image():
    grid = ag.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)

    lp_standard = ag.lp.Gaussian()
    lp_image_2d = lp_standard.image_2d_from(grid=grid)

    lp = ag.lp_operated.Gaussian()
    image_2d = lp.image_2d_from(grid=grid, operated_only=True)

    assert image_2d == pytest.approx(lp_image_2d.array, 1.0e-4)


def test__image_2d_from__operated_profile__operated_only_false__returns_zeros():
    grid = ag.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)

    lp = ag.lp_operated.Gaussian()

    image_2d = lp.image_2d_from(grid=grid, operated_only=False)

    assert image_2d == pytest.approx(np.zeros(shape=(9,)), 1.0e-4)
