import pytest

import autogalaxy as ag


def test__mask_interp():
    data = ag.Array2D.no_mask(
        values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], pixel_scales=1.0
    )
    noise_map = ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    dataset = ag.Imaging(data=data, noise_map=noise_map)

    interp = ag.DatasetInterp(dataset=dataset)

    assert interp.mask_interp((0.5, 0.5)) == pytest.approx(0.0, 1.0e-4)

    mask = ag.Mask2D(
        mask=[[False, False, False], [False, True, False], [False, False, False]],
        pixel_scales=1.0,
    )

    data = data.apply_mask(mask=mask)

    dataset = ag.Imaging(data=data, noise_map=noise_map)

    interp = ag.DatasetInterp(dataset=dataset)

    assert interp.mask_interp((0.5, 0.5)) == pytest.approx(0.25, 1.0e-4)


def test__data_interp():
    data = ag.Array2D.no_mask(
        values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], pixel_scales=1.0
    )
    noise_map = ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    dataset = ag.Imaging(data=data, noise_map=noise_map)

    interp = ag.DatasetInterp(dataset=dataset)

    assert interp.data_interp((0.5, 0.5)) == pytest.approx(7.0, 1.0e-4)


def test__noise_map_interp():
    data = ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)
    noise_map = ag.Array2D.no_mask(
        values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], pixel_scales=1.0
    )

    dataset = ag.Imaging(data=data, noise_map=noise_map)

    interp = ag.DatasetInterp(dataset=dataset)

    assert interp.noise_map_interp((0.5, 0.5)) == pytest.approx(7.0, 1.0e-4)
