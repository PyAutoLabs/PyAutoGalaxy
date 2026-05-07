import numpy as np
import os
import pytest
import shutil

import autogalaxy as ag
from pathlib import Path


def test__via_signal_to_noise_map__array_2d_data__correct_noise_map(
    dataset_quantity_7x7_array_2d, mask_2d_7x7
):
    data = ag.Array2D.no_mask(values=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0)
    signal_to_noise_map = ag.Array2D.no_mask(
        values=[[1.0, 5.0], [15.0, 40.0]], pixel_scales=1.0
    )

    dataset_quantity = ag.DatasetQuantity.via_signal_to_noise_map(
        data=data, signal_to_noise_map=signal_to_noise_map
    )

    assert dataset_quantity.signal_to_noise_map == pytest.approx(
        signal_to_noise_map, 1.0e-4
    )
    assert dataset_quantity.noise_map.native == pytest.approx(
        np.array([[1.0, 0.4], [0.2, 0.1]]), 1.0e-4
    )


def test__via_signal_to_noise_map__vector_yx_2d_data__correct_noise_map(
    dataset_quantity_7x7_array_2d, mask_2d_7x7
):
    data = ag.VectorYX2D.no_mask(
        values=[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]], pixel_scales=1.0
    )
    signal_to_noise_map = ag.Array2D.no_mask(
        values=[[1.0, 5.0], [15.0, 40.0]], pixel_scales=1.0
    )

    dataset_quantity = ag.DatasetQuantity.via_signal_to_noise_map(
        data=data, signal_to_noise_map=signal_to_noise_map
    )

    assert dataset_quantity.signal_to_noise_map == pytest.approx(
        np.array([[1.0, 1.0], [5.0, 5.0], [15.0, 15.0], [40.0, 40.0]]), 1.0e-4
    )
    assert dataset_quantity.noise_map.native == pytest.approx(
        np.array([[[1.0, 1.0], [0.4, 0.4]], [[0.2, 0.2], [0.1, 0.1]]]), 1.0e-4
    )


def test__apply_mask__array_2d_dataset__slim_and_native_values_correct(
    dataset_quantity_7x7_array_2d, mask_2d_7x7
):
    dataset_quantity_7x7 = dataset_quantity_7x7_array_2d.apply_mask(mask=mask_2d_7x7)

    assert (dataset_quantity_7x7.data.slim == np.ones(9)).all()
    assert (
        dataset_quantity_7x7.data.native == np.ones((7, 7)) * np.invert(mask_2d_7x7)
    ).all()

    assert (dataset_quantity_7x7.noise_map.slim == 2.0 * np.ones(9)).all()
    assert (
        dataset_quantity_7x7.noise_map.native
        == 2.0 * np.ones((7, 7)) * np.invert(mask_2d_7x7)
    ).all()


def test__apply_mask__vector_yx_2d_dataset__slim_values_correct(
    dataset_quantity_7x7_vector_yx_2d, mask_2d_7x7
):
    dataset_quantity_7x7 = dataset_quantity_7x7_vector_yx_2d.apply_mask(
        mask=mask_2d_7x7
    )

    assert (dataset_quantity_7x7.data.slim == np.ones((9, 2))).all()
    assert (dataset_quantity_7x7.noise_map.slim == 2.0 * np.ones((9, 2))).all()


def test__grid__default_over_sample__matches_fixture_grid(
    dataset_quantity_7x7_array_2d,
    mask_2d_7x7,
    grid_2d_7x7,
    blurring_grid_2d_7x7,
):
    dataset = dataset_quantity_7x7_array_2d.apply_mask(mask=mask_2d_7x7)

    assert isinstance(dataset.grids.lp, ag.Grid2D)
    assert (dataset.grids.lp == grid_2d_7x7).all()


def test__grid__custom_over_sample_size_4__matches_fixture_grid(
    mask_2d_7x7,
    grid_2d_7x7,
):
    dataset_quantity = ag.DatasetQuantity(
        data=ag.Array2D.ones(shape_native=(7, 7), pixel_scales=1.0),
        noise_map=ag.Array2D.full(
            fill_value=2.0, shape_native=(7, 7), pixel_scales=1.0
        ),
        over_sample_size_lp=4,
    )

    dataset = dataset_quantity.apply_mask(mask=mask_2d_7x7)

    assert (dataset.grids.lp == grid_2d_7x7).all()


def test__vector_data__y_x():
    data = ag.VectorYX2D.no_mask(
        values=[[[1.0, 5.0], [2.0, 6.0]], [[3.0, 7.0], [4.0, 8.0]]],
        pixel_scales=1.0,
    )

    noise_map = ag.VectorYX2D.no_mask(
        values=[[[1.1, 5.1], [2.1, 6.1]], [[3.1, 7.1], [4.1, 8.1]]],
        pixel_scales=1.0,
    )

    dataset_quantity = ag.DatasetQuantity(data=data, noise_map=noise_map)

    assert isinstance(dataset_quantity.y, ag.DatasetQuantity)
    assert (dataset_quantity.y.data.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert dataset_quantity.y.noise_map.slim == pytest.approx(
        np.array([1.1, 2.1, 3.1, 4.1]), 1.0e-4
    )

    assert isinstance(dataset_quantity.y, ag.DatasetQuantity)
    assert (dataset_quantity.x.data.slim == np.array([5.0, 6.0, 7.0, 8.0])).all()
    assert dataset_quantity.x.noise_map.slim == pytest.approx(
        np.array([5.1, 6.1, 7.1, 8.1]), 1.0e-4
    )


@pytest.fixture(name="test_data_path")
def make_test_data_path():
    test_data_path = Path(__file__).resolve().parent / "files" / "array" / "output_test"

    if test_data_path.exists():
        shutil.rmtree(test_data_path)

    os.makedirs(test_data_path)

    return test_data_path


def test__output_to_fits__array_2d_data__data_and_noise_map_written_correctly(
    dataset_quantity_7x7_array_2d, test_data_path
):
    from autoconf.fitsable import output_to_fits

    output_to_fits(
        values=dataset_quantity_7x7_array_2d.data.native.array.astype("float"),
        file_path=Path(test_data_path) / "data.fits",
        overwrite=True,
        header_dict=dataset_quantity_7x7_array_2d.data.mask.header_dict,
    )
    output_to_fits(
        values=dataset_quantity_7x7_array_2d.noise_map.native.array.astype("float"),
        file_path=Path(test_data_path) / "noise_map.fits",
        overwrite=True,
        header_dict=dataset_quantity_7x7_array_2d.noise_map.mask.header_dict,
    )

    data = ag.Array2D.from_fits(
        file_path=Path(test_data_path) / "data.fits", hdu=0, pixel_scales=1.0
    )
    noise_map = ag.Array2D.from_fits(
        file_path=Path(test_data_path) / "noise_map.fits", hdu=0, pixel_scales=1.0
    )

    assert (data.native == np.ones((7, 7))).all()
    assert (noise_map.native == 2.0 * np.ones((7, 7))).all()


def test__output_to_fits__vector_yx_2d_data__first_pixel_written_correctly(
    test_data_path,
):
    data = ag.VectorYX2D.no_mask(
        values=[[[1.0, 5.0], [2.0, 6.0]], [[3.0, 7.0], [4.0, 8.0]]],
        pixel_scales=1.0,
    )

    noise_map = ag.VectorYX2D.no_mask(
        values=[[[1.1, 5.1], [2.1, 6.1]], [[3.1, 7.1], [4.1, 8.1]]],
        pixel_scales=1.0,
    )

    dataset_quantity = ag.DatasetQuantity(data=data, noise_map=noise_map)

    from autoconf.fitsable import output_to_fits

    output_to_fits(
        values=dataset_quantity.data.native.array.astype("float"),
        file_path=Path(test_data_path) / "data.fits",
        overwrite=True,
        header_dict=dataset_quantity.data.mask.header_dict,
    )
    output_to_fits(
        values=dataset_quantity.noise_map.native.array.astype("float"),
        file_path=Path(test_data_path) / "noise_map.fits",
        overwrite=True,
        header_dict=dataset_quantity.noise_map.mask.header_dict,
    )

    data = ag.Array2D.from_fits(
        file_path=Path(test_data_path) / "data.fits", hdu=0, pixel_scales=1.0
    )

    assert data[0, 0] == pytest.approx([1.0, 5.0], 1.0e-4)
