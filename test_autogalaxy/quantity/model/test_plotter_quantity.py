import shutil
import pytest

import autogalaxy as ag

from autogalaxy.quantity.model.plotter import PlotterQuantity
from pathlib import Path

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_plotter_plotter_setup():
    return directory / "files"


def test__dataset(
    dataset_quantity_7x7_array_2d,
    plot_path,
    plot_patch,
):
    if Path(plot_path).exists():
        shutil.rmtree(plot_path)

    plotter = PlotterQuantity(image_path=plot_path)

    plotter.dataset_quantity(dataset=dataset_quantity_7x7_array_2d)

    image = ag.ndarray_via_fits_from(
        file_path=Path(plot_path) / "dataset.fits", hdu=1
    )

    assert image.shape == (7, 7)


def test__fit_quantity(
    fit_quantity_7x7_array_2d,
    fit_quantity_7x7_vector_yx_2d,
    plot_path,
    plot_patch,
):
    if Path(plot_path).exists():
        shutil.rmtree(plot_path)

    plotter = PlotterQuantity(image_path=plot_path)

    plotter.fit_quantity(fit=fit_quantity_7x7_array_2d)

    assert str(Path(plot_path) / "fit.png") not in plot_patch.paths
