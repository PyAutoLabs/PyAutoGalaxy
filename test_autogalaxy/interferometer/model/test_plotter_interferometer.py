import pytest
from pathlib import Path

import autogalaxy as ag

from autogalaxy.interferometer.model.plotter import (
    PlotterInterferometer,
)

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_plotter_plotter_setup():
    return directory / "files"


def test__interferometer(interferometer_7, plot_path, plot_patch):
    plotter = PlotterInterferometer(image_path=plot_path)

    plotter.interferometer(dataset=interferometer_7)

    assert str(Path(plot_path) / "dataset.png") in plot_patch.paths

    image = ag.ndarray_via_fits_from(
        file_path=Path(plot_path) / "dataset.fits", hdu=1
    )

    assert image.shape == (7, 2)


def test__fit_interferometer(
    interferometer_7,
    fit_interferometer_x2_galaxy_inversion_7x7,
    plot_path,
    plot_patch,
):
    plotter = PlotterInterferometer(image_path=plot_path)

    plotter.fit_interferometer(
        fit=fit_interferometer_x2_galaxy_inversion_7x7,
    )

    assert str(Path(plot_path) / "fit.png") in plot_patch.paths

    image = ag.ndarray_via_fits_from(
        file_path=Path(plot_path) / "galaxy_images.fits", hdu=1
    )

    assert image.shape == (5, 5)

    image = ag.ndarray_via_fits_from(
        file_path=Path(plot_path) / "fit_dirty_images.fits", hdu=1
    )

    assert image.shape == (5, 5)
