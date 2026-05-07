import shutil
import pytest

import autogalaxy as ag
from autogalaxy.imaging.model.plotter import PlotterImaging
from pathlib import Path

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_plotter_plotter_setup():
    return directory / "files"


def test__imaging(imaging_7x7, plot_path, plot_patch):
    if Path(plot_path).exists():
        shutil.rmtree(plot_path)

    plotter = PlotterImaging(image_path=plot_path)

    plotter.imaging(dataset=imaging_7x7)

    assert str(Path(plot_path) / "dataset.png") in plot_patch.paths

    image = ag.ndarray_via_fits_from(
        file_path=Path(plot_path) / "dataset.fits", hdu=1
    )

    assert image.shape == (7, 7)


def test__imaging_combined(imaging_7x7, plot_path, plot_patch):
    if Path(plot_path).exists():
        shutil.rmtree(plot_path)

    visualizer = PlotterImaging(image_path=plot_path)

    visualizer.imaging_combined(dataset_list=[imaging_7x7, imaging_7x7])

    assert str(Path(plot_path) / "dataset_combined.png") in plot_patch.paths


def test__fit_imaging(
    masked_imaging_7x7,
    fit_imaging_x2_galaxy_inversion_7x7,
    plot_path,
    plot_patch,
):
    if Path(plot_path).exists():
        shutil.rmtree(plot_path)

    plotter = PlotterImaging(image_path=plot_path)

    plotter.fit_imaging(
        fit=fit_imaging_x2_galaxy_inversion_7x7,
    )

    assert str(Path(plot_path) / "fit.png") in plot_patch.paths

    image = ag.ndarray_via_fits_from(file_path=Path(plot_path) / "fit.fits", hdu=1)

    assert image.shape == (5, 5)

    image = ag.ndarray_via_fits_from(
        file_path=Path(plot_path) / "model_galaxy_images.fits", hdu=1
    )

    assert image.shape == (5, 5)


def test__fit_imaging_combined(
    fit_imaging_x2_galaxy_inversion_7x7, plot_path, plot_patch
):
    if Path(plot_path).exists():
        shutil.rmtree(plot_path)

    visualizer = PlotterImaging(image_path=plot_path)

    visualizer.fit_imaging_combined(fit_list=2 * [fit_imaging_x2_galaxy_inversion_7x7])

    assert str(Path(plot_path) / "fit_combined.png") in plot_patch.paths
