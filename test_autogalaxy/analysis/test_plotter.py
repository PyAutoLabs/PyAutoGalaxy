import csv
import shutil
import numpy as np
from pathlib import Path
import pytest

import autogalaxy as ag

from autogalaxy.analysis.plotter import Plotter

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_plotter_plotter_setup():
    return directory / "files"


def test__galaxies(masked_imaging_7x7, galaxies_7x7, plot_path, plot_patch):
    if plot_path.exists():
        shutil.rmtree(plot_path)

    plotter = Plotter(image_path=plot_path)

    plotter.galaxies(
        galaxies=galaxies_7x7,
        grid=masked_imaging_7x7.grids.lp,
    )

    assert str(plot_path / "galaxies.png") in plot_patch.paths
    assert str(plot_path / "galaxy_images.png") in plot_patch.paths

    image = ag.ndarray_via_fits_from(
        file_path=plot_path / "galaxy_images.fits", hdu=1
    )

    assert image.shape == (5, 5)


def test__inversion(
    rectangular_inversion_7x7_3x3,
    plot_path,
    plot_patch,
):
    if plot_path.exists():
        shutil.rmtree(plot_path)

    plotter = Plotter(image_path=plot_path)

    plotter.inversion(
        inversion=rectangular_inversion_7x7_3x3,
    )

    assert str(plot_path / "inversion_0_0.png") in plot_patch.paths

    with open(
        plot_path / "source_plane_reconstruction_0.csv", mode="r"
    ) as file:
        reader = csv.reader(file)
        header_list = next(reader)  # ['y', 'x', 'reconstruction', 'noise_map']

        reconstruction_dict = {header: [] for header in header_list}

        for row in reader:
            for key, value in zip(header_list, row):
                print(value)
                reconstruction_dict[key].append(float(value))

        # Convert lists to NumPy arrays
        for key in reconstruction_dict:
            reconstruction_dict[key] = np.array(reconstruction_dict[key])

    assert reconstruction_dict["x"][0] == pytest.approx(-0.8333333333333334, rel=1.0e-2)


def test__adapt_images(
    masked_imaging_7x7,
    adapt_galaxy_name_image_dict_7x7,
    fit_imaging_x2_galaxy_inversion_7x7,
    plot_path,
    plot_patch,
):
    plotter = Plotter(image_path=plot_path)

    adapt_images = ag.AdaptImages(
        galaxy_name_image_dict=adapt_galaxy_name_image_dict_7x7,
    )

    plotter.adapt_images(
        adapt_images=adapt_images,
    )

    assert str(plot_path / "adapt_images.png") in plot_patch.paths

    image = ag.ndarray_via_fits_from(
        file_path=plot_path / "adapt_images.fits", hdu=1
    )

    assert image.shape == (5, 5)
