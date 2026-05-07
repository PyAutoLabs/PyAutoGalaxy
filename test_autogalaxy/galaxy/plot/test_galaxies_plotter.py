from pathlib import Path

import autogalaxy.plot as aplt
import pytest

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_plotter_setup():
    return Path(__file__).resolve().parent / "files" / "plots" / "galaxies"


def test__galaxies_sub_plot_output(galaxies_x2_7x7, grid_2d_7x7, plot_path, plot_patch):
    aplt.subplot_galaxies(
        galaxies=galaxies_x2_7x7,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "galaxies.png") in plot_patch.paths

    aplt.subplot_galaxy_images(
        galaxies=galaxies_x2_7x7,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "galaxy_images.png") in plot_patch.paths
