from pathlib import Path

import autogalaxy.plot as aplt
import pytest


@pytest.fixture(name="plot_path")
def make_adapt_plotter_setup():
    return Path(__file__).resolve().parent / "files" / "plots" / "adapt"


def test__plot_adapt_adapt_images(
    adapt_galaxy_name_image_dict_7x7, mask_2d_7x7, plot_path, plot_patch
):
    aplt.subplot_adapt_images(
        adapt_galaxy_name_image_dict=adapt_galaxy_name_image_dict_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "adapt_images.png") in plot_patch.paths
