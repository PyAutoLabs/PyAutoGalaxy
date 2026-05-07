
import autogalaxy.plot as aplt
import pytest
from pathlib import Path

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_profile_plotter_setup():
    return Path(Path(__file__).resolve().parent) / "files" / "plots" / "profiles"


def test__figures_2d__all_are_output(
    lp_0,
    grid_2d_7x7,
    plot_path,
    plot_patch,
):
    aplt.plot_array(
        array=lp_0.image_2d_from(grid=grid_2d_7x7),
        title="Image",
        output_path=plot_path,
        output_filename="image_2d",
        output_format="png",
    )

    assert str(Path(plot_path) / "image_2d.png") in plot_patch.paths
