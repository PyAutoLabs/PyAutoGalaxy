from pathlib import Path

import autogalaxy.plot as aplt
import pytest

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_galaxy_plotter_setup():
    return Path(__file__).resolve().parent / "files" / "plots" / "galaxy"


def test__subplots_galaxy_quantities__all_are_output(
    gal_x1_lp_x1_mp,
    grid_2d_7x7,
    plot_path,
    plot_patch,
):
    aplt.subplot_galaxy_light_profiles(
        galaxy=gal_x1_lp_x1_mp,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )

    assert str(plot_path / "image.png") in plot_patch.paths

    aplt.subplot_galaxy_mass_profiles(
        galaxy=gal_x1_lp_x1_mp,
        grid=grid_2d_7x7,
        convergence=True,
        potential=True,
        deflections_y=True,
        deflections_x=True,
        output_path=plot_path,
        output_format="png",
    )

    assert str(plot_path / "convergence.png") in plot_patch.paths
    assert str(plot_path / "potential.png") in plot_patch.paths
    assert str(plot_path / "deflections_y.png") in plot_patch.paths
    assert str(plot_path / "deflections_x.png") in plot_patch.paths
