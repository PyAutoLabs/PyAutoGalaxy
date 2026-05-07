from pathlib import Path

import pytest

import autogalaxy.plot as aplt

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_galaxy_fit_plotter_setup():
    return Path(__file__).resolve().parent / "files" / "plots" / "galaxy_fitting"


def test__fit_sub_plot__all_types_of_fit(
    fit_quantity_7x7_array_2d,
    fit_quantity_7x7_vector_yx_2d,
    plot_patch,
    plot_path,
):
    aplt.subplot_fit_quantity(
        fit=fit_quantity_7x7_array_2d,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "fit.png") in plot_patch.paths
