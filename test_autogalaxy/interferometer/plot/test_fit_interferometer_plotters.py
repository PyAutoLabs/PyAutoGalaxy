from pathlib import Path

import autogalaxy.plot as aplt
import pytest


@pytest.fixture(name="plot_path")
def make_fit_dataset_plotter_setup():
    return Path(__file__).resolve().parent / "files" / "plots" / "fit"


def test__fit_sub_plot_real_space(
    fit_interferometer_7x7,
    fit_interferometer_x2_galaxy_inversion_7x7,
    plot_path,
    plot_patch,
):
    aplt.subplot_fit_real_space(
        fit=fit_interferometer_7x7,
        output_path=plot_path,
        output_format="png",
    )

    assert str(plot_path / "fit_real_space.png") in plot_patch.paths

    plot_patch.paths = []

    aplt.subplot_fit_real_space(
        fit=fit_interferometer_x2_galaxy_inversion_7x7,
        output_path=plot_path,
        output_format="png",
    )

    assert str(plot_path / "fit_real_space.png") in plot_patch.paths
