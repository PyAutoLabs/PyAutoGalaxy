
import pytest

import autogalaxy.plot as aplt
from pathlib import Path

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_fit_imaging_plotter_setup():
    return Path(Path(__file__).resolve().parent) / "files" / "plots" / "fit"


def test__subplot_of_galaxy(fit_imaging_x2_galaxy_7x7, plot_path, plot_patch):
    aplt.subplot_fit_imaging_of_galaxy(
        fit=fit_imaging_x2_galaxy_7x7,
        galaxy_index=0,
        output_path=plot_path,
        output_format="png",
    )
    aplt.subplot_fit_imaging_of_galaxy(
        fit=fit_imaging_x2_galaxy_7x7,
        galaxy_index=1,
        output_path=plot_path,
        output_format="png",
    )
    assert str(Path(plot_path) / "of_galaxy_0.png") in plot_patch.paths
    assert str(Path(plot_path) / "of_galaxy_1.png") in plot_patch.paths

    plot_patch.paths = []

    aplt.subplot_fit_imaging_of_galaxy(
        fit=fit_imaging_x2_galaxy_7x7,
        galaxy_index=0,
        output_path=plot_path,
        output_format="png",
    )

    assert str(Path(plot_path) / "of_galaxy_0.png") in plot_patch.paths
    assert str(Path(plot_path) / "of_galaxy_1.png") not in plot_patch.paths
