from pathlib import Path

import autoarray as aa
import autogalaxy.plot as aplt
import pytest
from autogalaxy.operate.lens_calc import LensCalc

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_mp_plotter_setup():
    return Path(__file__).resolve().parent / "files" / "plots" / "profiles"


def test__figures_2d__all_are_output(
    mp_0,
    grid_2d_7x7,
    plot_path,
    plot_patch,
):
    aplt.plot_array(
        array=mp_0.convergence_2d_from(grid=grid_2d_7x7),
        title="Convergence",
        output_path=plot_path,
        output_filename="convergence_2d",
        output_format="png",
    )
    aplt.plot_array(
        array=mp_0.potential_2d_from(grid=grid_2d_7x7),
        title="Potential",
        output_path=plot_path,
        output_filename="potential_2d",
        output_format="png",
    )
    deflections = mp_0.deflections_yx_2d_from(grid=grid_2d_7x7)
    aplt.plot_array(
        array=aa.Array2D(values=deflections.slim[:, 0], mask=grid_2d_7x7.mask),
        title="Deflections Y",
        output_path=plot_path,
        output_filename="deflections_y_2d",
        output_format="png",
    )
    aplt.plot_array(
        array=aa.Array2D(values=deflections.slim[:, 1], mask=grid_2d_7x7.mask),
        title="Deflections X",
        output_path=plot_path,
        output_filename="deflections_x_2d",
        output_format="png",
    )
    aplt.plot_array(
        array=LensCalc.from_mass_obj(mp_0).magnification_2d_from(grid=grid_2d_7x7),
        title="Magnification",
        output_path=plot_path,
        output_filename="magnification_2d",
        output_format="png",
    )

    assert str(plot_path / "convergence_2d.png") in plot_patch.paths
    assert str(plot_path / "potential_2d.png") in plot_patch.paths
    assert str(plot_path / "deflections_y_2d.png") in plot_patch.paths
    assert str(plot_path / "deflections_x_2d.png") in plot_patch.paths
    assert str(plot_path / "magnification_2d.png") in plot_patch.paths
