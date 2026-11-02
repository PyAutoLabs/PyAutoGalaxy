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


def test__plot_magnification_2d__output(galaxies_x2_7x7, grid_2d_7x7, plot_path, plot_patch):
    aplt.plot_magnification_2d(
        mass_obj=galaxies_x2_7x7,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "magnification_2d.png") in plot_patch.paths


def _contour_levels_on(ax):
    return [
        list(collection.levels)
        for collection in ax.collections
        if hasattr(collection, "levels")
    ]


def test__plot_magnification_2d__contours_at_constant_mu_levels_in_range():
    import matplotlib.pyplot as plt
    import numpy as np

    import autogalaxy as ag

    grid = ag.Grid2D.uniform(shape_native=(40, 40), pixel_scales=0.1)
    galaxy = ag.Galaxy(
        redshift=0.5, mass=ag.mp.Isothermal(centre=(0.0, 0.0), einstein_radius=1.0)
    )

    fig, ax = plt.subplots()
    aplt.plot_magnification_2d(
        mass_obj=galaxy,
        grid=grid,
        contour_levels=(2.0, 10.0, 1.0e8),
        include_critical_curves=False,
        ax=ax,
    )

    assert _contour_levels_on(ax) == [list(np.log10([2.0, 10.0]))]

    fig, ax = plt.subplots()
    aplt.plot_magnification_2d(
        mass_obj=galaxy, grid=grid, contour_levels=None, include_critical_curves=False, ax=ax
    )

    assert _contour_levels_on(ax) == []

    plt.close("all")
