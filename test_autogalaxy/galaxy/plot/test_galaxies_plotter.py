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


def test__plot_magnification_source_plane__mesh_output(
    rectangular_mapper_7x7_3x3, galaxies_x2_7x7, grid_2d_7x7, plot_path, plot_patch
):
    aplt.plot_magnification_source_plane(
        mapper=rectangular_mapper_7x7_3x3,
        mass_obj=galaxies_x2_7x7,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "magnification_source_plane.png") in plot_patch.paths


def test__plot_magnification_source_plane_grid__output(
    galaxies_x2_7x7, grid_2d_7x7, plot_path, plot_patch
):
    aplt.plot_magnification_source_plane_grid(
        mass_obj=galaxies_x2_7x7,
        grid=grid_2d_7x7,
        source_shape_native=(10, 10),
        sub_size=2,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "magnification_source_plane_grid.png") in plot_patch.paths


def test__magnification_fractional_error_2d_from():
    import numpy as np

    import autogalaxy as ag

    mask = ag.Mask2D.all_false(shape_native=(2, 2), pixel_scales=0.1)
    best = ag.Array2D(values=np.array([2.0, -4.0, 10.0, 0.0]), mask=mask)

    constant = [best for _ in range(20)]

    assert np.allclose(
        aplt.magnification_fractional_error_2d_from(constant, best).slim.array[:3], 0.0
    )

    # |mu| uniform over a symmetric spread: the 16th-84th percentile half-range is 0.68 of the half-width.
    spread = np.linspace(-1.0, 1.0, 10001)
    draws = [
        ag.Array2D(values=np.array([2.0, -4.0, 10.0, 0.0]) * (1.0 + 0.1 * s), mask=mask)
        for s in spread
    ]

    fractional_error = aplt.magnification_fractional_error_2d_from(draws, best).slim.array

    assert fractional_error[:3] == pytest.approx([0.068, 0.068, 0.068], rel=1.0e-3)
    assert np.isinf(fractional_error[3])


def test__plot_magnification_uncertainty_2d__output(
    galaxies_x2_7x7, grid_2d_7x7, plot_path, plot_patch
):
    import autogalaxy as ag

    lens_calc = ag.LensCalc.from_mass_obj(galaxies_x2_7x7)
    best = lens_calc.magnification_2d_from(grid=grid_2d_7x7)

    aplt.plot_magnification_uncertainty_2d(
        magnification_list=[best * 0.95, best, best * 1.05],
        magnification_best=best,
        grid=grid_2d_7x7,
        mass_obj=galaxies_x2_7x7,
        output_path=plot_path,
        output_format="png",
    )
    assert str(plot_path / "magnification_uncertainty_2d.png") in plot_patch.paths
