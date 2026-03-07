from autoconf import cached_property

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.plot.mat_plot.two_d import MatPlot2D
from autogalaxy.plot.visuals.two_d import Visuals2D

from autogalaxy.plot.abstract_plotters import Plotter


class MassPlotter(Plotter):
    def __init__(
        self,
        mass_obj,
        grid: aa.type.Grid2DLike,
        mat_plot_2d: MatPlot2D = None,
        visuals_2d: Visuals2D = None,
    ):
        super().__init__(mat_plot_2d=mat_plot_2d, visuals_2d=visuals_2d)

        self.mass_obj = mass_obj
        self.grid = grid

    @cached_property
    def visuals_2d_with_critical_curves(self) -> aplt.Visuals2D:
        """
        Returns the `Visuals2D` of the plotter with critical curves and caustics added, which are used to plot
        the critical curves and caustics of the `Tracer` object.
        """
        return self.visuals_2d.add_critical_curves_or_caustics(
            mass_obj=self.mass_obj, grid=self.grid, plane_index=0
        )

    def figures_2d(
        self,
        convergence: bool = False,
        potential: bool = False,
        deflections_y: bool = False,
        deflections_x: bool = False,
        magnification: bool = False,
        title_suffix: str = "",
        filename_suffix: str = "",
    ):
        """
        Plots the individual attributes of the plotter's mass object in 2D, which are computed via the plotter's 2D
        grid object.

        The API is such that every plottable attribute of the `Imaging` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        convergence
            Whether to make a 2D plot (via `imshow`) of the convergence.
        potential
            Whether to make a 2D plot (via `imshow`) of the potential.
        deflections_y
            Whether to make a 2D plot (via `imshow`) of the y component of the deflection angles.
        deflections_x
            Whether to make a 2D plot (via `imshow`) of the x component of the deflection angles.
        magnification
            Whether to make a 2D plot (via `imshow`) of the magnification.
        """

        if convergence:
            self.mat_plot_2d.plot_array(
                array=self.mass_obj.convergence_2d_from(grid=self.grid),
                visuals_2d=self.visuals_2d_with_critical_curves,
                auto_labels=aplt.AutoLabels(
                    title=f"Convergence{title_suffix}",
                    filename=f"convergence_2d{filename_suffix}",
                    cb_unit="",
                ),
            )

        if potential:
            self.mat_plot_2d.plot_array(
                array=self.mass_obj.potential_2d_from(grid=self.grid),
                visuals_2d=self.visuals_2d_with_critical_curves,
                auto_labels=aplt.AutoLabels(
                    title=f"Potential{title_suffix}",
                    filename=f"potential_2d{filename_suffix}",
                    cb_unit="",
                ),
            )

        if deflections_y:
            deflections = self.mass_obj.deflections_yx_2d_from(grid=self.grid)
            deflections_y = aa.Array2D(
                values=deflections.slim[:, 0], mask=self.grid.mask
            )

            self.mat_plot_2d.plot_array(
                array=deflections_y,
                visuals_2d=self.visuals_2d_with_critical_curves,
                auto_labels=aplt.AutoLabels(
                    title=f"Deflections Y{title_suffix}",
                    filename=f"deflections_y_2d{filename_suffix}",
                    cb_unit="",
                ),
            )

        if deflections_x:
            deflections = self.mass_obj.deflections_yx_2d_from(grid=self.grid)
            deflections_x = aa.Array2D(
                values=deflections.slim[:, 1], mask=self.grid.mask
            )

            self.mat_plot_2d.plot_array(
                array=deflections_x,
                visuals_2d=self.visuals_2d_with_critical_curves,
                auto_labels=aplt.AutoLabels(
                    title=f"Deflections X{title_suffix}",
                    filename=f"deflections_x_2d{filename_suffix}",
                    cb_unit="",
                ),
            )

        if magnification:
            from autogalaxy.operate.lens_calc import LensCalc

            self.mat_plot_2d.plot_array(
                array=LensCalc.from_mass_obj(
                    self.mass_obj
                ).magnification_2d_from(grid=self.grid),
                visuals_2d=self.visuals_2d_with_critical_curves,
                auto_labels=aplt.AutoLabels(
                    title=f"Magnification{title_suffix}",
                    filename=f"magnification_2d{filename_suffix}",
                    cb_unit="",
                ),
            )
