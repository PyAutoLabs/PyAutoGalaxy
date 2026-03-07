from typing import List, Union, Optional

import autoarray as aa
import autoarray.plot as aplt


class Visuals2D(aplt.Visuals2D):
    def __init__(
        self,
        origin: aa.Grid2D = None,
        border: aa.Grid2D = None,
        mask: aa.Mask2D = None,
        lines: Optional[Union[List[aa.Array1D], aa.Grid2DIrregular]] = None,
        positions: Optional[Union[aa.Grid2DIrregular, List[aa.Grid2DIrregular]]] = None,
        grid: Union[aa.Grid2D] = None,
        mesh_grid: aa.Grid2D = None,
        vectors: aa.VectorYX2DIrregular = None,
        patches: "Union[ptch.Patch]" = None,
        fill_region: Optional[List] = None,
        array_overlay: aa.Array2D = None,
        light_profile_centres: aa.Grid2DIrregular = None,
        mass_profile_centres: aa.Grid2DIrregular = None,
        multiple_images: aa.Grid2DIrregular = None,
        tangential_critical_curves: Optional[
            Union[aa.Grid2DIrregular, List[aa.Grid2DIrregular]]
        ] = None,
        radial_critical_curves: Optional[
            Union[aa.Grid2DIrregular, List[aa.Grid2DIrregular]]
        ] = None,
        tangential_caustics: Optional[
            Union[aa.Grid2DIrregular, List[aa.Grid2DIrregular]]
        ] = None,
        radial_caustics: Optional[
            Union[aa.Grid2DIrregular, List[aa.Grid2DIrregular]]
        ] = None,
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
        indexes: Union[List[int], List[List[int]]] = None,
    ):
        super().__init__(
            mask=mask,
            positions=positions,
            grid=grid,
            lines=lines,
            mesh_grid=mesh_grid,
            vectors=vectors,
            patches=patches,
            fill_region=fill_region,
            array_overlay=array_overlay,
            origin=origin,
            border=border,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
            indexes=indexes,
        )

        self.light_profile_centres = light_profile_centres
        self.mass_profile_centres = mass_profile_centres
        self.multiple_images = multiple_images
        self.tangential_critical_curves = tangential_critical_curves
        self.radial_critical_curves = radial_critical_curves
        self.tangential_caustics = tangential_caustics
        self.radial_caustics = radial_caustics

    def plot_via_plotter(self, plotter, grid_indexes=None):
        super().plot_via_plotter(
            plotter=plotter,
            grid_indexes=grid_indexes,
        )

        if self.light_profile_centres is not None:
            plotter.light_profile_centres_scatter.scatter_grid(
                grid=self.light_profile_centres
            )

        if self.mass_profile_centres is not None:
            plotter.mass_profile_centres_scatter.scatter_grid(
                grid=self.mass_profile_centres
            )

        if self.multiple_images is not None:
            try:
                plotter.multiple_images_scatter.scatter_grid(
                    grid=self.multiple_images.array
                )
            except (AttributeError, ValueError):
                plotter.multiple_images_scatter.scatter_grid(grid=self.multiple_images)

        if self.tangential_critical_curves is not None:
            try:
                plotter.tangential_critical_curves_plot.plot_grid(
                    grid=self.tangential_critical_curves
                )
            except (AttributeError, TypeError):
                pass

        if self.radial_critical_curves is not None:
            try:
                plotter.radial_critical_curves_plot.plot_grid(
                    grid=self.radial_critical_curves
                )
            except (AttributeError, TypeError):
                pass

        if self.tangential_caustics is not None:
            try:
                try:
                    plotter.tangential_caustics_plot.plot_grid(
                        grid=self.tangential_caustics
                    )
                except (AttributeError, ValueError):
                    try:
                        plotter.tangential_caustics_plot.plot_grid(
                            grid=self.tangential_caustics.array
                        )
                    except (AttributeError, TypeError):
                        pass
            except (AttributeError, TypeError):
                pass

        if self.radial_caustics is not None:
            try:
                plotter.radial_caustics_plot.plot_grid(grid=self.radial_caustics)
            except (AttributeError, TypeError):
                pass

    def add_critical_curves_or_caustics(
        self, mass_obj, grid: aa.type.Grid2DLike, plane_index: int
    ):
        """
        From a object with mass profiles (e.g. mass profile, galaxy) extract the critical curves or caustics and
        returns them in a `Visuals2D` object.

        This includes support for a `plane_index`, which specifies the index of the plane in the tracer, which is
        an object used in PyAutoLens to represent a lensing system with multiple planes (e.g. an image plane and a
        source plane). The `plane_index` allows for the extraction of quantities from a specific plane in the tracer.

        When plotting a `Tracer` it is common for plots to only display quantities corresponding to one plane at a time
        (e.g. the convergence in the image plane, the source in the source plane). Therefore, quantities are only
        extracted from one plane, specified by the  input `plane_index`.

        Parameters
        ----------
        mass_obj
            The mass object (e.g. mass profile, galaxy, tracer) object which has attributes extracted for plotting.
        grid
            The 2D grid of (y,x) coordinates used to plot the tracer's quantities in 2D.
        plane_index
            The index of the plane in the tracer which is used to extract quantities, as only one plane is plotted
            at a time.

        Returns
        -------
        vis.Visuals2D
            A collection of attributes that can be plotted by a `Plotter` object.
        """
        if plane_index == 0:
            return self.add_critical_curves(mass_obj=mass_obj, grid=grid)
        return self.add_caustics(mass_obj=mass_obj, grid=grid)

    def add_critical_curves(self, mass_obj, grid: aa.type.Grid2DLike):
        """
        From a object with mass profiles (e.g. mass profile, galaxy) extract the critical curves and
        returns them in a `Visuals2D` object.

        When plotting a `Tracer` it is common for plots to only display quantities corresponding to one plane at a time
        (e.g. the convergence in the image plane, the source in the source plane). Therefore, quantities are only
        extracted from one plane, specified by the  input `plane_index`.

        Parameters
        ----------
        mass_obj
            The mass object (e.g. mass profile, galaxy, tracer) object which has attributes extracted for plotting.
        grid
            The 2D grid of (y,x) coordinates used to plot the tracer's quantities in 2D.
        plane_index
            The index of the plane in the tracer which is used to extract quantities, as only one plane is plotted
            at a time.

        Returns
        -------
        vis.Visuals2D
            A collection of attributes that can be plotted by a `Plotter` object.
        """
        from autogalaxy.operate.lens_calc import LensCalc

        od = LensCalc.from_mass_obj(mass_obj)

        tangential_critical_curves = od.tangential_critical_curve_list_from(grid=grid)

        radial_critical_curves = None
        radial_critical_curve_area_list = od.radial_critical_curve_area_list_from(
            grid=grid
        )

        if any([area > grid.pixel_scale for area in radial_critical_curve_area_list]):
            radial_critical_curves = od.radial_critical_curve_list_from(grid=grid)

        return self + self.__class__(
            tangential_critical_curves=tangential_critical_curves,
            radial_critical_curves=radial_critical_curves,
        )

    def add_caustics(self, mass_obj, grid: aa.type.Grid2DLike):
        """
        From a object with mass profiles (e.g. mass profile, galaxy) extract the caustics and
        returns them in a `Visuals2D` object.

        When plotting a `Tracer` it is common for plots to only display quantities corresponding to one plane at a time
        (e.g. the convergence in the image plane, the source in the source plane). Therefore, quantities are only
        extracted from one plane, specified by the  input `plane_index`.

        Parameters
        ----------
        mass_obj
            The mass object (e.g. mass profile, galaxy, tracer) object which has attributes extracted for plotting.
        grid
            The 2D grid of (y,x) coordinates used to plot the tracer's quantities in 2D.
        plane_index
            The index of the plane in the tracer which is used to extract quantities, as only one plane is plotted
            at a time.

        Returns
        -------
        vis.Visuals2D
            A collection of attributes that can be plotted by a `Plotter` object.
        """
        from autogalaxy.operate.lens_calc import LensCalc

        od = LensCalc.from_mass_obj(mass_obj)

        tangential_caustics = od.tangential_caustic_list_from(grid=grid)
        radial_caustics = od.radial_caustic_list_from(grid=grid)

        return self + self.__class__(
            tangential_caustics=tangential_caustics,
            radial_caustics=radial_caustics,
        )
