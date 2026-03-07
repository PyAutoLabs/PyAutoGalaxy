from __future__ import annotations

import numpy as np
from typing import List, Union, Optional, TYPE_CHECKING

import autoarray as aa
import autoarray.plot as aplt

if TYPE_CHECKING:

    from autogalaxy.galaxy.galaxy import Galaxy
    from autogalaxy.profiles.light.abstract import LightProfile
    from autogalaxy.profiles.mass.abstract.abstract import MassProfile

from autogalaxy.util import error_util


class Visuals1D(aplt.Visuals1D):
    def __init__(
        self,
        origin: Optional[aa.Grid1D] = None,
        mask: Optional[aa.Mask1D] = None,
        points: Optional[aa.Grid1D] = None,
        vertical_line: Optional[float] = None,
        shaded_region: Optional[List[Union[List, aa.Array1D, np.ndarray]]] = None,
        half_light_radius: Optional[float] = None,
        half_light_radius_errors: Optional[List[float]] = None,
        einstein_radius: Optional[float] = None,
        einstein_radius_errors: Optional[List[float]] = None,
        model_fluxes: Optional[aa.Grid1D] = None,
    ):
        super().__init__(
            origin=origin,
            mask=mask,
            points=points,
            vertical_line=vertical_line,
            shaded_region=shaded_region,
        )

        self.half_light_radius = half_light_radius
        self.half_light_radius_errors = half_light_radius_errors
        self.einstein_radius = einstein_radius
        self.einstein_radius_errors = einstein_radius_errors
        self.model_fluxes = model_fluxes

    def plot_via_plotter(self, plotter, grid_indexes=None, mapper=None):
        super().plot_via_plotter(plotter=plotter)

        if self.half_light_radius is not None:
            plotter.half_light_radius_axvline.axvline_vertical_line(
                vertical_line=self.half_light_radius,
                vertical_errors=self.half_light_radius_errors,
                label="Half-light Radius",
            )

        if self.einstein_radius is not None:
            plotter.einstein_radius_axvline.axvline_vertical_line(
                vertical_line=self.einstein_radius,
                vertical_errors=self.einstein_radius_errors,
                label="Einstein Radius",
            )

        if self.model_fluxes is not None:
            plotter.model_fluxes_yx_scatter.scatter_yx(
                y=self.model_fluxes, x=np.arange(len(self.model_fluxes))
            )

    def add_half_light_radius(
        self, light_obj: Union[LightProfile, Galaxy]
    ) -> "Visuals1D":
        """
        From an object with light profiles (e.g. a `LightProfile`, `Galaxy`) get its attributes that can be plotted
        and return them  in a `Visuals1D` object.

        Only attributes not already in `self` are extracted for plotting.

        From a light object the following 1D attributes can be extracted for plotting:

        - half_light_radius: the radius containing 50% of the light objects total integrated luminosity.

        Parameters
        ----------
        light_obj
            The light object (e.g. a `LightProfile`, `Galaxy`) whose attributes are extracted for plotting.

        Returns
        -------
        Visuals1D
            The collection of attributes that can be plotted by a `Plotter` object.
        """
        return self + self.__class__(half_light_radius=light_obj.half_light_radius)

    def add_half_light_radius_errors(
        self, light_obj_list: Union[List[LightProfile], List[Galaxy]], low_limit: float
    ) -> "Visuals1D":
        """
        From a list of objects with light profiles (e.g. a `LightProfile`, `Galaxy`) get its attributes that can be
        plotted and return them  in a `Visuals1D` object.

        Only attributes not already in `self` are extracted for plotting.

        This function iterates over all light objects in the list and averages over each attribute's values to estimate
        the mean value of the attribute and its error, both of which can then be plotted. This is typically used
        to plot 1D errors on a quantity that are estimated via a Probability Density Function.

        From a light object lust the following 1D attributes can be extracted for plotting:

        - half_light_radius: the radius containing 50% of the light objects total integrated luminosity.

        Parameters
        ----------
        light_obj_list
            The list of light objects (e.g. a `LightProfile`, `Galaxy`) whose mean attributes and error estimates are
            extracted for plotting.
        low_limit
            The value of sigma to which errors are estimated (e.g. 1.0 will estimate errors at the ~0.32 and ~0.68
            intervals of the probability distribution.

        Returns
        -------
        Visuals1D
            The mean value and errors of each attribute that are plotted in 1D by a `Plotter` object.
        """

        half_light_radius_list = [
            light_profile.half_light_radius for light_profile in light_obj_list
        ]

        if None in half_light_radius_list:
            half_light_radius = None
            half_light_radius_errors = None

        else:
            (
                half_light_radius,
                half_light_radius_errors,
            ) = error_util.value_median_and_error_region_via_quantile(
                value_list=half_light_radius_list, low_limit=low_limit
            )

        return self + self.__class__(
            half_light_radius=half_light_radius,
            half_light_radius_errors=half_light_radius_errors,
        )

    def add_einstein_radius(
        self, mass_obj: Union[MassProfile, Galaxy], grid: aa.type.Grid2DLike
    ) -> "Visuals1D":
        """
        From an object with mass profiles (e.g. a `MassProfile`, `Galaxy`) get its attributes that can be plotted
        and return them  in a `Visuals1D` object.

        Only attributes not already in `self` are extracted for plotting.

        From a mass object the following 1D attributes can be extracted for plotting:

        - einstein_radius: the einstein radius (e.g. area within critical curve) of the mass object.

        Mass profiles can be too shallow to do lensing and therefore an Einstein radius cannot be computed. This
        raises a TypeError which is accounted for below.

        Parameters
        ----------
        mass_obj
            The mass object (e.g. a `MassProfile`, `Galaxy`) whose attributes are extracted for plotting.

        Returns
        -------
        Visuals1D
            The collection of attributes that can be plotted by a `Plotter` object.
        """

        from autogalaxy.operate.lens_calc import LensCalc

        einstein_radius = None

        try:
            od = LensCalc.from_mass_obj(mass_obj)
            einstein_radius = od.einstein_radius_from(grid=grid)
        except (TypeError, AttributeError):
            pass

        return self + self.__class__(einstein_radius=einstein_radius)

    def add_einstein_radius_errors(
        self,
        mass_obj_list: Union[List[MassProfile], List[Galaxy]],
        grid: aa.type.Grid2DLike,
        low_limit: float,
    ) -> "Visuals1D":
        """
        From a list of objects with mass profiles (e.g. a `MassProfile`, `Galaxy`) get its attributes that can be
        plotted and return them  in a `Visuals1D` object.

        Only attributes not already in `self` are extracted for plotting.

        This function iterates over all mass objects in the list and averages over each attribute's values to estimate
        the mean value of the attribute and its error, both of which can then be plotted. This is typically used
        to plot 1D errors on a quantity that are estimated via a Probability Density Function.

        From a mass object lust the following 1D attributes can be extracted for plotting:

        - half_mass_radius: the radius containing 50% of the mass objects total integrated luminosity.

        Parameters
        ----------
        mass_obj_list
            The list of mass objects (e.g. a `MassProfile`, `Galaxy`) whose mean attributes and error estimates are
            extracted for plotting.
        low_limit
            The value of sigma to which errors are estimated (e.g. 1.0 will estimate errors at the ~0.32 and ~0.68
            intervals of the probability distribution.

        Returns
        -------
        Visuals1D
            The mean value and errors of each attribute that are plotted in 1D by a `Plotter` object.
        """

        from autogalaxy.operate.lens_calc import LensCalc

        einstein_radius_list = []

        for mass_obj in mass_obj_list:
            try:
                od = LensCalc.from_mass_obj(mass_obj)
                einstein_radius_list.append(od.einstein_radius_from(grid=grid))
            except TypeError:
                einstein_radius_list.append(None)

        einstein_radius_list = list(filter(None, einstein_radius_list))

        (
            einstein_radius,
            einstein_radius_errors,
        ) = error_util.value_median_and_error_region_via_quantile(
            value_list=einstein_radius_list, low_limit=low_limit
        )

        return self + self.__class__(
            einstein_radius=einstein_radius,
            einstein_radius_errors=einstein_radius_errors,
        )
