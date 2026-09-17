"""
The `MassProfileAggregate` mixin holds the aggregate-over-mass-profiles interface shared by every object which
stores `MassProfile` components as named attributes.

Both the `Galaxy` class (in `galaxy.py`) and the `MassField` class (in `mass_field.py`) hold their components as
named keyword-argument attributes, and both therefore sum the contributions of their mass profiles in exactly the
same way. That shared behaviour — `cls_list_from`, `deflections_yx_2d_from`, `convergence_2d_from`,
`potential_2d_from` and `extract_attribute` — lives here so there is one implementation of it.

The mixin deliberately does not define `has`. Both classes inherit `has` from `autofit`'s `ModelObject`, which
checks `self.__dict__` for instances of the input class, and the mixin must not shadow it.
"""

from typing import List, Optional, Type

import numpy as np

import autoarray as aa

from autogalaxy.profiles.mass.abstract import deflections_memo
from autogalaxy.profiles.mass.abstract.abstract import MassProfile


class MassProfileAggregate:
    """
    Aggregate methods over the `MassProfile` components an object holds as named attributes.

    Mixed into `Galaxy` and `MassField`. It assumes only that `self.__dict__` holds the object's components,
    and that `has` is provided by the class it is mixed into (via `af.ModelObject`).
    """

    def cls_list_from(self, cls: Type, cls_filtered: Optional[Type] = None) -> List:
        """
        Returns a list of objects in the galaxy which are an instance of the input `cls`.

        The optional `cls_filtered` input removes classes of an input instance type.

        For example:

        - If the input is `cls=ag.LightProfile`, a list containing all light profiles in the galaxy is returned.

        - If `cls=ag.LightProfile` and `cls_filtered=ag.LightProfileLinear`, a list of all light profiles
          excluding those which are linear light profiles will be returned.

        Parameters
        ----------
        cls
            The type of class that a list of instances of this class in the galaxy are returned for.
        cls_filtered
            A class type which is filtered and removed from the class list.

        Returns
        -------
            The list of objects in the galaxy that inherit from input `cls`.
        """
        return aa.util.misc.cls_list_from(
            values=self.__dict__.values(), cls=cls, cls_filtered=cls_filtered
        )

    @aa.decorators.to_vector_yx
    def deflections_yx_2d_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ) -> np.ndarray:
        """
        Returns the summed 2D deflection angles of the galaxy's mass profiles from a 2D grid of Cartesian (y,x)
        coordinates.

        If the galaxy has no mass profiles, a numpy array of zeros is returned.

        See the `autogalaxy.profiles.mass` package for details of how deflection angles are computed from a
        mass profile.

        The decorator `to_vector_yx` converts the output arrays from ndarrays to a `VectorYX2D` data structure
        using the input `grid`'s attributes.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the deflection angles are evaluated.
        """
        if self.has(cls=MassProfile):
            # Routed through `deflections_memo` (not called directly) so a fixed-geometry
            # profile's field is reused across likelihood evaluations; the helper falls
            # through to `p.deflections_yx_2d_from` whenever it cannot key the call
            # exactly, and is a no-op on the JAX path.
            return sum(
                map(
                    lambda p: deflections_memo.deflections_yx_2d_from(p, grid, xp),
                    self.cls_list_from(cls=MassProfile),
                )
            )

        return xp.zeros((grid.shape[0], 2))

    @aa.decorators.to_array
    def convergence_2d_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ) -> np.ndarray:
        """
        Returns the summed 2D convergence of the galaxy's mass profiles from a 2D grid of Cartesian (y,x) coordinates.

        If the galaxy has no mass profiles, a numpy array of zeros is returned.

        See the `autogalaxy.profiles.mass` package for details of how convergences are computed from a mass
        profile.

        The decorator `grid_2d_to_structure` converts the output arrays from ndarrays to an `Array2D` data structure
        using the input `grid`'s attributes.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the convergence are evaluated.
        """
        if self.has(cls=MassProfile):
            return sum(
                map(
                    lambda p: p.convergence_2d_from(grid=grid, xp=xp),
                    self.cls_list_from(cls=MassProfile),
                )
            )

        return xp.zeros((grid.shape[0],))

    @aa.decorators.to_array
    def potential_2d_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ) -> np.ndarray:
        """
        Returns the summed 2D potential of the galaxy's mass profiles from a 2D grid of Cartesian (y,x) coordinates.

        If the galaxy has no mass profiles, a numpy array of zeros is returned.

        See the `autogalaxy.profiles.mass` package for details of how potentials are computed from a mass
        profile.

        The decorator `grid_2d_to_structure` converts the output arrays from ndarrays to an `Array2D` data structure
        using the input `grid`'s attributes.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the potential are evaluated.
        """
        if self.has(cls=MassProfile):
            return sum(
                map(
                    lambda p: p.potential_2d_from(grid=grid, xp=xp),
                    self.cls_list_from(cls=MassProfile),
                )
            )
        return xp.zeros((grid.shape[0],))

    def extract_attribute(self, cls, attr_name):
        """
        Returns an attribute of a class and its children profiles in the galaxy as a `ValueIrregular`
        or `Grid2DIrregular` object.

        For example, if a galaxy has two light profiles and we want the `LightProfile` axis-ratios, the following:

        `galaxy.extract_attribute(cls=LightProfile, name="axis_ratio"`

        would return:

        ArrayIrregular(values=[axis_ratio_0, axis_ratio_1])

        If a galaxy has three mass profiles and we want the `MassProfile` centres, the following:

        `galaxy.extract_attribute(cls=MassProfile, name="centres"`

         would return:

        GridIrregular2D(grid=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1), (centre_y_2, centre_x_2)])

        This is used for visualization, for example plotting the centres of all light profiles colored by their profile.
        """

        def extract(value, name):
            try:
                return getattr(value, name)
            except (AttributeError, IndexError):
                return None

        attributes = [
            extract(value, attr_name)
            for value in self.__dict__.values()
            if isinstance(value, cls)
        ]

        attributes = list(filter(None, attributes))

        if attributes == []:
            return None
        elif isinstance(attributes[0], float):
            return aa.ArrayIrregular(values=attributes)
        elif isinstance(attributes[0], tuple):
            return aa.Grid2DIrregular(values=attributes)
