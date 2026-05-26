import copy
import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile


class PointMass(MassProfile):
    def __init__(
        self, centre: Tuple[float, float] = (0.0, 0.0), einstein_radius: float = 1.0
    ):
        """
        Represents a point-mass.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius
            The arc-second Einstein radius of the point-mass.
        """
        super().__init__(centre=centre, ell_comps=(0.0, 0.0))
        self.einstein_radius = einstein_radius

    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        squared_distances = np.square(grid[:, 0] - self.centre[0]) + np.square(
            grid[:, 1] - self.centre[1]
        )
        central_pixel = np.argmin(squared_distances)

        convergence = np.zeros(shape=grid.shape[0])
        #    convergence[central_pixel] = np.pi * self.einstein_radius ** 2.0
        return convergence

    @aa.decorators.to_array
    @aa.decorators.transform
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        r = xp.sqrt(grid.array[:, 0] ** 2 + grid.array[:, 1] ** 2 + 1e-20)
        return self.einstein_radius ** 2 * xp.log(r)

    @aa.decorators.to_vector_yx
    @aa.decorators.transform
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        grid_radii = self.radial_grid_from(grid=grid, xp=xp, **kwargs)
        return self._cartesian_grid_via_radial_from(
            grid=grid, radius=self.einstein_radius**2 / grid_radii, xp=xp
        )

    @property
    def is_point_mass(self):
        return True
