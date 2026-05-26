import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile


class MassSheet(MassProfile):
    def __init__(self, centre: Tuple[float, float] = (0.0, 0.0), kappa: float = 0.0):
        """
        Represents a mass-sheet

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        kappa
            The magnitude of the convergence of the mass-sheet.
        """
        super().__init__(centre=centre, ell_comps=(0.0, 0.0))
        self.kappa = kappa

    def convergence_func(self, grid_radius: float) -> float:
        return 0.0

    @aa.decorators.to_array
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        return xp.full(shape=grid.shape[0], fill_value=self.kappa)

    @aa.decorators.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        return 0.5 * self.kappa * (grid.array[:, 0] ** 2 + grid.array[:, 1] ** 2)

    @aa.decorators.to_vector_yx
    @aa.decorators.transform
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        grid_radii = self.radial_grid_from(grid=grid, xp=xp, **kwargs)
        return self._cartesian_grid_via_radial_from(
            grid=grid, radius=self.kappa * grid_radii, xp=xp
        )
