import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile


class MassSheet(MassProfile):
    r"""
    Uniform convergence mass sheet.

    A mass sheet produces a spatially constant convergence :math:`\kappa_{\rm ext}`
    across the lens plane.  It is the simplest realisation of the mass-sheet degeneracy
    (Gorenstein 1988): any lensing configuration can be transformed by adding a sheet
    without changing observed image positions or flux ratios.

    The convergence, lensing potential, and deflection field are:

    .. math::

        \kappa(\boldsymbol{\theta}) = \kappa_{\rm ext}

    .. math::

        \psi(\boldsymbol{\theta}) = \tfrac{1}{2} \kappa_{\rm ext} \, |\boldsymbol{\theta}|^2

    .. math::

        \boldsymbol{\alpha}(\boldsymbol{\theta}) = \kappa_{\rm ext} \, \boldsymbol{\theta}

    References
    ----------
    - Gorenstein, Falco & Shapiro 1988, ApJ, 327, 693
    - Schneider & Sluse 2013, A&A, 559, A37
    """

    def __init__(self, centre: Tuple[float, float] = (0.0, 0.0), kappa: float = 0.0):
        r"""
        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        kappa
            The uniform convergence :math:`\kappa_{\rm ext}` of the mass sheet.
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
