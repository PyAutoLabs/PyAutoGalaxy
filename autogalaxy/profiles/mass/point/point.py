import copy
import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile


class PointMass(MassProfile):
    r"""
    Point mass lens profile.

    A point mass produces a convergence that is a Dirac delta function at the lens
    centre, a logarithmic potential, and a deflection angle that falls off as
    :math:`1/r`:

    .. math::

        \kappa(\boldsymbol{\theta}) = \pi \theta_E^2 \, \delta^{(2)}(\boldsymbol{\theta})

    .. math::

        \psi(\boldsymbol{\theta}) = \theta_E^2 \ln r

    .. math::

        \boldsymbol{\alpha}(\boldsymbol{\theta}) = \frac{\theta_E^2}{r}\,\hat{r}

    where :math:`\theta_E` is the Einstein radius (``einstein_radius``) and :math:`r`
    is the angular distance from the lens centre.

    This profile is used to represent compact objects such as black holes or stars.
    In practice the convergence grid value at the centre pixel is set to zero; the
    point-mass nature is captured entirely through the deflection and potential.
    """

    def __init__(
        self, centre: Tuple[float, float] = (0.0, 0.0), einstein_radius: float = 1.0
    ):
        r"""
        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius
            The Einstein radius :math:`\theta_E` of the point mass (arcseconds).
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
