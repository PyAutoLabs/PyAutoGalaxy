"""
Shared helper for light profiles with m=3 and m=4 Fourier perturbations on the
eccentric radius.

Used by ``SersicMultipole`` (``sersic_multipole.py``) and ``GaussianMultipole``
(``gaussian_multipole.py``). The leading underscore signals "implementation
detail" — when the future generalisation lands (a single wrapper-style multipole
that perturbs any radial profile, unified with the ellipse-fitting API), this
module will be replaced rather than evolved in-place.
"""

from typing import Tuple

import numpy as np

import autoarray as aa


class _LightProfileMultipoleMixin:
    """
    Provides ``perturbed_radii_from`` to light profiles that:

    - set ``multipole_3_comps`` and ``multipole_4_comps`` on ``self`` in ``__init__``
    - inherit from a base profile providing ``eccentric_radii_grid_from``
    """

    multipole_3_comps: Tuple[float, float]
    multipole_4_comps: Tuple[float, float]

    def perturbed_radii_from(
        self,
        grid: aa.type.Grid2DLike,
        xp=np,
        **kwargs,
    ) -> np.ndarray:
        """
        Returns the eccentric radii of ``grid`` perturbed by the m=3 and m=4 Fourier
        multipoles, as a raw backend array (numpy or jax.numpy).

        The perturbation is

            r' = r * (1 + c_3 cos(3 theta) + s_3 sin(3 theta)
                        + c_4 cos(4 theta) + s_4 sin(4 theta))

        where ``theta = arctan2(y, x)`` is the polar angle in the profile's
        elliptical reference frame (the grid is already in that frame after the
        ``@aa.decorators.transform`` decorator on the caller's ``image_2d_from``).

        Result is floored at 1e-8 to keep ``r ** (1/n)`` evaluations finite when a
        large multipole component drives the radius through zero.
        """
        grid_radii = self.eccentric_radii_grid_from(grid=grid, xp=xp, **kwargs)
        grid_radii_array = (
            grid_radii.array if hasattr(grid_radii, "array") else grid_radii
        )
        y = grid.array[:, 0]
        x = grid.array[:, 1]
        theta = xp.arctan2(y, x)
        c3, s3 = self.multipole_3_comps
        c4, s4 = self.multipole_4_comps
        perturbation = (
            1.0
            + c3 * xp.cos(3.0 * theta)
            + s3 * xp.sin(3.0 * theta)
            + c4 * xp.cos(4.0 * theta)
            + s4 * xp.sin(4.0 * theta)
        )
        return xp.maximum(xp.multiply(grid_radii_array, perturbation), 1e-8)
