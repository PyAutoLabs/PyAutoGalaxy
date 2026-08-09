import numpy as np
from typing import Optional, Tuple

import autoarray as aa

from autogalaxy.profiles.light.abstract import LightProfile


class PointSource(LightProfile):
    """A discrete point source of light evaluated on an image-plane grid.

    ``intensity`` is the total flux assigned to the detector pixel containing
    ``centre``. When the grid is over-sampled, the flux is placed at the nearest
    sub-pixel and scaled before mean-binning so the total remains invariant.

    This profile describes direct image-plane emission, such as an unresolved
    star or AGN. Multiple lensed point images require the lens equation and are
    instead modelled with ``ag.ps.PointFlux`` and PyAutoLens's ``PointSolver``.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
    ):
        super().__init__(centre=centre, ell_comps=(0.0, 0.0), intensity=intensity)

    def _over_sampled_values_from(self, grid: aa.Grid2D, xp=np) -> np.ndarray:
        if grid.shape[0] == 0:
            return xp.zeros((0,))

        centre = xp.asarray(self.centre)
        pixel_centres = xp.asarray(grid.array)

        pixel_distances = xp.sum(xp.square(pixel_centres - centre), axis=1)
        nearest_pixel_index = xp.argmin(pixel_distances)
        nearest_pixel_centre = pixel_centres[nearest_pixel_index]

        pixel_scales = xp.asarray(grid.mask.pixel_scales)
        inside_pixel = xp.all(
            xp.abs(centre - nearest_pixel_centre) <= 0.5 * pixel_scales
        )

        over_sampled_grid = xp.asarray(grid.over_sampled.array)
        sub_to_pixel = xp.asarray(grid.over_sampler.slim_for_sub_slim)
        in_nearest_pixel = sub_to_pixel == nearest_pixel_index

        sub_distances = xp.sum(xp.square(over_sampled_grid - centre), axis=1)
        nearest_sub_index = xp.argmin(xp.where(in_nearest_pixel, sub_distances, xp.inf))

        local_sub_size = xp.asarray(grid.over_sample_size.array)[nearest_pixel_index]
        amplitude = self._intensity * xp.square(local_sub_size)

        return xp.where(
            xp.arange(over_sampled_grid.shape[0]) == nearest_sub_index,
            xp.where(inside_pixel, amplitude, 0.0),
            0.0,
        )

    def image_2d_from(
        self,
        grid: aa.type.Grid2DLike,
        xp=np,
        operated_only: Optional[bool] = None,
        binned: bool = True,
        **kwargs,
    ):
        """Return the point-source image on ``grid``.

        A uniform ``Grid2D`` provides the pixel geometry required for
        flux-conserving binning. For an irregular grid, the result is the
        corresponding discrete sample-space delta at the closest coordinate.
        """
        if isinstance(grid, aa.Grid2D):
            values = self._over_sampled_values_from(grid=grid, xp=xp)

            if operated_only is True:
                values = xp.zeros_like(values)

            if not binned:
                return values

            return grid.over_sampler.binned_array_2d_from(array=values, xp=xp)

        values = xp.asarray(grid.array if hasattr(grid, "array") else grid)

        if values.shape[0] == 0:
            result = xp.zeros((0,))
        else:
            distances = xp.sum(xp.square(values - xp.asarray(self.centre)), axis=1)
            nearest_index = xp.argmin(distances)
            result = xp.where(
                xp.arange(values.shape[0]) == nearest_index, self._intensity, 0.0
            )

        if operated_only is True:
            result = xp.zeros_like(result)

        if isinstance(grid, aa.Grid2DIrregular):
            return aa.ArrayIrregular(values=result)

        return result

    def image_2d_via_radii_from(self, grid_radii: np.ndarray, xp=np) -> np.ndarray:
        raise NotImplementedError(
            "PointSource is a discrete image-plane profile and cannot be "
            "evaluated from radial coordinates."
        )
