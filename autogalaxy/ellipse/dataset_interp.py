import numpy as np
from typing import Tuple

from autoconf import cached_property

import autoarray as aa


class DatasetInterp:
    def __init__(self, dataset: aa.Imaging):
        """
        An ellipse interpolator, which contains a dataset (e.g. the image data and noise-map) and performs interpo.aiton
        calculations used for ellipse fitting.

        This object is used by the input to the `FitEllipse` object, which fits the dataset with ellipses and quantifies
        the goodness-of-fit via a residual map, likelihood, chi-squared and other quantities.

        The following quantities of the ellipse data are interpolated and used for the following tasks:

        - `data`: The image data, which shows the signal that is analysed and fitted with ellipses.

        - `noise_map`: The RMS standard deviation error in every pixel, which is used to compute the chi-squared value
        and likelihood of a fit.

        The `data` and `noise_map` are typically the same images of a galaxy used to perform standard light-profile
        fitting.

        Parameters
        ----------
        dataset
            The imaging data, containing the image data, noise map.
        """
        self.dataset = dataset

    @cached_property
    def points_interp(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        The points on which the interpolation from the 2D grid of data is performed.
        """

        x = self.dataset.mask.derive_grid.all_false.native[0, :, 1]
        y = np.flip(self.dataset.mask.derive_grid.all_false.native[:, 0, 0])

        return (x, y)

    def mask_interp(self, points, xp=np):
        """
        Returns a 2D interpolation of the mask, which is used to determine whether inteprolated values use a masked
        pixel for the interpolation and thus should not be included in a fit.

        Backwards-compatibility note: this used to be a cached property that
        returned a scipy ``RegularGridInterpolator`` instance, also callable on
        ``(N, 2)`` points. Existing call sites in ``fit_ellipse.py`` that read
        ``self.interp.mask_interp(points)`` continue to work; the sentinel
        check ``self.interp.mask_interp is not None`` also continues to pass
        (a bound method is never None).

        Parameters
        ----------
        points
            An ``(N, 2)`` array of query coordinates. Out-of-bounds queries
            return 0.0 (treat outside-the-grid as unmasked).
        xp
            Array namespace (``numpy`` or ``jax.numpy``). Defaults to
            ``numpy``.
        """
        x_axis, y_axis = self.points_interp
        return aa.numerics.interp_2d(
            points,
            x_axis,
            y_axis,
            np.float64(self.dataset.data.mask),
            fill_value=0.0,
            xp=xp,
        )

    def data_interp(self, points, xp=np):
        """
        Returns a 2D interpolation of the data, which is used to evaluate the data at any point in 2D space.

        Parameters
        ----------
        points
            An ``(N, 2)`` array of query coordinates. Out-of-bounds queries
            return 0.0.
        xp
            Array namespace (``numpy`` or ``jax.numpy``). Defaults to
            ``numpy``.
        """
        x_axis, y_axis = self.points_interp
        return aa.numerics.interp_2d(
            points,
            x_axis,
            y_axis,
            np.float64(self.dataset.data.native),
            fill_value=0.0,
            xp=xp,
        )

    def noise_map_interp(self, points, xp=np):
        """
        Returns a 2D interpolation of the noise-map, which is used to evaluate the noise-map at any point in 2D space.

        Parameters
        ----------
        points
            An ``(N, 2)`` array of query coordinates. Out-of-bounds queries
            return 0.0.
        xp
            Array namespace (``numpy`` or ``jax.numpy``). Defaults to
            ``numpy``.
        """
        x_axis, y_axis = self.points_interp
        return aa.numerics.interp_2d(
            points,
            x_axis,
            y_axis,
            np.float64(self.dataset.noise_map.native),
            fill_value=0.0,
            xp=xp,
        )
