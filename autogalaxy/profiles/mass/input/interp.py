import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator


class LinearNDInterpolatorExt:
    def __init__(self, points, values):
        """
        Linear interpolation over a Delaunay triangulation of scattered 2D
        points, falling back to nearest-neighbour interpolation outside the
        convex hull so extrapolated values are never NaN.

        Ported from the ``potential_correction`` package of Cao et al. 2025
        (https://github.com/caoxiaoyue/lensing_potential_correction; cite via
        https://github.com/caoxiaoyue/potential_correction_paper).

        Parameters
        ----------
        points
            The (x, y) coordinates the values are defined at, as an
            [n_points, 2] array or a pre-built ``scipy.spatial.Delaunay``
            triangulation of them.
        values
            The values interpolated.
        """
        self.funcinterp = LinearNDInterpolator(points, values)
        self.funcnearest = NearestNDInterpolator(points, values)

    def __call__(self, *args):
        z = self.funcinterp(*args)
        chk = np.isnan(z)
        if chk.any():
            return np.where(chk, self.funcnearest(*args), z)
        return z
