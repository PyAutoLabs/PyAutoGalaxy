import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator


class LinearNDInterpolatorExt:
    def __init__(self, points, values, fill: str = "nearest"):
        """
        Linear interpolation over a Delaunay triangulation of scattered 2D
        points, with a choice of extrapolation behaviour outside the convex
        hull so extrapolated values are never NaN:

        - ``fill="nearest"`` (default): nearest-neighbour extrapolation —
          appropriate when the sampled field genuinely continues beyond the
          hull (e.g. a source brightness evaluated slightly off-mesh).
        - ``fill="zero"``: zero extrapolation — appropriate when the field is
          only defined on the sampled region and must vanish outside it
          (e.g. localized potential corrections: nearest extrapolation would
          smear constant non-zero values — and for their deflections,
          spurious constant deflections — across the whole grid).

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
        fill
            The extrapolation behaviour outside the convex hull:
            ``"nearest"`` or ``"zero"``.
        """
        if fill not in ("nearest", "zero"):
            raise ValueError(f"fill must be 'nearest' or 'zero', got {fill!r}")
        self.fill = fill
        self.funcinterp = LinearNDInterpolator(points, values)
        self.funcnearest = (
            NearestNDInterpolator(points, values) if fill == "nearest" else None
        )

    def __call__(self, *args):
        z = self.funcinterp(*args)
        chk = np.isnan(z)
        if chk.any():
            if self.fill == "nearest":
                return np.where(chk, self.funcnearest(*args), z)
            return np.where(chk, 0.0, z)
        return z
