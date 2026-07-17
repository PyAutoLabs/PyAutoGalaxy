from typing import Optional

import numpy as np
from scipy.sparse import spmatrix
from scipy.spatial import Delaunay

import autoarray as aa
from autoarray.operators import derivative_util

from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.profiles.mass.input.interp import LinearNDInterpolatorExt


class InputDeflections(MassProfile):
    def __init__(
        self,
        deflections_y: np.ndarray,
        deflections_x: np.ndarray,
        image_plane_grid: aa.type.Grid2DLike,
        mask: aa.type.Mask2D,
        extrapolate: str = "nearest",
        Hy: Optional[spmatrix] = None,
        Hx: Optional[spmatrix] = None,
    ):
        """
        A pixelized mass model defined by known deflection angles on the
        unmasked pixels of an image-plane grid (e.g. from a previous lens
        model, a particle simulation, or a potential-correction
        reconstruction).

        Deflections at arbitrary positions are evaluated by linear
        interpolation over a Delaunay triangulation of the unmasked pixels
        (nearest-neighbour fallback outside the convex hull). The convergence
        is derived from the input deflections via the sparse first-derivative
        operators of the mask, kappa = 0.5 * (dalpha_y/dy + dalpha_x/dx). The
        lensing potential is not derivable from deflections alone and returns
        zeros.

        Ported from the ``potential_correction`` package of Cao et al. 2025
        (https://github.com/caoxiaoyue/lensing_potential_correction). If you
        use this profile in your research, please cite Cao et al. 2025;
        citation materials are provided at
        https://github.com/caoxiaoyue/potential_correction_paper.

        Parameters
        ----------
        deflections_y
            The 1D (slim) array of the y components of the deflection angles
            on the unmasked pixels.
        deflections_x
            The 1D (slim) array of the x components of the deflection angles
            on the unmasked pixels.
        image_plane_grid
            The [n_unmasked, 2] (y, x) grid of the unmasked pixels the
            deflection angles are defined on.
        mask
            The cleaned 2D mask defining the unmasked pixels (see
            ``aa.util.derivative.cleaned_mask_from``); its ``pixel_scale``
            sets the finite-difference step of the derived convergence.
        extrapolate
            The extrapolation behaviour outside the unmasked pixels' convex
            hull: ``"nearest"`` (default; the field continues beyond the
            grid) or ``"zero"`` (the field vanishes outside it — required
            when the profile represents a localized correction on a
            sub-region of a larger grid, e.g. an arc-restricted dpsi mesh,
            where nearest extrapolation would produce spurious constant
            deflections everywhere else).
        Hy
            The sparse first-derivative operator along y of the mask; built
            from the mask if not input.
        Hx
            The sparse first-derivative operator along x of the mask; built
            from the mask if not input.
        """
        super().__init__()

        self.deflections_y = np.asarray(deflections_y)
        self.deflections_x = np.asarray(deflections_x)
        self.image_plane_grid = np.asarray(image_plane_grid)
        self.mask = mask
        self.extrapolate = extrapolate
        self.Hy = Hy
        self.Hx = Hx

        self._build_interpolators()

    def _build_interpolators(self):
        if self.Hy is None or self.Hx is None:
            self.Hy, self.Hx = derivative_util.derivative_1st_operators_from(
                np.asarray(self.mask), pixel_scale=self.mask.pixel_scale
            )

        self.convergence_slim = (
            self.Hy @ self.deflections_y + self.Hx @ self.deflections_x
        ) * 0.5

        self.tri = Delaunay(np.fliplr(self.image_plane_grid))
        self.interp_defl_y = LinearNDInterpolatorExt(self.tri, fill=self.extrapolate, values=self.deflections_y)
        self.interp_defl_x = LinearNDInterpolatorExt(self.tri, fill=self.extrapolate, values=self.deflections_x)
        self.interp_kappa = LinearNDInterpolatorExt(self.tri, fill=self.extrapolate, values=self.convergence_slim)

    @aa.decorators.to_array
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        grid = np.asarray(grid)
        return self.interp_kappa(grid[:, 1], grid[:, 0])

    @aa.decorators.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        return np.zeros(shape=np.asarray(grid).shape[0])

    @aa.decorators.to_vector_yx
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles at a given set of arc-second gridded
        coordinates, by interpolating the input deflection angles.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles
            are computed on.
        """
        grid = np.asarray(grid)
        deflections_y = self.interp_defl_y(grid[:, 1], grid[:, 0])
        deflections_x = self.interp_defl_x(grid[:, 1], grid[:, 0])
        return np.stack((deflections_y, deflections_x), axis=-1)
