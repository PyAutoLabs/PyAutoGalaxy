from typing import Optional

import numpy as np
from scipy.sparse import spmatrix
from scipy.spatial import Delaunay

import autoarray as aa
from autoarray.operators import derivative_util

from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.profiles.mass.input.interp import LinearNDInterpolatorExt


class InputPotential(MassProfile):
    def __init__(
        self,
        lensing_potential: np.ndarray,
        image_plane_grid: aa.type.Grid2DLike,
        mask: aa.type.Mask2D,
        extrapolate: str = "nearest",
        Hy: Optional[spmatrix] = None,
        Hx: Optional[spmatrix] = None,
        Hyy: Optional[spmatrix] = None,
        Hxx: Optional[spmatrix] = None,
    ):
        """
        A pixelized mass model defined by known lensing-potential values on
        the unmasked pixels of an image-plane grid (e.g. a Gaussian random
        field realization, or a potential-correction reconstruction).

        The deflection angles are derived from the input potential via the
        sparse first-derivative operators of the mask (alpha = grad psi) and
        the convergence via the second-derivative operators
        (kappa = 0.5 * laplacian psi). The potential, deflections and
        convergence at arbitrary positions are evaluated by linear
        interpolation over a Delaunay triangulation of the unmasked pixels
        (nearest-neighbour fallback outside the convex hull).

        Ported from the ``potential_correction`` package of Cao et al. 2025
        (https://github.com/caoxiaoyue/lensing_potential_correction). If you
        use this profile in your research, please cite Cao et al. 2025;
        citation materials are provided at
        https://github.com/caoxiaoyue/potential_correction_paper.

        Parameters
        ----------
        lensing_potential
            The 1D (slim) array of the lensing potential on the unmasked
            pixels.
        image_plane_grid
            The [n_unmasked, 2] (y, x) grid of the unmasked pixels the
            potential is defined on.
        mask
            The cleaned 2D mask defining the unmasked pixels (see
            ``aa.util.derivative.cleaned_mask_from``); its ``pixel_scale``
            sets the finite-difference step of the derived deflections and
            convergence.
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
        Hyy
            The sparse second-derivative operator along y of the mask; built
            from the mask if not input.
        Hxx
            The sparse second-derivative operator along x of the mask; built
            from the mask if not input.
        """
        super().__init__()

        self.lensing_potential = np.asarray(lensing_potential)
        self.image_plane_grid = np.asarray(image_plane_grid)
        self.mask = mask
        self.extrapolate = extrapolate
        self.Hy = Hy
        self.Hx = Hx
        self.Hyy = Hyy
        self.Hxx = Hxx

        self._build_interpolators()

    def _build_interpolators(self):
        if self.Hy is None or self.Hx is None:
            self.Hy, self.Hx = derivative_util.derivative_1st_operators_from(
                np.asarray(self.mask), pixel_scale=self.mask.pixel_scale
            )
        if self.Hyy is None or self.Hxx is None:
            self.Hyy, self.Hxx = derivative_util.derivative_2nd_operators_from(
                np.asarray(self.mask), pixel_scale=self.mask.pixel_scale
            )

        self.deflections_y = self.Hy @ self.lensing_potential
        self.deflections_x = self.Hx @ self.lensing_potential
        self.convergence_slim = (
            self.Hyy @ self.lensing_potential + self.Hxx @ self.lensing_potential
        ) * 0.5

        self.tri = Delaunay(np.fliplr(self.image_plane_grid))
        self.interp_psi = LinearNDInterpolatorExt(self.tri, fill=self.extrapolate, values=self.lensing_potential)
        self.interp_defl_y = LinearNDInterpolatorExt(self.tri, fill=self.extrapolate, values=self.deflections_y)
        self.interp_defl_x = LinearNDInterpolatorExt(self.tri, fill=self.extrapolate, values=self.deflections_x)
        self.interp_kappa = LinearNDInterpolatorExt(self.tri, fill=self.extrapolate, values=self.convergence_slim)

    @aa.decorators.to_array
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        grid = np.asarray(grid)
        return self.interp_kappa(grid[:, 1], grid[:, 0])

    @aa.decorators.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        grid = np.asarray(grid)
        return self.interp_psi(grid[:, 1], grid[:, 0])

    @aa.decorators.to_vector_yx
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles at a given set of arc-second gridded
        coordinates, by interpolating the deflections derived from the input
        lensing potential.

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
