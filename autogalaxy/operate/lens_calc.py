from functools import wraps
import logging
import numpy as np
from typing import List, Tuple, Union

from autoconf import conf

import autoarray as aa

from autogalaxy.util.shear_field import ShearYX2DIrregular

logger = logging.getLogger(__name__)


def grid_scaled_2d_for_marching_squares_from(
    grid_pixels_2d: aa.Grid2D,
    shape_native: Tuple[int, int],
    mask: aa.Mask2D,
) -> aa.Grid2DIrregular:
    pixel_scales = mask.pixel_scales
    origin = mask.origin

    grid_scaled_1d = aa.util.geometry.grid_scaled_2d_slim_from(
        grid_pixels_2d_slim=grid_pixels_2d,
        shape_native=shape_native,
        pixel_scales=pixel_scales,
        origin=origin,
    )

    grid_scaled_1d[:, 0] -= pixel_scales[0] / 2.0
    grid_scaled_1d[:, 1] += pixel_scales[1] / 2.0

    return aa.Grid2DIrregular(values=grid_scaled_1d)


def evaluation_grid(func):
    @wraps(func)
    def wrapper(
        lensing_obj, grid, pixel_scale: Union[Tuple[float, float], float] = 0.05
    ):
        if hasattr(grid, "is_evaluation_grid"):
            if grid.is_evaluation_grid:
                return func(lensing_obj, grid, pixel_scale)

        pixel_scale_ratio = grid.pixel_scale / pixel_scale

        zoom = aa.Zoom2D(mask=grid.mask)

        zoom_shape_native = zoom.shape_native
        shape_native = (
            int(pixel_scale_ratio * zoom_shape_native[0]),
            int(pixel_scale_ratio * zoom_shape_native[1]),
        )

        max_evaluation_grid_size = conf.instance["general"]["grid"][
            "max_evaluation_grid_size"
        ]

        # This is a hack to prevent the evaluation gird going beyond 1000 x 1000 pixels, which slows the code
        # down a lot. Need a better moe robust way to set this up for any general lens.

        if shape_native[0] > max_evaluation_grid_size:
            pixel_scale = pixel_scale_ratio / (
                shape_native[0] / float(max_evaluation_grid_size)
            )
            shape_native = (max_evaluation_grid_size, max_evaluation_grid_size)

        grid = aa.Grid2D.uniform(
            shape_native=shape_native,
            pixel_scales=(pixel_scale, pixel_scale),
            origin=zoom.offset_scaled,
        )

        grid.is_evaluation_grid = True

        return func(lensing_obj, grid, pixel_scale)

    return wrapper


class LensCalc:
    """
    Computes lensing quantities from a deflection-angle callable and an optional potential callable.

    The deflection callable is used to compute the Hessian, Jacobian, convergence, shear,
    magnification, critical curves, caustics, and Einstein radius/mass.  If a potential
    callable is also supplied, ``fermat_potential_from`` is available as well.

    Parameters
    ----------
    deflections_yx_2d_from
        A callable with signature ``(grid, xp=np, **kwargs)`` that returns the 2D deflection
        angles on the given grid.  Typically a bound method of a ``MassProfile``, ``Galaxy``,
        or ``Galaxies`` instance.
    potential_2d_from
        Optional callable with signature ``(grid, xp=np, **kwargs)`` that returns the 2D
        lensing potential on the given grid.  Required only for ``fermat_potential_from``.
    """

    def __init__(self, deflections_yx_2d_from, potential_2d_from=None):
        self.deflections_yx_2d_from = deflections_yx_2d_from
        self.potential_2d_from = potential_2d_from

    @classmethod
    def from_mass_obj(cls, mass_obj):
        """Construct from any object that has a ``deflections_yx_2d_from`` method.

        If the object also exposes ``potential_2d_from``, it is captured so that
        ``fermat_potential_from`` is available on the returned instance.
        """
        return cls(
            deflections_yx_2d_from=mass_obj.deflections_yx_2d_from,
            potential_2d_from=getattr(mass_obj, "potential_2d_from", None),
        )

    @classmethod
    def from_tracer(
        cls, tracer, use_multi_plane: bool = True, plane_i: int = 0, plane_j: int = -1
    ):
        """
        Construct from a PyAutoLens ``Tracer`` object.

        The ``Tracer`` type is deliberately left unannotated: ``autogalaxy`` does not
        depend on ``autolens``, so no import of ``Tracer`` is performed here.  Callers
        (which live inside ``autolens``) are responsible for passing the correct object.

        Parameters
        ----------
        tracer
            A PyAutoLens ``Tracer`` instance.  Must expose ``deflections_yx_2d_from``
            and, when ``use_multi_plane=True``, ``deflections_between_planes_from``.
        use_multi_plane
            If ``True`` the stored callable is
            ``tracer.deflections_between_planes_from`` with ``plane_i`` and ``plane_j``
            bound via ``functools.partial``, matching the multi-plane ray-tracing path.
            If ``False`` the stored callable is ``tracer.deflections_yx_2d_from``,
            i.e. the standard two-plane path.
        plane_i
            Index of the first plane used by ``deflections_between_planes_from``.
            Ignored when ``use_multi_plane=False``.  Defaults to ``0`` (image plane).
        plane_j
            Index of the second plane used by ``deflections_between_planes_from``.
            Ignored when ``use_multi_plane=False``.  Defaults to ``-1`` (source plane).
        """
        potential_2d_from = getattr(tracer, "potential_2d_from", None)

        if use_multi_plane:
            from functools import partial

            return cls(
                deflections_yx_2d_from=partial(
                    tracer.deflections_between_planes_from,
                    plane_i=plane_i,
                    plane_j=plane_j,
                ),
                potential_2d_from=potential_2d_from,
            )
        return cls(
            deflections_yx_2d_from=tracer.deflections_yx_2d_from,
            potential_2d_from=potential_2d_from,
        )

    def time_delay_geometry_term_from(self, grid, xp=np) -> aa.Array2D:
        """
        Returns the geometric time delay term of the Fermat potential for a given grid of image-plane positions.

        This term is given by:

        .. math::
            \[\tau_{\text{geom}}(\boldsymbol{\theta}) = \frac{1}{2} |\boldsymbol{\theta} - \boldsymbol{\beta}|^2\]

        where:
        - \( \boldsymbol{\theta} \) is the image-plane coordinate,
        - \( \boldsymbol{\beta} = \boldsymbol{\theta} - \boldsymbol{\alpha}(\boldsymbol{\theta}) \) is the source-plane coordinate,
        - \( \boldsymbol{\alpha} \) is the deflection angle at each image-plane coordinate.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and time delay geometric term are computed
            on.

        Returns
        -------
        The geometric time delay term at each grid position.
        """
        deflections = self.deflections_yx_2d_from(grid=grid, xp=xp)

        src_y = grid[:, 0] - deflections[:, 0]
        src_x = grid[:, 1] - deflections[:, 1]

        delay = 0.5 * ((grid[:, 0] - src_y) ** 2 + (grid[:, 1] - src_x) ** 2)

        if isinstance(grid, aa.Grid2DIrregular):
            return aa.ArrayIrregular(values=delay)
        return aa.Array2D(values=delay, mask=grid.mask)

    def fermat_potential_from(self, grid, xp=np) -> aa.Array2D:
        """
        Returns the Fermat potential for a given grid of image-plane positions.

        This is the sum of the geometric time delay term and the gravitational (Shapiro) delay
        term (i.e. the lensing potential), and is given by:

        .. math::
            \\phi(\\boldsymbol{\\theta}) = \\frac{1}{2} |\\boldsymbol{\\theta} - \\boldsymbol{\\beta}|^2
            - \\psi(\\boldsymbol{\\theta})

        Requires that ``potential_2d_from`` was supplied at construction (e.g. via
        ``LensCalc.from_mass_obj`` or ``LensCalc.from_tracer``).

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the Fermat potential is computed on.
        xp
            The array module (``numpy`` or ``jax.numpy``).
        """
        if self.potential_2d_from is None:
            raise ValueError(
                "fermat_potential_from requires a potential_2d_from callable. "
                "Construct LensCalc with potential_2d_from, or use from_mass_obj / from_tracer."
            )
        time_delay = self.time_delay_geometry_term_from(grid=grid, xp=xp)
        potential = self.potential_2d_from(grid=grid, xp=xp)
        fermat_potential = time_delay - potential
        if isinstance(grid, aa.Grid2DIrregular):
            return aa.ArrayIrregular(values=fermat_potential)
        return aa.Array2D(values=fermat_potential, mask=grid.mask)

    def tangential_eigen_value_from(self, grid, xp=np) -> aa.Array2D:
        """
        Returns the tangential eigen values of lensing jacobian, which are given by the expression:

        `tangential_eigen_value = 1 - convergence - shear`

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and tangential eigen values are computed
            on.
        xp
            The array module (``numpy`` or ``jax.numpy``). Passed through to ``convergence_2d_via_hessian_from``
            and ``shear_yx_2d_via_hessian_from``. When ``xp`` is not ``numpy`` the result is a raw array rather
            than an ``aa.Array2D`` wrapper.
        """
        convergence = self.convergence_2d_via_hessian_from(grid=grid, xp=xp)
        shear_yx = self.shear_yx_2d_via_hessian_from(grid=grid, xp=xp)

        if xp is np:
            return aa.Array2D(
                values=1 - convergence - shear_yx.magnitudes, mask=grid.mask
            )
        magnitudes = xp.sqrt(shear_yx[:, 0] ** 2 + shear_yx[:, 1] ** 2)
        return 1 - convergence - magnitudes

    def radial_eigen_value_from(self, grid, xp=np) -> aa.Array2D:
        """
        Returns the radial eigen values of lensing jacobian, which are given by the expression:

        `radial_eigen_value = 1 - convergence + shear`

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and radial eigen values are computed on.
        xp
            The array module (``numpy`` or ``jax.numpy``). Passed through to ``convergence_2d_via_hessian_from``
            and ``shear_yx_2d_via_hessian_from``. When ``xp`` is not ``numpy`` the result is a raw array rather
            than an ``aa.Array2D`` wrapper.
        """
        convergence = self.convergence_2d_via_hessian_from(grid=grid, xp=xp)
        shear = self.shear_yx_2d_via_hessian_from(grid=grid, xp=xp)

        if xp is np:
            return aa.Array2D(
                values=1 - convergence + shear.magnitudes, mask=grid.mask
            )
        magnitudes = xp.sqrt(shear[:, 0] ** 2 + shear[:, 1] ** 2)
        return 1 - convergence + magnitudes

    def magnification_2d_from(self, grid, xp=np) -> aa.Array2D:
        """
        Returns the 2D magnification map of lensing object, which is computed as the inverse of the determinant of the
        lensing Jacobian, expressed via the Hessian components.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and magnification map are computed on.
        xp
            The array module (``numpy`` or ``jax.numpy``). Passed through to ``hessian_from``. When ``xp`` is
            not ``numpy`` the result is a raw array rather than an ``aa.Array2D`` wrapper.
        """
        hessian_yy, hessian_xy, hessian_yx, hessian_xx = self.hessian_from(
            grid=grid, xp=xp
        )

        det_A = (1 - hessian_xx) * (1 - hessian_yy) - hessian_xy * hessian_yx

        if xp is np:
            return aa.Array2D(values=1 / det_A, mask=grid.mask)
        return 1 / det_A

    def deflections_yx_scalar(self, y, x, pixel_scales):
        """
        Returns the deflection angles at a single (y, x) arc-second coordinate as a JAX array of
        shape (2,), where index 0 is the y-deflection and index 1 is the x-deflection.

        This is an internal method used by `hessian_from` to enable JAX auto-differentiation via
        `jax.jacfwd`. The function must accept y and x as two separate scalar inputs (rather than
        a single combined array) so that JAX treats the function as R² -> R² and computes a proper
        2x2 Jacobian matrix.

        Parameters
        ----------
        y
            The y arc-second coordinate (scalar).
        x
            The x arc-second coordinate (scalar).
        pixel_scales
            The pixel scales used to construct the internal (1, 1) Mask2D.
        """
        import jax.numpy as jnp

        mask = aa.Mask2D.all_false(shape_native=(1, 1), pixel_scales=pixel_scales)
        grid = aa.Grid2D(
            values=jnp.stack((y.reshape(1), x.reshape(1)), axis=-1), mask=mask
        )
        return self.deflections_yx_2d_from(grid, xp=jnp).squeeze()

    def hessian_from(self, grid, xp=np) -> Tuple:
        """
        Returns the Hessian of the lensing object, where the Hessian is the second partial derivatives of the
        potential (see equation 55 https://inspirehep.net/literature/419263):

        `hessian_{i,j} = d^2 / dtheta_i dtheta_j`

        The Hessian is returned as a 4-entry tuple reflecting its structure as a 2x2 matrix:
        (hessian_yy, hessian_xy, hessian_yx, hessian_xx).

        Two computational paths are available, selected via the `xp` parameter:

        - **NumPy** (``xp=np``, default): finite-difference approximation. Deflection angles are
          evaluated at four shifted positions around each grid coordinate (±y, ±x) and the
          central difference is taken. JAX is not imported.

        - **JAX** (``xp=jnp``): exact derivatives via ``jax.jacfwd`` applied to
          ``deflections_yx_scalar``, vectorised over the grid with ``jnp.vectorize``.

        Both paths support uniform ``Grid2D`` and irregular ``Grid2DIrregular`` grids.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the Hessian is computed on.
        xp
            The array module (``numpy`` or ``jax.numpy``). Controls which computational path is
            used and the type of the returned arrays.
        """
        if xp is np:
            return self._hessian_via_finite_difference(grid=grid)
        return self._hessian_via_jax(grid=grid, xp=xp)

    def _hessian_via_jax(self, grid, xp) -> Tuple:
        import jax
        import jax.numpy as jnp

        pixel_scales = getattr(grid, "pixel_scales", (0.05, 0.05))

        y = jnp.array(grid[:, 0])
        x = jnp.array(grid[:, 1])

        def _hessian_single(y_scalar, x_scalar):
            return jnp.stack(
                jax.jacfwd(self.deflections_yx_scalar, argnums=(0, 1))(
                    y_scalar, x_scalar, pixel_scales
                )
            )

        h = jnp.vectorize(_hessian_single, signature="(),()->(i,i)")(y, x)

        # h has shape (N, 2, 2):
        #   h[..., 0, 0] = d(defl_y)/dy  = hessian_yy
        #   h[..., 0, 1] = d(defl_x)/dy  = hessian_xy
        #   h[..., 1, 0] = d(defl_y)/dx  = hessian_yx
        #   h[..., 1, 1] = d(defl_x)/dx  = hessian_xx
        return (
            xp.array(h[..., 0, 0]),
            xp.array(h[..., 0, 1]),
            xp.array(h[..., 1, 0]),
            xp.array(h[..., 1, 1]),
        )

    def _hessian_via_finite_difference(self, grid, buffer: float = 0.01) -> Tuple:
        grid_shift_y_up = aa.Grid2DIrregular(
            values=np.stack([grid[:, 0] + buffer, grid[:, 1]], axis=1)
        )
        grid_shift_y_down = aa.Grid2DIrregular(
            values=np.stack([grid[:, 0] - buffer, grid[:, 1]], axis=1)
        )
        grid_shift_x_left = aa.Grid2DIrregular(
            values=np.stack([grid[:, 0], grid[:, 1] - buffer], axis=1)
        )
        grid_shift_x_right = aa.Grid2DIrregular(
            values=np.stack([grid[:, 0], grid[:, 1] + buffer], axis=1)
        )

        deflections_up = self.deflections_yx_2d_from(grid=grid_shift_y_up)
        deflections_down = self.deflections_yx_2d_from(grid=grid_shift_y_down)
        deflections_left = self.deflections_yx_2d_from(grid=grid_shift_x_left)
        deflections_right = self.deflections_yx_2d_from(grid=grid_shift_x_right)

        hessian_yy = 0.5 * (deflections_up[:, 0] - deflections_down[:, 0]) / buffer
        hessian_xy = 0.5 * (deflections_up[:, 1] - deflections_down[:, 1]) / buffer
        hessian_yx = 0.5 * (deflections_right[:, 0] - deflections_left[:, 0]) / buffer
        hessian_xx = 0.5 * (deflections_right[:, 1] - deflections_left[:, 1]) / buffer

        return hessian_yy, hessian_xy, hessian_yx, hessian_xx

    def jacobian_from(self, grid, xp=np) -> List:
        """
        Returns the lensing Jacobian of the lensing object as a 2x2 list of lists.

        The Jacobian is the matrix `A = I - H`, where `H` is the Hessian matrix of the
        deflection angles:

        ``A = [[1 - hessian_xx, -hessian_xy], [-hessian_yx, 1 - hessian_yy]]``

        It is computed from `hessian_from`, so it supports both uniform and irregular
        grids and accepts the same `xp` parameter for JAX acceleration.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the Jacobian is computed on.
        xp
            The array module (``numpy`` or ``jax.numpy``). Passed through to
            ``hessian_from``.
        """
        hessian_yy, hessian_xy, hessian_yx, hessian_xx = self.hessian_from(
            grid=grid, xp=xp
        )

        a11 = 1 - hessian_xx
        a12 = -hessian_xy
        a21 = -hessian_yx
        a22 = 1 - hessian_yy

        return [[a11, a12], [a21, a22]]

    def convergence_2d_via_hessian_from(self, grid, xp=np) -> aa.ArrayIrregular:
        """
        Returns the convergence of the lensing object, which is computed from the 2D deflection angle map via the
        Hessian using the expression (see equation 56 https://inspirehep.net/literature/419263):

        `convergence = 0.5 * (hessian_{0,0} + hessian_{1,1}) = 0.5 * (hessian_xx + hessian_yy)`

        By going via the Hessian, the convergence can be calculated at any (y,x) coordinate therefore using either a
        2D uniform or irregular grid.

        This calculation of the convergence is independent of analytic calculations defined within `MassProfile` objects
        and can therefore be used as a cross-check.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and Hessian are computed on.
        xp
            The array module to use for the computation (e.g. `numpy` or `jax.numpy`). Passed through to
            `hessian_from`. When `xp` is not `numpy` (e.g. inside a `jax.jit` trace) the result is returned
            as a raw array rather than an `aa.ArrayIrregular` wrapper.
        """
        hessian_yy, hessian_xy, hessian_yx, hessian_xx = self.hessian_from(
            grid=grid, xp=xp
        )

        convergence = 0.5 * (hessian_yy + hessian_xx)

        if xp is np:
            return aa.ArrayIrregular(values=convergence)
        return convergence

    def shear_yx_2d_via_hessian_from(self, grid, xp=np) -> ShearYX2DIrregular:
        """
        Returns the 2D (y,x) shear vectors of the lensing object, which are computed from the 2D deflection angle map
        via the Hessian using the expressions (see equation 57 https://inspirehep.net/literature/419263):

        `shear_y = hessian_{1,0} =  hessian_{0,1} = hessian_yx = hessian_xy`
        `shear_x = 0.5 * (hessian_{0,0} - hessian_{1,1}) = 0.5 * (hessian_xx - hessian_yy)`

        By going via the Hessian, the shear vectors can be calculated at any (y,x) coordinate, therefore using either a
        2D uniform or irregular grid.

        This calculation of the shear vectors is independent of analytic calculations defined within `MassProfile`
        objects and can therefore be used as a cross-check.

        The result is returned as a `ShearYX2D` dats structure, which has shape [total_shear_vectors, 2], where
        entries for [:,0] are the gamma_2 values and entries for [:,1] are the gamma_1 values.

        Note therefore that this convention means the FIRST entries in the array are the gamma_2 values and the SECOND
        entries are the gamma_1 values.

        Parameters
        ----------
        grids
            The 2D grid of (y,x) arc-second coordinates the deflection angles and Hessian are computed on.
        xp
            The array module to use for the computation (e.g. `numpy` or `jax.numpy`). Passed through to
            `hessian_from`. When `xp` is not `numpy` (e.g. inside a `jax.jit` trace) the result is returned
            as a raw array of shape `(N, 2)` rather than a `ShearYX2DIrregular` wrapper.
        """

        hessian_yy, hessian_xy, hessian_yx, hessian_xx = self.hessian_from(
            grid=grid, xp=xp
        )

        gamma_1 = 0.5 * (hessian_xx - hessian_yy)
        gamma_2 = hessian_xy

        shear_yx_2d = xp.stack([gamma_2, gamma_1], axis=-1)

        if xp is np:
            return ShearYX2DIrregular(values=shear_yx_2d, grid=grid)
        return shear_yx_2d

    def magnification_2d_via_hessian_from(self, grid, xp=np) -> aa.ArrayIrregular:
        """
        Returns the 2D magnification map of lensing object, which is computed from the 2D deflection angle map
        via the Hessian using the expressions (see equation 60 https://inspirehep.net/literature/419263):

        `magnification = 1.0 / det(Jacobian) = 1.0 / abs((1.0 - convergence)**2.0 - shear**2.0)`
        `magnification = (1.0 - hessian_{0,0}) * (1.0 - hessian_{1, 1)) - hessian_{0,1}*hessian_{1,0}`
        `magnification = (1.0 - hessian_xx) * (1.0 - hessian_yy)) - hessian_xy*hessian_yx`

        By going via the Hessian, the magnification can be calculated at any (y,x) coordinate, therefore using either a
        2D uniform or irregular grid.

        This calculation of the magnification is independent of calculations using the Jacobian and can therefore be
        used as a cross-check.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and magnification map are computed on.
        """
        hessian_yy, hessian_xy, hessian_yx, hessian_xx = self.hessian_from(
            grid=grid, xp=xp
        )

        det_A = (1 - hessian_xx) * (1 - hessian_yy) - hessian_xy * hessian_yx

        if xp is np:
            return aa.ArrayIrregular(values=1.0 / det_A)
        return 1.0 / det_A

    def contour_list_from(self, grid, contour_array):
        grid_contour = aa.Grid2DContour(
            grid=grid,
            pixel_scales=grid.pixel_scales,
            shape_native=grid.shape_native,
            contour_array=contour_array.native,
        )

        return grid_contour.contour_list

    @evaluation_grid
    def tangential_critical_curve_list_from(
        self, grid, pixel_scale: Union[Tuple[float, float], float] = 0.05
    ) -> List[aa.Grid2DIrregular]:
        """
        Returns all tangential critical curves of the lensing system, which are computed as follows:

        1) Compute the tangential eigen values for every coordinate on the input grid via the Jacobian.
        2) Find contours of all values in the tangential eigen values that are zero using a marching squares algorithm.

        Due to the use of a marching squares algorithm that requires the zero values of the tangential eigen values to
        be computed, critical curves can only be calculated using the Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and tangential eigen values are computed
            on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            critical curve to be computed more accurately using a higher resolution grid.
        """
        tangential_eigen_values = self.tangential_eigen_value_from(grid=grid)

        return self.contour_list_from(grid=grid, contour_array=tangential_eigen_values)

    @evaluation_grid
    def radial_critical_curve_list_from(
        self, grid, pixel_scale: Union[Tuple[float, float], float] = 0.05
    ) -> List[aa.Grid2DIrregular]:
        """
        Returns all radial critical curves of the lensing system, which are computed as follows:

        1) Compute the radial eigen values for every coordinate on the input grid via the Jacobian.
        2) Find contours of all values in the radial eigen values that are zero using a marching squares algorithm.

        Due to the use of a marching squares algorithm that requires the zero values of the radial eigen values to
        be computed, this critical curves can only be calculated using the Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and radial eigen values are computed
            on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            critical curve to be computed more accurately using a higher resolution grid.
        """
        radial_eigen_values = self.radial_eigen_value_from(grid=grid)

        return self.contour_list_from(grid=grid, contour_array=radial_eigen_values)

    @evaluation_grid
    def tangential_caustic_list_from(
        self, grid, pixel_scale: Union[Tuple[float, float], float] = 0.05
    ) -> List[aa.Grid2DIrregular]:
        """
        Returns all tangential caustics of the lensing system, which are computed as follows:

        1) Compute the tangential eigen values for every coordinate on the input grid via the Jacobian.
        2) Find contours of all values in the tangential eigen values that are zero using a marching squares algorithm.
        3) Compute the lensing system's deflection angles at the (y,x) coordinates of the tangential critical curve
           contours and ray-trace it to the source-plane, therefore forming the tangential caustics.

        Due to the use of a marching squares algorithm that requires the zero values of the tangential eigen values to
        be computed, caustics can only be calculated using the Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and tangential eigen values are computed
            on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """

        tangential_critical_curve_list = self.tangential_critical_curve_list_from(
            grid=grid, pixel_scale=pixel_scale
        )

        tangential_caustic_list = []

        for tangential_critical_curve in tangential_critical_curve_list:
            deflections_critical_curve = self.deflections_yx_2d_from(
                grid=tangential_critical_curve
            )

            tangential_caustic_list.append(
                tangential_critical_curve - deflections_critical_curve
            )

        return tangential_caustic_list

    @evaluation_grid
    def radial_caustic_list_from(
        self, grid, pixel_scale: Union[Tuple[float, float], float] = 0.05
    ) -> List[aa.Grid2DIrregular]:
        """
        Returns all radial caustics of the lensing system, which are computed as follows:

        1) Compute the radial eigen values for every coordinate on the input grid via the Jacobian.
        2) Find contours of all values in the radial eigen values that are zero using a marching squares algorithm.
        3) Compute the lensing system's deflection angles at the (y,x) coordinates of the radial critical curve
           contours and ray-trace it to the source-plane, therefore forming the radial caustics.

        Due to the use of a marching squares algorithm that requires the zero values of the radial eigen values to
        be computed, this caustics can only be calculated using the Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and radial eigen values are computed
            on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """

        radial_critical_curve_list = self.radial_critical_curve_list_from(
            grid=grid, pixel_scale=pixel_scale
        )

        radial_caustic_list = []

        for radial_critical_curve in radial_critical_curve_list:
            deflections_critical_curve = self.deflections_yx_2d_from(
                grid=radial_critical_curve
            )

            radial_caustic_list.append(
                radial_critical_curve - deflections_critical_curve
            )

        return radial_caustic_list

    @evaluation_grid
    def radial_critical_curve_area_list_from(
        self, grid, pixel_scale: Union[Tuple[float, float], float]
    ) -> List[float]:
        """
        Returns the surface area within each radial critical curve as a list, the calculation of which is described in
        the function `radial_critical_curve_list_from()`.

        The area is computed via a line integral.

        Due to the use of a marching squares algorithm to estimate the critical curve, this function can only use the
        Jacobian and a uniform 2D grid.


        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles used to calculate the radial critical
            curve are computed on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """
        radial_critical_curve_list = self.radial_critical_curve_list_from(
            grid=grid, pixel_scale=pixel_scale
        )

        return self.area_within_curve_list_from(curve_list=radial_critical_curve_list)

    @evaluation_grid
    def tangential_critical_curve_area_list_from(
        self, grid, pixel_scale: Union[Tuple[float, float], float] = 0.05
    ) -> List[float]:
        """
        Returns the surface area within each tangential critical curve as a list, the calculation of which is
        described in the function `tangential_critical_curve_list_from()`.

        The area is computed via a line integral.

        Due to the use of a marching squares algorithm to estimate the critical curve, this function can only use the
        Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles used to calculate the tangential critical
            curve are computed on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """
        tangential_critical_curve_list = self.tangential_critical_curve_list_from(
            grid=grid, pixel_scale=pixel_scale
        )

        return self.area_within_curve_list_from(
            curve_list=tangential_critical_curve_list
        )

    def area_within_curve_list_from(
        self, curve_list: List[aa.Grid2DIrregular]
    ) -> List[float]:
        area_within_each_curve_list = []

        for curve in curve_list:
            x, y = curve[:, 0], curve[:, 1]
            area = np.abs(0.5 * np.sum(y[:-1] * np.diff(x) - x[:-1] * np.diff(y)))
            area_within_each_curve_list.append(area)

        return area_within_each_curve_list

    @evaluation_grid
    def einstein_radius_list_from(
        self, grid, pixel_scale: Union[Tuple[float, float], float] = 0.05
    ):
        """
        Returns a list of the Einstein radii corresponding to the area within each tangential critical curve.

        Each Einstein radius is defined as the radius of the circle which contains the same area as the area within
        each tangential critical curve.

        This definition is sometimes referred to as the "effective Einstein radius" in the literature and is commonly
        adopted in studies, for example the SLACS series of papers.

        The calculation of the tangential critical curves and their areas is described in the functions
         `tangential_critical_curve_list_from()` and `tangential_critical_curve_area_list_from()`.

        Due to the use of a marching squares algorithm to estimate the critical curve, this function can only use the
        Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles used to calculate the tangential critical
            curve are computed on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """
        try:
            area_list = self.tangential_critical_curve_area_list_from(
                grid=grid, pixel_scale=pixel_scale
            )
            return [np.sqrt(area / np.pi) for area in area_list]
        except TypeError:
            raise TypeError("The grid input was unable to estimate the Einstein Radius")

    @evaluation_grid
    def einstein_radius_from(
        self, grid, pixel_scale: Union[Tuple[float, float], float] = 0.05
    ):
        """
        Returns the Einstein radius corresponding to the area within the tangential critical curve.

        The Einstein radius is defined as the radius of the circle which contains the same area as the area within
        the tangential critical curve.

        This definition is sometimes referred to as the "effective Einstein radius" in the literature and is commonly
        adopted in studies, for example the SLACS series of papers.

        If there are multiple tangential critical curves (e.g. because the mass distribution is complex) this function
        raises an error, and the function `einstein_radius_list_from()` should be used instead.

        The calculation of the tangential critical curves and their areas is described in the functions
         `tangential_critical_curve_list_from()` and `tangential_critical_curve_area_list_from()`.

        Due to the use of a marching squares algorithm to estimate the critical curve, this function can only use the
        Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles used to calculate the tangential
            critical curve are computed on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """

        einstein_radii_list = self.einstein_radius_list_from(grid=grid)

        if len(einstein_radii_list) > 1:
            logger.info(
                """
                There are multiple tangential critical curves, and the computed Einstein radius is the sum of 
                all of them. Check the `einstein_radius_list_from` function for the individual Einstein. 
            """
            )

        return sum(einstein_radii_list)

    @evaluation_grid
    def einstein_mass_angular_list_from(
        self, grid, pixel_scale: Union[Tuple[float, float], float] = 0.05
    ) -> List[float]:
        """
        Returns a list of the angular Einstein massses corresponding to the area within each tangential critical curve.

        The angular Einstein mass is defined as: `einstein_mass = pi * einstein_radius ** 2.0` where the Einstein
        radius is the radius of the circle which contains the same area as the area within the tangential critical
        curve.

        The Einstein mass is returned in units of arcsecond**2.0 and requires division by the lensing critical surface
        density \sigma_cr to be converted to physical units like solar masses (see `autogalaxy.util.cosmology_util`).

        This definition of Eisntein radius (and therefore mass) is sometimes referred to as the "effective Einstein
        radius" in the literature and is commonly adopted in studies, for example the SLACS series of papers.

        The calculation of the einstein radius is described in the function `einstein_radius_from()`.

        Due to the use of a marching squares algorithm to estimate the critical curve, this function can only use the
        Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles used to calculate the tangential critical
            curve are computed on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """
        einstein_radius_list = self.einstein_radius_list_from(
            grid=grid, pixel_scale=pixel_scale
        )
        return [np.pi * einstein_radius**2 for einstein_radius in einstein_radius_list]

    @evaluation_grid
    def einstein_mass_angular_from(
        self, grid, pixel_scale: Union[Tuple[float, float], float] = 0.05
    ) -> float:
        """
        Returns the Einstein radius corresponding to the area within the tangential critical curve.

        The angular Einstein mass is defined as: `einstein_mass = pi * einstein_radius ** 2.0` where the Einstein
        radius is the radius of the circle which contains the same area as the area within the tangential critical
        curve.

        The Einstein mass is returned in units of arcsecond**2.0 and requires division by the lensing critical surface
        density \sigma_cr to be converted to physical units like solar masses (see `autogalaxy.util.cosmology_util`).

        This definition of Eisntein radius (and therefore mass) is sometimes referred to as the "effective Einstein
        radius" in the literature and is commonly adopted in studies, for example the SLACS series of papers.

        The calculation of the einstein radius is described in the function `einstein_radius_from()`.

        Due to the use of a marching squares algorithm to estimate the critical curve, this function can only use the
        Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles used to calculate the tangential critical
            curve are computed on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """
        einstein_mass_angular_list = self.einstein_mass_angular_list_from(
            grid=grid, pixel_scale=pixel_scale
        )

        if len(einstein_mass_angular_list) > 1:
            logger.info(
                """
                There are multiple tangential critical curves, and the computed Einstein mass is the sum of 
                all of them. Check the `einstein_mass_list_from` function for the individual Einstein. 
            """
            )

        return einstein_mass_angular_list[0]
