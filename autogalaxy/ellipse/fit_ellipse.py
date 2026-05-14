"""
Fit isophotal ellipses to a 2D image dataset.

`FitEllipse` uses a single `Ellipse` (plus optional `EllipseMultipole` perturbations) to fit an imaging
dataset.  Rather than convolving a model image with a PSF, the fit works by:

1. Computing the (y, x) sample points distributed around the ellipse's perimeter.
2. Interpolating the image data and noise-map at those sample points.
3. Computing residuals between the interpolated data values and the mean intensity along the ellipse.

This is the classical isophote-fitting approach used in tools such as IRAF/ELLIPSE and galfit, and is
appropriate for measuring galaxy morphology, position angles, axis ratios, and multipole perturbations
directly from imaging data without fitting a parametric light profile model.
"""
import numpy as np
from typing import List, Optional

from autoconf import cached_property

import autoarray as aa

from autogalaxy.ellipse.dataset_interp import DatasetInterp
from autogalaxy.ellipse.ellipse.ellipse import Ellipse
from autogalaxy.ellipse.ellipse.ellipse_multipole import EllipseMultipole


class FitEllipse(aa.FitDataset):
    def __init__(
        self,
        dataset: aa.Imaging,
        ellipse: Ellipse,
        multipole_list: Optional[List[EllipseMultipole]] = None,
        use_jax: bool = False,
    ):
        """
        A fit to a `DatasetInterp` dataset, using a model image to represent the observed data and noise-map.

        Parameters
        ----------
        dataset
            The dataset containing the signal and noise-map that is fitted.
        ellipse
            The ellipse profile used to fit the dataset.
        multipole_list
            Optional list of multipole perturbations applied to the ellipse points.
        use_jax
            If True, internal arrays are computed with JAX and downstream
            reductions return raw JAX arrays. The algorithm is identical to
            the NumPy path — both use a single perimeter-sampling pass with
            NaN-marking for masked positions.

        """
        super().__init__(dataset=dataset)

        self.ellipse = ellipse
        self.multipole_list = multipole_list
        self.use_jax = use_jax

    @property
    def _xp(self):
        """``jax.numpy`` if ``self.use_jax`` else ``numpy``."""
        if self.use_jax:
            import jax.numpy as jnp
            return jnp
        return np

    @cached_property
    def interp(self) -> DatasetInterp:
        """
        Returns a class which handles the interpolation of values from the image data and noise-map, so that they
        can be mapped to each ellipse for the fit.
        """
        return DatasetInterp(dataset=self.dataset)

    def points_from_major_axis_from(self) -> "np.ndarray | jax.Array":
        """
        Return the ``(N, 2)`` (y, x) coordinates on the ellipse perimeter where
        interpolation occurs.

        Sampling proceeds in one shot:

        1. ``N = Ellipse.total_points_from(pixel_scale)`` perimeter angles are
           generated.
        2. Multipole perturbations (if any) are applied in-place.
        3. The mask interpolator is evaluated at every point; positions that
           sit on (or partly on) masked pixels are marked as NaN via
           ``xp.where(keep, points, nan)``.

        The output shape is fixed at ``(N, 2)`` regardless of mask geometry —
        masked rows are NaN, unmasked rows hold their (y, x) coordinates.
        Downstream reductions (``residual_map``, ``chi_squared``,
        ``noise_normalization``) use ``nanmean`` / ``nansum`` and skip NaN
        positions correctly.

        The algorithm is identical for ``self.use_jax`` ``False`` and ``True``;
        only the numerics backend (``np`` vs ``jax.numpy``) differs.

        Returns
        -------
        The ``(N, 2)`` (y, x) coordinates with NaN at masked positions.
        """
        xp = self._xp
        pixel_scale = self.dataset.pixel_scales[0]

        points = self.ellipse.points_from_major_axis_from(
            pixel_scale=pixel_scale, xp=xp
        )

        if self.multipole_list is not None:
            for multipole in self.multipole_list:
                points = multipole.points_perturbed_from(
                    pixel_scale=pixel_scale,
                    points=points,
                    ellipse=self.ellipse,
                    xp=xp,
                )

        mask_values = self.interp.mask_interp(points, xp=xp)
        keep = mask_values == 0
        return xp.where(keep[:, None], points, xp.nan)

    @cached_property
    def _points_from_major_axis(self) -> np.ndarray:
        """
        Returns cached (y,x) coordinates on the ellipse that are used to interpolate the data and noise-map values.

        Returns
        -------
        The (y,x) coordinates on the ellipse where the interpolation occurs.
        """
        return self.points_from_major_axis_from()

    @property
    def data_interp(self) -> aa.ArrayIrregular:
        """
        Returns the data values of the dataset that the ellipse fits, which are computed by overlaying the ellipse over
        the 2D data and performing a 2D interpolation at discrete (y,x) coordinates on the ellipse.

        The (y,x) coordinates on the ellipse where the interpolation occurs are computed in the
        `points_from_major_axis` property of the `Ellipse` class, with the documentation describing how these points
        are computed.

        If the interpolation of an ellipse point uses one or more masked values, this point is not reliable, therefore
        the value is converted to `np.nan` and not used by other fitting quantities.

        Returns
        -------
        The data values of the ellipse fits, computed via a 2D interpolation of where the ellipse
        overlaps the data.
        """
        xp = self._xp
        data = self.interp.data_interp(self._points_from_major_axis, xp=xp)

        if xp is np:
            return aa.ArrayIrregular(values=data)
        return data

    @property
    def noise_map_interp(self) -> aa.ArrayIrregular:
        """
        Returns the noise-map values of the dataset that the ellipse fits, which are computed by overlaying the ellipse
        over the 2D noise-map and performing a 2D interpolation at discrete (y,x) coordinates on the ellipse.

        The (y,x) coordinates on the ellipse where the interpolation occurs are computed in the
        `points_from_major_axis` property of the `Ellipse` class, with the documentation describing how these points
        are computed.

        If the interpolation of an ellipse point uses one or more masked values, this point is not reliable, therefore
        the value is converted to `np.nan` and not used by other fitting quantities.

        Returns
        -------
        The noise-map values of the ellipse fits, computed via a 2D interpolation of where the ellipse
        overlaps the noise-map.
        """
        xp = self._xp
        noise_map = self.interp.noise_map_interp(self._points_from_major_axis, xp=xp)

        if xp is np:
            return aa.ArrayIrregular(values=noise_map)
        return noise_map

    @property
    def signal_to_noise_map_interp(self) -> aa.ArrayIrregular:
        """
        Returns the signal-to-noise-map of the dataset that the ellipse fits, which is computed by overlaying the ellipse
        over the 2D data and noise-map and performing a 2D interpolation at discrete (y,x) coordinates on the ellipse.

        Returns
        -------
        The signal-to-noise-map values of the ellipse fits, computed via a 2D interpolation of where
        the ellipse overlaps the data and noise-map.
        """
        xp = self._xp
        data = self.interp.data_interp(self._points_from_major_axis, xp=xp)
        noise_map = self.interp.noise_map_interp(self._points_from_major_axis, xp=xp)
        values = data / noise_map

        if xp is np:
            return aa.ArrayIrregular(values=values)
        return values

    @property
    def model_data(self) -> aa.ArrayIrregular:
        """
        Returns the model-data, which is the data values where the ellipse overlaps the data minus the mean
        value of these data values.

        By subtracting the mean of the data values from each data value, the model data quantifies how often there
        are large variations in the data values over the ellipse.

        For example, if every data value subtended by the ellipse are close to one another, the difference between
        the data values and the mean will be small.

        Conversely, if some data values are much higher or lower than the mean, the model data will be large.

        Returns
        -------
        The model data values of the ellipse fit, which are the data values minus the mean of the data values.
        """
        return self.data_interp

    @property
    def residual_map(self) -> aa.ArrayIrregular:
        """
        Returns the residual-map of the fit, which is the data minus the model data and therefore the same
        as the model data.

        Returns
        -------
        The residual-map of the fit, which is the data minus the model data and therefore the same as the model data.
        """
        xp = self._xp

        if xp is np:
            if not self.model_data:
                return aa.ArrayIrregular(values=xp.zeros(self.model_data.shape))

        values = self.model_data - xp.nanmean(self.model_data)

        if xp is np:
            return aa.ArrayIrregular(values=values)
        return values

    @property
    def normalized_residual_map(self) -> aa.ArrayIrregular:
        """
        Returns the normalized residual-map of the fit, which is the residual-map divided by the noise-map.

        The residual map and noise map are computed by overlaying the ellipse over the 2D data and noise-map and
        performing a 2D interpolation at discrete (y,x) coordinates on the ellipse. See the documentation of the
        `residual_map` and `noise_map` properties for more details.

        Returns
        -------
        The normalized residual-map of the fit, which is the residual-map divided by the noise-map.
        """
        xp = self._xp
        values = self.residual_map / self.noise_map_interp

        if xp is np:
            return aa.ArrayIrregular(values=values)
        return values

    @property
    def chi_squared_map(self) -> aa.ArrayIrregular:
        """
        Returns the chi-squared-map of the fit, which is the normalized residual-map squared.

        The normalized residual map is computed by overlaying the ellipse over the 2D data and noise-map and
        performing a 2D interpolation at discrete (y,x) coordinates on the ellipse. See the documentation of the
        `normalized_residual_map` property for more details.

        Returns
        -------
        The chi-squared-map of the fit, which is the normalized residual-map squared.
        """
        xp = self._xp
        values = self.normalized_residual_map**2.0

        if xp is np:
            return aa.ArrayIrregular(values=values)
        return values

    @property
    def chi_squared(self) -> float:
        """
        The sum of the chi-squared-map, which quantifies how well the model data represents the data and noise-map.

        The chi-squared-map is computed by overlaying the ellipse over the 2D data and noise-map and
        performing a 2D interpolation at discrete (y,x) coordinates on the ellipse. See the documentation of the
        `chi_squared_map` property for more details.

        Returns
        -------
        The chi-squared of the fit.
        """
        xp = self._xp
        return xp.nansum(self.chi_squared_map)

    @property
    def noise_normalization(self):
        """
        The noise normalization term of the log likelihood, which is the sum of the log noise-map values squared.

        Returns
        -------
        The noise normalization term of the log likelihood.
        """
        xp = self._xp
        noise_map = self.interp.noise_map_interp(self._points_from_major_axis, xp=xp)
        return xp.nansum(xp.log(2 * xp.pi * noise_map**2.0))

    @property
    def log_likelihood(self):
        """
        The log likelihood of the fit, which quantifies how well the model data represents the data and noise-map.

        Returns
        -------
        The log likelihood of the fit.
        """
        return -0.5 * (self.chi_squared)

    @property
    def figure_of_merit(self) -> float:
        """
        The figure of merit of the fit, which is passed by the `Analysis` class to the non-linear search to
        determine the best-fit solution.

        Returns
        -------
        The figure of merit of the fit.
        """
        return -0.5 * (self.chi_squared + self.noise_normalization)
