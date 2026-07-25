"""
Abstract base classes for all light profiles in **PyAutoGalaxy**.

A light profile has an analytic function that describes the surface brightness of a galaxy as a function of
(y,x) Cartesian coordinates. Each profile is associated with a spatial geometry (centre and ellipticity) and
an `intensity` normalisation that scales the overall brightness.

The `LightProfile` class is the root of the light profile hierarchy. All concrete profiles (e.g. `Sersic`,
`Exponential`, `Gaussian`) inherit from it and implement `image_2d_from` and `image_2d_via_radii_from`.
"""
import numpy as np
from typing import Optional, Tuple

import autoarray as aa

from autogalaxy.operate.image import OperateImage
from autogalaxy.profiles.geometry_profiles import EllProfile


class LightProfile(EllProfile, OperateImage):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
    ):
        """
        Abstract base class for an elliptical light-profile.

        Each light profile has an analytic equation associated with it that describes its 1D surface brightness.

        Given an input grid of 1D or 2D (y,x) coordinates the light profile can be used to evaluate its surface
        brightness in 1D or as a 2D image.

        Associated with a light profile is a spherical or elliptical geometry, which describes its `centre` of
        emission and ellipticity. Geometric transformations are performed by decorators linked to the **PyAutoArray**
        `geometry` package.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system (see the module
            `autogalaxy -> convert.py` for the convention).
        """
        super().__init__(centre=centre, ell_comps=ell_comps)
        self.intensity = intensity

    @property
    def coefficient_tag(self) -> str:
        """
        A short string tag used to label the intensity coefficient when this profile is used inside a `Basis`
        object (e.g. for multi-Gaussian expansion or shapelet decomposition).

        Returns an empty string for standard light profiles, and is overridden by linear light profile subclasses
        to label their solved-for intensity coefficient.
        """
        return ""

    def image_2d_from(
        self,
        grid: aa.type.Grid2DLike,
        xp=np,
        operated_only: Optional[bool] = None,
        **kwargs,
    ) -> aa.Array2D:
        """
        Returns the light profile's 2D image from a 2D grid of Cartesian (y,x) coordinates, which may have been
        transformed using the light profile's geometry.

        If the coordinates have not been transformed to the profile's geometry (e.g. translated to the
        profile `centre`), this is performed automatically.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the `LightProfile` evaluated at every (y,x) coordinate on the transformed grid.
        """
        raise NotImplementedError()

    def image_2d_via_radii_from(self, grid_radii: np.ndarray, xp=np) -> np.ndarray:
        """
        Returns the light profile's 2D image from a 1D grid of coordinates which are the radial distance of each
        coordinate from the light profile `centre`.

        Parameters
        ----------
        grid_radii
            The radial distances from the centre of the profile, for each coordinate on the grid.
        """
        raise NotImplementedError()

    def luminosity_within_circle_from(self, radius: float) -> float:
        """
        Integrate the light profile to compute the total luminosity within a circle of specified radius. This is
        centred on the light profile's `centre`.

        The `intensity` of a light profile is in dimension units, which are given physical meaning when the light
        profile is compared to data with physical units. The luminosity output by this function therefore is also
        dimensionless until compared to data.

        Parameters
        ----------
        radius
            The radius of the circle to compute the dimensionless luminosity within.
        """
        from scipy.integrate import quad

        return quad(func=self.luminosity_integral, a=0.0, b=radius)[0]

    def luminosity_integral(self, x: np.ndarray) -> np.ndarray:
        """
        Routine to integrate the luminosity of an elliptical light profile.

        The axis ratio is set to 1.0 for computing the luminosity within a circle

        Parameters
        ----------
        x
            The 1D (x) radial coordinates where the luminosity integral is evaluated.
        """
        from autoarray.structures.arrays.irregular import ArrayIrregular

        return 2 * np.pi * x * self.image_2d_via_radii_from(ArrayIrregular(x))

    @property
    def half_light_radius(self) -> float:
        """
        The radius that contains half of the total light of the profile (the half-light radius).

        For profiles with an `effective_radius` attribute (e.g. Sersic, Exponential) this returns the
        `effective_radius`. Returns `None` for profiles that do not define an effective radius.
        """
        if hasattr(self, "effective_radius"):
            return self.effective_radius

    @property
    def _intensity(self):
        """
        The normalisation intensity of the light profile used internally when evaluating the image.

        For standard light profiles this simply returns `self.intensity`. Linear light profiles override this
        property to return 1.0, because their intensity is solved for via a linear inversion rather than being
        a free parameter.
        """
        return self.intensity
