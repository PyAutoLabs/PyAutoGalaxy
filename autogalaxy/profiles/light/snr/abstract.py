import numpy as np
from typing import Optional

import autoarray as aa


class LightProfileSNR:
    def __init__(self, signal_to_noise_ratio: float = 10.0):
        """
        This light profile class sets the `intensity` of the light profile using input noise properties of a simulation
        (e.g. using the `exposure_time`, `background_sky_level`).

        This means that the intensities of the light profiles can be automatically adjusted when an `SimulatorImaging`
        object is used to simulate imaging data, whereby the intensity of each light profile is set to produce an
        image with the input `signal_to_noise_ratio` of this class.

        The brightest pixel of the image of the light profile is used to do this, thus the S/N in all other pixels
        away from the brightest pixel will be below the input `signal_to_noise_ratio`.

        The intensity is set using an input grid, meaning that for strong lensing calculations the ray-traced grid
        can be used such that the S/N accounts for the magnification of a source galaxy.

        Parameters
        ----------
        signal_to_noise_ratio
            The signal-to-noises ratio that the simulated light profile will produce.
        """
        self.signal_to_noise_ratio = signal_to_noise_ratio

    def image_2d_from(
        self,
        grid: aa.type.Grid2DLike,
        xp=np,
        operated_only: Optional[bool] = None,
        **kwargs,
    ) -> aa.Array2D:
        """
        Abstract method for obtaining intensity at a grid of Cartesian (y,x) coordinates.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the `LightProfile` evaluated at every (y,x) coordinate on the grid.
        """
        raise NotImplementedError()

    def set_intensity_from(
        self,
        grid: aa.type.Grid2DLike,
        exposure_time: float,
        background_sky_level: float = 0.0,
        psf: Optional[aa.Convolver] = None,
    ):
        """
        Set the `intensity` of the light profile as follows:

        - Evaluate the image of the light profile on an input grid.
        - Blur this image with a PSF, if included.
        - Take the value of the brightest pixel.
        - Use an input `exposure_time` and `background_sky` (e.g. from the `SimulatorImaging` object) to determine
        what value of `intensity` gives the desired signal to noise ratio for the image.

        The intensity is set using an input grid, meaning that for strong lensing calculations the ray-traced grid
        can be used such that the S/N accounts for the magnification of a source galaxy.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the original reference frame of the grid.
        exposure_time
            The exposure time of the simulated imaging.
        background_sky_level
            The level of the background sky of the simulated imaging.
        psf
            The psf of the simulated imaging which can change the S/N of the light profile due to spreading out
            the emission.
        """
        from scipy.optimize import root_scalar

        self.intensity = 1.0

        background_sky_level_counts = background_sky_level * exposure_time

        image_2d = self.image_2d_from(grid=grid)
        if psf is not None:
            image_2d = psf.convolved_image_from(image=image_2d, blurring_image=None)

        brightest_value = np.max(image_2d)

        def func(intensity_factor):
            signal = intensity_factor * brightest_value * exposure_time
            noise = np.sqrt(signal + background_sky_level_counts)

            signal_to_noise_ratio = signal / noise

            return signal_to_noise_ratio - self.signal_to_noise_ratio

        intensity_factor = root_scalar(func, bracket=[1.0e-8, 1.0e8]).root

        self.intensity *= intensity_factor
