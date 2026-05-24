"""
Extends the **PyAutoArray** `SimulatorInterferometer` class with galaxy-aware simulation.

`SimulatorInterferometer` (from `autoarray`) handles the low-level simulation: applying a Fourier
transform and adding visibility noise. This module adds a `via_galaxies_from` method that takes a list
of `Galaxy` objects and a 2D grid, evaluates the galaxy images, and passes them to the parent
simulation pipeline.
"""
import numpy as np
from typing import List

import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.galaxy.galaxies import Galaxies


class SimulatorInterferometer(aa.SimulatorInterferometer):
    def via_galaxies_from(self, galaxies: List[Galaxy], grid: aa.type.Grid2DLike, xp=None):
        """
        Returns a realistic simulated image by applying effects to a plain simulated image.

        Parameters
        ----------
        image
            The image before simulating (e.g. the lens and source galaxies before optics blurring and Imaging read-out).
        pixel_scales
            The scale of each pixel in arc seconds
        exposure_time_map
            An arrays representing the effective exposure time of each pixel.
        psf: PSF
            An arrays describing the PSF the simulated image is blurred with.
        add_poisson_noise_to_data: Bool
            If `True` poisson noise_maps is simulated and added to the image, based on the total counts in each image
            pixel
        noise_seed: int
            A seed for random noise_maps generation
        """

        if xp is None:
            xp = self._xp

        galaxies = Galaxies(galaxies=galaxies)

        image = galaxies.image_2d_from(grid=grid, xp=xp)

        return self.via_image_from(image=image, xp=xp)
