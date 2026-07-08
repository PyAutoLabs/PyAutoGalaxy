"""
Extends the **PyAutoArray** `SimulatorImaging` class with galaxy-aware simulation methods.

`SimulatorImaging` (from `autoarray`) handles the low-level simulation pipeline: applying a PSF, adding
Poisson noise, and adding a background sky. This module adds a `via_galaxies_from` method that takes a
list of `Galaxy` objects and a 2D grid, evaluates the galaxy images, and passes them to the parent
simulation pipeline.

It also handles `LightProfileSNR` objects, automatically scaling each profile's `intensity` so that its
peak signal-to-noise ratio matches the requested value given the simulation exposure time and sky level.
"""
import numpy as np
from typing import List

import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.galaxy.galaxies import Galaxies


class SimulatorImaging(aa.SimulatorImaging):
    def via_galaxies_from(
        self, galaxies: List[Galaxy], grid: aa.type.Grid2DLike, xp=None
    ) -> aa.Imaging:
        """
        Simulate an `Imaging` dataset from an input list of `Galaxy` objects and a 2D grid of (y,x) coordinates.

        The light profiles of each galaxy are used to generate the image of the galaxies which is simulated.

        The steps of the `SimulatorImaging` simulation process (e.g. PSF convolution, noise addition) are
        described in the `SimulatorImaging` `__init__` method docstring, found in the PyAutoArray project.

        If one of more galaxy light profiles are a `LightProfileSNR` object, the `intensity` of the light profile is
        automatically set such that the signal-to-noise ratio of the light profile is equal to its input
        `signal_to_noise_ratio` value.

        For example, if a `LightProfileSNR` object has a `signal_to_noise_ratio` of 5.0, the intensity of the light
        profile is set such that the peak surface brightness of the profile is 5.0 times the background noise level of
        the image.

        Parameters
        ----------
        galaxies
            The galaxies whose light profiles are evaluated using the input 2D grid of (y,x) coordinates in order to
            generate the image of the galaxies which is then simulated.
        grid
            The 2D grid of (y,x) coordinates which the light profiles of the galaxies are evaluated using in order
            to generate the image of the galaxies.
        """

        if xp is None:
            xp = self._xp

        galaxies = Galaxies(galaxies=galaxies)

        for galaxy in galaxies:
            galaxy.set_snr_of_snr_light_profiles(
                grid=grid,
                exposure_time=self.exposure_time,
                background_sky_level=self.background_sky_level,
                psf=self.psf,
            )

        if self.psf.convolve_over_sample_size > 1:

            image = galaxies.convolved_padded_image_2d_from(
                grid=grid, psf=self.psf, xp=xp
            )

            over_sample_size = grid.over_sample_size.resized_from(
                new_shape=image.shape_native, mask_pad_value=1
            )

            dataset = self.via_image_from(
                image=image,
                over_sample_size=over_sample_size,
                image_is_convolved=True,
                xp=xp,
            )

            return dataset.trimmed_after_convolution_from(
                kernel_shape=self.psf.kernel_shape_image_resolution
            )

        image = galaxies.padded_image_2d_from(
            grid=grid, psf_shape_2d=self.psf.kernel.shape_native, xp=xp
        )

        over_sample_size = grid.over_sample_size.resized_from(
            new_shape=image.shape_native, mask_pad_value=1
        )

        dataset = self.via_image_from(image=image, over_sample_size=over_sample_size, xp=xp)

        return dataset.trimmed_after_convolution_from(
            kernel_shape=self.psf.kernel.shape_native
        )
