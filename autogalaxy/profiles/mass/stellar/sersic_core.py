import numpy as np
from typing import Tuple

import autoarray as aa
from autogalaxy.profiles.mass import MGEDecomposer

from autogalaxy.profiles.mass.stellar.sersic import Sersic


class SersicCore(Sersic):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
        radius_break: float = 0.01,
        intensity: float = 0.05,
        gamma: float = 0.25,
        alpha: float = 3.0,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The elliptical cored-Sersic light profile.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        radius_break
            The break radius separating the inner power-law (with logarithmic slope gamma) and outer Sersic function.
        intensity
            The intensity at the break radius.
        gamma
            The logarithmic power-law slope of the inner core profiles
        alpha :
            Controls the sharpness of the transition between the inner core / outer Sersic profiles.
        """

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
        )

        self.radius_break = radius_break
        self.intensity = intensity
        self.alpha = alpha
        self.gamma = gamma

    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        return self.deflections_2d_via_mge_from(grid=grid, xp=xp, **kwargs)

    def deflections_2d_via_mge_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        radii_min = self.effective_radius / 50.0
        radii_max = self.effective_radius * 20.0
        log_sigmas = xp.linspace(xp.log(radii_min), xp.log(radii_max), 20)
        sigmas = xp.exp(log_sigmas)

        mge_decomp = MGEDecomposer(mass_profile=self)

        deflections_via_mge = mge_decomp.deflections_2d_via_mge_from(
            grid=grid,
            xp=xp,
            sigma_log_list=sigmas,
            ellipticity_convention="circularised",
            three_D=False,
        )
        return deflections_via_mge

    def image_2d_via_radii_from(self, grid_radii: np.ndarray, xp=np):
        """
        Calculate the intensity of the cored-Sersic light profile on a grid of radial coordinates.

        Parameters
        ----------
        grid_radii
            The radial distance from the centre of the profile. for each coordinate on the grid.
        """
        return xp.multiply(
            xp.multiply(
                self.intensity_prime(xp),
                xp.power(
                    xp.add(
                        1,
                        xp.power(
                            xp.divide(self.radius_break, grid_radii.array), self.alpha
                        ),
                    ),
                    (self.gamma / self.alpha),
                ),
            ),
            xp.exp(
                xp.multiply(
                    -self.sersic_constant,
                    (
                        xp.power(
                            xp.divide(
                                xp.add(
                                    xp.power(grid_radii.array, self.alpha),
                                    (self.radius_break**self.alpha),
                                ),
                                (self.effective_radius**self.alpha),
                            ),
                            (1.0 / (self.alpha * self.sersic_index)),
                        )
                    ),
                )
            ),
        )

    def convergence_func(self, grid_radius: float, xp=np) -> float:
        return (
            self.mass_to_light_ratio
            * self.intensity_prime(xp=xp)
            * (1.0 + (self.radius_break / grid_radius.array) ** self.alpha)
            ** (self.gamma / self.alpha)
            * xp.exp(
                -self.sersic_constant
                * (
                    (grid_radius.array**self.alpha + self.radius_break**self.alpha)
                    / self.effective_radius**self.alpha
                )
                ** (1.0 / (self.sersic_index * self.alpha))
            )
        )

    def intensity_prime(self, xp=np):
        """Overall intensity normalisation in the rescaled Core-Sersic light profiles (electrons per second)"""
        return (
            self.intensity
            * (2.0 ** (-self.gamma / self.alpha))
            * xp.exp(
                self.sersic_constant
                * (
                    ((2.0 ** (1.0 / self.alpha)) * self.radius_break)
                    / self.effective_radius
                )
                ** (1.0 / self.sersic_index)
            )
        )


class SersicCoreSph(SersicCore):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
        radius_break: float = 0.01,
        intensity: float = 0.05,
        gamma: float = 0.25,
        alpha: float = 3.0,
    ):
        """
        The elliptical cored-Sersic light profile.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        radius_break
            The break radius separating the inner power-law (with logarithmic slope gamma) and outer Sersic function.
        intensity
            The intensity at the break radius.
        gamma
            The logarithmic power-law slope of the inner core profiles
        alpha :
            Controls the sharpness of the transition between the inner core / outer Sersic profiles.
        """
        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            radius_break=radius_break,
            intensity=intensity,
            gamma=gamma,
            alpha=alpha,
        )
        self.radius_break = radius_break
        self.intensity = intensity
        self.alpha = alpha
        self.gamma = gamma
