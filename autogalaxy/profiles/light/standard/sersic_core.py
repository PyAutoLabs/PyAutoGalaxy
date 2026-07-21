import numpy as np
from typing import Tuple

from autogalaxy.profiles.light.standard.sersic import Sersic


class SersicCore(Sersic):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
        radius_break: float = 0.025,
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
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
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
        alpha
            Controls the sharpness of the transition between the inner core / outer Sersic profiles.
        """

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )

        self.radius_break = radius_break
        self.intensity = intensity
        self.alpha = alpha
        self.gamma = gamma

    def intensity_prime(self, xp=np) -> float:
        """
        Overall intensity normalisation in the rescaled cored Sersic light profile.

        Like the `intensity` parameter, the units of `intensity_prime` are dimensionless and derived from the data
        the light profile's image is compared too, which are expected to be electrons per second.
        """
        return (
            self._intensity
            * (2.0 ** (-self.gamma / self.alpha))
            * xp.exp(
                self.sersic_constant
                * xp.divide(
                    (2.0 ** (1.0 / self.alpha)) * self.radius_break,
                    self.effective_radius,
                )
                ** (1.0 / self.sersic_index)
            )
        )

    def image_2d_via_radii_from(
        self, grid_radii: np.ndarray, xp=np, **kwargs
    ) -> np.ndarray:
        """
        Returns the 2D image of the Sersic light profile from a grid of coordinates which are the radial distances of
        each coordinate from the its `centre`.

        Parameters
        ----------
        grid_radii
            The radial distances from the centre of the profile, for each coordinate on the grid.
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
