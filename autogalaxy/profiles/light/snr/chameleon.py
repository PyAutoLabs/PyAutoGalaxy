from typing import Tuple

from autogalaxy.profiles.light.snr.abstract import LightProfileSNR

from autogalaxy.profiles.light import standard as lp


class Chameleon(lp.Chameleon, LightProfileSNR):
    def __init__(
        self,
        signal_to_noise_ratio: float = 10.0,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.05,
    ):
        """
        The elliptical Chameleon light profile.

        Profile form:
            mass_to_light_ratio * intensity *\
                (1.0 / Sqrt(x^2 + (y/q)^2 + rc^2) - 1.0 / Sqrt(x^2 + (y/q)^2 + (rc + dr)**2.0))

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        core_radius_0 : the core size of the first elliptical cored Isothermal profile.
        core_radius_1 : rc + dr is the core size of the second elliptical cored Isothermal profile.
             We use dr here is to avoid negative values.
        """

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            intensity=0.0,
            core_radius_0=core_radius_0,
            core_radius_1=core_radius_1,
        )
        LightProfileSNR.__init__(self, signal_to_noise_ratio=signal_to_noise_ratio)


class ChameleonSph(lp.ChameleonSph, LightProfileSNR):
    def __init__(
        self,
        signal_to_noise_ratio: float = 10.0,
        centre: Tuple[float, float] = (0.0, 0.0),
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.05,
    ):
        """
        The spherical Chameleon light profile.

        Profile form:
            mass_to_light_ratio * intensity *\
                (1.0 / Sqrt(x^2 + (y/q)^2 + rc^2) - 1.0 / Sqrt(x^2 + (y/q)^2 + (rc + dr)**2.0))

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        core_radius_0 : the core size of the first elliptical cored Isothermal profile.
        core_radius_1 : rc + dr is the core size of the second elliptical cored Isothermal profile.
             We use dr here is to avoid negative values.
        """

        super().__init__(
            centre=centre,
            intensity=0.0,
            core_radius_0=core_radius_0,
            core_radius_1=core_radius_1,
        )
        LightProfileSNR.__init__(self, signal_to_noise_ratio=signal_to_noise_ratio)
