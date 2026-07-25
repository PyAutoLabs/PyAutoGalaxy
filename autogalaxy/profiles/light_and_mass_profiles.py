from typing import Tuple

from autogalaxy.profiles.light import standard as lp
from autogalaxy.profiles import mass as mp


class LightMassProfile:
    """
    Mass and light profiles describe both mass distributions and light distributions with a single set of parameters. This
    means that the light and mass of these profiles are tied together. Galaxy instances interpret these
    objects as being both mass and light profiles.
    """

    pass


class Gaussian(lp.Gaussian, mp.Gaussian, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        sigma: float = 1.0,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The elliptical Gaussian light and mass profile.

        This simultaneously represents the luminous emission and stellar mass of a galaxy.

        Parameters
        ----------
        centre
            The grid of The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius
            The radius containing half the light of this light profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles.
        """
        lp.Gaussian.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            sigma=sigma,
        )
        mp.Gaussian.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            sigma=sigma,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class GaussianGradient(lp.Gaussian, mp.GaussianGradient, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        sigma: float = 1.0,
        mass_to_light_ratio_base: float = 1.0,
        mass_to_light_gradient: float = 0.0,
        mass_to_light_radius: float = 1.0,
    ):
        """
        The elliptical Gaussian light profile with a gradient in its mass to light conversion.

        This simultaneously represents the luminous emission and stellar mass of a galaxy.

        Parameters
        ----------
        centre
            The grid of The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius
            The radius containing half the light of this light profile.
        mass_to_light_ratio_base
            The base mass-to-light ratio of the profile, which is the mass-to-light ratio of the Gaussian before it
            is scaled by values that adjust its mass-to-light ratio based on the reference radius and gradient.
        mass_to_light_gradient
            The mass-to-light radial gradient of the profile, whereby positive values means there is more mass
            per unit light within the reference radius.
        mass_to_light_radius
            The radius where the mass-to-light ratio is equal to the base mass-to-light ratio, such that there will be
            more of less mass per unit light within this radius depending on the mass-to-light gradient.
        """
        lp.Gaussian.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            sigma=sigma,
        )
        mp.GaussianGradient.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            sigma=sigma,
            mass_to_light_ratio_base=mass_to_light_ratio_base,
            mass_to_light_gradient=mass_to_light_gradient,
            mass_to_light_radius=mass_to_light_radius,
        )


class Sersic(lp.Sersic, mp.Sersic, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The elliptical Sersic light and mass profile.

        This simultaneously represents the luminous emission and stellar mass of a galaxy.

        Parameters
        ----------
        centre
            The grid of The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius
            The radius containing half the light of this light profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles.
        """
        lp.Sersic.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )
        mp.Sersic.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class SersicSph(Sersic, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The spherical Sersic light and mass profile.

        This simultaneously represents the luminous emission and stellar mass of a galaxy.

        Parameters
        ----------
        centre
            The grid of The (y,x) arc-second coordinates of the profile centre..
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius
            The radius containing half the light of this light profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles.
        """
        Sersic.__init__(
            self,
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class Exponential(Sersic, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The elliptical Exponential light and mass profile.

        This simultaneously represents the luminous emission and stellar mass of a galaxy.

        Parameters
        ----------
        centre
            The grid of The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The radius containing half the light of this light profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        """
        Sersic.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=1.0,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class ExponentialSph(Exponential, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The spherical Exponential light and mass profile.

        This simultaneously represents the luminous emission and stellar mass of a galaxy.

        Parameters
        ----------
        centre
            The grid of The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The radius containing half the light of this light profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        """
        Exponential.__init__(
            self,
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class DevVaucouleurs(Sersic, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The elliptical Dev Vaucouleurs light and mass profile.

        This simultaneously represents the luminous emission and stellar mass of a galaxy.

        Parameters
        ----------
        centre
            The grid of The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The radius containing half the light of this light profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        """
        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=4.0,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class DevVaucouleursSph(DevVaucouleurs, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The spherical Dev Vaucouleurs light and mass profile.

        This simultaneously represents the luminous emission and stellar mass of a galaxy.

        Parameters
        ----------
        centre
            The grid of The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The radius containing half the light of this light profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        """
        DevVaucouleurs.__init__(
            self,
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class SersicGradient(lp.Sersic, mp.SersicGradient, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: float = 1.0,
        mass_to_light_gradient: float = 0.0,
    ):
        """
        The elliptical Sersic light and mass profile, with a radial gradient in the conversion of light to mass..

        This simultaneously represents the luminous emission and stellar mass of a galaxy.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre..
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The radius containing half the light of this light profile.
        sersic_index
            The concentration of the light profiles.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles.
        mass_to_light_gradient
            The mass-to-light radial gradient.
        """
        lp.Sersic.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )
        mp.SersicGradient.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
            mass_to_light_gradient=mass_to_light_gradient,
        )


class SersicGradientSph(SersicGradient, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: float = 1.0,
        mass_to_light_gradient: float = 0.0,
    ):
        """
        The spherical Sersic light and mass profile, with a radial gradient in the conversion of light to mass..

        This simultaneously represents the luminous emission and stellar mass of a galaxy.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The radius containing half the light of this light profile.
        sersic_index
            The concentration of the light profiles
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        mass_to_light_gradient
            The mass-to-light radial gradient.
        """

        SersicGradient.__init__(
            self,
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
            mass_to_light_gradient=mass_to_light_gradient,
        )


class ExponentialGradient(SersicGradient, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
        mass_to_light_gradient: float = 0.0,
    ):
        """
        The elliptical Exponential light and mass profile, with a radial gradient in the conversion of light to mass..

        This simultaneously represents the luminous emission and stellar mass of a galaxy.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The radius containing half the light of this light profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        mass_to_light_gradient
            The mass-to-light radial gradient.
        """

        SersicGradient.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=1.0,
            mass_to_light_ratio=mass_to_light_ratio,
            mass_to_light_gradient=mass_to_light_gradient,
        )


class ExponentialGradientSph(SersicGradientSph, LightMassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
        mass_to_light_gradient: float = 0.0,
    ):
        """
        The spherical Exponential light and mass profile, with a radial gradient in the conversion of light to mass..

        This simultaneously represents the luminous emission and stellar mass of a galaxy.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The radius containing half the light of this light profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        mass_to_light_gradient
            The mass-to-light radial gradient.
        """

        SersicGradientSph.__init__(
            self,
            centre=centre,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=1.0,
            mass_to_light_ratio=mass_to_light_ratio,
            mass_to_light_gradient=mass_to_light_gradient,
        )


class SersicCore(lp.SersicCore, mp.SersicCore, LightMassProfile):
    """
    The elliptical cored-Sersic light and mass profile.

    This simultaneously represents the luminous emission and stellar mass of a galaxy.

    Parameters
    ----------
    centre
        The grid of The (y,x) arc-second coordinates of the profile centre.
    ell_comps
        The first and second ellipticity components of the elliptical coordinate system.
    intensity
        Overall flux intensity normalisation in the light profiles (electrons per second).
    effective_radius
        The radius containing half the light of this light profile.
    radius_break
        The break radius separating the inner power-law (with logarithmic slope gamma) and outer Sersic function.
    gamma
        The logarithmic power-law slope of the inner core profiles
    alpha
        Controls the sharpness of the transition between the inner core / outer Sersic profiles.
    mass_to_light_ratio
        The mass-to-light ratio of the light profiles.
    """

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
        lp.SersicCore.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            radius_break=radius_break,
            intensity=intensity,
            gamma=gamma,
            alpha=alpha,
        )
        mp.SersicCore.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            radius_break=radius_break,
            intensity=intensity,
            gamma=gamma,
            alpha=alpha,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class SersicCoreSph(SersicCore, LightMassProfile):
    """
    The spherical cored-Sersic light and mass profile.

    This simultaneously represents the luminous emission and stellar mass of a galaxy.

    Parameters
    ----------
    centre
        The grid of The (y,x) arc-second coordinates of the profile centre.
    ell_comps
        The first and second ellipticity components of the elliptical coordinate system.
    intensity
        Overall flux intensity normalisation in the light profiles (electrons per second).
    effective_radius
        The radius containing half the light of this light profile.
    radius_break
        The break radius separating the inner power-law (with logarithmic slope gamma) and outer Sersic function.
    gamma
        The logarithmic power-law slope of the inner core profiles
    alpha
        Controls the sharpness of the transition between the inner core / outer Sersic profiles.
    mass_to_light_ratio
        The mass-to-light ratio of the light profiles.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
        radius_break: float = 0.01,
        intensity: float = 0.05,
        gamma: float = 0.25,
        alpha: float = 3.0,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The SersicSph mass profile, the mass profiles of the light profiles that are used to fit_normal and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre
            The grid of The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The radius containing half the light of this light profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        """
        SersicCore.__init__(
            self,
            centre=centre,
            ell_comps=(0.0, 0.0),
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            radius_break=radius_break,
            intensity=intensity,
            gamma=gamma,
            alpha=alpha,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class Chameleon(lp.Chameleon, mp.Chameleon, LightMassProfile):
    """
    The elliptical Chameleon light and mass profile.

    This simultaneously represents the luminous emission and stellar mass of a galaxy.

    This light profile closely approximates the Elliptical Sersic light profile, by representing it as two cored
    elliptical isothermal profiles. This is convenient for lensing calculations, because the deflection angles of
    an isothermal profile can be evaluated analyticially efficiently.

    Parameters
    ----------
    centre
        The (y,x) arc-second coordinates of the profile centre.
    ell_comps
        The first and second ellipticity components of the elliptical coordinate system.
    intensity
        Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
        the light profile's image is compared too, which is expected to be electrons per second).
    core_radius_0
        The core size of the first elliptical cored Isothermal profile.
    core_radius_1
        The core size of the second elliptical cored Isothermal profile.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.005,
        mass_to_light_ratio: float = 1.0,
    ):
        lp.Chameleon.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            core_radius_0=core_radius_0,
            core_radius_1=core_radius_1,
        )
        mp.Chameleon.__init__(
            self,
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            core_radius_0=core_radius_0,
            core_radius_1=core_radius_1,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class ChameleonSph(Chameleon, LightMassProfile):
    """
    The spherical Chameleon light and mass profile.

    This simultaneously represents the luminous emission and stellar mass of a galaxy.

    This light profile closely approximates the Elliptical Sersic light profile, by representing it as two cored
    elliptical isothermal profiles. This is convenient for lensing calculations, because the deflection angles of
    an isothermal profile can be evaluated analyticially efficiently.

    Parameters
    ----------
    centre
        The (y,x) arc-second coordinates of the profile centre.
    intensity
        Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
        the light profile's image is compared too, which is expected to be electrons per second).
    core_radius_0
        The core size of the first elliptical cored Isothermal profile.
    core_radius_1
        The core size of the second elliptical cored Isothermal profile.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.005,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The ChameleonSph mass profile, the mass profiles of the light profiles that are used to fit_normal and
        subtract the lens model_galaxy's light.

        Parameters
        ----------
        centre
            The grid of The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The radius containing half the light of this light profile.
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        """
        Chameleon.__init__(
            self,
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            core_radius_0=core_radius_0,
            core_radius_1=core_radius_1,
            mass_to_light_ratio=mass_to_light_ratio,
        )
