from typing import Tuple

from autogalaxy.profiles.mass.stellar.sersic import Sersic


class Exponential(Sersic):
    r"""
    Elliptical exponential stellar mass profile.

    A special case of the :class:`Sersic` mass profile with Sérsic index :math:`n = 1`,
    corresponding to an exponential disc surface brightness profile:

    .. math::

        \kappa(R) = \Upsilon \, I_e \exp\!\left(-b_1 \left[\frac{R}{R_e} - 1\right]\right)

    where :math:`b_1 \approx 1.678` ensures :math:`R_e` is the half-light radius.
    This profile is commonly used to model disc-dominated galaxies.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        r"""
        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall intensity normalisation :math:`I_e` at the effective radius
            (electrons per second).
        effective_radius
            The effective (half-light) radius :math:`R_e` in arcseconds.
        mass_to_light_ratio
            The mass-to-light ratio :math:`\Upsilon` in solar units.
        """
        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=1.0,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class ExponentialSph(Exponential):
    r"""
    Spherical exponential stellar mass profile.

    A special case of :class:`Exponential` with no ellipticity (:math:`q = 1`),
    i.e., a :class:`Sersic` profile with :math:`n = 1` evaluated on a circular grid:

    .. math::

        \kappa(r) = \Upsilon \, I_e \exp\!\left(-b_1 \left[\frac{r}{R_e} - 1\right]\right)
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        r"""
        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall intensity normalisation :math:`I_e` at the effective radius
            (electrons per second).
        effective_radius
            The effective (half-light) radius :math:`R_e` in arcseconds.
        mass_to_light_ratio
            The mass-to-light ratio :math:`\Upsilon` in solar units.
        """
        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            mass_to_light_ratio=mass_to_light_ratio,
        )
