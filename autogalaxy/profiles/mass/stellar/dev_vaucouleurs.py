from typing import Tuple

from autogalaxy.profiles.mass.stellar.sersic import Sersic


class DevVaucouleurs(Sersic):
    r"""
    Elliptical de Vaucouleurs stellar mass profile (de Vaucouleurs 1948).

    A special case of the :class:`Sersic` mass profile with Sérsic index :math:`n = 4`,
    corresponding to the classical de Vaucouleurs :math:`R^{1/4}` surface brightness law
    that describes the light distribution of massive elliptical galaxies:

    .. math::

        \kappa(R) = \Upsilon \, I_e \exp\!\left\{
            -b_4 \left[\left(\frac{R}{R_e}\right)^{1/4} - 1\right]
        \right\}

    where :math:`b_4 \approx 7.669` ensures :math:`R_e` is the half-light radius.

    References
    ----------
    - de Vaucouleurs 1948, Ann. Astrophys., 11, 247
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
            sersic_index=4.0,
            mass_to_light_ratio=mass_to_light_ratio,
        )


class DevVaucouleursSph(DevVaucouleurs):
    r"""
    Spherical de Vaucouleurs stellar mass profile.

    A special case of :class:`DevVaucouleurs` with no ellipticity (:math:`q = 1`),
    i.e., a :class:`Sersic` profile with :math:`n = 4` evaluated on a circular grid:

    .. math::

        \kappa(r) = \Upsilon \, I_e \exp\!\left\{
            -b_4 \left[\left(\frac{r}{R_e}\right)^{1/4} - 1\right]
        \right\}

    References
    ----------
    - de Vaucouleurs 1948, Ann. Astrophys., 11, 247
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
