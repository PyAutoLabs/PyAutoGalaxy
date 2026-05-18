"""
Linear elliptical Sersic light profile with m=3 and m=4 Fourier multipole
perturbations on the eccentric radius.

This is the linear-light-profile counterpart of ``autogalaxy.lp.SersicMultipole``:
``intensity`` is hardcoded to 1.0 in the constructor because it is solved
analytically via the linear inversion at fit time.
"""

from typing import Tuple

from autogalaxy.profiles.light.linear.abstract import LightProfileLinear

from autogalaxy.profiles.light import standard as lp


class SersicMultipole(lp.SersicMultipole, LightProfileLinear):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
        multipole_3_comps: Tuple[float, float] = (0.0, 0.0),
        multipole_4_comps: Tuple[float, float] = (0.0, 0.0),
    ):
        """
        The linear elliptical Sersic light profile with m=3 and m=4 Fourier multipole
        perturbations on the eccentric radius.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate
            system.
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile.
        multipole_3_comps
            The ``(cos, sin)`` components of the m=3 Fourier perturbation. Defaults to
            ``(0.0, 0.0)`` which reduces the profile to ``lp_linear.Sersic``.
        multipole_4_comps
            The ``(cos, sin)`` components of the m=4 Fourier perturbation. Defaults to
            ``(0.0, 0.0)`` which reduces the profile to ``lp_linear.Sersic``.
        """
        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            intensity=1.0,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            multipole_3_comps=multipole_3_comps,
            multipole_4_comps=multipole_4_comps,
        )
