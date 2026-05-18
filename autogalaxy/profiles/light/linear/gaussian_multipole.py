"""
Linear elliptical Gaussian light profile with m=3 and m=4 Fourier multipole
perturbations on the eccentric radius.

This is the linear-light-profile counterpart of ``autogalaxy.lp.GaussianMultipole``:
``intensity`` is hardcoded to 1.0 in the constructor because it is solved
analytically via the linear inversion at fit time.
"""

from typing import Tuple

from autogalaxy.profiles.light.linear.abstract import LightProfileLinear

from autogalaxy.profiles.light import standard as lp


class GaussianMultipole(lp.GaussianMultipole, LightProfileLinear):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        sigma: float = 1.0,
        multipole_3_comps: Tuple[float, float] = (0.0, 0.0),
        multipole_4_comps: Tuple[float, float] = (0.0, 0.0),
    ):
        """
        The linear elliptical Gaussian light profile with m=3 and m=4 Fourier multipole
        perturbations on the eccentric radius.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate
            system.
        sigma
            The sigma value of the Gaussian.
        multipole_3_comps
            The ``(cos, sin)`` components of the m=3 Fourier perturbation. Defaults to
            ``(0.0, 0.0)`` which reduces the profile to ``lp_linear.Gaussian``.
        multipole_4_comps
            The ``(cos, sin)`` components of the m=4 Fourier perturbation. Defaults to
            ``(0.0, 0.0)`` which reduces the profile to ``lp_linear.Gaussian``.
        """
        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            intensity=1.0,
            sigma=sigma,
            multipole_3_comps=multipole_3_comps,
            multipole_4_comps=multipole_4_comps,
        )
