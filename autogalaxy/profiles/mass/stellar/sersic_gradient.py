import numpy as np
from typing import List, Tuple

import autoarray as aa

from autogalaxy.profiles.mass.stellar.sersic import AbstractSersic

from autogalaxy.profiles.mass.stellar.sersic import cse_settings_from


class SersicGradient(AbstractSersic):
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
        Setup a Sersic mass and light profiles.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        mass_to_light_ratio
            The mass-to-light ratio of the light profile.
        mass_to_light_gradient
            The mass-to-light radial gradient.
        """
        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
        )
        self.mass_to_light_gradient = mass_to_light_gradient

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """
        return self.convergence_func(
            self.eccentric_radii_grid_from(grid=grid, xp=xp, **kwargs)
        )

    def convergence_func(self, grid_radius: float, xp=np) -> float:
        return (
            self.mass_to_light_ratio
            * (
                ((self.axis_ratio() * grid_radius) / self.effective_radius)
                ** -self.mass_to_light_gradient
            )
            * self.image_2d_via_radii_from(grid_radius)
        )

    def decompose_convergence_via_mge(self, **kwargs):
        radii_min = self.effective_radius / 100.0
        radii_max = self.effective_radius * 20.0

        def sersic_gradient_2D(r):
            return (
                self.mass_to_light_ratio
                * self.intensity
                * (
                    ((self.axis_ratio() * r) / self.effective_radius)
                    ** -self.mass_to_light_gradient
                )
                * np.exp(
                    -self.sersic_constant
                    * (((r / self.effective_radius) ** (1.0 / self.sersic_index)) - 1.0)
                )
            )

        return self._decompose_convergence_via_mge(
            func=sersic_gradient_2D, radii_min=radii_min, radii_max=radii_max
        )

    def decompose_convergence_via_cse(
        self, grid_radii: np.ndarray
    ) -> Tuple[List, List]:
        """
        Decompose the convergence of the Sersic profile into singular isothermal elliptical (sie) profiles.

        This decomposition uses the standard 2d profile of a Sersic mass profile.

        Parameters
        ----------
        func
            The function representing the profile that is decomposed into CSEs.
        radii_min:
            The minimum radius to fit
        radii_max:
            The maximum radius to fit
        total_sies
            The number of SIEs used to approximate the input func.
        sample_points: int (should be larger than 'total_sies')
            The number of data points to fit

        Returns
        -------
        Tuple[List, List]
            A list of amplitudes and core radii of every singular isothernal ellipsoids (sie) the mass profile is decomposed
            into.
        """

        upper_dex, lower_dex, total_cses, sample_points = cse_settings_from(
            effective_radius=self.effective_radius,
            sersic_index=self.sersic_index,
            sersic_constant=self.sersic_constant,
            mass_to_light_gradient=self.mass_to_light_gradient,
        )

        scaled_effective_radius = self.effective_radius / np.sqrt(self.axis_ratio())
        radii_min = scaled_effective_radius / 10.0**lower_dex
        radii_max = scaled_effective_radius * 10.0**upper_dex

        def sersic_gradient_2D(r):
            return (
                self.mass_to_light_ratio
                * self.intensity
                * (
                    ((self.axis_ratio() * r) / scaled_effective_radius)
                    ** -self.mass_to_light_gradient
                )
                * np.exp(
                    -self.sersic_constant
                    * (
                        ((r / scaled_effective_radius) ** (1.0 / self.sersic_index))
                        - 1.0
                    )
                )
            )

        return self._decompose_convergence_via_cse_from(
            func=sersic_gradient_2D,
            radii_min=radii_min,
            radii_max=radii_max,
            total_cses=total_cses,
            sample_points=sample_points,
        )


class SersicGradientSph(SersicGradient):
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
        Setup a Sersic mass and light profiles.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        mass_to_light_ratio
            The mass-to-light ratio of the light profile.
        mass_to_light_gradient
            The mass-to-light radial gradient.
        """
        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
            mass_to_light_gradient=mass_to_light_gradient,
        )
