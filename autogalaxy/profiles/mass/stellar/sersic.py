import numpy as np

from typing import List, Tuple

from autoarray import Grid2D

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile

from autogalaxy.profiles.mass.abstract.cse import (
    MassProfileCSE,
)
from autogalaxy.profiles.mass.stellar.abstract import StellarProfile


def cse_settings_from(
    effective_radius, sersic_index, sersic_constant, mass_to_light_gradient
):
    if mass_to_light_gradient > 0.5:
        if effective_radius > 0.2:
            lower_dex = 6.0
            upper_dex = np.min(
                [np.log10((18.0 / sersic_constant) ** sersic_index), 1.1]
            )

            if sersic_index <= 1.2:
                total_cses = 50
                sample_points = 80
            elif sersic_index > 3.8:
                total_cses = 40
                sample_points = 50
                lower_dex = 6.5
            else:
                total_cses = 30
                sample_points = 50

        else:
            if sersic_index <= 1.2:
                upper_dex = 1.0
                total_cses = 50
                sample_points = 80
                lower_dex = 4.5

            elif sersic_index > 3.8:
                total_cses = 40
                sample_points = 50
                lower_dex = 6.0
                upper_dex = 1.5
            else:
                upper_dex = 1.1
                lower_dex = 6.0
                total_cses = 30
                sample_points = 50
    else:
        upper_dex = np.min(
            [
                np.log10((23.0 / sersic_constant) ** sersic_index),
                0.85 - np.log10(effective_radius),
            ]
        )

        if (sersic_index <= 0.9) and (sersic_index > 0.8):
            total_cses = 50
            sample_points = 80
            upper_dex = np.log10((18.0 / sersic_constant) ** sersic_index)
            lower_dex = 4.3 + np.log10(effective_radius)
        elif sersic_index <= 0.8:
            total_cses = 50
            sample_points = 80
            upper_dex = np.log10((16.0 / sersic_constant) ** sersic_index)
            lower_dex = 4.0 + np.log10(effective_radius)
        elif sersic_index > 3.8:
            total_cses = 40
            sample_points = 50
            lower_dex = 4.5 + np.log10(effective_radius)
        else:
            lower_dex = 3.5 + np.log10(effective_radius)
            total_cses = 30
            sample_points = 50

    return upper_dex, lower_dex, total_cses, sample_points


class AbstractSersic(MassProfile, MassProfileCSE, StellarProfile):
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
        The Sersic mass profile, the mass profiles of the light profiles that are used to fit and subtract the lens \
        model_galaxy's light.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius
            The radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        mass_to_light_ratio
            The mass-to-light ratio of the light profiles
        """
        super(AbstractSersic, self).__init__(centre=centre, ell_comps=ell_comps)
        super(MassProfile, self).__init__(centre=centre, ell_comps=ell_comps)
        super(MassProfileCSE, self).__init__()
        self.mass_to_light_ratio = mass_to_light_ratio
        self.intensity = intensity
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index

    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        return self.deflections_2d_via_cse_from(grid=grid, xp=xp, **kwargs)

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    def deflections_2d_via_cse_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the projected 2D deflection angles from a grid of (y,x) arc second coordinates, by computing and
        summing the convergence of each individual cse used to decompose the mass profile.

        The cored steep elliptical (cse) decomposition of a the elliptical NFW mass
        profile (e.g. `decompose_convergence_via_cse`) is using equation (12) of
        Oguri 2021 (https://arxiv.org/abs/2106.11464).

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.
        """
        return self._deflections_2d_via_cse_from(grid=grid, xp=xp, **kwargs)

    @aa.over_sample
    @aa.grid_dec.to_array
    @aa.grid_dec.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """
        return self.convergence_func(
            self.eccentric_radii_grid_from(grid=grid, xp=xp, **kwargs), xp=xp
        )

    @aa.over_sample
    @aa.grid_dec.to_array
    @aa.grid_dec.transform
    def convergence_2d_via_cse_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the projected 2D convergence from a grid of (y,x) arc second coordinates, by computing and summing
        the convergence of each individual cse used to decompose the mass profile.

        The cored steep elliptical (cse) decomposition of a the elliptical NFW mass
        profile (e.g. `decompose_convergence_via_cse`) is using equation (12) of
        Oguri 2021 (https://arxiv.org/abs/2106.11464).

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.
        """

        elliptical_radii = self.elliptical_radii_grid_from(grid=grid, xp=xp, **kwargs)

        return self._convergence_2d_via_cse_from(grid_radii=elliptical_radii)

    def convergence_func(self, grid_radius: float, xp=np) -> float:
        return self.mass_to_light_ratio * self.image_2d_via_radii_from(
            grid_radius, xp=xp
        )

    @aa.grid_dec.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        return np.zeros(shape=grid.shape[0])

    def image_2d_via_radii_from(self, radius: np.ndarray, xp=np):
        """
        Returns the intensity of the profile at a given radius.

            Parameters
            ----------
            radius
                The distance from the centre of the profile.
        """
        return self.intensity * xp.exp(
            -self.sersic_constant
            * (
                ((radius.array / self.effective_radius) ** (1.0 / self.sersic_index))
                - 1
            )
        )

    def decompose_convergence_via_cse(
        self, grid_radii: np.ndarray
    ) -> Tuple[List, List]:
        """
        Decompose the convergence of the Sersic profile into cored steep elliptical (cse) profiles.

        This decomposition uses the standard 2d profile of a Sersic mass profile.

        Parameters
        ----------
        func
            The function representing the profile that is decomposed into CSEs.
        radii_min:
            The minimum radius to fit
        radii_max:
            The maximum radius to fit
        total_cses
            The number of CSEs used to approximate the input func.
        sample_points: int (should be larger than 'total_cses')
            The number of data points to fit

        Returns
        -------
        Tuple[List, List]
            A list of amplitudes and core radii of every cored steep elliptical (cse) the mass profile is decomposed
            into.
        """

        upper_dex, lower_dex, total_cses, sample_points = cse_settings_from(
            effective_radius=self.effective_radius,
            sersic_index=self.sersic_index,
            sersic_constant=self.sersic_constant,
            mass_to_light_gradient=0.0,
        )

        scaled_effective_radius = self.effective_radius / np.sqrt(self.axis_ratio())
        radii_min = scaled_effective_radius / 10.0**lower_dex
        radii_max = scaled_effective_radius * 10.0**upper_dex

        def sersic_2d(r):
            return (
                self.mass_to_light_ratio
                * self.intensity
                * np.exp(
                    -self.sersic_constant
                    * (
                        ((r / scaled_effective_radius) ** (1.0 / self.sersic_index))
                        - 1.0
                    )
                )
            )

        return self._decompose_convergence_via_cse_from(
            func=sersic_2d,
            radii_min=radii_min,
            radii_max=radii_max,
            total_cses=total_cses,
            sample_points=sample_points,
        )

    @property
    def sersic_constant(self):
        """A parameter derived from Sersic index which ensures that effective radius contains 50% of the profile's
        total integrated light.
        """
        return (
            (2 * self.sersic_index)
            - (1.0 / 3.0)
            + (4.0 / (405.0 * self.sersic_index))
            + (46.0 / (25515.0 * self.sersic_index**2))
            + (131.0 / (1148175.0 * self.sersic_index**3))
            - (2194697.0 / (30690717750.0 * self.sersic_index**4))
        )

    @property
    def ellipticity_rescale(self):
        return 1.0 - ((1.0 - self.axis_ratio()) / 2.0)

    @property
    def elliptical_effective_radius(self):
        """
        The effective_radius of a Sersic light profile is defined as the circular effective radius. This is the \
        radius within which a circular aperture contains half the profiles's total integrated light. For elliptical \
        systems, this won't robustly capture the light profile's elliptical shape.

        The elliptical effective radius instead describes the major-axis radius of the ellipse containing \
        half the light, and may be more appropriate for highly flattened systems like disk galaxies.
        """
        return self.effective_radius / np.sqrt(self.axis_ratio())


class Sersic(AbstractSersic, MassProfileCSE):
    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    def deflections_2d_via_integral_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        from scipy.integrate import quad

        def calculate_deflection_component(npow, index):
            sersic_constant = self.sersic_constant

            deflection_grid = self.axis_ratio() * grid.array[:, index]

            for i in range(grid.shape[0]):
                deflection_grid[i] = deflection_grid[i] * (
                    self.intensity
                    * self.mass_to_light_ratio
                    * quad(
                        self.deflection_func,
                        a=0.0,
                        b=1.0,
                        args=(
                            grid.array[i, 0],
                            grid.array[i, 1],
                            npow,
                            self.axis_ratio(),
                            self.sersic_index,
                            self.effective_radius,
                            sersic_constant,
                        ),
                    )[0]
                )

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotated_grid_from_reference_frame_from(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T), xp=xp
        )

    @staticmethod
    def deflection_func(
        u, y, x, npow, axis_ratio, sersic_index, effective_radius, sersic_constant
    ):
        _eta_u = np.sqrt(axis_ratio) * np.sqrt(
            (u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u))))
        )

        return np.exp(
            -sersic_constant
            * (((_eta_u / effective_radius) ** (1.0 / sersic_index)) - 1)
        ) / ((1 - (1 - axis_ratio**2) * u) ** (npow + 0.5))


class SersicSph(Sersic):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The Sersic mass profile, the mass profiles of the light profiles that are used to fit and subtract the lens
        model_galaxy's light.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second)
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        mass_to_light_ratio
            The mass-to-light ratio of the light profile.
        """
        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
        )
