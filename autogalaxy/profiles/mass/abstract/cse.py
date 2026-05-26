from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, List, Tuple


class MassProfileCSE(ABC):
    @staticmethod
    def convergence_cse_1d_from(
        grid_radii: np.ndarray, core_radius: float, xp=np
    ) -> np.ndarray:
        """
        One dimensional function which is solved to decompose a convergence profile in cored steep ellipsoids, given by
        equation (14) of Oguri 2021 (https://arxiv.org/abs/2106.11464).

        Parameters
        ----------
        grid_radii
            The 1D radial coordinates the decomposition is performed for.
        core_radius
            The core radius of the cored steep ellisoid used for this decomposition.

        """
        return 1.0 / (2.0 * (core_radius**2.0 + grid_radii**2.0) ** (1.5))

    @staticmethod
    def deflections_via_cse_from(
        term1: float,
        term2: float,
        term3: float,
        term4: float,
        axis_ratio_squared: float,
        core_radius: float,
        xp=np,
    ) -> np.ndarray:
        """
        Returns the deflection angles of a 1d cored steep ellisoid (CSE) profile, given by equation (19) and (20) of
        Oguri 2021 (https://arxiv.org/abs/2106.11464).

        To accelerate the deflection angle computation terms are computed separated, defined as term1, 2, 3, 4.

        Parameters
        ----------
        """
        phi = xp.sqrt(axis_ratio_squared * core_radius**2.0 + term1)
        Psi = (phi + core_radius) ** 2.0 + term2
        bottom = core_radius * phi * Psi
        defl_x = (term3 * (phi + axis_ratio_squared * core_radius)) / bottom
        defl_y = (term4 * (phi + core_radius)) / bottom
        return xp.vstack((defl_y, defl_x))

    @abstractmethod
    def decompose_convergence_via_cse(self, grid_radii: np.ndarray):
        pass

    def _decompose_convergence_via_cse_from(
        self,
        func: Callable,
        radii_min: float,
        radii_max: float,
        total_cses: int = 25,
        sample_points: int = 100,
    ) -> Tuple[List, List]:
        """
        Decompose the convergence of a mass profile into cored steep elliptical (cse) profiles.

        This uses an input function `func` which is specific to the inherited mass profile, and defines the function
        which is solved for in order to decompose its convergence into cses.

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
        from scipy.linalg import lstsq

        error_sigma = 0.1  # error spread. Could be any value.

        r_samples = np.logspace(np.log10(radii_min), np.log10(radii_max), sample_points)
        y_samples = np.ones_like(r_samples) / error_sigma
        y_samples_func = func(r_samples)

        core_radius_list = np.logspace(
            np.log10(radii_min), np.log10(radii_max), total_cses
        )

        # Different from Masamune's (2106.11464) method, I set S to a series fixed values. So that
        # the decomposition can be solved linearly.

        coefficient_matrix = np.zeros((sample_points, total_cses))

        for j in range(total_cses):
            coefficient_matrix[:, j] = self.convergence_cse_1d_from(
                r_samples, core_radius_list[j]
            )

        for k in range(sample_points):
            coefficient_matrix[k] /= y_samples_func[k] * error_sigma

        results = lstsq(coefficient_matrix, y_samples.T)

        amplitude_list = results[0]

        return amplitude_list, core_radius_list

    def convergence_2d_via_cse_from(
        self, grid_radii: np.ndarray, **kwargs
    ) -> np.ndarray:
        pass

    def _convergence_2d_via_cse_from(
        self, grid_radii: np.ndarray, xp=np, **kwargs
    ) -> np.ndarray:
        """
        Calculate the projected 2D convergence from a grid of radial coordinates, by computing and summing the
        convergence of each individual cse used to decompose the mass profile.

        The cored steep elliptical (cse) decomposition of a given mass profile (e.g. `convergence_cse_1d_from`) is
        defined for every mass profile and defines how it is efficiently decomposed its cses.

        Parameters
        ----------
        grid_radii
            The grid of 1D radial arc-second coordinates the convergence is computed on.
        """

        amplitude_list, core_radius_list = self.decompose_convergence_via_cse(
            grid_radii=grid_radii
        )

        return sum(
            amplitude
            * self.convergence_cse_1d_from(
                grid_radii=grid_radii, core_radius=core_radius, xp=xp
            )
            for amplitude, core_radius in zip(amplitude_list, core_radius_list)
        )

    def _deflections_2d_via_cse_from(self, grid: np.ndarray, xp=np, **kwargs) -> np.ndarray:
        """
        Calculate the projected 2D deflection angles from a grid of radial coordinates, by computing and summing the
        deflections of each individual cse used to decompose the mass profile.

        The cored steep elliptical (cse) decomposition of a given mass profile (e.g. `convergence_cse_1d_from`) is
        defined for every mass profile and defines how it is efficiently decomposed its cses.

        Parameters
        ----------
        grid_radii
            The grid of 1D radial arc-second coordinates the convergence is computed on.
        """

        amplitude_list, core_radius_list = self.decompose_convergence_via_cse(
            grid_radii=self.radial_grid_from(grid=grid, **kwargs)
        )
        q = self.axis_ratio()
        q2 = q**2.0
        grid_y = grid.array[:, 0]
        grid_x = grid.array[:, 1]
        gridx2 = grid_x**2.0
        gridy2 = grid_y**2.0
        term1 = q2 * gridx2 + gridy2
        term2 = (1.0 - q2) * gridx2
        term3 = q * grid_x
        term4 = q * grid_y

        # To accelarate deflection angle computation, I define term1, term2, term3, term4 to avoid
        # repeated matrix operation. There might be still space for optimization.

        deflections_2d = sum(
            amplitude
            * self.deflections_via_cse_from(
                axis_ratio_squared=q2,
                core_radius=core_radius,
                term1=term1,
                term2=term2,
                term3=term3,
                term4=term4,
                xp=xp,
            )
            for amplitude, core_radius in zip(amplitude_list, core_radius_list)
        )

        return deflections_2d.T
