import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile


class PowerLawCore(MassProfile):
    r"""Cored elliptical power-law (cored EPL / SPEMD) mass profile.

    The convergence of the cored elliptical power-law is:

    .. math::

        \kappa(R) = \frac{3 - \gamma}{2}
                    \left(s^2 + R^2\right)^{-({\gamma - 1})/2}
                    \theta_{\rm E}^{\,\gamma - 1}

    where :math:`\gamma` is the logarithmic density slope, :math:`\theta_{\rm E}`
    is the Einstein radius, :math:`R` is the elliptical radius, and :math:`s` is
    the core radius that prevents the central divergence of the pure power-law.

    Parameters
    ----------
    centre : (float, float)
        (y, x) arc-second coordinates of the profile centre.
    ell_comps : (float, float)
        Ellipticity components (e1, e2) of the elliptical coordinate system.
    einstein_radius : float
        Einstein radius in arcseconds.
    slope : float
        Logarithmic density slope :math:`\gamma`; shallower profiles have
        lower values, steeper profiles have higher values.
    core_radius : float
        Core radius :math:`s` in arcseconds.

    References
    ----------
    Keeton (2001), arXiv:astro-ph/0102341.
    Tessore & Metcalf (2015), A&A, 580, A79.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
        core_radius: float = 0.01,
    ):
        """
        Represents a cored elliptical power-law density distribution

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        einstein_radius
            The arc-second Einstein radius.
        slope
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        core_radius
            The arc-second radius of the inner core.
        """
        super().__init__(centre=centre, ell_comps=ell_comps)

        self.einstein_radius = einstein_radius
        self.slope = slope
        self.core_radius = core_radius

    def einstein_radius_major_from(self, xp=np):
        """
        The Einstein radius in the convention the profile's convergence and deflection formulae are written in,
        where the elliptical radius is the major-axis radius of the isodensity contour.

        For this class and its standard subclasses this is the ``einstein_radius`` parameter itself; subclasses
        which parameterise the Einstein radius in a different convention (e.g. ``PowerLawIntermediate``) override
        this with the conversion, which every calculation then inherits.
        """
        return self.einstein_radius

    def einstein_radius_rescaled(self, xp=np):
        """
        Rescale the einstein radius by slope and axis_ratio, to reduce its degeneracy with other mass-profiles
        parameters.
        """
        return (
            (3 - self.slope) / (1 + self.axis_ratio(xp))
        ) * self.einstein_radius_major_from(xp) ** (self.slope - 1)

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Returns the two dimensional projected convergence on a grid of (y,x) arc-second coordinates.

        The `grid_2d_to_structure` decorator reshapes the ndarrays the convergence is outputted on. See
        *aa.grid_2d_to_structure* for a description of the output.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.
        """
        grid_eta = self.elliptical_radii_grid_from(grid=grid, xp=xp, **kwargs)

        return self.convergence_func(grid_radius=grid_eta)

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the potential on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        from scipy.integrate import quad

        potential_grid = np.zeros(grid.shape[0])

        for i in range(grid.shape[0]):
            potential_grid[i] = quad(
                self.potential_func,
                a=0.0,
                b=1.0,
                args=(
                    grid.array[i, 0],
                    grid.array[i, 1],
                    self.axis_ratio(xp),
                    self.slope,
                    self.core_radius,
                ),
            )[0]

        return self.einstein_radius_rescaled(xp) * self.axis_ratio(xp) * potential_grid

    @aa.decorators.to_vector_yx
    @aa.decorators.transform(rotate_back=True)
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        from scipy.integrate import quad

        def calculate_deflection_component(npow, index):
            einstein_radius_rescaled = self.einstein_radius_rescaled(xp)

            deflection_grid = np.array(self.axis_ratio(xp) * grid.array[:, index])

            for i in range(grid.shape[0]):
                deflection_grid[i] *= (
                    einstein_radius_rescaled
                    * quad(
                        self.deflection_func,
                        a=0.0,
                        b=1.0,
                        args=(
                            grid.array[i, 0],
                            grid.array[i, 1],
                            npow,
                            self.axis_ratio(xp),
                            self.slope,
                            self.core_radius,
                        ),
                    )[0]
                )

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return np.vstack((deflection_y, deflection_x)).T

    def convergence_func(self, grid_radius: float, xp=np) -> float:
        return self.einstein_radius_rescaled(xp) * (
            self.core_radius**2 + grid_radius**2
        ) ** (-(self.slope - 1) / 2.0)

    @staticmethod
    def potential_func(u, y, x, axis_ratio, slope, core_radius):
        eta = np.sqrt((u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u)))))
        return (
            (eta / u)
            * ((3.0 - slope) * eta) ** -1.0
            * (
                (core_radius**2.0 + eta**2.0) ** ((3.0 - slope) / 2.0)
                - core_radius ** (3 - slope)
            )
            / ((1 - (1 - axis_ratio**2) * u) ** 0.5)
        )

    @staticmethod
    def deflection_func(u, y, x, npow, axis_ratio, slope, core_radius):
        _eta_u = np.sqrt((u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u)))))
        return (core_radius**2 + _eta_u**2) ** (-(slope - 1) / 2.0) / (
            (1 - (1 - axis_ratio**2) * u) ** (npow + 0.5)
        )

    def ellipticity_rescale(self, xp=np):
        return (1.0 + self.axis_ratio(xp=xp)) / 2.0

    @property
    def unit_mass(self):
        return "angular"


class PowerLawCoreSph(PowerLawCore):
    r"""Cored spherical power-law mass profile.

    The spherical limit of :class:`PowerLawCore`.  The convergence is:

    .. math::

        \kappa(r) = \frac{3 - \gamma}{2}
                    \left(s^2 + r^2\right)^{-({\gamma - 1})/2}
                    \theta_{\rm E}^{\,\gamma - 1}

    where :math:`\gamma` is the logarithmic density slope, :math:`\theta_{\rm E}`
    is the Einstein radius, :math:`r` is the circular projected radius, and
    :math:`s` is the core radius.

    Parameters
    ----------
    centre : (float, float)
        (y, x) arc-second coordinates of the profile centre.
    einstein_radius : float
        Einstein radius in arcseconds.
    slope : float
        Logarithmic density slope :math:`\gamma`; shallower profiles have
        lower values, steeper profiles have higher values.
    core_radius : float
        Core radius :math:`s` in arcseconds.

    References
    ----------
    Keeton (2001), arXiv:astro-ph/0102341.
    Tessore & Metcalf (2015), A&A, 580, A79.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
        core_radius: float = 0.01,
    ):
        """
        Represents a cored spherical power-law density distribution

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius
            The arc-second Einstein radius.
        slope
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        core_radius
            The arc-second radius of the inner core.
        """
        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            einstein_radius=einstein_radius,
            slope=slope,
            core_radius=core_radius,
        )

    @aa.decorators.to_vector_yx
    @aa.decorators.transform
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        eta = self.radial_grid_from(grid=grid, xp=xp, **kwargs)
        deflection = xp.multiply(
            2.0 * self.einstein_radius_rescaled(xp),
            xp.divide(
                xp.add(
                    xp.power(
                        xp.add(self.core_radius**2, xp.square(eta.array)),
                        (3.0 - self.slope) / 2.0,
                    ),
                    -self.core_radius ** (3 - self.slope),
                ),
                xp.multiply((3.0 - self.slope), eta.array),
            ),
        )
        return self._cartesian_grid_via_radial_from(grid=grid, radius=deflection, xp=xp)
