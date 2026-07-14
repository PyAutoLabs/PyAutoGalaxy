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
    effective_radius, sersic_index, sersic_constant, mass_to_light_gradient, xp=np
):
    """
    Return the radial fitting range (``upper_dex`` / ``lower_dex``) and the CSE
    decomposition resolution (``total_cses`` / ``sample_points``) used to decompose
    a Sersic convergence profile into cored steep ellipsoids.

    The standard (``mass_to_light_gradient <= 0.5``) path is written branch-free
    using ``xp`` (``jnp.where`` / ``jnp.log10`` under JAX) so it traces cleanly
    inside ``jax.jit`` when ``sersic_index`` / ``effective_radius`` are tracers:
    Python ``if`` statements cannot branch on a tracer value, and ``jax.jit``
    requires static array shapes, so ``total_cses`` / ``sample_points`` are frozen
    to the conservative maximum previously used (50 / 80) rather than tuned per
    Sersic index. The per-index ``upper_dex`` / ``lower_dex`` *ranges* are preserved
    exactly via ``xp.where`` so accuracy is unchanged (only the CSE count can rise).

    The ``mass_to_light_gradient > 0.5`` path (radial-gradient Sersic profiles) is
    left as the original NumPy branching — it is only reached with a concrete
    gradient value and is not part of the standard JAX-traced Sersic path.
    """
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
        # Static shapes for jax.jit: frozen to the former per-index maximum.
        total_cses = 50
        sample_points = 80

        log_effective_radius = xp.log10(effective_radius)
        base_upper_dex = xp.minimum(
            xp.log10((23.0 / sersic_constant) ** sersic_index),
            0.85 - log_effective_radius,
        )

        # upper_dex / lower_dex select the same per-index ranges as the original
        # if/elif ladder, expressed branch-free so a tracer sersic_index traces.
        upper_dex = xp.where(
            sersic_index <= 0.8,
            xp.log10((16.0 / sersic_constant) ** sersic_index),
            xp.where(
                sersic_index <= 0.9,
                xp.log10((18.0 / sersic_constant) ** sersic_index),
                base_upper_dex,
            ),
        )
        lower_dex = xp.where(
            sersic_index <= 0.8,
            4.0 + log_effective_radius,
            xp.where(
                sersic_index <= 0.9,
                4.3 + log_effective_radius,
                xp.where(
                    sersic_index > 3.8,
                    4.5 + log_effective_radius,
                    3.5 + log_effective_radius,
                ),
            ),
        )

    return upper_dex, lower_dex, total_cses, sample_points


class AbstractSersic(MassProfile, MassProfileCSE, StellarProfile):
    r"""
    Abstract base class for Sérsic stellar mass profiles.

    The convergence of a Sérsic mass profile is proportional to the Sérsic surface
    brightness profile scaled by a constant mass-to-light ratio:

    .. math::

        \kappa(R) = \Upsilon \, I_e \exp\!\left\{
            -b_n \left[\left(\frac{R}{R_e}\right)^{1/n} - 1\right]
        \right\}

    where :math:`\Upsilon` is the mass-to-light ratio (``mass_to_light_ratio``),
    :math:`I_e` is the intensity at the effective radius (``intensity``),
    :math:`R_e` is the effective (half-light) radius (``effective_radius``), :math:`n` is
    the Sérsic index (``sersic_index``), and :math:`b_n` is a constant that ensures the
    effective radius encloses half the total luminosity (approximated by a polynomial in
    :math:`n`).

    Deflection angles are computed via a cored-steep-ellipsoid (CSE) decomposition
    following Oguri (2021).

    References
    ----------
    - Sérsic 1963, Boletin de la Asociacion Argentina de Astronomia, 6, 41
    - Oguri 2021, PASP, 133, 074504  (arXiv:2106.11464)
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 0.6,
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
        sersic_index
            The Sérsic index :math:`n` controlling profile concentration
            (lower -> less concentrated, higher -> more concentrated).
        mass_to_light_ratio
            The mass-to-light ratio :math:`\Upsilon` in solar units.
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

    @aa.decorators.to_vector_yx
    @aa.decorators.transform(rotate_back=True)
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
            self.eccentric_radii_grid_from(grid=grid, xp=xp, **kwargs), xp=xp
        )

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
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

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        from autogalaxy.profiles.mass.abstract.mge import MGEDecomposer

        radii_min = self.effective_radius / 100.0
        radii_max = self.effective_radius * 20.0
        sigmas = xp.exp(xp.linspace(xp.log(radii_min), xp.log(radii_max), 30))
        mge_decomp = MGEDecomposer(mass_profile=self)
        return mge_decomp.potential_2d_via_mge_from(
            grid=grid, xp=xp, sigma_log_list=sigmas,
            ellipticity_convention="circularised", three_D=False,
        )

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
                ((radius / self.effective_radius) ** (1.0 / self.sersic_index))
                - 1
            )
        )

    def decompose_convergence_via_cse(
        self, grid_radii: np.ndarray, xp=np
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
            xp=xp,
        )

        scaled_effective_radius = self.effective_radius / xp.sqrt(
            self.axis_ratio(xp)
        )
        radii_min = scaled_effective_radius / 10.0**lower_dex
        radii_max = scaled_effective_radius * 10.0**upper_dex

        def sersic_2d(r):
            return (
                self.mass_to_light_ratio
                * self.intensity
                * xp.exp(
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
            xp=xp,
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
    r"""
    Elliptical Sérsic stellar mass profile.

    Inherits the full Sérsic convergence and CSE deflection-angle machinery from
    :class:`AbstractSersic`.  The convergence is:

    .. math::

        \kappa(R) = \Upsilon \, I_e \exp\!\left\{
            -b_n \left[\left(\frac{R}{R_e}\right)^{1/n} - 1\right]
        \right\}

    where :math:`R` is the elliptical radius, :math:`\Upsilon` is the mass-to-light
    ratio, :math:`I_e` is the intensity at the effective radius :math:`R_e`, and
    :math:`b_n` is determined from the Sérsic index :math:`n`.

    References
    ----------
    - Sérsic 1963, Boletin de la Asociacion Argentina de Astronomia, 6, 41
    - Oguri 2021, PASP, 133, 074504  (arXiv:2106.11464)
    """

    pass


class SersicSph(Sersic):
    r"""
    Spherical Sérsic stellar mass profile.

    A special case of :class:`Sersic` with no ellipticity (:math:`q = 1`).  The
    convergence is evaluated on a circular radial grid:

    .. math::

        \kappa(r) = \Upsilon \, I_e \exp\!\left\{
            -b_n \left[\left(\frac{r}{R_e}\right)^{1/n} - 1\right]
        \right\}

    References
    ----------
    - Sérsic 1963, Boletin de la Asociacion Argentina de Astronomia, 6, 41
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 0.6,
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
        sersic_index
            The Sérsic index :math:`n` controlling profile concentration
            (lower -> less concentrated, higher -> more concentrated).
        mass_to_light_ratio
            The mass-to-light ratio :math:`\Upsilon` in solar units.
        """
        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
        )
