"""
Abstract base class for all mass profiles in **PyAutoGalaxy** and **PyAutoLens**.

A mass profile describes the projected mass distribution of a galaxy and exposes three fundamental lensing
quantities:

- `deflections_yx_2d_from` — the deflection angles α(θ) that describe how light rays are bent.
- `convergence_2d_from` — the dimensionless surface mass density κ(θ) = Σ(θ) / Σ_cr.
- `potential_2d_from` — the lensing (Shapiro) potential ψ(θ).

Every other lensing observable (shear, magnification, critical curves, Einstein radius, Fermat potential) can
be derived from these three quantities. See the `autogalaxy.operate.lens_calc` module for the `LensCalc` class
that derives these secondary quantities.
"""
import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.geometry_profiles import EllProfile


class MassProfile(EllProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
    ):
        """
        Abstract class for elliptical mass profiles.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        """
        super().__init__(centre=centre, ell_comps=ell_comps)

    @staticmethod
    def radial_deflection_from(r, params, xp):
        raise NotImplementedError(
            "vmapped_deflections_from is not yet supported for this profile. "
            "Only spherical profiles with a radial_deflection_from "
            "implementation are currently supported. If you need this for "
            "your profile, please contact James "
            "(jnightingale2211@gmail.com) or open a GitHub issue."
        )

    @classmethod
    def vmapped_deflections_from(cls, grid, params_batch, mask, xp=None):
        import jax
        import jax.numpy as jnp

        if xp is None:
            xp = jnp

        def single(params):
            centred = grid - params[:2]
            r = xp.sqrt(centred[:, 0] ** 2 + centred[:, 1] ** 2)
            defl_r = cls.radial_deflection_from(r, params[2:], xp)
            angle = xp.arctan2(centred[:, 0], centred[:, 1])
            return defl_r[:, None] * xp.stack(
                [xp.sin(angle), xp.cos(angle)], axis=-1
            )

        all_defl = jax.vmap(single)(params_batch)
        return xp.sum(
            xp.where(mask[:, None, None], all_defl, 0.0), axis=0
        )

    def deflections_yx_2d_from(self, grid):
        """
        Returns the 2D deflection angles of the mass profile from a 2D grid of Cartesian (y,x) coordinates.

        The deflection angle α(θ) at image-plane position θ describes how a light ray is bent by the
        gravitational field of the lens. The source-plane position β is then:

            β = θ − α(θ)

        Deflection angles are the single most important output of a mass profile — every other lensing quantity
        (convergence, shear, magnification, critical curves, caustics) can be derived from them.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where the deflection angles are evaluated.

        Returns
        -------
        aa.VectorYX2D
            The (y, x) deflection angles at every coordinate on the input grid.
        """
        raise NotImplementedError

    def deflections_2d_via_potential_2d_from(self, grid):
        """
        Returns the 2D deflection angles of the mass profile by numerically differentiating the lensing
        potential on the input grid.

        This is a fallback implementation that computes deflection angles as the gradient of the potential via
        finite differences:

            α_y = ∂ψ/∂y,  α_x = ∂ψ/∂x

        Most concrete mass profiles override `deflections_yx_2d_from` with an analytic expression. This
        method is provided for cross-checking and for profiles where only the potential is known analytically.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where the deflection angles are evaluated.

        Returns
        -------
        aa.Grid2D
            The (y, x) deflection angles at every coordinate on the input grid, computed via finite differences
            of the lensing potential.
        """
        potential = self.potential_2d_from(grid=grid)

        deflections_y_2d = np.gradient(
            potential.native.array, grid.native.array[:, 0, 0], axis=0
        )
        deflections_x_2d = np.gradient(
            potential.native.array, grid.native.array[0, :, 1], axis=1
        )

        return aa.Grid2D(
            values=np.stack((deflections_y_2d, deflections_x_2d), axis=-1),
            mask=grid.mask,
        )

    def convergence_2d_from(self, grid, xp=np):
        """
        Returns the 2D convergence of the mass profile from a 2D grid of Cartesian (y,x) coordinates.

        The convergence κ(θ) is the dimensionless surface mass density of the lens, defined as the projected
        surface mass density Σ(θ) divided by the critical surface mass density Σ_cr:

            κ(θ) = Σ(θ) / Σ_cr

        Physically, κ = 1 on the Einstein ring. Regions with κ > 1 produce multiple images.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where the convergence is evaluated.

        Returns
        -------
        aa.Array2D
            The convergence κ(θ) at every coordinate on the input grid.
        """
        raise NotImplementedError

    def convergence_func(self, grid_radius: float) -> float:
        """
        Returns the convergence of the mass profile as a function of the radial coordinate.

        This is used to integrate the convergence profile to compute enclosed masses and the Einstein radius.

        Parameters
        ----------
        grid_radius
            The radial distance from the profile centre at which the convergence is evaluated.

        Returns
        -------
        float
            The convergence at the input radial distance.
        """
        raise NotImplementedError

    def potential_2d_from(self, grid):
        """
        Returns the 2D lensing potential of the mass profile from a 2D grid of Cartesian (y,x) coordinates.

        The lensing potential ψ(θ) is the gravitational (Shapiro) time-delay term. It quantifies how much the
        passage of light through the gravitational field delays its arrival relative to a straight-line path in
        empty space.

        The potential enters directly into the Fermat potential:

            φ(θ) = ½ |θ − β|²  −  ψ(θ)

        which governs time delays between multiple lensed images of the same source.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where the lensing potential is evaluated.

        Returns
        -------
        aa.Array2D
            The lensing potential ψ(θ) at every coordinate on the input grid.
        """
        raise NotImplementedError

    def potential_func(self, u, y, x):
        """
        Returns the integrand of the lensing potential at a single point, used in numerical integration schemes
        for computing the potential from the mass profile's convergence.

        Parameters
        ----------
        u
            The integration variable.
        y
            The y-coordinate of the point at which the potential is evaluated.
        x
            The x-coordinate of the point at which the potential is evaluated.
        """
        raise NotImplementedError

    def mass_integral(self, x, xp=np):
        """
        Integrand used by `mass_angular_within_circle_from` to compute the total projected mass within a circle.

        This integrates 2π r κ(r) to give the enclosed convergence (dimensionless mass) at radius `x`.

        Parameters
        ----------
        x
            The radial coordinate at which the integrand is evaluated.
        """
        return 2 * xp.pi * x * self.convergence_func(grid_radius=aa.ArrayIrregular(x))

    @property
    def ellipticity_rescale(self):
        """
        A rescaling factor applied to account for the ellipticity of the mass profile when computing the
        Einstein radius from the average convergence equals unity criterion.

        For a spherical profile this is 1.0. Elliptical profiles return a factor that maps the elliptical
        enclosed mass to an equivalent circular Einstein radius.
        """
        return NotImplementedError()

    def mass_angular_within_circle_from(self, radius: float):
        """
        Integrate the mass profiles's convergence profile to compute the total mass within a circle of
        specified radius. This is centred on the mass profile.

        Parameters
        ----------
        radius : dim.Length
            The radius of the circle to compute the dimensionless mass within.
        """
        from scipy.integrate import quad

        return quad(self.mass_integral, a=0.0, b=radius)[0]

    def density_between_circular_annuli(
        self, inner_annuli_radius: float, outer_annuli_radius: float
    ):
        """
        Calculate the mass between two circular annuli and compute the density by dividing by the annuli surface
        area.

        The value returned by the mass integral is dimensionless, therefore the density between annuli is returned in \
        unit_label of inverse radius squared. A conversion factor can be specified to convert this to a physical value \
        (e.g. the critical surface mass density).

        Parameters
        ----------
        inner_annuli_radius
            The radius of the inner annulus outside of which the density are estimated.
        outer_annuli_radius
            The radius of the outer annulus inside of which the density is estimated.
        """
        annuli_area = (np.pi * outer_annuli_radius**2.0) - (
            np.pi * inner_annuli_radius**2.0
        )

        outer_mass = self.mass_angular_within_circle_from(radius=outer_annuli_radius)

        inner_mass = self.mass_angular_within_circle_from(radius=inner_annuli_radius)

        return (outer_mass - inner_mass) / annuli_area

    @property
    def average_convergence_of_1_radius(self):
        """
        The radius a critical curve forms for this mass profile, e.g. where the mean convergence is equal to 1.0.

        In case of ellipitical mass profiles, the 'average' critical curve is used, whereby the convergence is \
        rescaled into a circle using the axis ratio.

        This radius corresponds to the Einstein radius of the mass profile, and is a property of a number of \
        mass profiles below.
        """
        from scipy.optimize import root_scalar

        def func(radius):
            return (
                self.mass_angular_within_circle_from(radius=radius)
                - np.pi * radius**2.0
            )

        return self.ellipticity_rescale() * root_scalar(func, bracket=[1e-4, 1e4]).root

    def extract_attribute(self, cls, attr_name):
        """
        Returns an attribute of a class and its children profiles in the galaxy as a `ValueIrregular`
        or `Grid2DIrregular` object.

        For example, if a galaxy has two light profiles and we want the `LightProfile` axis-ratios, the following:

        `galaxy.extract_attribute(cls=LightProfile, name="axis_ratio"`

        would return:

        ArrayIrregular(values=[axis_ratio_0, axis_ratio_1])

        If a galaxy has three mass profiles and we want the `MassProfile` centres, the following:

        `galaxy.extract_attribute(cls=MassProfile, name="centres"`

         would return:

        GridIrregular2D(grid=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1), (centre_y_2, centre_x_2)])

        This is used for visualization, for example plotting the centres of all light profiles colored by their profile.
        """

        if isinstance(self, cls):
            if hasattr(self, attr_name):
                attribute = getattr(self, attr_name)

                if isinstance(attribute, float):
                    return aa.ArrayIrregular(values=[attribute])
                if isinstance(attribute, tuple):
                    return aa.Grid2DIrregular(values=[attribute])
