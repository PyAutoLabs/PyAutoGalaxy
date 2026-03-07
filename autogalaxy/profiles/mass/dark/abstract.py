import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.cosmology.model import LensingCosmology


from autogalaxy import exc


class DarkProfile:
    pass


class AbstractgNFW(MassProfile, DarkProfile):
    epsrel = 1.49e-5

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        kappa_s: float = 0.05,
        inner_slope: float = 1.0,
        scale_radius: float = 1.0,
    ):
        """
        The elliptical NFW profiles, used to fit the dark matter halo of the lens.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        kappa_s
            The overall normalization of the dark matter halo \
            (kappa_s = (rho_s * scale_radius)/lensing_critical_density)
        inner_slope
            The inner slope of the dark matter halo
        scale_radius
            The NFW scale radius as an angle on the sky in arc-seconds. For a regular NFW halo, \
            the scale radius is roughly where the log-slope of the profile changes from -1 to -3.
        """

        super().__init__(centre=centre, ell_comps=ell_comps)

        self.kappa_s = kappa_s
        self.scale_radius = scale_radius
        self.inner_slope = inner_slope

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

        grid_eta = self.elliptical_radii_grid_from(grid=grid, xp=xp, **kwargs)

        return self.convergence_func(grid_radius=grid_eta, xp=xp)

    def tabulate_integral(self, grid, tabulate_bins, **kwargs):
        """Tabulate an integral over the convergence of deflection potential of a mass profile. This is used in \
        the gNFW profile classes to speed up the integration procedure.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the potential / deflection_stacks are computed on.
        tabulate_bins
            The number of bins to tabulate the inner integral of this profile.
        """
        eta_min = 1.0e-4
        eta_max = 1.05 * np.max(self.elliptical_radii_grid_from(grid=grid, **kwargs))

        minimum_log_eta = np.log10(eta_min)
        maximum_log_eta = np.log10(eta_max)
        bin_size = (maximum_log_eta - minimum_log_eta) / (tabulate_bins - 1)

        return eta_min, eta_max, minimum_log_eta, maximum_log_eta, bin_size

    def density_3d_func(self, r, xp=np):
        x = r.array / self.scale_radius

        rho_at_scale_radius = (
            self.kappa_s / self.scale_radius
        )  # density parameter of 3D gNFW

        return (
            rho_at_scale_radius
            * x ** (-self.inner_slope)
            * (1.0 + x) ** (self.inner_slope - 3.0)
        )

    def coord_func_f(self, grid_radius: np.ndarray, xp=np) -> np.ndarray:
        """
        Given an array `grid_radius` and a work array `f`, fill f[i] with

          • (1 / sqrt(r_i^2 − 1)) * arccos(1 / r_i)   if r_i > 1
          • (1 / sqrt(1 − r_i^2)) * arccosh(1 / r_i)  if r_i < 1
          • leave f[i] unchanged if r_i == 1 (you can adjust this if you want a different convention)

        This version uses only pure JAX ops, so it can be jitted or grad’ed without
        any Python control flow on tracer values.
        """
        if isinstance(grid_radius, float) or isinstance(grid_radius, complex):
            grid_radius = xp.array([grid_radius])

        f = xp.ones(shape=grid_radius.shape[0], dtype="complex64")

        # compute both branches
        r = grid_radius
        inv_r = 1.0 / r

        # branch for r > 1
        out_gt = (1.0 / xp.sqrt(r**2 - 1.0)) * xp.arccos(inv_r)

        # branch for r < 1
        out_lt = (1.0 / xp.sqrt(1.0 - r**2)) * xp.arccosh(inv_r)

        # combine: if r>1 pick out_gt, elif r<1 pick out_lt, else keep original f
        return xp.where(r > 1.0, out_gt, xp.where(r < 1.0, out_lt, f))

    def coord_func_g(self, grid_radius: np.ndarray, xp=np) -> np.ndarray:
        """
        Vectorized version of the original looped `coord_func_g_jit`.

        Applies conditional logic element-wise:
          - if r > 1: g = (1 - f_r) / (r^2 - 1)
          - if r < 1: g = (f_r - 1) / (1 - r^2)
          - else:     g = 1/3

        Parameters
        ----------
        grid_radius : np.ndarray
            The input grid radius values.
        f_r : np.ndarray
            Precomputed values from `coord_func_f`.
        g : np.ndarray
            Output array (will be overwritten).

        Returns
        -------
        np.ndarray
            The updated `g` array.
        """
        # Convert single values to JAX arrays
        if isinstance(grid_radius, (float, complex)):
            grid_radius = xp.array([grid_radius], dtype=xp.complex64)

        # Evaluate f_r
        f_r = self.coord_func_f(grid_radius=grid_radius, xp=xp)

        r = xp.real(grid_radius)
        r2 = r**2

        return xp.where(
            r > 1.0,
            (1.0 - f_r) / (r2 - 1.0),
            xp.where(
                r < 1.0,
                (f_r - 1.0) / (1.0 - r2),
                1.0 / 3.0,
            ),
        )

    def coord_func_h(self, grid_radius, xp=np):
        return xp.log(grid_radius / 2.0) + self.coord_func_f(
            grid_radius=grid_radius, xp=xp
        )

    def rho_at_scale_radius_solar_mass_per_kpc3(
        self, redshift_object, redshift_source, cosmology: LensingCosmology = None
    ):
        """
        The Cosmic average density is defined at the redshift of the profile.
        """

        from autogalaxy.cosmology.model import Planck15

        cosmology = cosmology or Planck15()

        critical_surface_density = cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
            redshift_0=redshift_object, redshift_1=redshift_source
        )

        kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=redshift_object)

        return (
            self.kappa_s
            * critical_surface_density
            / (self.scale_radius * kpc_per_arcsec)
        )

    def delta_concentration(
        self,
        redshift_object,
        redshift_source,
        redshift_of_cosmic_average_density="profile",
        cosmology: LensingCosmology = None,
    ):

        from autogalaxy.cosmology.model import Planck15

        cosmology = cosmology or Planck15()

        if redshift_of_cosmic_average_density == "profile":
            redshift_calc = redshift_object
        elif redshift_of_cosmic_average_density == "local":
            redshift_calc = 0.0
        else:
            raise exc.UnitsException(
                "The redshift of the cosmic average density haas been specified as an invalid "
                "string. Must be {local, profile}"
            )

        cosmic_average_density = (
            cosmology.cosmic_average_density_solar_mass_per_kpc3_from(
                redshift=redshift_calc
            )
        )

        rho_scale_radius = self.rho_at_scale_radius_solar_mass_per_kpc3(
            redshift_object=redshift_object,
            redshift_source=redshift_source,
            cosmology=cosmology,
        )

        return rho_scale_radius / cosmic_average_density

    def concentration(
        self,
        redshift_profile,
        redshift_source,
        redshift_of_cosmic_average_density="profile",
        cosmology: LensingCosmology = None,
        xp=np,
    ):

        from autogalaxy.cosmology.model import Planck15

        cosmology = cosmology or Planck15()

        from scipy.optimize import fsolve

        """
        Computes the NFW halo concentration, `c_{200m}`
        """
        delta_concentration = self.delta_concentration(
            redshift_object=redshift_profile,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
        )

        return fsolve(
            func=self.concentration_func,
            x0=10.0,
            args=(delta_concentration, xp),
        )[0]

    @staticmethod
    def concentration_func(concentration, delta_concentration, xp=np):
        return (
            200.0
            / 3.0
            * (
                concentration
                * concentration
                * concentration
                / (xp.log(1 + concentration) - concentration / (1 + concentration))
            )
            - delta_concentration
        )

    def radius_at_200(
        self,
        redshift_object,
        redshift_source,
        redshift_of_cosmic_average_density="profile",
        cosmology: LensingCosmology = None,
        xp=np,
    ):
        """
        Returns `r_{200m}` for this halo in **arcseconds**
        """
        from autogalaxy.cosmology.model import Planck15

        cosmology = cosmology or Planck15()

        concentration = self.concentration(
            redshift_profile=redshift_object,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
            xp=xp,
        )

        return concentration * self.scale_radius

    def mass_at_200_solar_masses(
        self,
        redshift_object,
        redshift_source,
        redshift_of_cosmic_average_density="profile",
        cosmology: LensingCosmology = None,
        xp=np,
    ):
        from autogalaxy.cosmology.model import Planck15

        cosmology = cosmology or Planck15()

        """
        Returns `M_{200m}` of this NFW halo, in solar masses, at the given cosmology.
        """
        if redshift_of_cosmic_average_density == "profile":
            redshift_calc = redshift_object
        elif redshift_of_cosmic_average_density == "local":
            redshift_calc = 0.0
        else:
            raise exc.UnitsException(
                "The redshift of the cosmic average density haas been specified as an invalid "
                "string. Must be {local, profile}"
            )

        cosmic_average_density = (
            cosmology.cosmic_average_density_solar_mass_per_kpc3_from(
                redshift=redshift_calc
            )
        )

        radius_at_200 = self.radius_at_200(
            redshift_object=redshift_object,
            redshift_source=redshift_source,
            redshift_of_cosmic_average_density=redshift_of_cosmic_average_density,
            cosmology=cosmology,
            xp=xp,
        )

        kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=redshift_object)

        radius_at_200_kpc = radius_at_200 * kpc_per_arcsec

        return (
            200.0
            * ((4.0 / 3.0) * xp.pi)
            * cosmic_average_density
            * (radius_at_200_kpc**3.0)
        )

    @property
    def ellipticity_rescale(self):
        return 1.0 - ((1.0 - self.axis_ratio()) / 2.0)
