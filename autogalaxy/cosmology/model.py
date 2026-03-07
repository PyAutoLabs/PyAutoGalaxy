import numpy as np


class LensingCosmology:
    """
    Class containing specific functions for performing gravitational lensing cosmology calculations.

    This version is JAX-compatible by using an explicit `xp` backend (NumPy or jax.numpy).
    """

    def arcsec_per_kpc_proper(self, z: float, xp=np) -> float:
        """
        Angular separation in arcsec corresponding to 1 proper kpc at redshift z.

        This matches astropy.cosmology.arcsec_per_kpc_proper.

        Proper transverse distance uses the angular diameter distance D_A(z):

            arcsec_per_kpc_proper = 206264.806247 / D_A(z)

        where D_A(z) is in kpc.
        """

        angular_diameter_distance_kpc = self.angular_diameter_distance_kpc_z1z2(
            0.0, z, xp=xp
        )

        return xp.asarray(206264.806247) / angular_diameter_distance_kpc

    def kpc_proper_per_arcsec(self, z: float, xp=np) -> float:
        """
        Proper transverse separation in kpc corresponding to 1 arcsec at redshift z.

        This matches the inverse of astropy.cosmology.arcsec_per_kpc_proper:

            kpc_proper_per_arcsec = D_A(z) / 206264.806247
        """

        angular_diameter_distance_kpc = self.angular_diameter_distance_kpc_z1z2(
            0.0, z, xp=xp
        )

        return angular_diameter_distance_kpc / xp.asarray(206264.806247)

    def arcsec_per_kpc_from(self, redshift: float, xp=np) -> float:
        """
        Angular separation in arcsec corresponding to a proper kpc at redshift `z`.

        For simplicity, **PyAutoLens** internally uses only certain units to perform lensing cosmology calculations.

        This is a thin convenience wrapper around `arcsec_per_kpc_proper`.
        """
        return self.arcsec_per_kpc_proper(z=redshift, xp=xp)

    def kpc_per_arcsec_from(self, redshift: float, xp=np) -> float:
        """
        Separation in transverse proper kpc corresponding to an arcsec at redshift `z`.

        For simplicity, **PyAutoLens** internally uses only certain units to perform lensing cosmology calculations.

        This is a thin convenience wrapper around `kpc_proper_per_arcsec`.
        """
        return self.kpc_proper_per_arcsec(z=redshift, xp=xp)

    def luminosity_distance(self, z: float, xp=np) -> float:
        """
        Luminosity distance to redshift z in Mpc.

        For a flat universe:

            D_L(z) = (1 + z)^2 * D_A(0, z)

        where D_A(0, z) is the angular diameter distance from Earth.

        Returns Mpc, matching the convention of astropy.cosmology.FlatLambdaCDM.luminosity_distance(z).value.

        Parameters
        ----------
        z
            Redshift at which the luminosity distance is calculated.
        """
        D_A_kpc = self.angular_diameter_distance_to_earth_in_kpc_from(redshift=z, xp=xp)
        D_A_Mpc = D_A_kpc / xp.asarray(1.0e3)
        return (xp.asarray(1.0) + xp.asarray(z)) ** 2 * D_A_Mpc

    def angular_diameter_distance_to_earth_in_kpc_from(
        self, redshift: float, xp=np
    ) -> float:
        """
        Angular diameter distance from the input `redshift` to redshift zero (e.g. us, the observer on earth) in
        kiloparsecs.

        This gives the proper (sometimes called 'physical') transverse distance corresponding to an angle of 1 radian
        for an object at redshift `z`.

        Weinberg, 1972, pp 421-424; Weedman, 1986, pp 65-67; Peebles, 1993, pp 325-327.

        For simplicity, **PyAutoLens** internally uses only certain units to perform lensing cosmology calculations.
        This function therefore returns only the value of the astropy function it wraps, omitting the units instance.

        Parameters
        ----------
        redshift
            Input redshift from which the angular diameter distance to Earth is calculated.
        """
        return self.angular_diameter_distance_kpc_z1z2(0.0, redshift, xp=xp)

    def angular_diameter_distance_between_redshifts_in_kpc_from(
        self, redshift_0: float, redshift_1: float, xp=np
    ) -> float:
        """
        Angular diameter distance from an input `redshift_0` to another input `redshift_1`.

        For simplicity, **PyAutoLens** internally uses only certain units to perform lensing cosmology calculations.
        This function therefore returns only the value of the astropy function it wraps, omitting the units instance.

        Parameters
        ----------
        redshift_0
            Redshift from which the angular diameter distance to the other redshift is calculated.
        redshift_1
            Redshift from which the angular diameter distance to the other redshift is calculated.
        """
        return self.angular_diameter_distance_kpc_z1z2(redshift_0, redshift_1, xp=xp)

    def cosmic_average_density_from(self, redshift: float, xp=np):
        """
        Critical density of the Universe at redshift z in units of solar masses,
        scaled into arcsec-based internal lensing units.

        Parameters
        ----------
        redshift
            Redshift at which the critiical density in solMass of the Universe is calculated.
        """

        rho_kpc3 = self.cosmic_average_density_solar_mass_per_kpc3_from(redshift, xp=xp)
        kpc_per_arcsec = self.kpc_per_arcsec_from(redshift, xp=xp)
        return rho_kpc3 * kpc_per_arcsec**3

    def cosmic_average_density_solar_mass_per_kpc3_from(
        self,
        redshift: float,
        xp=np,
    ):
        """
        Critical density of the Universe at redshift z in units of Msun / kpc^3.

        JAX / NumPy compatible via `xp`.

        Computes:

            rho_c(z) = 3 H(z)^2 / (8 pi G)

        Returns physical density (not scaled into arcsec units).
        """

        # -----------------------------
        # Physical constants
        # -----------------------------

        # Gravitational constant in kpc^3 / (Msun s^2)
        G = xp.asarray(4.30091e-6)  # kpc (km/s)^2 / Msun
        G = G / xp.asarray((3.085677581e16) ** 2)

        # H0 in km/s/Mpc → convert to 1/s
        H0_km_s_Mpc = xp.asarray(self.H0)
        H0_s = H0_km_s_Mpc / xp.asarray(3.085677581e19)

        # Dimensionless expansion factor
        Ez = self.E(redshift, xp=xp)

        # H(z) in 1/s
        Hz = H0_s * Ez

        # -----------------------------
        # Critical density in Msun/kpc^3
        # -----------------------------

        rho_crit = (3.0 * Hz**2) / (8.0 * xp.pi * G)

        return rho_crit

    def critical_density(self, z: float, xp=np):
        """
        Critical density of the Universe at redshift z, returned in Msun / kpc^3.

        This is an xp (NumPy / JAX) drop-in for the Astropy method:
            astropy.cosmology.FLRW.critical_density(z)

        Astropy returns g/cm^3, but in AutoLens you were immediately converting to:
            solMass / kpc^3

        So this function returns Msun / kpc^3 directly.

        Requires:
            self.H0  (km/s/Mpc)
            self.E(z, xp=xp)  where E(z) = H(z)/H0
        """

        z = xp.asarray(z)

        # H0 in km/s/Mpc -> 1/s
        # 1 Mpc = 3.085677581e19 km
        H0_s = xp.asarray(self.H0) / xp.asarray(3.085677581e19)

        # H(z) in 1/s
        Hz = H0_s * self.E(z, xp=xp)

        # Gravitational constant G in kpc^3 / (Msun s^2)
        # Start from: G = 4.30091e-6 kpc (km/s)^2 / Msun
        # Convert (km/s)^2 -> (kpc/s)^2 by dividing by (km per kpc)^2
        km_per_kpc = xp.asarray(3.085677581e16)
        G = xp.asarray(4.30091e-6) / (km_per_kpc**2)

        # rho_c = 3 H(z)^2 / (8 pi G)  [Msun / kpc^3]
        return (xp.asarray(3.0) * Hz**2) / (xp.asarray(8.0) * xp.pi * G)

    def critical_surface_density_between_redshifts_from(
        self,
        redshift_0: float,
        redshift_1: float,
        xp=np,
    ):
        """
        Critical surface density scaled into AutoLens angular units (Msun / arcsec^2).

        This is:
            Sigma_crit_arcsec2 = Sigma_crit_kpc2 * (kpc_per_arcsec(z_l))^2
        """
        sigma_crit_kpc2 = (
            self.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
                redshift_0=redshift_0, redshift_1=redshift_1, xp=xp
            )
        )

        kpc_per_arcsec = self.kpc_per_arcsec_from(redshift=redshift_0, xp=xp)

        return sigma_crit_kpc2 * kpc_per_arcsec**2.0

    def critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
        self,
        redshift_0: float,
        redshift_1: float,
        xp=np,
    ):
        """
        Critical surface density in physical units (Msun / kpc^2):

            Sigma_crit = (c^2 / (4*pi*G)) * D_s / (D_l * D_ls)

        Distances must be angular diameter distances in kpc.

        JAX/NumPy compatible via `xp` (pass `jax.numpy` as xp).
        """

        # Distances in kpc
        D_l_kpc = self.angular_diameter_distance_to_earth_in_kpc_from(
            redshift=redshift_0, xp=xp
        )
        D_s_kpc = self.angular_diameter_distance_to_earth_in_kpc_from(
            redshift=redshift_1, xp=xp
        )
        D_ls_kpc = self.angular_diameter_distance_between_redshifts_in_kpc_from(
            redshift_0=redshift_0, redshift_1=redshift_1, xp=xp
        )

        # kpc -> m
        kpc_to_m = xp.asarray(3.085677581491367e19)
        D_l = D_l_kpc * kpc_to_m
        D_s = D_s_kpc * kpc_to_m
        D_ls = D_ls_kpc * kpc_to_m

        # SI constants
        c = xp.asarray(299792458.0)  # m/s
        G = xp.asarray(6.67430e-11)  # m^3/(kg s^2)
        Msun = xp.asarray(1.98847e30)  # kg

        # Sigma_crit in kg / m^2
        prefac = (c * c) / (xp.asarray(4.0) * xp.pi * G)
        sigma_SI = prefac * D_s / (D_l * D_ls)

        # kg/m^2 -> Msun/kpc^2
        sigma = sigma_SI * (kpc_to_m * kpc_to_m) / Msun

        return xp.where(D_ls_kpc == xp.asarray(0.0), xp.asarray(np.inf), sigma)

    def scaling_factor_between_redshifts_from(
        self,
        redshift_0: float,
        redshift_1: float,
        redshift_final: float,
        xp=np,
    ):
        """
        For strong lens systems with more than 2 planes, the deflection angles between different planes must be scaled
        by the angular diameter distances between the planes in order to properly perform multi-plane ray-tracing. This
        function computes the factor to scale deflections between `redshift_0` and `reshift_final`, to deflections between
        `redshift_0` and `redshift_1`.

        The second redshift should be strictly larger than the first. The scaling factor is unity when `redshift_1`
        is `redshift_final`, and 0 when `redshift_0` is equal to `redshift_1`.

        For a system with a first lens galaxy l0 at `redshift_0`, second lens galaxy l1 at `redshift_1` and final
        source galaxy at `redshift_final` this scaling factor is given by:

        (D_l0l1 * D_s) / (D_l1* D_l0s)

        The critical surface density for lensing, often written as $\\sigma_{cr}$, is given by:

        critical_surface_density = (c^2 * D_s) / (4 * pi * G * D_ls * D_l)

        D_l0l1 = Angular diameter distance of first lens redshift to second lens redshift.
        D_s = Angular diameter distance of source redshift to earth
        D_l1 = Angular diameter distance of second lens redshift to Earth.
        D_l0s = Angular diameter distance of first lens redshift to source redshift

        For systems with more planes this scaling factor is computed multiple times for the different redshift
        combinations and applied recursively when scaling the deflection angles.

        Parameters
        ----------
        redshift_0
            The redshift of the first strong lens galaxy (the first lens plane).
        redshift_1
            The redshift of the second strong lens galaxy (the second lens plane).
        redshift_final
            The redshift of the final source galaxy (the final source plane).
        """

        # D_l0l1 : between lens plane 0 and lens plane 1
        D_l0l1 = self.angular_diameter_distance_between_redshifts_in_kpc_from(
            redshift_0=redshift_0,
            redshift_1=redshift_1,
            xp=xp,
        )

        # D_s : observer to final source plane
        D_s = self.angular_diameter_distance_to_earth_in_kpc_from(
            redshift=redshift_final,
            xp=xp,
        )

        # D_l1 : observer to lens plane 1
        D_l1 = self.angular_diameter_distance_to_earth_in_kpc_from(
            redshift=redshift_1,
            xp=xp,
        )

        # D_l0s : between lens plane 0 and final source plane
        D_l0s = self.angular_diameter_distance_between_redshifts_in_kpc_from(
            redshift_0=redshift_0,
            redshift_1=redshift_final,
            xp=xp,
        )

        return (D_l0l1 * D_s) / (D_l1 * D_l0s)

    def velocity_dispersion_from(
        self, redshift_0: float, redshift_1: float, einstein_radius: float, xp=np
    ) -> float:
        """
        For a strong lens galaxy with an Einstein radius in arcseconds, the corresponding velocity dispersion of the
        lens galaxy can be computed (assuming an isothermal mass distribution).

        The velocity dispersion is given by:

        sigma_v = c * sqrt((R_Ein * D_s) / (4 * pi * D_l * D_ls))

        c = speed of light
        D_s = Angular diameter distance of source redshift to earth
        D_ls = Angular diameter distance of lens redshift to source redshift
        D_l = Angular diameter distance of lens redshift to earth

        Parameters
        ----------
        redshift_0
            The redshift of the first strong lens galaxy (the lens).
        redshift_1
            The redshift of the second strong lens galaxy (the source).
        """

        # Speed of light in km / s
        c_km_s = xp.asarray(299792.458)

        # Angular diameter distances in kpc
        D_l = self.angular_diameter_distance_to_earth_in_kpc_from(
            redshift=redshift_0, xp=xp
        )

        D_s = self.angular_diameter_distance_to_earth_in_kpc_from(
            redshift=redshift_1, xp=xp
        )

        D_ls = self.angular_diameter_distance_between_redshifts_in_kpc_from(
            redshift_0=redshift_0, redshift_1=redshift_1, xp=xp
        )

        # Convert Einstein radius to kpc
        kpc_per_arcsec = self.kpc_per_arcsec_from(redshift=redshift_0, xp=xp)
        R_ein_kpc = xp.asarray(einstein_radius) * kpc_per_arcsec

        # Velocity dispersion (km/s)
        sigma_v = c_km_s * xp.sqrt(
            (R_ein_kpc * D_s) / (xp.asarray(4.0) * xp.pi * D_l * D_ls)
        )

        return sigma_v


class FlatLambdaCDM(LensingCosmology):
    def __init__(
        self,
        H0: float = 67.66,
        Om0: float = 0.30966,
        Tcmb0: float = 2.7255,
        Neff: float = 3.046,
        m_nu: float = 0.0,
        Ob0: float = 0.04897,
    ):
        """
        A JAX-compatible implementation of the FlatLambdaCDM cosmology model, which allows it to be used for
        modeling such that the cosmological parameters are free parameters which can be fitted for.

        This is a custom implementation designed to work with both NumPy and JAX backends, enabling automatic
        differentiation and GPU acceleration when needed. The interface and parameter naming conventions follow
        the astropy `FlatLambdaCDM` class for compatibility.

        The class inherits from `LensingCosmology`, which provides additional functionality for calculating
        lensing specific quantities in the cosmology.

        Parameters
        ----------
        H0
            The Hubble constant at z=0.
        Om0
            The total matter density at z=0.
        Tcmb0
            The CMB temperature at z=0.
        Neff
            The effective number of neutrinos.
        m_nu
            The sum of the neutrino masses.
        Ob0
            The baryon density at z=0.
        """
        self.H0 = H0
        self.Om0 = Om0
        self.Tcmb0 = Tcmb0
        self.Neff = Neff
        self.m_nu = m_nu
        self.Ob0 = Ob0

        # Make ΛCDM a special case of wCDM
        self.w0 = -1.0

    @staticmethod
    def _simpson_1d(y, x, xp=np):
        """
        Composite Simpson's rule on a 1D grid.

        Requirements:
        - x is 1D, evenly spaced
        - len(x) is odd  (i.e. number of intervals is even)

        Works with numpy or jax.numpy passed as xp.
        """
        n = x.shape[0]
        if (n % 2) == 0:
            raise ValueError(
                "Simpson's rule requires an odd number of samples (even number of intervals)."
            )

        h = (x[-1] - x[0]) / (n - 1)

        # y0 + yN
        s = y[0] + y[-1]

        # 4 * sum of odd indices
        s = s + 4.0 * xp.sum(y[1:-1:2])

        # 2 * sum of even indices (excluding endpoints)
        s = s + 2.0 * xp.sum(y[2:-1:2])

        return (h / 3.0) * s

    def _m_nu_sum_eV(self, xp=np):
        m = xp.asarray(getattr(self, "m_nu", 0.0))
        if getattr(m, "ndim", 0) == 0:
            # scalar means per-species mass (astropy convention)
            n = int(np.floor(float(getattr(self, "Neff", 3.046))))
            return m * xp.asarray(float(n))
        return xp.sum(m)

    def _radiation_and_massive_nu_densities(self, h, xp=np):
        """
        Returns (Or0, Onu_m0) where:
        - Or0 is photons + *massless* neutrino radiation (dimensionless today)
        - Onu_m0 is massive neutrino density today treated as matter-like (dimensionless today)

        Uses a simple split: N_eff_massless = Neff - n_massive.
        """
        Tcmb = xp.asarray(self.Tcmb0)
        Ogamma_h2 = xp.asarray(2.469e-5) * (Tcmb / xp.asarray(2.7255)) ** 4
        Ogamma0 = Ogamma_h2 / (h * h)

        # Massive neutrinos: Omega_nu h^2 ≈ sum(m_nu)/93.14 eV
        m_nu = getattr(self, "m_nu", 0.0)
        m_arr = xp.asarray(m_nu)
        if getattr(m_arr, "ndim", 0) == 0:
            # interpret scalar as total sum (your current convention)
            m_sum = m_arr
            n_massive = xp.asarray(1.0) if m_arr > 0 else xp.asarray(0.0)
        else:
            m_sum = xp.sum(m_arr)
            n_massive = xp.sum(m_arr > xp.asarray(0.0))

        Onu_m_h2 = m_sum / xp.asarray(93.14)
        Onu_m0 = Onu_m_h2 / (h * h)

        # Only the *massless* share of Neff contributes as radiation if we're separately adding massive nu matter
        Neff = xp.asarray(self.Neff)
        Neff_massless = xp.maximum(Neff - n_massive, xp.asarray(0.0))
        Onu_rad0 = Ogamma0 * xp.asarray(0.2271) * Neff_massless

        Or0 = Ogamma0 + Onu_rad0
        return Or0, Onu_m0

    def angular_diameter_distance_kpc_z1z2(
        self,
        z1: float,
        z2: float,
        n_steps: int = 8193,  # odd by default
        xp=np,
    ):
        """
        D_A(z1,z2) in kpc for flat wCDM using Simpson's rule.

        Includes:
        - photons via Tcmb0
        - neutrinos split into:
            * massive part from m_nu (approx matter-like today)
            * massless part from Neff, with the number of massive species subtracted to avoid double-counting

        Notes:
        - Flat universe: Omega_k = 0
        - Dark energy equation of state constant w0
        """
        # Ensure odd number of samples for Simpson (safe: n_steps is a Python int)
        if (n_steps % 2) == 0:
            n_steps += 1

        z1a = xp.asarray(z1)
        z2a = xp.asarray(z2)
        same = z1a == z2a

        c_km_s = xp.asarray(299792.458)

        H0 = xp.asarray(self.H0)
        h = H0 / xp.asarray(100.0)

        Om0 = xp.asarray(self.Om0)
        w0 = xp.asarray(self.w0)

        # ---- photons ----
        Tcmb = xp.asarray(getattr(self, "Tcmb0", 0.0))
        Ogamma_h2 = xp.asarray(2.469e-5) * (Tcmb / xp.asarray(2.7255)) ** 4
        Ogamma0 = Ogamma_h2 / (h * h)

        # ---- massive neutrinos (approx matter-like today) ----
        m_nu_sum = self._m_nu_sum_eV(xp=xp)
        Onu_m_h2 = m_nu_sum / xp.asarray(93.14)
        Onu_m0 = Onu_m_h2 / (h * h)

        # ---- massless neutrino radiation via Neff (avoid double-counting) ----
        Neff = xp.asarray(getattr(self, "Neff", 0.0))

        m_nu = getattr(self, "m_nu", 0.0)
        m_arr = xp.asarray(m_nu)

        if getattr(m_arr, "ndim", 0) == 0:
            n_massive = xp.where(
                m_arr > xp.asarray(0.0), xp.asarray(1.0), xp.asarray(0.0)
            )
        else:
            n_massive = xp.sum(m_arr > xp.asarray(0.0))

        Neff_massless = xp.maximum(Neff - n_massive, xp.asarray(0.0))

        Onu_rad0 = Ogamma0 * xp.asarray(0.2271) * Neff_massless
        Or0 = Ogamma0 + Onu_rad0

        # ---- flatness: Omega_de ----
        Ode0 = xp.asarray(1.0) - Om0 - Or0 - Onu_m0

        def E_local(z):
            zp1 = xp.asarray(1.0) + z
            return xp.sqrt(
                (Om0 + Onu_m0) * zp1**3
                + Or0 * zp1**4
                + Ode0 * zp1 ** (xp.asarray(3.0) * (xp.asarray(1.0) + w0))
            )

        z_grid = xp.linspace(z1a, z2a, n_steps)
        integrand = xp.asarray(1.0) / E_local(z_grid)

        integral = self._simpson_1d(integrand, z_grid, xp=xp)

        Dc_Mpc = (c_km_s / H0) * integral
        Da_Mpc = Dc_Mpc / (xp.asarray(1.0) + z2a)
        Da_kpc = Da_Mpc * xp.asarray(1.0e3)

        return xp.where(same, xp.asarray(0.0), Da_kpc)

    def E(self, z: float, xp=np):
        """
        Dimensionless Hubble parameter E(z) = H(z) / H0.

        JAX/NumPy compatible via `xp` (pass `jax.numpy` as xp).

        Components:
        - Photons: Omega_gamma from Tcmb0
        - Neutrinos:
            * massive part from m_nu (approx matter-like today)
            * massless part from Neff, but we subtract the number of massive species to avoid double-counting
        - Dark energy: constant w0, with Omega_de set by flatness
        """

        z = xp.asarray(z)

        H0 = xp.asarray(self.H0)
        h = H0 / xp.asarray(100.0)

        Om0 = xp.asarray(self.Om0)
        w0 = xp.asarray(getattr(self, "w0", -1.0))

        # ---- photons ----
        Tcmb = xp.asarray(getattr(self, "Tcmb0", 0.0))
        Ogamma_h2 = xp.asarray(2.469e-5) * (Tcmb / xp.asarray(2.7255)) ** 4
        Ogamma0 = Ogamma_h2 / (h * h)

        # ---- massive neutrinos (approx matter-like today) ----
        # NOTE: _m_nu_sum_eV should follow YOUR convention (scalar=total, list=sum, etc.)
        m_nu_sum = self._m_nu_sum_eV(xp=xp)
        Onu_m_h2 = m_nu_sum / xp.asarray(93.14)
        Onu_m0 = Onu_m_h2 / (h * h)

        # ---- massless neutrino radiation via Neff (avoid double-counting) ----
        # Approx: subtract number of massive species from Neff
        Neff = xp.asarray(getattr(self, "Neff", 0.0))

        m_nu = getattr(self, "m_nu", 0.0)
        m_arr = xp.asarray(m_nu)

        if getattr(m_arr, "ndim", 0) == 0:
            n_massive = xp.where(
                m_arr > xp.asarray(0.0), xp.asarray(1.0), xp.asarray(0.0)
            )
        else:
            n_massive = xp.sum(m_arr > xp.asarray(0.0))

        Neff_massless = xp.maximum(Neff - n_massive, xp.asarray(0.0))

        Onu_rad0 = Ogamma0 * xp.asarray(0.2271) * Neff_massless
        Or0 = Ogamma0 + Onu_rad0

        # ---- flatness: Omega_de ----
        Ode0 = xp.asarray(1.0) - Om0 - Or0 - Onu_m0

        zp1 = xp.asarray(1.0) + z

        Ez2 = (
            (Om0 + Onu_m0) * zp1**3
            + Or0 * zp1**4
            + Ode0 * zp1 ** (xp.asarray(3.0) * (xp.asarray(1.0) + w0))
        )

        return xp.sqrt(Ez2)

    def Om(self, z: float, xp=np):
        """
        Matter density parameter at redshift z: Omega_m(z).

        JAX / NumPy compatible via `xp`.

        For flat models using your E(z):
            Om(z) = Om0 (1+z)^3 / E(z)^2
        """
        z = xp.asarray(z)
        zp1 = xp.asarray(1.0) + z

        Ez = self.E(z, xp=xp)
        return xp.asarray(self.Om0) * zp1**3 / (Ez * Ez)


class Planck15(FlatLambdaCDM):

    def __init__(self):

        super().__init__(
            H0=67.74,
            Om0=0.3075,
            Tcmb0=2.7255,
            Neff=3.046,
            m_nu=[0.0, 0.0, 0.06],
            Ob0=0.0486,
        )
