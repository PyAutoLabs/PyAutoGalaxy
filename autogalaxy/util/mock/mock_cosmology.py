"""
Mock cosmology object for use in unit tests.

``MockCosmology`` implements the same interface as the real ``LensingCosmology`` class
(``autogalaxy/cosmology/model.py``) but returns hard-coded constant values rather than
calling ``astropy.cosmology``.  This makes unit tests fast and free of external
dependencies while still exercising code paths that accept a cosmology argument.

The ``Value`` helper wraps a scalar so that ``.to(unit)`` calls (which real astropy
``Quantity`` objects support) are accepted without error.
"""
import math
import numpy as np


class Value:
    def __init__(self, value):
        self.value = value

    def to(self, *args, **kwargs):
        return Value(value=self.value)


class MockCosmology:
    def __init__(
        self,
        arcsec_per_kpc=0.5,
        kpc_per_arcsec=2.0,
        critical_surface_density=2.0,
        cosmic_average_density=2.0,
    ):

        self.arcsec_per_kpc = arcsec_per_kpc
        self.kpc_per_arcsec = kpc_per_arcsec
        self.critical_surface_density = critical_surface_density
        self.cosmic_average_density = cosmic_average_density

    def arcsec_per_kpc_proper(self, z):
        return Value(value=self.arcsec_per_kpc)

    def kpc_per_arcsec_proper(self, z):
        return Value(value=self.kpc_per_arcsec)

    def angular_diameter_distance(self, z):
        return Value(value=1.0)

    def angular_diameter_distance_z1z2(self, z1, z2, xp=np):
        """
        Replacement for the astropy-constants version.

        Computes the factor:

            const = c^2 / (4*pi*G)

        in units of Msun / kpc.

        No astropy.constants required.
        """

        # Speed of light in kpc/s
        c_kpc_s = xp.asarray(299792.458) / xp.asarray(3.085677581e16)

        # Gravitational constant in kpc^3 / (Msun s^2)
        G_kpc3_Msun_s2 = xp.asarray(4.30091e-6) / xp.asarray((3.085677581e16) ** 2)

        const = (c_kpc_s**2) / (xp.asarray(4.0) * xp.pi * G_kpc3_Msun_s2)

        return Value(value=self.critical_surface_density * const)

    def critical_density(self, z):
        return Value(value=self.cosmic_average_density)

    def cosmic_average_density_solar_mass_per_kpc3_from(self, redshift: float) -> float:
        """
        Critical density of the Universe at an input `redshift` in units of solar masses per kiloparsecs**3.

        For simplicity, **PyAutoLens** internally uses only certain units to perform lensing cosmology calculations.
        This function therefore returns only the value of the astropy function it wraps, omitting the units instance.

        Parameters
        ----------
        redshift
            Redshift at which the critiical density in solMass/kpc^3 of the Universe is calculated.
        """
        cosmic_average_density_kpc = (
            self.critical_density(z=redshift).to("solMass / kpc^3").value
        )

        return cosmic_average_density_kpc

    def critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
        self, redshift_0: float, redshift_1: float
    ) -> float:
        """
        The critical surface density for lensing, often written as $\sigma_{cr}$, is given by:

        critical_surface_density = (c^2 * D_s) / (4 * pi * G * D_ls * D_l)

        c = speed of light
        G = Newton's gravity constant
        D_s = Angular diameter distance of source redshift to earth
        D_ls = Angular diameter distance of lens redshift to source redshift
        D_l = Angular diameter distance of lens redshift to earth

        This function returns the critical surface density in units of solar masses / kpc^2, which are convenient
        units for converting the inferred masses of a model from angular units (e.g. dimensionless units inferred
        from data in arcseconds) to solar masses.

        Parameters
        ----------
        redshift_0
            The redshift of the first strong lens galaxy (E.g. the lens galaxy) for which the critical surface
            density is calculated.
        redshift_1
            The redshift of the second strong lens galaxy (E.g. the lens galaxy) for which the critical surface
            density is calculated.
        """
        from astropy import constants

        const = constants.c.to("kpc / s") ** 2.0 / (
            4 * math.pi * constants.G.to("kpc3 / (solMass s2)")
        )

        angular_diameter_distance_of_redshift_0_to_earth_kpc = (
            self.angular_diameter_distance_to_earth_in_kpc_from(redshift=redshift_0)
        )

        angular_diameter_distance_of_redshift_1_to_earth_kpc = (
            self.angular_diameter_distance_to_earth_in_kpc_from(redshift=redshift_1)
        )

        angular_diameter_distance_between_redshifts_kpc = (
            self.angular_diameter_distance_between_redshifts_in_kpc_from(
                redshift_0=redshift_0, redshift_1=redshift_1
            )
        )

        return (
            const
            * angular_diameter_distance_of_redshift_1_to_earth_kpc
            / (
                angular_diameter_distance_between_redshifts_kpc
                * angular_diameter_distance_of_redshift_0_to_earth_kpc
            )
        ).value

    def angular_diameter_distance_to_earth_in_kpc_from(self, redshift: float) -> float:
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
        angular_diameter_distance_kpc = self.angular_diameter_distance(z=redshift).to(
            "kpc"
        )

        return angular_diameter_distance_kpc.value

    def angular_diameter_distance_between_redshifts_in_kpc_from(
        self, redshift_0: float, redshift_1: float
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
        angular_diameter_distance_between_redshifts_kpc = (
            self.angular_diameter_distance_z1z2(redshift_0, redshift_1).to("kpc")
        )

        return angular_diameter_distance_between_redshifts_kpc.value

    def kpc_per_arcsec_from(self, redshift: float) -> float:
        """
        Separation in transverse proper kpc corresponding to an arcminute at redshift `z`.

        For simplicity, **PyAutoLens** internally uses only certain units to perform lensing cosmology calculations.
        This function therefore returns only the value of the astropy function it wraps, omitting the units instance.

        Parameters
        ----------
        redshift
            Input redshift from which the transverse proper kpc value is calculated at.
        """
        return 1.0 / self.arcsec_per_kpc_proper(z=redshift).value
