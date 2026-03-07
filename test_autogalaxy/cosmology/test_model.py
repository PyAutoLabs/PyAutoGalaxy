import pytest
import numpy as np

import autogalaxy as ag


def test__arcsec_to_kpc_conversion():

    cosmology = ag.cosmology.Planck15()

    arcsec_per_kpc = cosmology.arcsec_per_kpc_from(redshift=0.1)

    assert arcsec_per_kpc == pytest.approx(0.525060, 1e-5)

    kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=0.1)

    assert kpc_per_arcsec == pytest.approx(1.904544, 1e-5)

    arcsec_per_kpc = cosmology.arcsec_per_kpc_from(redshift=1.0)

    assert arcsec_per_kpc == pytest.approx(0.1214785, 5e-5)

    kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=1.0)

    assert kpc_per_arcsec == pytest.approx(8.231907, 5e-5)


def test__angular_diameter_distance_z1z2():

    cosmology = ag.cosmology.FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

    angular_diameter_distance = cosmology.angular_diameter_distance_kpc_z1z2(0.1, 1.0)

    assert angular_diameter_distance == pytest.approx(1442537.80833, 1.0e-4)


def test__angular_diameter_distances():

    cosmology = ag.cosmology.Planck15()

    angular_diameter_distance_to_earth_kpc = (
        cosmology.angular_diameter_distance_to_earth_in_kpc_from(redshift=0.1)
    )

    assert angular_diameter_distance_to_earth_kpc == pytest.approx(392840, 1e-5)

    angular_diameter_distance_between_redshifts_kpc = (
        cosmology.angular_diameter_distance_between_redshifts_in_kpc_from(
            redshift_0=0.1, redshift_1=1.0
        )
    )

    assert angular_diameter_distance_between_redshifts_kpc == pytest.approx(
        1481890.4, 5e-5
    )


def test__cosmic_average_densities_solar_mass_per_kpc3():

    cosmology = ag.cosmology.Planck15()

    cosmic_average_density = cosmology.cosmic_average_density_from(redshift=0.6)

    assert cosmic_average_density == pytest.approx(81280.09116133313, 1.0e-4)

    cosmic_average_density = cosmology.cosmic_average_density_solar_mass_per_kpc3_from(
        redshift=0.6
    )

    assert cosmic_average_density == pytest.approx(249.20874, 1.0e-4)


def test__critical_surface_mass_densities():

    cosmology = ag.cosmology.Planck15()

    critical_surface_density = (
        cosmology.critical_surface_density_between_redshifts_from(
            redshift_0=0.1, redshift_1=1.0
        )
    )

    assert critical_surface_density == pytest.approx(17593241668, 1e-2)

    critical_surface_density = (
        cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
            redshift_0=0.1, redshift_1=1.0
        )
    )

    assert critical_surface_density == pytest.approx(4.85e9, 1e-2)


def test__critical_surface_density():

    cosmology = ag.cosmology.FlatLambdaCDM()

    critical_surface_density = (
        cosmology.critical_surface_density_between_redshifts_from(
            redshift_0=0.1, redshift_1=1.0
        )
    )

    assert critical_surface_density == pytest.approx(17613991217.945473, 1.0e-4)


def test__scaling_factor_between_redshifts_from():

    cosmology = ag.cosmology.FlatLambdaCDM()

    scaling_factor = cosmology.scaling_factor_between_redshifts_from(
        redshift_0=0.5, redshift_1=1.0, redshift_final=2.0
    )

    assert scaling_factor == pytest.approx(0.6739452581456, 1e-4)


def test__luminosity_distance():

    cosmology = ag.cosmology.Planck15()

    # D_A(0, 0.1) = 392840 kpc = 392.840 Mpc (verified by test__angular_diameter_distances)
    # D_L(0.1) = (1 + 0.1)^2 * 392.840 = 1.21 * 392.840 = 475.3364 Mpc
    luminosity_distance = cosmology.luminosity_distance(z=0.1)

    assert luminosity_distance == pytest.approx(475.336, 1e-4)


def test__velocity_dispersion_from():

    cosmology = ag.cosmology.Planck15()

    velocity_dispersion = cosmology.velocity_dispersion_from(
        redshift_0=0.5, redshift_1=1.0, einstein_radius=1.0
    )

    assert velocity_dispersion == pytest.approx(284.935499691701, 1.0e-4)

    velocity_dispersion = cosmology.velocity_dispersion_from(
        redshift_0=0.5, redshift_1=1.0, einstein_radius=2.0
    )

    assert velocity_dispersion == pytest.approx(np.sqrt(2) * 284.935499691701, 1.0e-4)
