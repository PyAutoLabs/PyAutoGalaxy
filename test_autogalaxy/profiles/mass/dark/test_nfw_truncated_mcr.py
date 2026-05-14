import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__duffy__mass_and_concentration_consistent_with_normal_truncated_nfw():

    from autogalaxy.cosmology.model import FlatLambdaCDM

    cosmology = FlatLambdaCDM(H0=70.0, Om0=0.3)

    mp = ag.mp.NFWTruncatedMCRDuffySph(
        centre=(1.0, 2.0),
        mass_at_200=1.0e9,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    mass_at_200_via_mass = mp.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_mass = mp.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    truncated_nfw_kappa_s = ag.mp.NFWTruncatedSph(
        centre=(1.0, 2.0),
        kappa_s=mp.kappa_s,
        scale_radius=mp.scale_radius,
        truncation_radius=mp.truncation_radius,
    )

    mass_at_200_via_kappa_s = truncated_nfw_kappa_s.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_kappa_s = truncated_nfw_kappa_s.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    # We uare using the NFWTruncatedSph to check the mass gives a conosistnt kappa_s, given certain radii.

    assert mass_at_200_via_kappa_s == mass_at_200_via_mass
    assert concentration_via_kappa_s == concentration_via_mass

    assert mp.centre == (1.0, 2.0)
    assert mp.axis_ratio() == 1.0
    assert mp.angle() == 0.0
    assert mp.inner_slope == 1.0

    assert mp.scale_radius == pytest.approx(0.273382, 1.0e-3)


def test__ludlow__mass_and_concentration_consistent_with_normal_truncated_nfw__scatter_0():

    from autogalaxy.cosmology.model import FlatLambdaCDM

    cosmology = FlatLambdaCDM(H0=70.0, Om0=0.3)

    mp = ag.mp.NFWTruncatedMCRLudlowSph(
        centre=(1.0, 2.0),
        mass_at_200=1.0e9,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    mass_at_200_via_mass = mp.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_mass = mp.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    truncated_nfw_kappa_s = ag.mp.NFWTruncatedSph(
        centre=(1.0, 2.0),
        kappa_s=mp.kappa_s,
        scale_radius=mp.scale_radius,
        truncation_radius=mp.truncation_radius,
    )

    mass_at_200_via_kappa_s = truncated_nfw_kappa_s.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_kappa_s = truncated_nfw_kappa_s.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    # We uare using the NFWTruncatedSph to check the mass gives a conosistnt kappa_s, given certain radii.

    assert mass_at_200_via_kappa_s == mass_at_200_via_mass
    assert concentration_via_kappa_s == concentration_via_mass

    assert mp.centre == (1.0, 2.0)
    assert mp.axis_ratio() == 1.0
    assert mp.angle() == 0.0
    assert mp.inner_slope == 1.0

    assert mp.scale_radius == pytest.approx(0.21157, 1.0e-3)
    assert mp.truncation_radius == pytest.approx(33.7134116, 1.0e-3)
