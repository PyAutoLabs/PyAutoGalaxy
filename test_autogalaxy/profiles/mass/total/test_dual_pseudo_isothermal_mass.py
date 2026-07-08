import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from__sph_config_1():
    mp = ag.mp.dPIEMassSph(centre=(-0.7, 0.5), b0=5.2, ra=2.0, rs=3.0)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(1.033080741, 1e-4)
    assert deflections[0, 1] == pytest.approx(-0.39286169026, 1e-4)


def test__deflections_yx_2d_from__sph_config_2():
    mp = ag.mp.dPIEMassSph(centre=(-0.1, 0.1), b0=20.0, ra=2.0, rs=3.0)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(1.4212977207, 1e-4)
    assert deflections[0, 1] == pytest.approx(0.308977765378, 1e-4)


def test__deflections_yx_2d_from__elliptical():
    # First deviation from potential case due to ellipticity

    mp = ag.mp.dPIEMass(
        centre=(0, 0), ell_comps=(0.0, 0.333333), b0=4.0, ra=2.0, rs=3.0
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(0.21461366, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.10753914, 1e-3)


def test__deflections_yx_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.dPIEMass(
        centre=(1.1, 1.1), ell_comps=(0.000001, 0.0000001), b0=12.0, ra=2.0, rs=3.0
    )
    spherical = ag.mp.dPIEMassSph(centre=(1.1, 1.1), b0=12.0, ra=2.0, rs=3.0)

    assert elliptical.deflections_yx_2d_from(grid=grid).array == pytest.approx(
        spherical.deflections_yx_2d_from(grid=grid).array, 1e-1
    )


def test__convergence_func__matches_private_helper():
    """Regression: dPIEMass must override the abstract `convergence_func`
    so MGEDecomposer.decompose_convergence_via_mge (which walks the
    convergence radially during MGE potential decomposition) doesn't
    fall through to the abstract NotImplementedError. The shim delegates
    to the existing `_convergence` radial helper that `convergence_2d_from`
    already uses."""

    import numpy as np

    mp = ag.mp.dPIEMass(centre=(0.0, 0.0), b0=5.2, ra=2.0, rs=3.0)

    # Scalar radius: equals the _convergence formula directly.
    assert mp.convergence_func(1.5) == pytest.approx(mp._convergence(1.5), 1e-12)

    # 1-D array of radii: shape preserved, element-wise equality.
    radii = np.array([0.1, 0.5, 1.0, 2.5, 5.0])
    expected = mp._convergence(radii)
    actual = mp.convergence_func(radii)
    assert actual.shape == radii.shape
    assert actual == pytest.approx(expected, 1e-12)

    # dPIEMassSph inherits the override from dPIEMass.
    sph = ag.mp.dPIEMassSph(centre=(0.0, 0.0), b0=5.2, ra=2.0, rs=3.0)
    assert sph.convergence_func(1.5) == pytest.approx(sph._convergence(1.5), 1e-12)


def test__from_lenstool__b0_conversion__matches_isothermal_relation():
    # b0 = 6 * 648000 * (sigma_LT / c)^2 * (D_LS / D_S); cross-checked via the independent
    # cosmology.velocity_dispersion_from (isothermal theta_E -> sigma_0), which must
    # recover the central dispersion sigma_0 = sqrt(3/2) * sigma_LT.

    cosmology = ag.cosmo.Planck15()

    mp = ag.mp.dPIEMassSph.from_lenstool(
        sigma=1000.0,
        r_core=0.1,
        r_cut=20.0,
        redshift_object=0.5,
        redshift_source=2.0,
        cosmology=cosmology,
    )

    sigma_0 = cosmology.velocity_dispersion_from(
        redshift_0=0.5, redshift_1=2.0, einstein_radius=mp.b0
    )

    assert sigma_0 == pytest.approx(1000.0 * (1.5**0.5), rel=1e-6)


def test__from_lenstool__b0_explicit_value():
    # Pure prefactor check with the distance ratio divided out:
    # b0 / (D_LS / D_S) = 6 * 648000 * (sigma / c)^2.

    cosmology = ag.cosmo.Planck15()

    mp = ag.mp.dPIEMassSph.from_lenstool(
        sigma=250.0,
        redshift_object=0.3,
        redshift_source=1.5,
        cosmology=cosmology,
    )

    d_s = cosmology.angular_diameter_distance_to_earth_in_kpc_from(redshift=1.5)
    d_ls = cosmology.angular_diameter_distance_between_redshifts_in_kpc_from(
        redshift_0=0.3, redshift_1=1.5
    )

    assert mp.b0 / (d_ls / d_s) == pytest.approx(
        6.0 * 648000.0 * (250.0 / 299792.458) ** 2, rel=1e-8
    )


def test__from_lenstool__ellipticity_conversion__matches_lenstool_internal():
    # Lenstool converts emass = (a^2-b^2)/(a^2+b^2) to epot = (1 - sqrt(1 - emass^2)) / emass
    # (set_lens.c) and passes epot to ci05f; the port's _ellip() = |ell_comps| must equal it.

    emass = 0.4

    mp = ag.mp.dPIEMass.from_lenstool(ellipticity=emass, angle_pos=0.0)

    epot_lenstool = (1.0 - (1.0 - emass**2) ** 0.5) / emass

    assert mp._ellip() == pytest.approx(epot_lenstool, rel=1e-10)

    axis_ratio = ((1.0 - emass) / (1.0 + emass)) ** 0.5

    assert mp._ellip() == pytest.approx(
        (1.0 - axis_ratio) / (1.0 + axis_ratio), rel=1e-10
    )


def test__from_lenstool__angle_and_radii_passthrough():
    mp = ag.mp.dPIEMass.from_lenstool(
        centre=(0.1, -0.2),
        ellipticity=0.3,
        angle_pos=45.0,
        sigma=200.0,
        r_core=0.15,
        r_cut=12.0,
    )

    assert mp.centre == (0.1, -0.2)
    assert mp.ra == 0.15
    assert mp.rs == 12.0

    axis_ratio = ((1.0 - 0.3) / (1.0 + 0.3)) ** 0.5
    ell_comps = ag.convert.ell_comps_from(axis_ratio=axis_ratio, angle=45.0)

    assert mp.ell_comps[0] == pytest.approx(ell_comps[0], rel=1e-10)
    assert mp.ell_comps[1] == pytest.approx(ell_comps[1], rel=1e-10)


def test__from_lenstool__sph_matches_elliptical_zero_ellipticity():
    kwargs = dict(
        sigma=300.0, r_core=0.2, r_cut=15.0, redshift_object=0.4, redshift_source=1.8
    )

    sph = ag.mp.dPIEMassSph.from_lenstool(**kwargs)
    ell = ag.mp.dPIEMass.from_lenstool(ellipticity=0.0, angle_pos=0.0, **kwargs)

    grid_check = ag.Grid2DIrregular([[0.5, 0.8], [1.5, -0.3]])

    sph_deflections = sph.deflections_yx_2d_from(grid=grid_check)
    ell_deflections = ell.deflections_yx_2d_from(grid=grid_check)

    assert sph_deflections[0, 0] == pytest.approx(ell_deflections[0, 0], rel=1e-4)
    assert sph_deflections[1, 1] == pytest.approx(ell_deflections[1, 1], rel=1e-4)


def test__from_lenstool__isothermal_limit_deflection_equals_b0():
    # ra -> 0, rs -> inf, e = 0: the dPIE tends to a SIS whose deflection magnitude is b0
    # everywhere, so b0 is the Einstein radius in this limit.

    mp = ag.mp.dPIEMassSph.from_lenstool(
        sigma=800.0, r_core=1e-5, r_cut=1e6, redshift_object=0.5, redshift_source=2.0
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.0, 3.0]]))

    assert deflections[0, 1] == pytest.approx(mp.b0, rel=1e-3)


def test__potential_2d_from__gradient_matches_deflections():
    # The analytic potential (Lenstool pi05 port) must be exactly consistent with the
    # ci05f deflections: finite-difference grad(psi) = alpha, including rotation.

    mp = ag.mp.dPIEMass(
        centre=(0.1, -0.3), ell_comps=(0.15, 0.1), ra=1.5, rs=30.0, b0=10.0
    )

    eps_fd = 1e-6
    y, x = 0.8, 1.3

    grid_fd = ag.Grid2DIrregular(
        [
            [y, x + eps_fd],
            [y, x - eps_fd],
            [y + eps_fd, x],
            [y - eps_fd, x],
        ]
    )

    psi = mp.potential_2d_from(grid=grid_fd)

    alpha_fd_x = (psi[0] - psi[1]) / (2 * eps_fd)
    alpha_fd_y = (psi[2] - psi[3]) / (2 * eps_fd)

    alpha = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[y, x]]))

    assert alpha_fd_y == pytest.approx(alpha[0, 0], rel=1e-5)
    assert alpha_fd_x == pytest.approx(alpha[0, 1], rel=1e-5)


def test__potential_2d_from__spherical_limit_matches_quadrature():
    # For a near-circular profile psi(R) = 2 * int_0^R kappa(r) r ln(R/r) dr + ... ;
    # simpler and equivalent: d(psi)/dR must equal the spherical deflection alpha(R).

    import numpy as np

    mp = ag.mp.dPIEMassSph(centre=(0.0, 0.0), ra=2.0, rs=20.0, b0=5.0)

    eps_fd = 1e-6
    R = 4.0

    psi = mp.potential_2d_from(
        grid=ag.Grid2DIrregular([[0.0, R + eps_fd], [0.0, R - eps_fd]])
    )
    alpha_fd = (psi[0] - psi[1]) / (2 * eps_fd)

    alpha = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.0, R]]))

    assert alpha_fd == pytest.approx(alpha[0, 1], rel=1e-5)


def test__lenstool_wrapper__matches_from_lenstool_constructor():
    # The model-fittable wrapper class must produce the identical profile to the
    # from_lenstool classmethod with the (fixed Planck15) cosmology.

    kwargs = dict(
        centre=(0.1, -0.2),
        ellipticity=0.3,
        angle_pos=45.0,
        sigma=350.0,
        r_core=0.15,
        r_cut=12.0,
        redshift_object=0.4,
        redshift_source=1.8,
    )

    wrapper = ag.mp.dPIEMassLenstool(**kwargs)
    constructed = ag.mp.dPIEMass.from_lenstool(**kwargs)

    assert wrapper.b0 == pytest.approx(constructed.b0, rel=1e-10)
    assert wrapper.ra == constructed.ra
    assert wrapper.rs == constructed.rs
    assert wrapper.ell_comps[0] == pytest.approx(constructed.ell_comps[0], rel=1e-10)
    assert wrapper.ell_comps[1] == pytest.approx(constructed.ell_comps[1], rel=1e-10)

    grid_check = ag.Grid2DIrregular([[0.5, 0.8]])

    assert wrapper.deflections_yx_2d_from(grid=grid_check)[0, 0] == pytest.approx(
        constructed.deflections_yx_2d_from(grid=grid_check)[0, 0], rel=1e-10
    )

    sph_kwargs = {
        k: v for k, v in kwargs.items() if k not in ("ellipticity", "angle_pos")
    }

    wrapper_sph = ag.mp.dPIEMassLenstoolSph(**sph_kwargs)
    constructed_sph = ag.mp.dPIEMassSph.from_lenstool(**sph_kwargs)

    assert wrapper_sph.b0 == pytest.approx(constructed_sph.b0, rel=1e-10)


def test__lenstool_wrapper__supports_model_composition():
    # Fitting in Lenstool parameters: af.Model must resolve priors for every __init__
    # arg from the config, and fixing the redshifts must leave the Lenstool free
    # parameters (centre_0, centre_1, ellipticity, angle, sigma, r_core, r_cut).

    import autofit as af

    model = af.Model(ag.mp.dPIEMassLenstool)

    assert model.prior_count == 9

    model.redshift_object = 0.5
    model.redshift_source = 2.0

    assert model.prior_count == 7

    instance = model.instance_from_unit_vector([0.5] * model.prior_count)

    assert isinstance(instance, ag.mp.dPIEMassLenstool)
    assert instance.b0 > 0.0

    model_sph = af.Model(ag.mp.dPIEMassLenstoolSph)

    assert model_sph.prior_count == 7
