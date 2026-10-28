import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_via_analytical_from__nfw_config_1():
    nfw = ag.mp.NFW(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        kappa_s=1.0,
        scale_radius=1.0,
    )

    deflections = nfw.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections[0, 0] == pytest.approx(0.56194, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.56194, 1e-3)


def test__deflections_via_analytical_from__nfw_config_2():
    nfw = ag.mp.NFW(
        centre=(0.3, 0.2),
        ell_comps=(0.03669, 0.172614),
        kappa_s=2.5,
        scale_radius=4.0,
    )

    deflections = nfw.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([(0.1625, 0.1625)])
    )

    assert deflections[0, 0] == pytest.approx(-2.59480, 1e-3)
    assert deflections[0, 1] == pytest.approx(-0.44204, 1e-3)


def test__deflections_via_analytical_from__analytic_vs_cse_config_1():
    nfw = ag.mp.NFW(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        kappa_s=1.0,
        scale_radius=1.0,
    )

    deflections_via_analytic = nfw.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[0.0, 0.0]])
    )
    deflections_via_cse = nfw.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.0, 0.0]])
    )

    assert deflections_via_analytic == pytest.approx(deflections_via_cse.array, 1.0e-4)


def test__deflections_via_analytical_from__analytic_vs_cse_config_2():
    nfw = ag.mp.NFW(
        centre=(0.3, -0.2),
        ell_comps=(0.09, 0.172614),
        kappa_s=0.05,
        scale_radius=4.0,
    )

    deflections_via_analytic = nfw.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[0.1625, -0.1625]])
    )
    deflections_via_cse = nfw.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, -0.1625]])
    )

    assert deflections_via_analytic == pytest.approx(deflections_via_cse.array, 1.0e-4)


def test__deflections_2d_via_cse_from__nfw_sph_config_1():
    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    deflections_via_integral = np.array([[0.56194, 0.56194]])
    deflections_via_cse = nfw.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-4)


def test__deflections_2d_via_cse_from__nfw_sph_config_2():
    nfw = ag.mp.NFWSph(centre=(0.3, 0.2), kappa_s=2.5, scale_radius=4.0)

    deflections_via_integral = np.array([[-2.4275514589, -0.6620594888]])
    deflections_via_cse = nfw.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-4)


def test__deflections_2d_via_cse_from__nfw_ell_config_1():
    nfw = ag.mp.NFW(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        kappa_s=1.0,
        scale_radius=1.0,
    )

    deflections_via_integral = np.array([[0.56194, 0.56194]])
    deflections_via_cse = nfw.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-4)


def test__deflections_2d_via_cse_from__nfw_ell_config_2():
    nfw = ag.mp.NFW(
        centre=(0.3, 0.2),
        ell_comps=(0.03669, 0.172614),
        kappa_s=2.5,
        scale_radius=4.0,
    )

    deflections_via_integral = np.array([[-2.59480, -0.44204]])
    deflections_via_cse = nfw.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-4)


def test__deflections_2d_numerical_precision__config_1():
    nfw = ag.mp.NFW(
        centre=(0.3, 0.2),
        ell_comps=(0.03669, 0.172614),
        kappa_s=2.5,
        scale_radius=4.0,
    )

    deflections_via_integral = np.array([[3.7984113660, 7.7612406998]])
    deflections_via_cse = nfw.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[1.0, 2.0]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-4)


def test__deflections_2d_numerical_precision__config_2():
    nfw = ag.mp.NFW(
        centre=(0.3, 0.2),
        ell_comps=(0.2, 0.3),
        kappa_s=3.5,
        scale_radius=40.0,
    )

    deflections_via_integral = np.array([[36.8433139810, 60.3308895404]])
    deflections_via_cse = nfw.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[100.0, 200.0]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-4)


def test__deflections_2d_numerical_precision__config_3():
    nfw = ag.mp.NFW(
        centre=(0.3, 0.2),
        ell_comps=(0.2, 0.3),
        kappa_s=3.5,
        scale_radius=40.0,
    )

    deflections_via_integral = np.array([[-8.4199994072, -15.0162989892]])
    deflections_via_cse = nfw.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[-1000.0, -2000.0]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-4)


def test__deflections_yx_2d_from():
    nfw = ag.mp.NFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    deflections = nfw.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflections_via_integral = nfw.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_integral.array, 1.0e-4)


def test__convergence_2d_via_cse_from__nfw_grid_2():
    # r = 2.0 (> 1.0)
    # F(r) = (1/(sqrt(3))*atan(sqrt(3)) = 0.60459978807
    # kappa(r) = 2 * kappa_s * (1 - 0.60459978807) / (4-1) = 0.263600141

    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    convergence = nfw.convergence_2d_via_cse_from(grid=ag.Grid2DIrregular([[2.0, 0.0]]))

    assert convergence == pytest.approx(0.263600141, 1e-2)


def test__convergence_2d_via_cse_from__nfw_grid_05():
    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    convergence = nfw.convergence_2d_via_cse_from(grid=ag.Grid2DIrregular([[0.5, 0.0]]))

    assert convergence == pytest.approx(1.388511, 1e-2)


def test__convergence_2d_via_cse_from__nfw_kappa_s_2():
    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=1.0)

    convergence = nfw.convergence_2d_via_cse_from(grid=ag.Grid2DIrregular([[0.5, 0.0]]))

    assert convergence == pytest.approx(2.0 * 1.388511, 1e-2)


def test__convergence_2d_via_cse_from__nfw_scale_radius_2():
    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=2.0)

    convergence = nfw.convergence_2d_via_cse_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert convergence == pytest.approx(1.388511, 1e-2)


def test__convergence_2d_via_cse_from__nfw_ell():
    nfw = ag.mp.NFW(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        kappa_s=1.0,
        scale_radius=1.0,
    )

    convergence = nfw.convergence_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.25, 0.0]])
    )

    assert convergence == pytest.approx(1.388511, 1e-3)


def test__convergence_2d_from__nfw_sph_grid_2():
    # r = 2.0 (> 1.0)
    # F(r) = (1/(sqrt(3))*atan(sqrt(3)) = 0.60459978807
    # kappa(r) = 2 * kappa_s * (1 - 0.60459978807) / (4-1) = 0.263600141

    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    convergence = nfw.convergence_2d_from(grid=ag.Grid2DIrregular([[2.0, 0.0]]))

    assert convergence == pytest.approx(0.263600141, 1e-3)


def test__convergence_2d_from__nfw_sph_grid_05():
    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    convergence = nfw.convergence_2d_from(grid=ag.Grid2DIrregular([[0.5, 0.0]]))

    assert convergence == pytest.approx(1.388511, 1e-3)


def test__convergence_2d_from__nfw_sph_kappa_s_2():
    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=1.0)

    convergence = nfw.convergence_2d_from(grid=ag.Grid2DIrregular([[0.5, 0.0]]))

    assert convergence == pytest.approx(2.0 * 1.388511, 1e-3)


def test__convergence_2d_from__nfw_sph_scale_radius_2():
    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=2.0)

    convergence = nfw.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert convergence == pytest.approx(1.388511, 1e-3)


def test__convergence_2d_from__nfw_ell():
    nfw = ag.mp.NFW(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        kappa_s=1.0,
        scale_radius=1.0,
    )

    convergence = nfw.convergence_2d_from(grid=ag.Grid2DIrregular([[0.25, 0.0]]))

    assert convergence == pytest.approx(1.388511, 1e-3)


def test__potential_2d_from__nfw_sph():
    nfw = ag.mp.NFWSph(centre=(0.3, 0.2), kappa_s=2.5, scale_radius=4.0)

    potential = nfw.potential_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert potential == pytest.approx(0.148108, 1e-3)


def test__potential_2d_from__elliptical_reduces_to_spherical():
    # At zero ellipticity the elliptical NFW (MGE potential) must match the
    # spherical NFW (analytic potential) — a cross-check of two independent
    # implementations.
    spherical = ag.mp.NFWSph(centre=(0.3, 0.2), kappa_s=2.5, scale_radius=4.0)
    elliptical = ag.mp.NFW(
        centre=(0.3, 0.2), ell_comps=(0.0, 0.0), kappa_s=2.5, scale_radius=4.0
    )

    grid = ag.Grid2DIrregular([[0.1875, 0.1625]])

    assert elliptical.potential_2d_from(grid=grid) == pytest.approx(
        spherical.potential_2d_from(grid=grid).array, 1e-3
    )


def test__shear_yx_2d_from__nfw_sph_config_1():
    mp = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    shear = mp.shear_yx_2d_from(grid=ag.Grid2DIrregular([[2.0, 1.0]]))

    assert shear[0, 0] == pytest.approx(-0.24712463, abs=1e-3)
    assert shear[0, 1] == pytest.approx(0.185340150, abs=1e-3)


def test__shear_yx_2d_from__nfw_sph_config_2():
    mp = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    shear = mp.shear_yx_2d_from(grid=ag.Grid2DIrregular([[3.0, 5.0]]))

    assert shear[0, 0] == pytest.approx(-0.09588857, abs=1e-3)
    assert shear[0, 1] == pytest.approx(-0.05114060, abs=1e-3)


def test__shear_yx_2d_from__nfw_ell():
    mp = ag.mp.NFW(
        centre=(0.0, 0.0), ell_comps=(0.3, 0.4), kappa_s=1.0, scale_radius=1.0
    )

    shear = mp.shear_yx_2d_from(grid=ag.Grid2DIrregular([[2.0, 1.0]]))

    assert shear[0, 0] == pytest.approx(-0.08554797, abs=1e-3)
    assert shear[0, 1] == pytest.approx(0.111356360, abs=1e-3)


def _nfw_mass_angular_within_circle_analytic(kappa_s, scale_radius, radius):
    """
    Closed-form projected NFW mass inside a circle, in convergence (angular) units:
    M(<R) = 4 pi kappa_s r_s^2 h(x), x = R / r_s (Bartelmann 1996, eq. 11), with
    h(x) = ln(x/2) + arccosh(1/x) / sqrt(1 - x^2) for x < 1 and arccos(1/x) / sqrt(x^2 - 1) for x > 1.
    """
    x = radius / scale_radius

    if x < 1.0:
        h = np.log(x / 2.0) + np.arccosh(1.0 / x) / np.sqrt(1.0 - x**2)
    elif x > 1.0:
        h = np.log(x / 2.0) + np.arccos(1.0 / x) / np.sqrt(x**2 - 1.0)
    else:
        h = np.log(0.5) + 1.0

    return 4.0 * np.pi * kappa_s * scale_radius**2 * h


def test__mass_angular_within_circle_from__nfw_sph__matches_analytic_closed_form():
    """
    Regression: `mass_integral` fed `convergence_func` a 0-d radius, which the NFW
    `coord_func_g` promoted to a length-1 (complex64) array that `scipy.integrate.quad`
    rejected with `TypeError: only 0-dimensional arrays can be converted to Python scalars`.
    """
    nfw = ag.mp.NFWSph(kappa_s=0.2, scale_radius=5.0)

    # Inside the scale radius (x = 0.2) ...
    mass = nfw.mass_angular_within_circle_from(radius=1.0)

    assert mass == pytest.approx(
        _nfw_mass_angular_within_circle_analytic(
            kappa_s=0.2, scale_radius=5.0, radius=1.0
        ),
        rel=1e-6,
    )
    assert mass == pytest.approx(2.3322042, rel=1e-6)

    # ... and outside it (x = 1.6).
    assert nfw.mass_angular_within_circle_from(radius=8.0) == pytest.approx(
        _nfw_mass_angular_within_circle_analytic(
            kappa_s=0.2, scale_radius=5.0, radius=8.0
        ),
        rel=1e-6,
    )


def test__mass_angular_within_circle_from__nfw_sph__matches_grid_integration():
    nfw = ag.mp.NFWSph(kappa_s=0.2, scale_radius=5.0)

    grid = ag.Grid2D.uniform(shape_native=(200, 200), pixel_scales=0.02)

    mass_via_grid = ag.mass_angular_within_circle_via_grid_from(
        nfw, radius=1.0, centre=(0.0, 0.0), grid=grid
    )

    assert nfw.mass_angular_within_circle_from(radius=1.0) == pytest.approx(
        mass_via_grid, rel=1e-4
    )


def test__mass_angular_within_circle_from__gnfw_sph_inner_slope_1__matches_nfw_closed_form():
    """
    Regression: the gNFW `convergence_func` indexed `grid_radius.shape[0]`, which raised
    `IndexError` on the 0-d radius `mass_integral` used to pass it.
    """
    gnfw = ag.mp.gNFWSph(kappa_s=0.2, scale_radius=5.0, inner_slope=1.0)

    assert gnfw.mass_angular_within_circle_from(radius=1.0) == pytest.approx(
        _nfw_mass_angular_within_circle_analytic(
            kappa_s=0.2, scale_radius=5.0, radius=1.0
        ),
        rel=1e-6,
    )
