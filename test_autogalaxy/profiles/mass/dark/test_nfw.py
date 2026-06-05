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
