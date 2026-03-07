import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_via_analytical_from():

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


def test__deflections_via_integral_from():
    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    deflections = nfw.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections[0, 0] == pytest.approx(0.56194, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.56194, 1e-3)

    nfw = ag.mp.NFWSph(centre=(0.3, 0.2), kappa_s=2.5, scale_radius=4.0)

    deflections = nfw.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1875, 0.1625]])
    )

    assert deflections[0, 0] == pytest.approx(-2.08909, 1e-3)
    assert deflections[0, 1] == pytest.approx(-0.69636, 1e-3)

    nfw = ag.mp.NFW(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        kappa_s=1.0,
        scale_radius=1.0,
    )

    deflections = nfw.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections[0, 0] == pytest.approx(0.56194, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.56194, 1e-3)

    nfw = ag.mp.NFW(
        centre=(0.3, 0.2),
        ell_comps=(0.03669, 0.172614),
        kappa_s=2.5,
        scale_radius=4.0,
    )

    deflections = nfw.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([(0.1625, 0.1625)])
    )

    assert deflections[0, 0] == pytest.approx(-2.59480, 1e-3)
    assert deflections[0, 1] == pytest.approx(-0.44204, 1e-3)


def test__deflections_2d_via_cse_from():
    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    deflections_via_integral = nfw.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )
    deflections_via_cse = nfw.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-4)

    nfw = ag.mp.NFWSph(centre=(0.3, 0.2), kappa_s=2.5, scale_radius=4.0)

    deflections_via_integral = nfw.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )
    deflections_via_cse = nfw.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-4)

    nfw = ag.mp.NFW(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        kappa_s=1.0,
        scale_radius=1.0,
    )

    deflections_via_integral = nfw.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )
    deflections_via_cse = nfw.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-4)

    nfw = ag.mp.NFW(
        centre=(0.3, 0.2),
        ell_comps=(0.03669, 0.172614),
        kappa_s=2.5,
        scale_radius=4.0,
    )

    deflections_via_integral = nfw.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )
    deflections_via_cse = nfw.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-4)


def test__deflections_2d__numerical_precision_of_csv_compared_to_integral():
    nfw = ag.mp.NFW(
        centre=(0.3, 0.2),
        ell_comps=(0.03669, 0.172614),
        kappa_s=2.5,
        scale_radius=4.0,
    )

    deflections_via_integral = nfw.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[1.0, 2.0]])
    )
    deflections_via_cse = nfw.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[1.0, 2.0]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-4)

    nfw = ag.mp.NFW(
        centre=(0.3, 0.2),
        ell_comps=(0.2, 0.3),
        kappa_s=3.5,
        scale_radius=40.0,
    )

    deflections_via_integral = nfw.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[100.0, 200.0]])
    )
    deflections_via_cse = nfw.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[100.0, 200.0]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-4)

    deflections_via_integral = nfw.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[-1000.0, -2000.0]])
    )
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


def test__convergence_2d_via_cse_from():
    # r = 2.0 (> 1.0)
    # F(r) = (1/(sqrt(3))*atan(sqrt(3)) = 0.60459978807
    # kappa(r) = 2 * kappa_s * (1 - 0.60459978807) / (4-1) = 0.263600141

    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    convergence = nfw.convergence_2d_via_cse_from(grid=ag.Grid2DIrregular([[2.0, 0.0]]))

    assert convergence == pytest.approx(0.263600141, 1e-2)

    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    convergence = nfw.convergence_2d_via_cse_from(grid=ag.Grid2DIrregular([[0.5, 0.0]]))

    assert convergence == pytest.approx(1.388511, 1e-2)

    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=1.0)

    convergence = nfw.convergence_2d_via_cse_from(grid=ag.Grid2DIrregular([[0.5, 0.0]]))

    assert convergence == pytest.approx(2.0 * 1.388511, 1e-2)

    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=2.0)

    convergence = nfw.convergence_2d_via_cse_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert convergence == pytest.approx(1.388511, 1e-2)

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


def test__convergence_2d_from():
    # r = 2.0 (> 1.0)
    # F(r) = (1/(sqrt(3))*atan(sqrt(3)) = 0.60459978807
    # kappa(r) = 2 * kappa_s * (1 - 0.60459978807) / (4-1) = 0.263600141

    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    convergence = nfw.convergence_2d_from(grid=ag.Grid2DIrregular([[2.0, 0.0]]))

    assert convergence == pytest.approx(0.263600141, 1e-3)

    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    convergence = nfw.convergence_2d_from(grid=ag.Grid2DIrregular([[0.5, 0.0]]))

    assert convergence == pytest.approx(1.388511, 1e-3)

    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=1.0)

    convergence = nfw.convergence_2d_from(grid=ag.Grid2DIrregular([[0.5, 0.0]]))

    assert convergence == pytest.approx(2.0 * 1.388511, 1e-3)

    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=2.0)

    convergence = nfw.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert convergence == pytest.approx(1.388511, 1e-3)

    nfw = ag.mp.NFW(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        kappa_s=1.0,
        scale_radius=1.0,
    )

    convergence = nfw.convergence_2d_from(grid=ag.Grid2DIrregular([[0.25, 0.0]]))

    assert convergence == pytest.approx(1.388511, 1e-3)


# def test__potential_2d_from():
#     nfw = ag.mp.NFWSph(centre=(0.3, 0.2), kappa_s=2.5, scale_radius=4.0)
#
#     potential = nfw.potential_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))
#
#     assert potential == pytest.approx(0.03702, 1e-3)
#
#     nfw = ag.mp.NFWSph(centre=(0.3, 0.2), kappa_s=2.5, scale_radius=4.0)
#
#     potential = nfw.potential_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))
#
#     assert potential == pytest.approx(0.03702, 1e-3)
#
#     nfw = ag.mp.NFW(
#         centre=(0.3, 0.2),
#         ell_comps=(0.03669, 0.172614),
#         kappa_s=2.5,
#         scale_radius=4.0,
#     )
#
#     potential = nfw.potential_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))
#
#     assert potential == pytest.approx(0.05380, 1e-3)
#
#     nfw_spherical = ag.mp.NFWSph(centre=(0.3, 0.2), kappa_s=2.5, scale_radius=4.0)
#     nfw_elliptical = ag.mp.NFW(
#         centre=(0.3, 0.2),
#         ell_comps=(0.0, 0.0),
#         kappa_s=2.5,
#         scale_radius=4.0,
#     )
#
#     potential_spherical = nfw_spherical.potential_2d_from(
#         grid=ag.Grid2DIrregular([[0.1875, 0.1625]])
#     )
#     potential_elliptical = nfw_elliptical.potential_2d_from(
#         grid=ag.Grid2DIrregular([[0.1875, 0.1625]])
#     )
#
#     assert potential_spherical == pytest.approx(potential_elliptical, 1e-3)


def test__shear_yx_2d_from():
    mp = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    shear = mp.shear_yx_2d_from(grid=ag.Grid2DIrregular([[2.0, 1.0]]))

    assert shear[0, 0] == pytest.approx(-0.24712463, abs=1e-3)
    assert shear[0, 1] == pytest.approx(0.185340150, abs=1e-3)

    shear = mp.shear_yx_2d_from(grid=ag.Grid2DIrregular([[3.0, 5.0]]))

    assert shear[0, 0] == pytest.approx(-0.09588857, abs=1e-3)
    assert shear[0, 1] == pytest.approx(-0.05114060, abs=1e-3)

    mp = ag.mp.NFW(
        centre=(0.0, 0.0), ell_comps=(0.3, 0.4), kappa_s=1.0, scale_radius=1.0
    )

    shear = mp.shear_yx_2d_from(grid=ag.Grid2DIrregular([[2.0, 1.0]]))

    assert shear[0, 0] == pytest.approx(-0.08554797, abs=1e-3)
    assert shear[0, 1] == pytest.approx(0.111356360, abs=1e-3)
