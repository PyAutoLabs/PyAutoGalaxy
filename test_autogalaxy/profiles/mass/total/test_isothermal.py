import numpy as np
import pytest

import autogalaxy as ag

from autogalaxy.operate.lens_calc import LensCalc

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from__isothermal_sph_config_1():
    mp = ag.mp.IsothermalSph(centre=(-0.7, 0.5), einstein_radius=1.3)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(1.21510, 1e-4)
    assert deflections[0, 1] == pytest.approx(-0.46208, 1e-4)


def test__deflections_yx_2d_from__isothermal_sph_config_2():
    mp = ag.mp.IsothermalSph(centre=(-0.1, 0.1), einstein_radius=5.0)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(4.88588, 1e-4)
    assert deflections[0, 1] == pytest.approx(1.06214, 1e-4)


def test__deflections_yx_2d_from__isothermal_ell_config_1():
    mp = ag.mp.Isothermal(centre=(0, 0), ell_comps=(0.0, 0.333333), einstein_radius=1.0)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(0.79421, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.50734, 1e-3)


def test__deflections_yx_2d_from__isothermal_ell_config_2():
    mp = ag.mp.Isothermal(centre=(0, 0), ell_comps=(0.0, 0.333333), einstein_radius=1.0)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(0.79421, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.50734, 1e-3)


def test__deflections_yx_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.Isothermal(
        centre=(1.1, 1.1), ell_comps=(0.0, 0.0), einstein_radius=3.0
    )
    spherical = ag.mp.IsothermalSph(centre=(1.1, 1.1), einstein_radius=3.0)

    assert elliptical.deflections_yx_2d_from(grid=grid) == pytest.approx(
        spherical.deflections_yx_2d_from(grid=grid).array, 1e-4
    )


def test__convergence_2d_from__isothermal_sph():
    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.5 * 2.0, 1e-3)


def test__convergence_2d_from__isothermal_no_ell():
    mp = ag.mp.Isothermal(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), einstein_radius=1.0)

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.5, 1e-3)


def test__convergence_2d_from__isothermal_einstein_radius_2():
    mp = ag.mp.Isothermal(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), einstein_radius=2.0)

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.5 * 2.0, 1e-3)


def test__convergence_2d_from__isothermal_with_ell_comps():
    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.333333), einstein_radius=1.0
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.66666, 1e-3)


def test__convergence_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.Isothermal(
        centre=(1.1, 1.1), ell_comps=(0.0, 0.0), einstein_radius=3.0
    )
    spherical = ag.mp.IsothermalSph(centre=(1.1, 1.1), einstein_radius=3.0)

    assert elliptical.convergence_2d_from(grid=grid).array == pytest.approx(
        spherical.convergence_2d_from(grid=grid).array, 1e-4
    )


def test__potential_2d_from__isothermal_sph():
    mp = ag.mp.IsothermalSph(centre=(-0.7, 0.5), einstein_radius=1.3)

    potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert potential == pytest.approx(1.23435, 1e-3)


def test__potential_2d_from__isothermal_elliptical():
    mp = ag.mp.Isothermal(
        centre=(-0.7, 0.5),
        ell_comps=(0.152828, -0.088235),
        einstein_radius=1.3,
    )

    potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert potential == pytest.approx(1.19268, 1e-3)


def test__potential_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.Isothermal(
        centre=(1.1, 1.1), ell_comps=(0.0, 0.0), einstein_radius=3.0
    )
    spherical = ag.mp.IsothermalSph(centre=(1.1, 1.1), einstein_radius=3.0)

    assert elliptical.potential_2d_from(grid=grid) == pytest.approx(
        spherical.potential_2d_from(grid=grid).array, 1e-4
    )


def test__shear_yx_2d_from__isothermal_sph_grid_1():
    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))
    shear = mp.shear_yx_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert shear[0, 0] == pytest.approx(0.0, 1e-4)
    assert shear[0, 1] == pytest.approx(-convergence.array[0], 1e-4)


def test__shear_yx_2d_from__isothermal_sph_grid_2():
    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[2.0, 1.0]]))
    shear = mp.shear_yx_2d_from(grid=ag.Grid2DIrregular([[2.0, 1.0]]))

    assert shear[0, 0] == pytest.approx(-(4.0 / 5.0) * convergence.array[0], 1e-4)
    assert shear[0, 1] == pytest.approx((3.0 / 5.0) * convergence.array[0], 1e-4)


def test__shear_yx_2d_from__isothermal_sph_grid_3():
    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[3.0, 5.0]]))
    shear = mp.shear_yx_2d_from(grid=ag.Grid2DIrregular([[3.0, 5.0]]))

    assert shear[0, 0] == pytest.approx(-(30.0 / 34.0) * convergence.array[0], 1e-4)
    assert shear[0, 1] == pytest.approx(-(16.0 / 34.0) * convergence.array[0], 1e-4)


def test__shear_yx_2d_from__isothermal_no_ell():
    mp = ag.mp.Isothermal(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), einstein_radius=2.0)

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))
    shear = mp.shear_yx_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert shear[0, 0] == pytest.approx(0.0, 1e-4)
    assert shear[0, 1] == pytest.approx(-convergence.array[0], 1e-4)


def test__shear_yx_2d_from__isothermal_with_ell_comps():
    mp = ag.mp.Isothermal(centre=(0.0, 0.0), ell_comps=(0.3, 0.4), einstein_radius=2.0)

    shear = mp.shear_yx_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert shear[0, 0] == pytest.approx(0.0, abs=1e-4)
    assert shear[0, 1] == pytest.approx(-1.11803398874, 1e-4)


def test__shear_yx_2d_from__matches_via_hessian():
    """
    The analytic ``Isothermal.shear_yx_2d_from`` and the numerical
    ``LensCalc.shear_yx_2d_via_hessian_from`` must agree to within finite-difference accuracy at every grid
    point: both compute the same physical shear in the same ``[gamma_2, gamma_1]`` convention, the former
    via a closed-form formula and the latter via Richardson-extrapolated derivatives of
    ``deflections_yx_2d_from``.

    This cross-check guards against either path silently drifting (e.g. a sign flip, a column swap, or a
    rotation-frame mistake) and pins the convention that downstream weak-lensing code relies on.
    """
    grid = ag.Grid2DIrregular(values=[(0.7, 0.5), (1.0, 1.0), (-0.3, 0.6), (1.5, -0.4)])

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.1, -0.1), einstein_radius=2.0
    )

    shear_analytic = mp.shear_yx_2d_from(grid=grid)
    shear_via_hessian = LensCalc.from_mass_obj(mp).shear_yx_2d_via_hessian_from(grid=grid)

    np.testing.assert_allclose(
        np.asarray(shear_analytic), np.asarray(shear_via_hessian), rtol=1e-3, atol=1e-6
    )
