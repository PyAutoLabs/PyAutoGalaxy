import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from__isothermal_core_sph_config_1():
    mp = ag.mp.IsothermalCoreSph(
        centre=(-0.7, 0.5), einstein_radius=1.3, core_radius=0.2
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(0.98582, 1e-3)
    assert deflections[0, 1] == pytest.approx(-0.37489, 1e-3)


def test__deflections_yx_2d_from__isothermal_core_sph_config_2():
    mp = ag.mp.IsothermalCoreSph(
        centre=(0.2, -0.2), einstein_radius=0.5, core_radius=0.5
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(-0.00559, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.16216, 1e-3)


def test__deflections_yx_2d_from__isothermal_core_ell_config_1():
    mp = ag.mp.IsothermalCore(
        centre=(-0.7, 0.5),
        ell_comps=(0.152828, -0.088235),
        einstein_radius=1.3,
        core_radius=0.2,
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(0.95429, 1e-3)
    assert deflections[0, 1] == pytest.approx(-0.52047, 1e-3)


def test__deflections_yx_2d_from__isothermal_core_ell_config_2():
    mp = ag.mp.IsothermalCore(
        centre=(0.2, -0.2),
        ell_comps=(-0.216506, -0.125),
        einstein_radius=0.5,
        core_radius=0.5,
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(0.02097, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.20500, 1e-3)


def test__deflections_yx_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.IsothermalCore(
        centre=(1.1, 1.1),
        ell_comps=(0.0, 0.0),
        einstein_radius=3.0,
        core_radius=1.0,
    )
    spherical = ag.mp.IsothermalCoreSph(
        centre=(1.1, 1.1), einstein_radius=3.0, core_radius=1.0
    )

    assert elliptical.deflections_yx_2d_from(grid=grid) == pytest.approx(
        spherical.deflections_yx_2d_from(grid=grid).array, 1e-4
    )


def test__convergence_2d_from__isothermal_core_sph_convergence_func():
    mp = ag.mp.IsothermalCoreSph(centre=(1, 1), einstein_radius=1.0, core_radius=0.1)

    convergence = mp.convergence_func(grid_radius=1.0)

    assert convergence == pytest.approx(0.49752, 1e-4)


def test__convergence_2d_from__isothermal_core_sph_config_2():
    mp = ag.mp.IsothermalCoreSph(centre=(1, 1), einstein_radius=1.0, core_radius=0.1)

    convergence = mp.convergence_func(grid_radius=1.0)

    assert convergence == pytest.approx(0.49752, 1e-4)


def test__convergence_2d_from__isothermal_core_sph_core_radius_02():
    mp = ag.mp.IsothermalCoreSph(
        centre=(0.0, 0.0), einstein_radius=1.0, core_radius=0.2
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert convergence == pytest.approx(0.49029, 1e-3)


def test__convergence_2d_from__isothermal_core_sph_einstein_radius_2():
    mp = ag.mp.IsothermalCoreSph(
        centre=(0.0, 0.0), einstein_radius=2.0, core_radius=0.2
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert convergence == pytest.approx(2.0 * 0.49029, 1e-3)


def test__convergence_2d_from__isothermal_core_sph_y_axis():
    mp = ag.mp.IsothermalCoreSph(
        centre=(0.0, 0.0), einstein_radius=1.0, core_radius=0.2
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.49029, 1e-3)


def test__convergence_2d_from__isothermal_core_ell_config_1():
    # axis ratio changes only einstein_rescaled, so we can use the above value and times by 1.0/1.5.
    mp = ag.mp.IsothermalCore(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        einstein_radius=1.0,
        core_radius=0.2,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.49029 * 1.33333, 1e-3)


def test__convergence_2d_from__isothermal_core_ell_config_2():
    mp = ag.mp.IsothermalCore(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        einstein_radius=2.0,
        core_radius=0.2,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(2.0 * 0.49029, 1e-3)


def test__convergence_2d_from__isothermal_core_ell_config_3():
    # for axis_ratio = 1.0, the factor is 1/2
    # for axis_ratio = 0.5, the factor is 1/(1.5)
    # So the change in the value is 0.5 / (1/1.5) = 1.0 / 0.75
    # axis ratio changes only einstein_rescaled, so we can use the above value and times by 1.0/1.5.

    mp = ag.mp.IsothermalCore(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        einstein_radius=1.0,
        core_radius=0.2,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx((1.0 / 0.75) * 0.49029, 1e-3)


def test__convergence_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.IsothermalCore(
        centre=(1.1, 1.1),
        ell_comps=(0.0, 0.0),
        einstein_radius=3.0,
        core_radius=1.0,
    )
    spherical = ag.mp.IsothermalCoreSph(
        centre=(1.1, 1.1), einstein_radius=3.0, core_radius=1.0
    )

    assert elliptical.convergence_2d_from(grid=grid).array == pytest.approx(
        spherical.convergence_2d_from(grid=grid).array, 1e-4
    )


def test__potential_2d_from__isothermal_core_sph_config_1():
    mp = ag.mp.IsothermalCoreSph(
        centre=(-0.7, 0.5), einstein_radius=1.3, core_radius=0.2
    )

    potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert potential == pytest.approx(0.72231, 1e-3)


def test__potential_2d_from__isothermal_core_sph_config_2():
    mp = ag.mp.IsothermalCoreSph(
        centre=(0.2, -0.2), einstein_radius=0.5, core_radius=0.5
    )

    potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert potential == pytest.approx(0.03103, 1e-3)


def test__potential_2d_from__isothermal_core_ell_config_1():
    mp = ag.mp.IsothermalCore(
        centre=(-0.7, 0.5),
        ell_comps=(0.152828, -0.088235),
        einstein_radius=1.3,
        core_radius=0.2,
    )

    potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert potential == pytest.approx(0.74354, 1e-3)


def test__potential_2d_from__isothermal_core_ell_config_2():
    mp = ag.mp.IsothermalCore(
        centre=(0.2, -0.2),
        ell_comps=(-0.216506, -0.125),
        einstein_radius=0.5,
        core_radius=0.5,
    )

    potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert potential == pytest.approx(0.04024, 1e-3)


def test__potential_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.IsothermalCore(
        centre=(1.1, 1.1),
        ell_comps=(0.0, 0.0),
        einstein_radius=3.0,
        core_radius=1.0,
    )
    spherical = ag.mp.IsothermalCoreSph(
        centre=(1.1, 1.1), einstein_radius=3.0, core_radius=1.0
    )

    assert elliptical.potential_2d_from(grid=grid) == pytest.approx(
        spherical.potential_2d_from(grid=grid).array, 1e-4
    )
