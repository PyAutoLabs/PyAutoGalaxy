import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from__power_law_core_sph_config_1():
    mp = ag.mp.PowerLawCoreSph(
        centre=(-0.7, 0.5), einstein_radius=1.0, slope=1.8, core_radius=0.2
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(0.80677, 1e-3)
    assert deflections[0, 1] == pytest.approx(-0.30680, 1e-3)


def test__deflections_yx_2d_from__power_law_core_sph_config_2():
    mp = ag.mp.PowerLawCoreSph(
        centre=(0.2, -0.2), einstein_radius=0.5, slope=2.4, core_radius=0.5
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(-0.00321, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.09316, 1e-3)


def test__deflections_yx_2d_from__power_law_core_ell_config_1():
    cored_power_law = ag.mp.PowerLawCore(
        centre=(-0.7, 0.5),
        ell_comps=(0.152828, -0.088235),
        einstein_radius=1.3,
        slope=1.8,
        core_radius=0.2,
    )

    deflections = cored_power_law.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections[0, 0] == pytest.approx(0.9869, 1e-3)
    assert deflections[0, 1] == pytest.approx(-0.54882, 1e-3)


def test__deflections_yx_2d_from__power_law_core_ell_config_2():
    cored_power_law = ag.mp.PowerLawCore(
        centre=(0.2, -0.2),
        ell_comps=(-0.216506, -0.125),
        einstein_radius=0.5,
        slope=2.4,
        core_radius=0.5,
    )

    deflections = cored_power_law.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )
    assert deflections[0, 0] == pytest.approx(0.01111, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.11403, 1e-3)


def test__deflections_yx_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.PowerLawCore(
        centre=(1.1, 1.1),
        ell_comps=(0.0, 0.0),
        einstein_radius=3.0,
        slope=2.2,
        core_radius=0.1,
    )
    spherical = ag.mp.PowerLawCoreSph(
        centre=(1.1, 1.1), einstein_radius=3.0, slope=2.2, core_radius=0.1
    )

    assert elliptical.deflections_yx_2d_from(grid=grid) == pytest.approx(
        spherical.deflections_yx_2d_from(grid=grid).array, 1e-4
    )


def test__convergence_2d_from__power_law_core_sph_convergence_func():
    mp = ag.mp.PowerLawCoreSph(
        centre=(1, 1), einstein_radius=1.0, slope=2.2, core_radius=0.1
    )

    convergence = mp.convergence_func(grid_radius=1.0)

    assert convergence == pytest.approx(0.39762, 1e-4)


def test__convergence_2d_from__power_law_core_ell_config_1():
    mp = ag.mp.PowerLawCore(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        einstein_radius=1.0,
        slope=2.3,
        core_radius=0.2,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.45492, 1e-3)


def test__convergence_2d_from__power_law_core_ell_config_2():
    mp = ag.mp.PowerLawCore(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        einstein_radius=2.0,
        slope=1.7,
        core_radius=0.2,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(1.3887, 1e-3)


def test__convergence_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.PowerLawCore(
        centre=(1.1, 1.1),
        ell_comps=(0.0, 0.0),
        einstein_radius=3.0,
        slope=2.2,
        core_radius=0.1,
    )
    spherical = ag.mp.PowerLawCoreSph(
        centre=(1.1, 1.1), einstein_radius=3.0, slope=2.2, core_radius=0.1
    )

    assert elliptical.convergence_2d_from(grid=grid).array == pytest.approx(
        spherical.convergence_2d_from(grid=grid).array, 1e-4
    )


def test__potential_2d_from__power_law_core_sph_config_1():
    mp = ag.mp.PowerLawCoreSph(
        centre=(-0.7, 0.5), einstein_radius=1.0, slope=1.8, core_radius=0.2
    )

    potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert potential == pytest.approx(0.54913, 1e-3)


def test__potential_2d_from__power_law_core_sph_config_2():
    mp = ag.mp.PowerLawCoreSph(
        centre=(0.2, -0.2), einstein_radius=0.5, slope=2.4, core_radius=0.5
    )

    potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert potential == pytest.approx(0.01820, 1e-3)


def test__potential_2d_from__power_law_core_ell_config_1():
    cored_power_law = ag.mp.PowerLawCore(
        centre=(0.2, -0.2),
        ell_comps=(-0.216506, -0.125),
        einstein_radius=0.5,
        slope=2.4,
        core_radius=0.5,
    )

    potential = cored_power_law.potential_2d_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert potential == pytest.approx(0.02319, 1e-3)


def test__potential_2d_from__power_law_core_ell_config_2():
    cored_power_law = ag.mp.PowerLawCore(
        centre=(-0.7, 0.5),
        ell_comps=(0.152828, -0.088235),
        einstein_radius=1.3,
        slope=1.8,
        core_radius=0.2,
    )

    potential = cored_power_law.potential_2d_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert potential == pytest.approx(0.71185, 1e-3)


def test__potential_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.PowerLawCore(
        centre=(1.1, 1.1),
        ell_comps=(0.0, 0.0),
        einstein_radius=3.0,
        slope=2.2,
        core_radius=0.1,
    )
    spherical = ag.mp.PowerLawCoreSph(
        centre=(1.1, 1.1), einstein_radius=3.0, slope=2.2, core_radius=0.1
    )

    assert elliptical.potential_2d_from(grid=grid) == pytest.approx(
        spherical.potential_2d_from(grid=grid).array, 1e-4
    )
