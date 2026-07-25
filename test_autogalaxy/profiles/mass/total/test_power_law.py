import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from__power_law_sph_slope_2():
    mp = ag.mp.PowerLawSph(centre=(0.2, 0.2), einstein_radius=1.0, slope=2.0)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(-0.31622, 1e-3)
    assert deflections[0, 1] == pytest.approx(-0.94868, 1e-3)


def test__deflections_yx_2d_from__power_law_sph_slope_25():
    mp = ag.mp.PowerLawSph(centre=(0.2, 0.2), einstein_radius=1.0, slope=2.5)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(-1.59054, 1e-3)
    assert deflections[0, 1] == pytest.approx(-4.77162, 1e-3)


def test__deflections_yx_2d_from__power_law_sph_slope_15():
    mp = ag.mp.PowerLawSph(centre=(0.2, 0.2), einstein_radius=1.0, slope=1.5)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(-0.06287, 1e-3)
    assert deflections[0, 1] == pytest.approx(-0.18861, 1e-3)


def test__deflections_yx_2d_from__power_law_ell_slope_2():
    mp = ag.mp.PowerLaw(
        centre=(0, 0),
        ell_comps=(0.0, 0.333333),
        einstein_radius=1.0,
        slope=2.0,
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(0.79421, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.50734, 1e-3)


def test__deflections_yx_2d_from__power_law_ell_slope_25():
    mp = ag.mp.PowerLaw(
        centre=(0, 0),
        ell_comps=(0.0, 0.333333),
        einstein_radius=1.0,
        slope=2.5,
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(1.29641, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.99629, 1e-3)


def test__deflections_yx_2d_from__power_law_ell_slope_15():
    mp = ag.mp.PowerLaw(
        centre=(0, 0),
        ell_comps=(0.0, 0.333333),
        einstein_radius=1.0,
        slope=1.5,
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(0.48036, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.26729, 1e-3)


def test__deflections_yx_2d_from__power_law_ell_slope_19():
    mp = ag.mp.PowerLaw(
        centre=(-0.7, 0.5),
        ell_comps=(0.152828, -0.088235),
        einstein_radius=1.3,
        slope=1.9,
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    # assert deflections[0, 0] == pytest.approx(1.12841, 1e-3)
    # assert deflections[0, 1] == pytest.approx(-0.60205, 1e-3)


def test__deflections_yx_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.PowerLaw(
        centre=(1.1, 1.1),
        ell_comps=(0.0, 0.0),
        einstein_radius=3.0,
        slope=2.4,
    )

    spherical = ag.mp.PowerLawSph(centre=(1.1, 1.1), einstein_radius=3.0, slope=2.4)

    assert elliptical.deflections_yx_2d_from(grid=grid) == pytest.approx(
        spherical.deflections_yx_2d_from(grid=grid).array, 1e-4
    )


def test__convergence_2d_from__power_law_sph_config_1():
    mp = ag.mp.PowerLawSph(centre=(0.0, 0.0), einstein_radius=1.0, slope=2.0)

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert convergence == pytest.approx(0.5, 1e-3)


def test__convergence_2d_from__power_law_sph_config_2():
    mp = ag.mp.PowerLawSph(centre=(0.0, 0.0), einstein_radius=2.0, slope=2.2)

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[2.0, 0.0]]))

    assert convergence == pytest.approx(0.4, 1e-3)


def test__convergence_2d_from__power_law_sph_config_3():
    mp = ag.mp.PowerLawSph(centre=(0.0, 0.0), einstein_radius=2.0, slope=2.2)

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[2.0, 0.0]]))

    assert convergence == pytest.approx(0.4, 1e-3)


def test__convergence_2d_from__power_law_ell_config_1():
    mp = ag.mp.PowerLaw(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        einstein_radius=1.0,
        slope=2.3,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.466666, 1e-3)


def test__convergence_2d_from__power_law_ell_config_2():
    mp = ag.mp.PowerLaw(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        einstein_radius=2.0,
        slope=1.7,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(1.4079, 1e-3)


def test__convergence_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.PowerLaw(
        centre=(1.1, 1.1),
        ell_comps=(0.0, 0.0),
        einstein_radius=3.0,
        slope=2.4,
    )

    spherical = ag.mp.PowerLawSph(centre=(1.1, 1.1), einstein_radius=3.0, slope=2.4)

    assert elliptical.convergence_2d_from(grid=grid).array == pytest.approx(
        spherical.convergence_2d_from(grid=grid).array, 1e-4
    )


def test__potential_2d_from__power_law_sph_config_1():
    mp = ag.mp.PowerLawSph(centre=(-0.7, 0.5), einstein_radius=1.3, slope=2.3)

    potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert potential == pytest.approx(1.90421, 1e-3)


def test__potential_2d_from__power_law_sph_config_2():
    mp = ag.mp.PowerLawSph(centre=(-0.7, 0.5), einstein_radius=1.3, slope=1.8)

    potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert potential == pytest.approx(0.93758, 1e-3)


def test__potential_2d_from__power_law_ell_config_1():
    mp = ag.mp.PowerLaw(
        centre=(-0.7, 0.5),
        ell_comps=(0.152828, -0.088235),
        einstein_radius=1.3,
        slope=2.2,
    )

    potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert potential == pytest.approx(1.53341, 1e-3)


def test__potential_2d_from__power_law_ell_config_2():
    mp = ag.mp.PowerLaw(
        centre=(-0.7, 0.5),
        ell_comps=(0.152828, -0.088235),
        einstein_radius=1.3,
        slope=1.8,
    )

    potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert potential == pytest.approx(0.96723, 1e-3)


def test__potential_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.PowerLaw(
        centre=(1.1, 1.1),
        ell_comps=(0.0, 0.0),
        einstein_radius=3.0,
        slope=2.4,
    )

    spherical = ag.mp.PowerLawSph(centre=(1.1, 1.1), einstein_radius=3.0, slope=2.4)

    assert elliptical.potential_2d_from(grid=grid) == pytest.approx(
        spherical.potential_2d_from(grid=grid).array, 1e-4
    )
