import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_2d_via_analytic_from():
    mp = ag.mp.Chameleon(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        core_radius_0=0.2,
        core_radius_1=0.4,
        mass_to_light_ratio=3.0,
    )

    deflections = mp.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections[0, 0] == pytest.approx(2.12608, 1e-3)
    assert deflections[0, 1] == pytest.approx(1.55252, 1e-3)


def test__deflections_yx_2d_from__chameleon():
    mp = ag.mp.Chameleon()

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflections_via_integral = mp.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_integral.array, 1.0e-4)


def test__deflections_yx_2d_from__chameleon_sph():
    mp = ag.mp.ChameleonSph()

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflections_via_integral = mp.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_integral.array, 1.0e-4)


def test__spherical_and_elliptical_identical():
    elliptical = ag.mp.Chameleon(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        mass_to_light_ratio=1.0,
    )

    spherical = ag.mp.ChameleonSph(
        centre=(0.0, 0.0), intensity=1.0, mass_to_light_ratio=1.0
    )

    elliptical_deflections = elliptical.deflections_yx_2d_from(grid=grid)
    spherical_deflections = spherical.deflections_yx_2d_from(grid=grid)

    assert elliptical_deflections == pytest.approx(spherical_deflections.array, 1.0e-4)


def test__convergence_2d_from__chameleon_config_1():
    mp = ag.mp.Chameleon(
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        core_radius_0=0.1,
        core_radius_1=0.3,
        mass_to_light_ratio=2.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(2.0 * 0.018605, 1e-3)


def test__convergence_2d_from__chameleon_config_2():
    mp = ag.mp.Chameleon(
        ell_comps=(0.5, 0.0),
        intensity=3.0,
        core_radius_0=0.2,
        core_radius_1=0.4,
        mass_to_light_ratio=1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.5]]))

    assert convergence == pytest.approx(0.007814, 1e-3)


def test__convergence_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.Chameleon(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        mass_to_light_ratio=1.0,
    )

    spherical = ag.mp.ChameleonSph(
        centre=(0.0, 0.0), intensity=1.0, mass_to_light_ratio=1.0
    )

    assert elliptical.convergence_2d_from(grid=grid) == pytest.approx(
        spherical.convergence_2d_from(grid=grid).array, 1.0e-4
    )
