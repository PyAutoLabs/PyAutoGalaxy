import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from():
    mp = ag.mp.Exponential()

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflections_via_cse = mp.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_cse.array, 1.0e-4)

    mp = ag.mp.ExponentialSph()

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflections_via_cse = mp.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_cse.array, 1.0e-4)


def test__deflections_2d_via_integral_from():
    mp = ag.mp.Exponential(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        mass_to_light_ratio=1.0,
    )

    deflections = mp.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections[0, 0] == pytest.approx(0.90493, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.62569, 1e-3)

    mp = ag.mp.Exponential(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        mass_to_light_ratio=1.0,
    )

    deflections = mp.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([(0.1625, 0.1625)])
    )

    assert deflections[0, 0] == pytest.approx(0.90493, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.62569, 1e-3)


def test__deflections_2d_via_cse_from():
    mp = ag.mp.Exponential(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.8,
        mass_to_light_ratio=1.0,
    )

    deflections_via_integral = mp.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )
    deflections_via_cse = mp.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-4)

    mp = ag.mp.Exponential(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.8,
        mass_to_light_ratio=1.0,
    )

    deflections_via_integral = mp.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )
    deflections_via_cse = mp.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-4)


def test__convergence_2d_from():
    mp = ag.mp.Exponential(
        ell_comps=(0.0, 0.333333),
        intensity=3.0,
        effective_radius=2.0,
        mass_to_light_ratio=1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert convergence == pytest.approx(4.9047, 1e-3)

    mp = ag.mp.Exponential(
        ell_comps=(0.0, -0.333333),
        intensity=2.0,
        effective_radius=3.0,
        mass_to_light_ratio=1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(4.8566, 1e-3)

    mp = ag.mp.Exponential(
        ell_comps=(0.0, -0.333333),
        intensity=4.0,
        effective_radius=3.0,
        mass_to_light_ratio=1.0,
    )
    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(2.0 * 4.8566, 1e-3)

    mp = ag.mp.Exponential(
        ell_comps=(0.0, -0.333333),
        intensity=2.0,
        effective_radius=3.0,
        mass_to_light_ratio=2.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(2.0 * 4.8566, 1e-3)

    mp = ag.mp.Exponential(
        ell_comps=(0.0, -0.333333),
        intensity=2.0,
        effective_radius=3.0,
        mass_to_light_ratio=1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(4.8566, 1e-3)

    elliptical = ag.mp.Exponential(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        mass_to_light_ratio=1.0,
    )

    spherical = ag.mp.Exponential(
        centre=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        mass_to_light_ratio=1.0,
    )

    ell_convergence_2d = elliptical.convergence_2d_from(grid=grid)
    sph_convergence_2d = spherical.convergence_2d_from(grid=grid)

    assert ell_convergence_2d == pytest.approx(sph_convergence_2d, 1.0e-4)
