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


def _potential_gradient_from(mp, points, h=1.0e-4):
    def potential(shift):
        return np.asarray(
            mp.potential_2d_from(grid=ag.Grid2DIrregular(points + shift))
        )

    gradient_y = (potential([h, 0.0]) - potential([-h, 0.0])) / (2.0 * h)
    gradient_x = (potential([0.0, h]) - potential([0.0, -h])) / (2.0 * h)

    return np.stack([gradient_y, gradient_x], axis=-1)


_potential_points = np.array([[0.7, 0.3], [-0.5, 0.9], [1.2, -1.0], [2.5, 1.5]])

_potential_profiles = [
    ag.mp.Chameleon(ell_comps=(0.2, -0.1), intensity=1.0),
    ag.mp.ChameleonSph(intensity=1.0),
    ag.mp.Chameleon(
        centre=(0.1, -0.2),
        ell_comps=(0.2, -0.1),
        intensity=1.0,
        core_radius_0=0.1,
        core_radius_1=0.5,
        mass_to_light_ratio=2.0,
    ),
]


@pytest.mark.parametrize("mp", _potential_profiles)
def test__potential_2d_from__gradient_matches_deflections(mp):
    deflections = np.asarray(
        mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular(_potential_points))
    )

    gradient = _potential_gradient_from(mp=mp, points=_potential_points)

    assert np.max(np.abs(gradient - deflections)) < 1.0e-4 * np.max(
        np.abs(deflections)
    )


@pytest.mark.parametrize("mp", _potential_profiles)
def test__potential_2d_from__laplacian_is_twice_convergence(mp):
    h = 1.0e-3

    def potential(shift):
        return np.asarray(
            mp.potential_2d_from(grid=ag.Grid2DIrregular(_potential_points + shift))
        )

    laplacian = (
        potential([h, 0.0])
        + potential([-h, 0.0])
        + potential([0.0, h])
        + potential([0.0, -h])
        - 4.0 * potential([0.0, 0.0])
    ) / h**2

    convergence = np.asarray(
        mp.convergence_2d_from(grid=ag.Grid2DIrregular(_potential_points))
    )

    assert laplacian == pytest.approx(2.0 * convergence, rel=1.0e-3)


def test__potential_2d_from():
    mp = ag.mp.Chameleon(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        core_radius_0=0.2,
        core_radius_1=0.4,
        mass_to_light_ratio=3.0,
    )

    potential = mp.potential_2d_from(
        grid=ag.Grid2DIrregular([[-0.4, -0.2], [0.1625, 0.1625]])
    )

    assert potential[0] == pytest.approx(0.0, abs=1.0e-8)
    assert potential[1] == pytest.approx(1.68326, 1e-4)
