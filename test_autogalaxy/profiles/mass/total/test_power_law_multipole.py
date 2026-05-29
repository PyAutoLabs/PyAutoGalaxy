import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from__config_1():
    mp = ag.mp.PowerLawMultipole(
        m=4,
        centre=(0.1, 0.2),
        einstein_radius=2.0,
        slope=2.2,
        multipole_comps=(0.1, 0.2),
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert deflections[0, 0] == pytest.approx(-0.036120991, 1e-3)
    assert deflections[0, 1] == pytest.approx(-0.0476260676, 1e-3)


def test__deflections_yx_2d_from__config_2():
    mp = ag.mp.PowerLawMultipole(
        m=4,
        centre=(0.2, 0.3),
        einstein_radius=3.0,
        slope=1.7,
        multipole_comps=(0.2, 0.3),
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert deflections[0, 0] == pytest.approx(-0.096376665, 1e-3)
    assert deflections[0, 1] == pytest.approx(-0.1298677210, 1e-3)


def test__convergence_2d_from__config_1():
    mp = ag.mp.PowerLawMultipole(
        m=4,
        centre=(0.1, 0.2),
        einstein_radius=2.0,
        slope=2.2,
        multipole_comps=(0.1, 0.2),
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert convergence[0] == pytest.approx(0.25958037, 1e-3)


def test__convergence_2d_from__config_2():
    mp = ag.mp.PowerLawMultipole(
        m=4,
        centre=(0.2, 0.3),
        einstein_radius=3.0,
        slope=1.7,
        multipole_comps=(0.2, 0.3),
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert convergence[0] == pytest.approx(0.2875647, 1e-3)


def test__potential_2d_from():
    mp = ag.mp.PowerLawMultipole(
        m=4,
        centre=(0.1, 0.2),
        einstein_radius=2.0,
        slope=2.2,
        multipole_comps=(0.1, 0.2),
    )

    potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert potential[0] == pytest.approx(0.0, 1e-3)


def test__convergence_func__returns_zero_monopole():
    """Regression: PowerLawMultipole overrides the abstract `convergence_func` to return
    the zero monopole. The multipole convergence is angle-dependent (cos(m(phi - phi_m)))
    so its azimuthal average is identically zero — a pure multipole encloses zero net
    azimuthally-symmetric mass. `convergence_func` is reached only by radial mass
    integration (`mass_integral` -> `mass_angular_within_circle_from`)."""

    import numpy as np

    mp = ag.mp.PowerLawMultipole(
        m=4,
        centre=(0.0, 0.0),
        einstein_radius=1.0,
        slope=2.0,
        multipole_comps=(0.1, 0.2),
    )

    assert mp.convergence_func(1.5) == pytest.approx(0.0, 1e-12)

    radii = np.array([0.1, 0.5, 1.0, 2.5])
    actual = mp.convergence_func(radii)
    assert actual.shape == radii.shape
    assert actual == pytest.approx(np.zeros_like(radii), 1e-12)

    # A pure multipole encloses zero net mass.
    assert mp.mass_angular_within_circle_from(radius=2.0) == pytest.approx(0.0, 1e-9)
