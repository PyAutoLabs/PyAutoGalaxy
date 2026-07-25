import autogalaxy as ag
import numpy as np
import pytest

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__potential_2d_from__gamma_2_only():
    mp = ag.mp.ExternalShear(gamma_1=0.0, gamma_2=0.1)
    potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[0.1, 0.1]]))

    assert potential == pytest.approx(np.array([0.001]), 1.0e-4)


def test__potential_2d_from__multiple_grid_points():
    mp = ag.mp.ExternalShear(gamma_1=0.0, gamma_2=0.1)
    potential = mp.potential_2d_from(
        grid=ag.Grid2DIrregular([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
    )
    assert potential == pytest.approx(np.array([0.001, 0.004, 0.009]), 1.0e-4)


def test__potential_2d_from__gamma_1_and_gamma_2():
    mp = ag.mp.ExternalShear(gamma_1=0.1, gamma_2=-0.05)
    potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[0.1, 0.1]]))

    assert potential == pytest.approx(np.array([-0.0005]), 1.0e-4)


def test__deflections_yx_2d_from__gamma_2_only():
    mp = ag.mp.ExternalShear(gamma_1=0.0, gamma_2=0.1)
    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))
    assert deflections[0, 0] == pytest.approx(0.01625, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.01625, 1e-3)


def test__deflections_yx_2d_from__gamma_1_and_gamma_2():
    mp = ag.mp.ExternalShear(gamma_1=-0.17320, gamma_2=0.1)
    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))
    assert deflections[0, 0] == pytest.approx(0.04439, 1e-3)
    assert deflections[0, 1] == pytest.approx(-0.011895, 1e-3)


def test__convergence_returns_zeros__single_point():
    mp = ag.mp.ExternalShear(gamma_1=0.0, gamma_2=0.1)
    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.1, 0.1]]))
    assert (convergence == np.array([0.0])).all()


def test__convergence_returns_zeros__multiple_grid_points():
    mp = ag.mp.ExternalShear(gamma_1=0.0, gamma_2=0.1)
    convergence = mp.convergence_2d_from(
        grid=ag.Grid2DIrregular([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
    )
    assert (convergence == np.array([0.0, 0.0, 0.0])).all()


def test__convergence_returns_zeros__y_axis():
    mp = ag.mp.ExternalShear(gamma_1=0.0, gamma_2=0.1)
    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert convergence[0] == pytest.approx(0.0, 1e-3)
