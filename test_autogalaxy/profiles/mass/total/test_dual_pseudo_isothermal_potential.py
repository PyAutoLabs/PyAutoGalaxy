import pytest

import autoarray as aa
import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__convergence_2d_from__returns_array2d_not_vector():
    # Regression test for a copy-paste typo: convergence_2d_from used to be
    # decorated with @aa.decorators.to_vector_yx (the deflections decorator
    # directly above it) which wrapped the scalar convergence in a VectorYX2D.
    # The correct decorator is @aa.decorators.to_array.
    mp = ag.mp.dPIEPotential(centre=(0.0, 0.0), ell_comps=(0.05, 0.0), ra=0.2, rs=2.0, b0=1.0)

    convergence = mp.convergence_2d_from(grid=grid)

    assert isinstance(convergence, aa.ArrayIrregular)


def test__deflections_yx_2d_from__sph_config_1():
    mp = ag.mp.dPIEPotentialSph(centre=(-0.7, 0.5), b0=5.2, ra=2.0, rs=3.0)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(1.033080741, 1e-4)
    assert deflections[0, 1] == pytest.approx(-0.39286169026, 1e-4)


def test__deflections_yx_2d_from__sph_config_2():
    mp = ag.mp.dPIEPotentialSph(centre=(-0.1, 0.1), b0=20.0, ra=2.0, rs=3.0)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(1.4212977207, 1e-4)
    assert deflections[0, 1] == pytest.approx(0.308977765378, 1e-4)


def test__deflections_yx_2d_from__elliptical():
    mp = ag.mp.dPIEPotential(
        centre=(0, 0), ell_comps=(0.0, 0.333333), b0=4.0, ra=2.0, rs=3.0
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(0.186341843, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.13176363087, 1e-3)


def test__deflections_yx_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.dPIEPotential(
        centre=(1.1, 1.1), ell_comps=(0.0, 0.0), b0=12.0, ra=2.0, rs=3.0
    )
    spherical = ag.mp.dPIEPotentialSph(centre=(1.1, 1.1), b0=12.0, ra=2.0, rs=3.0)

    assert elliptical.deflections_yx_2d_from(grid=grid) == pytest.approx(
        spherical.deflections_yx_2d_from(grid=grid), 1e-4
    )


def test__convergence_2d_from__sph():
    # eta = 1.0
    # kappa = 0.5 * 1.0 ** 1.0

    mp = ag.mp.dPIEPotentialSph(centre=(0.0, 0.0), b0=8.0, ra=2.0, rs=3.0)

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(1.57182995, 1e-3)


def test__convergence_2d_from__elliptical_no_ell_comps_b0_4():
    mp = ag.mp.dPIEPotential(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), b0=4.0, ra=2.0, rs=3.0
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.78591498, 1e-3)


def test__convergence_2d_from__elliptical_no_ell_comps_b0_8():
    mp = ag.mp.dPIEPotential(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), b0=8.0, ra=2.0, rs=3.0
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(1.57182995, 1e-3)


def test__convergence_2d_from__elliptical_with_ell_comps():
    mp = ag.mp.dPIEPotential(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.333333), b0=4.0, ra=2.0, rs=3.0
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.87182837, 1e-3)


def test__convergence_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.dPIEPotential(
        centre=(1.1, 1.1), ell_comps=(0.0, 0.0), b0=12.0, ra=2.0, rs=3.0
    )
    spherical = ag.mp.dPIEPotentialSph(centre=(1.1, 1.1), b0=12.0, ra=2.0, rs=3.0)

    assert elliptical.convergence_2d_from(grid=grid).array == pytest.approx(
        spherical.convergence_2d_from(grid=grid).array, 1e-4
    )
