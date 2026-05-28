import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from__sph_config_1():
    mp = ag.mp.dPIEMassSph(centre=(-0.7, 0.5), b0=5.2, ra=2.0, rs=3.0)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(1.033080741, 1e-4)
    assert deflections[0, 1] == pytest.approx(-0.39286169026, 1e-4)


def test__deflections_yx_2d_from__sph_config_2():
    mp = ag.mp.dPIEMassSph(centre=(-0.1, 0.1), b0=20.0, ra=2.0, rs=3.0)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(1.4212977207, 1e-4)
    assert deflections[0, 1] == pytest.approx(0.308977765378, 1e-4)


def test__deflections_yx_2d_from__elliptical():
    # First deviation from potential case due to ellipticity

    mp = ag.mp.dPIEMass(
        centre=(0, 0), ell_comps=(0.0, 0.333333), b0=4.0, ra=2.0, rs=3.0
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(0.21461366, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.10753914, 1e-3)


def test__deflections_yx_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.dPIEMass(
        centre=(1.1, 1.1), ell_comps=(0.000001, 0.0000001), b0=12.0, ra=2.0, rs=3.0
    )
    spherical = ag.mp.dPIEMassSph(centre=(1.1, 1.1), b0=12.0, ra=2.0, rs=3.0)

    assert elliptical.deflections_yx_2d_from(grid=grid).array == pytest.approx(
        spherical.deflections_yx_2d_from(grid=grid).array, 1e-1
    )


def test__convergence_func__matches_private_helper():
    """Regression: dPIEMass must override the abstract `convergence_func`
    so MGEDecomposer.decompose_convergence_via_mge (which walks the
    convergence radially during MGE potential decomposition) doesn't
    fall through to the abstract NotImplementedError. The shim delegates
    to the existing `_convergence` radial helper that `convergence_2d_from`
    already uses."""

    import numpy as np

    mp = ag.mp.dPIEMass(centre=(0.0, 0.0), b0=5.2, ra=2.0, rs=3.0)

    # Scalar radius: equals the _convergence formula directly.
    assert mp.convergence_func(1.5) == pytest.approx(mp._convergence(1.5), 1e-12)

    # 1-D array of radii: shape preserved, element-wise equality.
    radii = np.array([0.1, 0.5, 1.0, 2.5, 5.0])
    expected = mp._convergence(radii)
    actual = mp.convergence_func(radii)
    assert actual.shape == radii.shape
    assert actual == pytest.approx(expected, 1e-12)

    # dPIEMassSph inherits the override from dPIEMass.
    sph = ag.mp.dPIEMassSph(centre=(0.0, 0.0), b0=5.2, ra=2.0, rs=3.0)
    assert sph.convergence_func(1.5) == pytest.approx(sph._convergence(1.5), 1e-12)
