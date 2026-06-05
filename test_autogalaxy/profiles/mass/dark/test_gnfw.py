import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from__gnfw():
    mp = ag.mp.gNFW()

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflections_via_integral = mp.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_integral.array, 1.0e-4)


def test__deflections_yx_2d_from__gnfw_sph():
    mp = ag.mp.gNFWSph()

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflections_via_integral = mp.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_integral.array, 1.0e-4)


def test__deflections_yx_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.gNFW(
        centre=(0.1, 0.2),
        ell_comps=(0.0, 0.0),
        kappa_s=2.0,
        inner_slope=1.5,
        scale_radius=3.0,
    )
    spherical = ag.mp.gNFWSph(
        centre=(0.1, 0.2), kappa_s=2.0, inner_slope=1.5, scale_radius=3.0
    )

    assert elliptical.deflections_yx_2d_from(grid) == pytest.approx(
        spherical.deflections_yx_2d_from(grid).array, 1e-4
    )


def test__convergence_2d_from__gnfw_sph():
    mp = ag.mp.gNFWSph(
        centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5, scale_radius=1.0
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[2.0, 0.0]]))

    assert convergence == pytest.approx(0.30840, 1e-2)


def test__potential_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.gNFW(
        centre=(0.1, 0.2),
        ell_comps=(0.0, 0.0),
        kappa_s=2.0,
        inner_slope=1.5,
        scale_radius=3.0,
    )
    spherical = ag.mp.gNFWSph(
        centre=(0.1, 0.2), kappa_s=2.0, inner_slope=1.5, scale_radius=3.0
    )

    assert elliptical.potential_2d_from(grid) == pytest.approx(
        spherical.potential_2d_from(grid).array, 1e-4
    )


def test__compare_to_nfw():
    nfw = ag.mp.NFW(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.111111),
        kappa_s=1.0,
        scale_radius=20.0,
    )
    gnfw = ag.mp.gNFW(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.111111),
        kappa_s=1.0,
        inner_slope=1.0,
        scale_radius=20.0,
    )

    assert nfw.deflections_yx_2d_from(grid) == pytest.approx(
        gnfw.deflections_yx_2d_from(grid).array, 1e-3
    )
    assert nfw.deflections_yx_2d_from(grid) == pytest.approx(
        gnfw.deflections_yx_2d_from(grid).array, 1e-3
    )
