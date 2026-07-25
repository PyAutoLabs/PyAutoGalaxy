import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


# def test__deflections_2d_via_mge_from():
#     mp = ag.mp.SersicCore(
#         centre=(1.0, 2.0),
#         ell_comps=ag.convert.ell_comps_from(axis_ratio=0.5, angle=70.0),
#         intensity=0.45,
#         effective_radius=0.5,
#         radius_break=0.01,
#         gamma=0.0,
#         alpha=2.0,
#         sersic_index=2.2,
#     )
#
#     deflections = mp.deflections_2d_via_mge_from(grid=ag.Grid2DIrregular([[2.5, -2.5]]))
#
#     assert deflections[0, 0] == pytest.approx(0.0015047, 1e-4)
#     assert deflections[0, 1] == pytest.approx(-0.004493, 1e-4)
#
#     mp = ag.mp.SersicCore(
#         centre=(1.0, 2.0),
#         ell_comps=ag.convert.ell_comps_from(axis_ratio=0.5, angle=70.0),
#         intensity=2.0 * 0.45,
#         effective_radius=0.5,
#         radius_break=0.01,
#         gamma=0.0,
#         alpha=2.0,
#         sersic_index=2.2,
#     )
#
#     deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[2.5, -2.5]]))
#
#     assert deflections[0, 0] == pytest.approx(2.0 * 0.0015047, 1e-4)
#     assert deflections[0, 1] == pytest.approx(2.0 * -0.004493, 1e-4)
#
#     mp = ag.mp.SersicCore(
#         centre=(1.0, 2.0),
#         ell_comps=ag.convert.ell_comps_from(axis_ratio=0.5, angle=70.0),
#         intensity=0.45,
#         effective_radius=0.5,
#         radius_break=0.01,
#         gamma=0.0,
#         alpha=2.0,
#         sersic_index=2.2,
#         mass_to_light_ratio=2.0,
#     )
#
#     deflections = mp.deflections_2d_via_mge_from(grid=ag.Grid2DIrregular([[2.5, -2.5]]))
#
#     assert deflections[0, 0] == pytest.approx(2.0 * 0.0015047, 1e-4)
#     assert deflections[0, 1] == pytest.approx(2.0 * -0.004493, 1e-4)


# def test__deflections_yx_2d_from():
#     mp = ag.mp.SersicCore()
#
#     deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
#     deflections_via_integral = mp.deflections_2d_via_mge_from(
#         grid=ag.Grid2DIrregular([[1.0, 0.0]])
#     )
#
#     assert deflections == pytest.approx(deflections_via_integral, 1.0e-4)
#
#     mp = ag.mp.SersicCoreSph()
#
#     deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
#     deflections_via_integral = mp.deflections_2d_via_mge_from(
#         grid=ag.Grid2DIrregular([[1.0, 0.0]])
#     )
#
#     assert deflections == pytest.approx(deflections_via_integral, 1.0e-4)
#
#     elliptical = ag.mp.SersicCore(
#         centre=(0.0, 0.0),
#         ell_comps=(0.0, 0.0),
#         effective_radius=1.0,
#         sersic_index=4.0,
#         mass_to_light_ratio=1.0,
#     )
#
#     spherical = ag.mp.SersicCore(
#         centre=(0.0, 0.0),
#         effective_radius=1.0,
#         sersic_index=4.0,
#         mass_to_light_ratio=1.0,
#     )
#
#     elliptical_deflections = elliptical.deflections_2d_via_integral_from(grid=grid)
#     spherical_deflections = spherical.deflections_2d_via_integral_from(grid=grid)
#
#     assert elliptical_deflections == pytest.approx(spherical_deflections, 1.0e-4)
#


def test__convergence_2d_from__sersic_core_mass_to_light_1():
    mp = ag.mp.SersicCore(
        ell_comps=(0.0, 0.0),
        effective_radius=5.0,
        sersic_index=4.0,
        radius_break=0.01,
        intensity=0.1,
        gamma=1.0,
        alpha=1.0,
        mass_to_light_ratio=1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 0.01]]))

    assert convergence == pytest.approx(0.1, 1e-3)


def test__convergence_2d_from__sersic_core_mass_to_light_2():
    mp = ag.mp.SersicCore(
        ell_comps=(0.0, 0.0),
        effective_radius=5.0,
        sersic_index=4.0,
        radius_break=0.01,
        intensity=0.1,
        gamma=1.0,
        alpha=1.0,
        mass_to_light_ratio=2.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 0.01]]))

    assert convergence == pytest.approx(0.2, 1e-3)


def test__convergence_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        effective_radius=1.0,
        sersic_index=4.0,
        mass_to_light_ratio=1.0,
    )

    spherical = ag.mp.SersicCore(
        centre=(0.0, 0.0),
        effective_radius=1.0,
        sersic_index=4.0,
        mass_to_light_ratio=1.0,
    )

    ell_convergence_2d = elliptical.convergence_2d_from(grid=grid)
    sph_convergence_2d = spherical.convergence_2d_from(grid=grid)

    assert ell_convergence_2d == pytest.approx(sph_convergence_2d.array, 1.0e-4)


# def test__convergence_2d_via_mge_from():
#     mp = ag.mp.SersicCore(
#         ell_comps=(0.2, 0.4),
#         effective_radius=5.0,
#         sersic_index=4.0,
#         radius_break=0.01,
#         intensity=0.1,
#         gamma=1.0,
#         alpha=1.0,
#         mass_to_light_ratio=1.0,
#     )
#
#     convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))
#     convergence_via_mge = mp.convergence_2d_via_mge_from(
#         grid=ag.Grid2DIrregular([[0.0, 1.0]])
#     )
#
#     assert convergence == pytest.approx(convergence_via_mge, 1e-3)
