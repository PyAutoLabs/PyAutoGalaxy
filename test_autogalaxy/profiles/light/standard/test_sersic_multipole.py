from __future__ import division, print_function

import numpy as np
import pytest

import autogalaxy as ag


grid = ag.Grid2DIrregular([[1.0, 0.0], [0.5, 0.5], [2.0, 1.0], [-1.0, 0.3]])


def test__sersic_multipole__zero_perturbation__matches_sersic_exactly():
    kwargs = dict(
        centre=(0.0, 0.0),
        ell_comps=(0.1, 0.2),
        intensity=2.5,
        effective_radius=1.3,
        sersic_index=3.0,
    )
    base = ag.lp.Sersic(**kwargs)
    mp = ag.lp.SersicMultipole(
        **kwargs,
        multipole_3_comps=(0.0, 0.0),
        multipole_4_comps=(0.0, 0.0),
    )

    image_base = np.asarray(base.image_2d_from(grid=grid))
    image_mp = np.asarray(mp.image_2d_from(grid=grid))

    assert image_base == pytest.approx(image_mp, rel=1e-12, abs=1e-12)


def test__sersic_multipole__defaults__match_default_sersic():
    image_base = np.asarray(ag.lp.Sersic().image_2d_from(grid=grid))
    image_mp = np.asarray(ag.lp.SersicMultipole().image_2d_from(grid=grid))

    assert image_base == pytest.approx(image_mp, rel=1e-12, abs=1e-12)


def test__sersic_multipole__m4_only__has_fourfold_symmetry_on_circular_base():
    mp = ag.lp.SersicMultipole(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=4.0,
        multipole_4_comps=(0.05, 0.02),
    )
    pts = np.array([[1.0, 0.5], [0.3, 1.2], [-0.5, 0.8]])
    rotated = np.column_stack([-pts[:, 1], pts[:, 0]])

    image_a = np.asarray(mp.image_2d_from(grid=ag.Grid2DIrregular(pts)))
    image_b = np.asarray(mp.image_2d_from(grid=ag.Grid2DIrregular(rotated)))

    assert image_a == pytest.approx(image_b, rel=1e-10, abs=1e-12)


def test__sersic_multipole__m3_only__has_threefold_symmetry_on_circular_base():
    mp = ag.lp.SersicMultipole(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=4.0,
        multipole_3_comps=(0.03, 0.04),
    )
    c120 = np.cos(2.0 * np.pi / 3.0)
    s120 = np.sin(2.0 * np.pi / 3.0)
    pts = np.array([[1.0, 0.5], [0.3, 1.2], [-0.5, 0.8]])
    rotated = np.column_stack(
        [
            c120 * pts[:, 0] - s120 * pts[:, 1],
            s120 * pts[:, 0] + c120 * pts[:, 1],
        ]
    )

    image_a = np.asarray(mp.image_2d_from(grid=ag.Grid2DIrregular(pts)))
    image_b = np.asarray(mp.image_2d_from(grid=ag.Grid2DIrregular(rotated)))

    assert image_a == pytest.approx(image_b, rel=1e-10, abs=1e-12)


def test__sersic_multipole__centre_translation__shifts_image():
    mp_origin = ag.lp.SersicMultipole(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=4.0,
        multipole_4_comps=(0.05, 0.0),
    )
    mp_translated = ag.lp.SersicMultipole(
        centre=(0.3, -0.2),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=4.0,
        multipole_4_comps=(0.05, 0.0),
    )
    pts = np.array([[1.0, 0.5], [-0.4, 0.7]])
    pts_shifted = pts + np.array([0.3, -0.2])

    image_origin = np.asarray(mp_origin.image_2d_from(grid=ag.Grid2DIrregular(pts)))
    image_translated = np.asarray(
        mp_translated.image_2d_from(grid=ag.Grid2DIrregular(pts_shifted))
    )

    assert image_origin == pytest.approx(image_translated, rel=1e-10, abs=1e-12)


def test__sersic_multipole__nonzero_multipole__differs_from_sersic():
    sersic = ag.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=4.0,
    )
    mp = ag.lp.SersicMultipole(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=4.0,
        multipole_4_comps=(0.1, 0.0),
    )
    diff = np.asarray(mp.image_2d_from(grid=grid)) - np.asarray(
        sersic.image_2d_from(grid=grid)
    )

    assert np.max(np.abs(diff)) > 1e-5


def test__sersic_multipole__finite_and_nonnegative_under_large_perturbation():
    mp = ag.lp.SersicMultipole(
        centre=(0.0, 0.0),
        ell_comps=(0.1, 0.1),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=4.0,
        multipole_3_comps=(0.2, -0.15),
        multipole_4_comps=(0.3, 0.1),
    )
    uniform_grid = ag.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2)
    image = np.asarray(mp.image_2d_from(grid=uniform_grid))

    assert np.all(np.isfinite(image))
    assert np.all(image >= 0.0)
