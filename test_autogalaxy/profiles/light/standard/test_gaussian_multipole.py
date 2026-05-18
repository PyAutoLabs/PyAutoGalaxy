from __future__ import division, print_function

import numpy as np
import pytest

import autogalaxy as ag


grid = ag.Grid2DIrregular([[1.0, 0.0], [0.5, 0.5], [2.0, 1.0], [-1.0, 0.3]])


def test__gaussian_multipole__zero_perturbation__matches_gaussian_exactly():
    kwargs = dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.2), intensity=1.5, sigma=0.7)
    base = ag.lp.Gaussian(**kwargs)
    mp = ag.lp.GaussianMultipole(
        **kwargs,
        multipole_3_comps=(0.0, 0.0),
        multipole_4_comps=(0.0, 0.0),
    )

    image_base = np.asarray(base.image_2d_from(grid=grid))
    image_mp = np.asarray(mp.image_2d_from(grid=grid))

    assert image_base == pytest.approx(image_mp, rel=1e-12, abs=1e-12)


def test__gaussian_multipole__defaults__match_default_gaussian():
    image_base = np.asarray(ag.lp.Gaussian().image_2d_from(grid=grid))
    image_mp = np.asarray(ag.lp.GaussianMultipole().image_2d_from(grid=grid))

    assert image_base == pytest.approx(image_mp, rel=1e-12, abs=1e-12)


def test__gaussian_multipole__m4_only__has_fourfold_symmetry_on_circular_base():
    mp = ag.lp.GaussianMultipole(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        sigma=0.5,
        multipole_4_comps=(0.05, 0.02),
    )
    pts = np.array([[1.0, 0.5], [0.3, 1.2], [-0.5, 0.8]])
    rotated = np.column_stack([-pts[:, 1], pts[:, 0]])

    image_a = np.asarray(mp.image_2d_from(grid=ag.Grid2DIrregular(pts)))
    image_b = np.asarray(mp.image_2d_from(grid=ag.Grid2DIrregular(rotated)))

    assert image_a == pytest.approx(image_b, rel=1e-10, abs=1e-12)


def test__gaussian_multipole__nonzero_multipole__differs_from_gaussian():
    gauss = ag.lp.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=1.0, sigma=0.5
    )
    mp = ag.lp.GaussianMultipole(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        sigma=0.5,
        multipole_4_comps=(0.1, 0.0),
    )
    diff = np.asarray(mp.image_2d_from(grid=grid)) - np.asarray(
        gauss.image_2d_from(grid=grid)
    )

    assert np.max(np.abs(diff)) > 1e-5


def test__gaussian_multipole__finite_and_nonnegative_under_large_perturbation():
    mp = ag.lp.GaussianMultipole(
        centre=(0.0, 0.0),
        ell_comps=(0.1, 0.1),
        intensity=1.0,
        sigma=0.5,
        multipole_3_comps=(0.2, -0.15),
        multipole_4_comps=(0.3, 0.1),
    )
    uniform_grid = ag.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2)
    image = np.asarray(mp.image_2d_from(grid=uniform_grid))

    assert np.all(np.isfinite(image))
    assert np.all(image >= 0.0)
