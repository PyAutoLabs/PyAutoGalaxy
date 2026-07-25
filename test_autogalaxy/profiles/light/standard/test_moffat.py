from __future__ import division, print_function
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__image_2d_from__circular__alpha_1_beta_1__correct_value():
    lp = ag.lp.Moffat(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        alpha=1.0,
        beta=1.0,
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert image == pytest.approx(0.5, 1e-4)


def test__image_2d_from__elliptical__alpha_1p8_beta_0p75__correct_value():
    lp = ag.lp.Moffat(
        centre=(0.0, 0.0),
        ell_comps=(0.2, 0.2),
        intensity=1.0,
        alpha=1.8,
        beta=0.75,
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 2.0]]))

    assert image == pytest.approx(0.7340746, 1e-4)


def test__image_2d_from__spherical__alpha_1p8_beta_0p75__correct_value():
    lp = ag.lp.MoffatSph(centre=(0.0, 0.0), intensity=1.0, alpha=1.8, beta=0.75)

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 2.0]]))

    assert image == pytest.approx(0.5471480213, 1e-4)
