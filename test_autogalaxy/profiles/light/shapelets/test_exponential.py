from __future__ import division, print_function
import numpy as np
import pytest

import autogalaxy as ag


def test__image_2d_from__spherical__centred_at_origin__correct_values():
    shapelet = ag.lp_linear.ShapeletExponentialSph(
        n=2, m=0, centre=(0.0, 0.0), beta=1.0
    )

    image = shapelet.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(np.array([0.05784, 0.17962]), 1e-4)


def test__image_2d_from__spherical__offset_centre__correct_values():
    shapelet = ag.lp_linear.ShapeletExponentialSph(
        n=2, m=0, centre=(0.2, 0.4), beta=1.0
    )

    image = shapelet.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(np.array([0.16136, 0.22476]), 1e-4)


def test__image_2d_from__elliptical__ell_comps_0p1_0p2__correct_values():
    shapelet = ag.lp_linear.ShapeletExponential(
        n=2, m=0, centre=(0.0, 0.0), ell_comps=(0.1, 0.2), beta=1.0
    )

    image = shapelet.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(np.array([0.05784, 0.17961]), 1e-4)


def test__image_2d_from__elliptical__ell_comps_0p5_0p7__correct_values():
    shapelet = ag.lp_linear.ShapeletExponential(
        n=2, m=0, centre=(0.0, 0.0), ell_comps=(0.5, 0.7), beta=1.0
    )

    image = shapelet.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(np.array([0.05784, 0.17962]), 1e-4)
