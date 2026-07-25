from __future__ import division, print_function
import numpy as np
import pytest

import autogalaxy as ag


def test__image_2d_from__spherical__centred_at_origin__correct_values():
    shapelet = ag.lp_linear.ShapeletCartesianSph(
        n_y=2, n_x=3, centre=(0.0, 0.0), beta=1.0
    )

    image = shapelet.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(np.array([0.1397, 0.0708009]), 1e-4)


def test__image_2d_from__spherical__offset_centre__correct_values():
    shapelet = ag.lp_linear.ShapeletCartesianSph(
        n_y=2, n_x=3, centre=(0.2, 0.4), beta=1.0
    )

    image = shapelet.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(np.array([0.23733, -0.07913]), 1e-4)


def test__image_2d_from__elliptical__ell_comps_0p1_0p2__correct_values():
    shapelet = ag.lp_linear.ShapeletCartesian(
        n_y=2, n_x=3, centre=(0.0, 0.0), ell_comps=(0.1, 0.2), beta=1.0
    )

    image = shapelet.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(np.array([0.1066423886714124, 0.014346163370]), 1e-4)


def test__image_2d_from__elliptical__ell_comps_0p2_0p3__correct_values():
    shapelet = ag.lp_linear.ShapeletCartesian(
        n_y=2, n_x=3, centre=(0.0, 0.0), ell_comps=(0.2, 0.3), beta=1.0
    )

    image = shapelet.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(np.array([0.0322749900, -0.075487038]), 1e-4)
