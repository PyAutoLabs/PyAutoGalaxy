from __future__ import division, print_function
import numpy as np
import pytest

import autogalaxy as ag


def test__image_2d_from__spherical__centred_at_origin__correct_values():
    shapelet = ag.lp_linear.ShapeletPolarSph(n=2, m=0, centre=(0.0, 0.0), beta=1.0)

    image = shapelet.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(np.array([0.0, -0.33177]), 1e-4)


def test__image_2d_from__spherical__offset_centre__correct_values():
    shapelet = ag.lp_linear.ShapeletPolarSph(n=2, m=0, centre=(0.2, 0.4), beta=1.0)

    image = shapelet.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(np.array([-0.27715, -0.47333]), 1e-4)


def test__image_2d_from__elliptical__ell_comps_0p1_0p2__correct_values():
    shapelet = ag.lp_linear.ShapeletPolar(
        n=2, m=0, centre=(0.0, 0.0), ell_comps=(0.1, 0.2), beta=1.0
    )

    image = shapelet.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(
        np.array([0.02577349206014249, -0.17434262753]), abs=1e-4
    )


def test__image_2d_from__elliptical__ell_comps_0p5_0p7__correct_values():
    shapelet = ag.lp_linear.ShapeletPolar(
        n=2, m=0, centre=(0.0, 0.0), ell_comps=(0.5, 0.7), beta=1.0
    )

    image = shapelet.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(np.array([0.001538188813, 0.0]), abs=1e-4)
