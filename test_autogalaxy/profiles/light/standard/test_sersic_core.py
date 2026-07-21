from __future__ import division, print_function
import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__image_2d_from__elliptical__ell_comps_nonzero__correct_value():
    lp = ag.lp.SersicCore(
        ell_comps=(0.0, 0.333333),
        effective_radius=5.0,
        sersic_index=4.0,
        radius_break=0.01,
        intensity=0.1,
        gamma=1.0,
        alpha=1.0,
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 0.1]]))

    assert image == pytest.approx(0.0255173, 1.0e-4)


def test__intensity_prime__effective_radius_zero__non_finite_not_zero_division():
    # A degenerate `effective_radius=0` (e.g. a non-linear search proposing an
    # unphysical value, or a wavelength-relation instance evaluating to 0) must
    # yield a non-finite value the search resamples on, not a hard
    # ``ZeroDivisionError`` — matching every other profile's array division.
    lp = ag.lp.SersicCore(
        effective_radius=0.0,
        sersic_index=4.0,
        radius_break=0.01,
        intensity=0.1,
        gamma=1.0,
        alpha=1.0,
    )

    value = lp.intensity_prime()

    assert not np.isfinite(value)

    # The full image path must not raise either.
    lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 0.1]]))


def test__image_2d_from__spherical_profile__matches_elliptical_with_zero_ellipticity():
    elliptical = ag.lp.SersicCore(
        ell_comps=(0.0, 0.0),
        effective_radius=5.0,
        sersic_index=4.0,
        radius_break=0.01,
        intensity=0.1,
        gamma=1.0,
        alpha=1.0,
    )

    spherical = ag.lp.SersicCoreSph(
        effective_radius=5.0,
        sersic_index=4.0,
        radius_break=0.01,
        intensity=0.1,
        gamma=1.0,
        alpha=1.0,
    )

    image_elliptical = elliptical.image_2d_from(grid=grid)
    image_spherical = spherical.image_2d_from(grid=grid)

    assert image_elliptical.array == pytest.approx(image_spherical.array, 1.0e-4)
