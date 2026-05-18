from __future__ import division, print_function

import numpy as np
import pytest

import autogalaxy as ag

from autogalaxy.profiles.light.linear import LightProfileLinear


grid = ag.Grid2DIrregular([[1.0, 0.0], [0.5, 0.5], [2.0, 1.0], [-1.0, 0.3]])


def test__is_light_profile_linear_and_gaussian_multipole():
    mp = ag.lp_linear.GaussianMultipole()

    assert isinstance(mp, LightProfileLinear)
    assert isinstance(mp, ag.lp.GaussianMultipole)


def test__intensity_is_one__not_a_constructor_parameter():
    mp = ag.lp_linear.GaussianMultipole(
        centre=(0.0, 0.0),
        ell_comps=(0.1, 0.0),
        sigma=0.8,
    )

    assert mp.intensity == 1.0
    with pytest.raises(TypeError):
        ag.lp_linear.GaussianMultipole(intensity=2.0)


def test__multipole_components_propagate():
    mp = ag.lp_linear.GaussianMultipole(
        multipole_3_comps=(0.03, 0.04),
        multipole_4_comps=(0.05, -0.02),
    )

    assert mp.multipole_3_comps == (0.03, 0.04)
    assert mp.multipole_4_comps == (0.05, -0.02)


def test__zero_perturbation__matches_linear_gaussian_exactly():
    kwargs = dict(centre=(0.0, 0.0), ell_comps=(0.1, 0.2), sigma=0.7)
    base = ag.lp_linear.Gaussian(**kwargs)
    mp = ag.lp_linear.GaussianMultipole(**kwargs)

    image_base = np.asarray(base.image_2d_from(grid=grid))
    image_mp = np.asarray(mp.image_2d_from(grid=grid))

    assert image_base == pytest.approx(image_mp, rel=1e-12, abs=1e-12)


def test__image_matches_standard_gaussian_multipole_at_intensity_one():
    kwargs = dict(
        centre=(0.0, 0.0),
        ell_comps=(0.1, 0.0),
        sigma=0.8,
        multipole_4_comps=(0.05, 0.0),
    )
    standard = ag.lp.GaussianMultipole(intensity=1.0, **kwargs)
    linear = ag.lp_linear.GaussianMultipole(**kwargs)

    image_standard = np.asarray(standard.image_2d_from(grid=grid))
    image_linear = np.asarray(linear.image_2d_from(grid=grid))

    assert image_standard == pytest.approx(image_linear, rel=1e-12, abs=1e-12)
