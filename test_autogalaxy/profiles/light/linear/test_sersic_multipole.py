from __future__ import division, print_function

import numpy as np
import pytest

import autogalaxy as ag

from autogalaxy.profiles.light.linear import LightProfileLinear


grid = ag.Grid2DIrregular([[1.0, 0.0], [0.5, 0.5], [2.0, 1.0], [-1.0, 0.3]])


def test__is_light_profile_linear_and_sersic_multipole():
    mp = ag.lp_linear.SersicMultipole()

    assert isinstance(mp, LightProfileLinear)
    assert isinstance(mp, ag.lp.SersicMultipole)


def test__intensity_is_one__not_a_constructor_parameter():
    mp = ag.lp_linear.SersicMultipole(
        centre=(0.0, 0.0),
        ell_comps=(0.1, 0.0),
        effective_radius=0.6,
        sersic_index=4.0,
    )

    assert mp.intensity == 1.0
    with pytest.raises(TypeError):
        ag.lp_linear.SersicMultipole(intensity=2.0)


def test__multipole_components_propagate():
    mp = ag.lp_linear.SersicMultipole(
        multipole_3_comps=(0.03, 0.04),
        multipole_4_comps=(0.05, -0.02),
    )

    assert mp.multipole_3_comps == (0.03, 0.04)
    assert mp.multipole_4_comps == (0.05, -0.02)


def test__zero_perturbation__matches_linear_sersic_exactly():
    kwargs = dict(
        centre=(0.0, 0.0),
        ell_comps=(0.1, 0.2),
        effective_radius=1.3,
        sersic_index=3.0,
    )
    base = ag.lp_linear.Sersic(**kwargs)
    mp = ag.lp_linear.SersicMultipole(**kwargs)

    image_base = np.asarray(base.image_2d_from(grid=grid))
    image_mp = np.asarray(mp.image_2d_from(grid=grid))

    assert image_base == pytest.approx(image_mp, rel=1e-12, abs=1e-12)


def test__image_matches_standard_sersic_multipole_at_intensity_one():
    kwargs = dict(
        centre=(0.0, 0.0),
        ell_comps=(0.1, 0.0),
        effective_radius=0.6,
        sersic_index=4.0,
        multipole_4_comps=(0.05, 0.0),
    )
    standard = ag.lp.SersicMultipole(intensity=1.0, **kwargs)
    linear = ag.lp_linear.SersicMultipole(**kwargs)

    image_standard = np.asarray(standard.image_2d_from(grid=grid))
    image_linear = np.asarray(linear.image_2d_from(grid=grid))

    assert image_standard == pytest.approx(image_linear, rel=1e-12, abs=1e-12)
