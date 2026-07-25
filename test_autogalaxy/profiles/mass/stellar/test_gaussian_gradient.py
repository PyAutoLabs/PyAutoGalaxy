import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__init_constructor__gradient_0_scales_values_correctly():
    mp = ag.mp.GaussianGradient(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.05),
        intensity=1.0,
        sigma=3.0,
        mass_to_light_ratio_base=1.0,
        mass_to_light_gradient=0.0,
        mass_to_light_radius=1.0,
    )

    assert mp.centre == (0.0, 0.0)
    assert mp.ell_comps == (0.0, 0.05)
    assert mp.intensity == 1.0
    assert mp.sigma == 3.0
    assert mp.mass_to_light_ratio_base == 1.0
    assert mp.mass_to_light_gradient == 0.0
    assert mp.mass_to_light_radius == 1.0
    assert mp.mass_to_light_ratio == 1.0


def test__init_constructor__gradient_1_scales_values_correctly():
    mp = ag.mp.GaussianGradient(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.05),
        intensity=1.0,
        sigma=3.0,
        mass_to_light_ratio_base=1.0,
        mass_to_light_gradient=1.0,
        mass_to_light_radius=5.0,
    )

    assert mp.centre == (0.0, 0.0)
    assert mp.ell_comps == (0.0, 0.05)
    assert mp.intensity == 1.0
    assert mp.sigma == 3.0
    assert mp.mass_to_light_ratio_base == 1.0
    assert mp.mass_to_light_gradient == 1.0
    assert mp.mass_to_light_radius == 5.0
    assert mp.mass_to_light_ratio == 0.602
