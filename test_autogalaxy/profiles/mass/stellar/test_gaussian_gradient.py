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
    # (sigma / mass_to_light_radius) ** -mass_to_light_gradient = (3 / 5) ** -1
    assert mp.mass_to_light_ratio == pytest.approx(5.0 / 3.0, 1.0e-8)


def test__mass_to_light_ratio__equals_base_at_reference_radius():
    mp = ag.mp.GaussianGradient(
        sigma=0.7,
        mass_to_light_ratio_base=2.0,
        mass_to_light_gradient=1.3,
        mass_to_light_radius=0.7,
    )

    assert mp.mass_to_light_ratio == pytest.approx(2.0, 1.0e-12)


def test__mass_to_light_ratio__same_sign_convention_as_sersic_gradient():
    # The same positive gradient makes both profiles' M/L fall outward, with the same power law.
    gaussian_ratios = [
        ag.mp.GaussianGradient(
            sigma=sigma,
            mass_to_light_ratio_base=1.0,
            mass_to_light_gradient=1.0,
            mass_to_light_radius=1.0,
        ).mass_to_light_ratio
        for sigma in (0.5, 2.0)
    ]

    sersic_kwargs = dict(
        intensity=1.0, effective_radius=1.0, sersic_index=2.0, mass_to_light_ratio=1.0
    )
    grid = ag.Grid2DIrregular([[0.0, 0.5], [0.0, 2.0]])
    sersic_ratios = np.asarray(
        ag.mp.SersicGradient(
            **sersic_kwargs, mass_to_light_gradient=1.0
        ).convergence_2d_from(grid=grid)
    ) / np.asarray(ag.mp.Sersic(**sersic_kwargs).convergence_2d_from(grid=grid))

    assert gaussian_ratios == pytest.approx([2.0, 0.5], 1.0e-12)
    assert sersic_ratios == pytest.approx(gaussian_ratios, 1.0e-6)
