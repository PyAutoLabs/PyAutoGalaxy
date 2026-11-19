import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__scatter_is_nonzero__scatter_positive():
    mp = ag.mp.NFWTruncatedMCRScatterLudlowSph(
        centre=(1.0, 2.0),
        mass_at_200=1.0e9,
        scatter_sigma=1.0,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    assert mp.scale_radius == pytest.approx(0.14978, 1.0e-3)
    assert mp.truncation_radius == pytest.approx(4.897742, 1.0e-3)


def test__scatter_is_nonzero__scatter_negative():
    mp = ag.mp.NFWTruncatedMCRScatterLudlowSph(
        centre=(1.0, 2.0),
        mass_at_200=1.0e9,
        scatter_sigma=-1.0,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    assert mp.scale_radius == pytest.approx(0.29886, 1.0e-3)
    assert mp.truncation_radius == pytest.approx(4.897742, 1.0e-3)


@pytest.mark.parametrize("scatter_sigma", [-1.0, 0.0, 1.0])
def test__truncation_radius_is_twice_r200__tau_is_twice_concentration(scatter_sigma):
    from autogalaxy.profiles.mass.dark.mcr_util import ludlow16_cosmology

    mp = ag.mp.NFWTruncatedMCRScatterLudlowSph(
        mass_at_200=1.0e9,
        scatter_sigma=scatter_sigma,
        redshift_object=0.5,
        redshift_source=1.0,
    )

    concentration = ludlow16_cosmology(
        mass_at_200=1.0e9, redshift_object=0.5, redshift_source=1.0
    )[0]
    concentration = 10.0 ** (np.log10(concentration) + scatter_sigma * 0.15)

    assert mp.tau == pytest.approx(2.0 * concentration, 1.0e-6)
