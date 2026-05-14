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
    assert mp.truncation_radius == pytest.approx(33.7134116, 1.0e-3)


def test__scatter_is_nonzero__scatter_negative():
    mp = ag.mp.NFWTruncatedMCRScatterLudlowSph(
        centre=(1.0, 2.0),
        mass_at_200=1.0e9,
        scatter_sigma=-1.0,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    assert mp.scale_radius == pytest.approx(0.29886, 1.0e-3)
    assert mp.truncation_radius == pytest.approx(33.7134116, 1.0e-3)
