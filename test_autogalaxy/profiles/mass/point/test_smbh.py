import pytest

import autogalaxy as ag


def test__mass_to_einstein_radius__init():
    mp = ag.mp.SMBH(
        centre=(0.0, 0.0), mass=0.513e10, redshift_object=0.169, redshift_source=0.451
    )

    assert mp.einstein_radius == pytest.approx(0.2014, 1e-3)
