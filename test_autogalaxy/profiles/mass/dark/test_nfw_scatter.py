import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__scatter_is_nonzero():
    mp = ag.mp.NFWMCRScatterLudlowSph(
        mass_at_200=1.0e9,
        scatter_sigma=1.0,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    # We uare using the NFWTruncatedSph to check the mass gives a conosistnt kappa_s, given certain radii.

    assert mp.scale_radius == pytest.approx(0.14978, 1.0e-4)

    mp = ag.mp.NFWMCRScatterLudlowSph(
        mass_at_200=1.0e9,
        scatter_sigma=-1.0,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    # We uare using the NFWTruncatedSph to check the mass gives a conosistnt kappa_s, given certain radii.

    assert mp.scale_radius == pytest.approx(0.29886, 1.0e-4)

    nfw_ell = ag.mp.NFWMCRScatterLudlow(
        ell_comps=(0.5, 0.5),
        mass_at_200=1.0e9,
        scatter_sigma=1.0,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    # We uare using the NFWTruncated to check the mass gives a conosistnt kappa_s, given certain radii.

    assert nfw_ell.ell_comps == (0.5, 0.5)
    assert nfw_ell.scale_radius == pytest.approx(0.14978, 1.0e-4)

    nfw_ell = ag.mp.NFWMCRScatterLudlow(
        ell_comps=(0.5, 0.5),
        mass_at_200=1.0e9,
        scatter_sigma=-1.0,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    # We uare using the NFWTruncated to check the mass gives a conosistnt kappa_s, given certain radii.

    assert nfw_ell.ell_comps == (0.5, 0.5)
    assert nfw_ell.scale_radius == pytest.approx(0.29886, 1.0e-4)

    deflections_sph = mp.deflections_yx_2d_from(grid=grid)
    deflections_ell = nfw_ell.deflections_yx_2d_from(grid=grid)

    assert deflections_sph[0] != pytest.approx(deflections_ell[0], 1.0e-4)


def test__scatter_is_nonzero_cored():
    cnfw_sph = ag.mp.cNFWMCRScatterLudlowSph(
        mass_at_200=1.0e9,
        scatter_sigma=1.0,
        f_c=0.01,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    assert cnfw_sph.scale_radius == pytest.approx(0.14978, 1.0e-4)

    cnfw_sph = ag.mp.cNFWMCRScatterLudlowSph(
        mass_at_200=1.0e9,
        scatter_sigma=-1.0,
        f_c=0.01,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    assert cnfw_sph.scale_radius == pytest.approx(0.29886, 1.0e-4)

    cnfw = ag.mp.cNFWMCRScatterLudlow(
        ell_comps=(-0.1, 0.2),
        mass_at_200=1.0e9,
        scatter_sigma=1.0,
        f_c=0.01,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    assert cnfw.scale_radius == pytest.approx(0.14978, 1.0e-4)

    cnfw_sph = ag.mp.cNFWMCRScatterLudlow(
        mass_at_200=1.0e9,
        scatter_sigma=-1.0,
        f_c=0.01,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    assert cnfw_sph.scale_radius == pytest.approx(0.29886, 1.0e-4)

