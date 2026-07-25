import numpy as np
import pytest

import autogalaxy as ag


def test__x2_smbhs__centres_correct_based_on_angle__angle_0001():
    mp = ag.mp.SMBHBinary(
        centre=(0.0, 0.0),
        separation=1.0,
        angle_binary=0.0001,
    )

    assert mp.smbh_0.centre == pytest.approx((8.726646259967217e-07, 0.5), 1e-2)
    assert mp.smbh_1.centre == pytest.approx((-8.726646259967217e-07, -0.5), 1e-2)


def test__x2_smbhs__centres_correct_based_on_angle__angle_90():
    mp = ag.mp.SMBHBinary(
        centre=(0.0, 0.0),
        separation=1.0,
        angle_binary=90.0,
        mass=3.0,
        mass_ratio=2.0,
        redshift_object=0.169,
        redshift_source=0.451,
    )

    assert mp.smbh_0.centre == pytest.approx((0.5, 0.0), 1e-2)
    assert mp.smbh_1.centre == pytest.approx((-0.5, 0.0), 1e-2)


def test__x2_smbhs__centres_correct_based_on_angle__angle_180():
    mp = ag.mp.SMBHBinary(
        centre=(0.0, 0.0),
        separation=1.0,
        angle_binary=180.0,
    )

    assert mp.smbh_0.centre == pytest.approx((0.0, -0.5), 1e-2)
    assert mp.smbh_1.centre == pytest.approx((0.0, 0.5), 1e-2)


def test__x2_smbhs__centres_correct_based_on_angle__angle_270():
    mp = ag.mp.SMBHBinary(
        centre=(0.0, 0.0),
        separation=1.0,
        angle_binary=270.0,
    )

    assert mp.smbh_0.centre == pytest.approx((-0.5, 0.0), 1e-2)
    assert mp.smbh_1.centre == pytest.approx((0.5, 0.0), 1e-2)


def test__x2_smbhs__centres_correct_based_on_angle__angle_360():
    mp = ag.mp.SMBHBinary(
        centre=(0.0, 0.0),
        separation=1.0,
        angle_binary=359.999,
    )

    assert mp.smbh_0.centre == pytest.approx((-8.726646259517295e-06, 0.5), 1e-2)
    assert mp.smbh_1.centre == pytest.approx((8.726646259456063e-06, -0.5), 1e-2)


def test__x2_smbhs__separation__separation_2():
    mp = ag.mp.SMBHBinary(
        centre=(0.0, 0.0),
        separation=2.0,
        angle_binary=0.000,
    )

    assert mp.smbh_0.centre == pytest.approx((0.0, 1.0), 1e-2)
    assert mp.smbh_1.centre == pytest.approx((0.0, -1.0), 1e-2)


def test__x2_smbhs__centres_shifted_based_on_centre__positive_centre():
    mp = ag.mp.SMBHBinary(
        centre=(3.0, 1.0),
        separation=1.0,
        angle_binary=0.0001,
    )

    assert mp.smbh_0.centre == pytest.approx((3.0, 1.5), 1e-2)
    assert mp.smbh_1.centre == pytest.approx((3.0, 0.5), 1e-2)


def test__x2_smbhs__centres_shifted_based_on_centre__negative_centre():
    mp = ag.mp.SMBHBinary(
        centre=(-3.0, -1.0),
        separation=1.0,
        angle_binary=0.0001,
    )

    assert mp.smbh_0.centre == pytest.approx((-3.0, -0.5), 1e-2)
    assert mp.smbh_1.centre == pytest.approx((-3.0, -1.5), 1e-2)


def test__x2_smbhs__masses_corrected_based_on_mass_and_ratio__ratio_2():
    mp = ag.mp.SMBHBinary(
        mass=3.0,
        mass_ratio=2.0,
    )

    assert mp.smbh_0.mass == pytest.approx(2.0, 1e-2)
    assert mp.smbh_1.mass == pytest.approx(1.0, 1e-2)


def test__x2_smbhs__masses_corrected_based_on_mass_and_ratio__ratio_05():
    mp = ag.mp.SMBHBinary(
        mass=3.0,
        mass_ratio=0.5,
    )

    assert mp.smbh_0.mass == pytest.approx(1.0, 1e-2)
    assert mp.smbh_1.mass == pytest.approx(2.0, 1e-2)


def test__convergence_2d_from():
    mp = ag.mp.SMBHBinary(
        centre=(0.0, 0.0),
        separation=2.0,
        angle_binary=0.000,
    )

    smbh_0 = ag.mp.SMBH(centre=mp.smbh_0.centre, mass=mp.smbh_0.mass)
    smbh_1 = ag.mp.SMBH(centre=mp.smbh_1.centre, mass=mp.smbh_1.mass)

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 1.0]]))

    convergence_0 = smbh_0.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 1.0]]))
    convergence_1 = smbh_1.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 1.0]]))

    assert convergence == pytest.approx(convergence_0 + convergence_1, 1e-2)


def test__potential_2d_from():
    mp = ag.mp.SMBHBinary(
        centre=(0.0, 0.0),
        separation=2.0,
        angle_binary=0.000,
    )

    smbh_0 = ag.mp.SMBH(centre=mp.smbh_0.centre, mass=mp.smbh_0.mass)
    smbh_1 = ag.mp.SMBH(centre=mp.smbh_1.centre, mass=mp.smbh_1.mass)

    potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[1.0, 1.0]]))

    potential_0 = smbh_0.potential_2d_from(grid=ag.Grid2DIrregular([[1.0, 1.0]]))
    potential_1 = smbh_1.potential_2d_from(grid=ag.Grid2DIrregular([[1.0, 1.0]]))

    assert potential == pytest.approx(potential_0 + potential_1, 1e-2)


def test__deflections_yx_2d_from():
    mp = ag.mp.SMBHBinary(
        centre=(0.0, 0.0),
        separation=2.0,
        angle_binary=0.000,
    )

    smbh_0 = ag.mp.SMBH(centre=mp.smbh_0.centre, mass=mp.smbh_0.mass)
    smbh_1 = ag.mp.SMBH(centre=mp.smbh_1.centre, mass=mp.smbh_1.mass)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 1.0]]))

    deflections_0 = smbh_0.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 1.0]]))
    deflections_1 = smbh_1.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 1.0]]))

    assert deflections[0, 0] == pytest.approx(
        deflections_0.array[0, 0] + deflections_1.array[0, 0], 1e-2
    )
