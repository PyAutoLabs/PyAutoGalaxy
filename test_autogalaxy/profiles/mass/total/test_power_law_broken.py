import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from__power_law_broken_sph__single_and_multi_grid():
    mp = ag.mp.PowerLawBrokenSph(
        centre=(0, 0),
        einstein_radius=1.0,
        inner_slope=1.5,
        outer_slope=2.5,
        break_radius=0.1,
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    assert deflections[0, 0] == pytest.approx(0.404076, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.808152, 1e-3)

    deflections = mp.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.5, 1.0], [0.5, 1.0]])
    )

    assert deflections[0, 0] == pytest.approx(0.404076, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.808152, 1e-3)
    assert deflections[1, 0] == pytest.approx(0.404076, 1e-3)
    assert deflections[1, 1] == pytest.approx(0.808152, 1e-3)


def test__deflections_yx_2d_from__power_law_broken_ell_config_1():
    mp = ag.mp.PowerLawBroken(
        centre=(0, 0),
        ell_comps=(0.096225, 0.055555),
        einstein_radius=1.0,
        inner_slope=1.5,
        outer_slope=2.5,
        break_radius=0.1,
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    assert deflections[0, 0] == pytest.approx(0.40392, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.811619, 1e-3)


def test__deflections_yx_2d_from__power_law_broken_ell_config_2():
    mp = ag.mp.PowerLawBroken(
        centre=(0, 0),
        ell_comps=(-0.07142, -0.085116),
        einstein_radius=1.0,
        inner_slope=1.5,
        outer_slope=2.5,
        break_radius=0.1,
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    assert deflections[0, 0] == pytest.approx(0.4005338, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.8067221, 1e-3)


def test__deflections_yx_2d_from__power_law_broken_ell_config_3():
    mp = ag.mp.PowerLawBroken(
        centre=(0, 0),
        ell_comps=(0.109423, 0.019294),
        einstein_radius=1.0,
        inner_slope=1.5,
        outer_slope=2.5,
        break_radius=0.1,
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    assert deflections[0, 0] == pytest.approx(0.399651, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.813372, 1e-3)


def test__deflections_yx_2d_from__power_law_broken_ell_config_4():
    mp = ag.mp.PowerLawBroken(
        centre=(0, 0),
        ell_comps=(-0.216506, -0.125),
        einstein_radius=1.0,
        inner_slope=1.5,
        outer_slope=2.5,
        break_radius=0.1,
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    assert deflections[0, 0] == pytest.approx(0.402629, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.798795, 1e-3)


def test__convergence_2d_from__power_law_broken_sph_single_grid():
    mp = ag.mp.PowerLawBrokenSph(
        centre=(0, 0),
        einstein_radius=1.0,
        inner_slope=1.5,
        outer_slope=2.5,
        break_radius=0.1,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    assert convergence == pytest.approx(0.0355237, 1e-4)


def test__convergence_2d_from__power_law_broken_sph_two_grids():
    mp = ag.mp.PowerLawBrokenSph(
        centre=(0, 0),
        einstein_radius=1.0,
        inner_slope=1.5,
        outer_slope=2.5,
        break_radius=0.1,
    )

    convergence = mp.convergence_2d_from(
        grid=ag.Grid2DIrregular([[0.5, 1.0], [0.5, 1.0]])
    )

    assert convergence == pytest.approx([0.0355237, 0.0355237], 1e-4)


def test__convergence_2d_from__power_law_broken_ell_config_1():
    mp = ag.mp.PowerLawBroken(
        centre=(0, 0),
        ell_comps=(0.096225, 0.055555),
        einstein_radius=1.0,
        inner_slope=1.5,
        outer_slope=2.5,
        break_radius=0.1,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    assert convergence == pytest.approx(0.05006035, 1e-4)


def test__convergence_2d_from__power_law_broken_ell_config_2():
    mp = ag.mp.PowerLawBroken(
        centre=(0, 0),
        ell_comps=(-0.113433, 0.135184),
        einstein_radius=1.0,
        inner_slope=1.8,
        outer_slope=2.2,
        break_radius=0.1,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    assert convergence == pytest.approx(0.034768, 1e-4)


def test__convergence_2d_from__power_law_broken_ell_config_3():
    mp = ag.mp.PowerLawBroken(
        centre=(0, 0),
        ell_comps=(0.113433, -0.135184),
        einstein_radius=1.0,
        inner_slope=1.8,
        outer_slope=2.2,
        break_radius=0.1,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    assert convergence == pytest.approx(0.03622852, 1e-4)


def test__convergence_2d_from__power_law_broken_ell_config_4():
    mp = ag.mp.PowerLawBroken(
        centre=(0, 0),
        ell_comps=(-0.173789, -0.030643),
        einstein_radius=1.0,
        inner_slope=1.8,
        outer_slope=2.2,
        break_radius=0.1,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    assert convergence == pytest.approx(0.026469, 1e-4)


def test__deflections_yx_2d_from__compare_to_power_law__slope_2():
    mp = ag.mp.PowerLawBrokenSph(
        centre=(0, 0),
        einstein_radius=2.0,
        inner_slope=1.999,
        outer_slope=2.0001,
        break_radius=0.0001,
    )
    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    # Use of ratio avoids normalization definition difference effects

    broken_yx_ratio = deflections[0, 0] / deflections[0, 1]

    power_law = ag.mp.PowerLawSph(centre=(0, 0), einstein_radius=2.0, slope=2.0)
    deflections = power_law.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.5, 1.0]])
    )

    power_law_yx_ratio = deflections[0, 0] / deflections[0, 1]

    assert broken_yx_ratio == pytest.approx(power_law_yx_ratio, 1.0e-4)


def test__deflections_yx_2d_from__compare_to_power_law__slope_24():
    mp = ag.mp.PowerLawBrokenSph(
        centre=(0, 0),
        einstein_radius=2.0,
        inner_slope=2.399,
        outer_slope=2.4001,
        break_radius=0.0001,
    )
    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    # Use of ratio avoids normalization difference effects

    broken_yx_ratio = deflections[0, 0] / deflections[0, 1]

    power_law = ag.mp.PowerLawSph(centre=(0, 0), einstein_radius=2.0, slope=2.4)
    deflections = power_law.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.5, 1.0]])
    )

    power_law_yx_ratio = deflections[0, 0] / deflections[0, 1]

    assert broken_yx_ratio == pytest.approx(power_law_yx_ratio, 1.0e-4)


def test__convergence_func__matches_private_helper():
    """Regression: PowerLawBroken must override the abstract `convergence_func` so
    `MGEDecomposer.decompose_convergence_via_mge` (reached by `potential_2d_from`,
    which uses the `three_D=False` MGE path) doesn't fall through to the abstract
    NotImplementedError. The shim delegates to the `_convergence` radial helper that
    `convergence_2d_from` already uses."""

    mp = ag.mp.PowerLawBroken(
        centre=(0.0, 0.0),
        ell_comps=(0.1, 0.05),
        einstein_radius=1.0,
        inner_slope=1.5,
        outer_slope=2.5,
        break_radius=0.4,
    )

    # Scalar radius, inside and outside the break radius.
    assert mp.convergence_func(0.2) == pytest.approx(mp._convergence(0.2), 1e-12)
    assert mp.convergence_func(1.5) == pytest.approx(mp._convergence(1.5), 1e-12)

    # 1-D array of radii spanning the break: shape preserved, element-wise equality.
    radii = np.array([0.1, 0.4, 0.5, 1.0, 2.5])
    expected = mp._convergence(radii)
    actual = mp.convergence_func(radii)
    assert actual.shape == radii.shape
    assert actual == pytest.approx(expected, 1e-12)

    # PowerLawBrokenSph inherits the override from PowerLawBroken.
    sph = ag.mp.PowerLawBrokenSph(
        centre=(0.0, 0.0),
        einstein_radius=1.0,
        inner_slope=1.5,
        outer_slope=2.5,
        break_radius=0.4,
    )
    assert sph.convergence_func(0.2) == pytest.approx(sph._convergence(0.2), 1e-12)


def test__potential_2d_from__raises_not_implemented():
    """The broken power law's convergence is piecewise (a slope kink at break_radius), which
    is incompatible with the MGE potential decomposition (it integrates the convergence along
    a complex contour and requires an analytic profile, producing values wrong by orders of
    magnitude otherwise). `potential_2d_from` therefore raises a clear NotImplementedError
    rather than returning numerically-invalid values. Convergence/deflections/convergence_func
    remain available."""

    mp = ag.mp.PowerLawBroken(
        centre=(0.0, 0.0),
        ell_comps=(0.1, 0.05),
        einstein_radius=1.0,
        inner_slope=1.5,
        outer_slope=2.5,
        break_radius=0.4,
    )

    grid = ag.Grid2D.uniform(shape_native=(5, 5), pixel_scales=0.2)

    with pytest.raises(NotImplementedError):
        mp.potential_2d_from(grid=grid)
