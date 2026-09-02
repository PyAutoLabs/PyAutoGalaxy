import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_2d_via_analytic_from__config_1_positive_y():
    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.05263),
        intensity=1.0,
        sigma=3.0,
        mass_to_light_ratio=1.0,
    )

    deflections = mp.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections[0, 0] == pytest.approx(1.024423, 1.0e-4)
    assert deflections[0, 1] == pytest.approx(0.0, abs=1.0e-4)


def test__deflections_2d_via_analytic_from__config_1_negative_y():
    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.05263),
        intensity=1.0,
        sigma=3.0,
        mass_to_light_ratio=1.0,
    )

    deflections = mp.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[-1.0, 0.0]])
    )

    assert deflections[0, 0] == pytest.approx(-1.024423, 1.0e-4)
    assert deflections[0, 1] == pytest.approx(0.0, abs=1.0e-4)


def test__deflections_2d_via_analytic_from__config_2_positive():
    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.111111),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio=1.0,
    )

    deflections = mp.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[0.5, 0.2]])
    )

    assert deflections[0, 0] == pytest.approx(0.554062, 1.0e-4)
    assert deflections[0, 1] == pytest.approx(0.177336, 1.0e-4)


def test__deflections_2d_via_analytic_from__config_2_negative():
    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.111111),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio=1.0,
    )

    deflections = mp.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[-0.5, -0.2]])
    )

    assert deflections[0, 0] == pytest.approx(-0.554062, 1.0e-4)
    assert deflections[0, 1] == pytest.approx(-0.177336, 1.0e-4)


def test__deflections_2d_via_analytic_from__mass_to_light_2():
    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.111111),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio=2.0,
    )

    deflections = mp.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[0.5, 0.2]])
    )

    assert deflections[0, 0] == pytest.approx(1.108125, 1.0e-4)
    assert deflections[0, 1] == pytest.approx(0.35467, 1.0e-4)


def test__deflections_2d_via_analytic_from__intensity_2():
    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.111111),
        intensity=2.0,
        sigma=5.0,
        mass_to_light_ratio=1.0,
    )

    deflections = mp.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[0.5, 0.2]])
    )

    assert deflections[0, 0] == pytest.approx(1.10812, 1.0e-4)
    assert deflections[0, 1] == pytest.approx(0.35467, 1.0e-4)


def test__deflections_yx_2d_from():
    mp = ag.mp.Gaussian()

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflections_via_integral = mp.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_integral.array, 1.0e-4)


def test__convergence_2d_from__gaussian_config_1():
    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        sigma=1.0,
        mass_to_light_ratio=1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.60653, 1e-2)


def test__convergence_2d_from__gaussian_mass_to_light_2():
    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        sigma=1.0,
        mass_to_light_ratio=2.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(2.0 * 0.60653, 1e-2)


def test__convergence_2d_from__gaussian_elliptical():
    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        intensity=2.0,
        sigma=3.0,
        mass_to_light_ratio=4.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(7.88965, 1e-2)


def test__intensity_and_convergence_match_for_mass_light_ratio_1():
    lp = ag.lp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        intensity=2.0,
        sigma=3.0,
    )

    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        intensity=2.0,
        sigma=3.0,
        mass_to_light_ratio=1.0,
    )

    intensity = lp.image_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert (intensity == convergence).all()


def test__image_2d_via_radii_from__config_1():
    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=1.0, sigma=1.0
    )

    intensity = mp.image_2d_via_radii_from(grid_radii=ag.ArrayIrregular(1.0))

    assert intensity == pytest.approx(0.60653, 1e-2)


def test__image_2d_via_radii_from__intensity_2():
    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=2.0, sigma=1.0
    )

    intensity = mp.image_2d_via_radii_from(grid_radii=ag.ArrayIrregular(1.0))

    assert intensity == pytest.approx(2.0 * 0.60653, 1e-2)


def test__image_2d_via_radii_from__sigma_2():
    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=1.0, sigma=2.0
    )
    intensity = mp.image_2d_via_radii_from(grid_radii=ag.ArrayIrregular(1.0))

    assert intensity == pytest.approx(0.882496, 1e-2)


def test__image_2d_via_radii_from__sigma_2_radii_3():
    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=1.0, sigma=2.0
    )

    intensity = mp.image_2d_via_radii_from(grid_radii=ag.ArrayIrregular(3.0))

    assert intensity == pytest.approx(0.32465, 1e-2)


def test__wofz__regions_1_2_3():
    """The hand-rolled rational approximation is the JAX-path routine (numpy goes to
    `scipy.special.wofz`), so these region pins target it directly -- the suite never
    imports jax, and `MGEDecomposer.wofz` would dispatch away from the branches."""
    from scipy.special import wofz

    from autogalaxy.profiles.mass.abstract.mge import _wofz_rational as mp_wofz

    wofz_approx_reg_1 = mp_wofz(20.0 + 1j * 0.001)
    wofz_approx_reg_2 = mp_wofz(2.0 + 1j * 0.001)
    wofz_approx_reg_3 = mp_wofz(1.0 + 1j * 0.001)

    assert wofz_approx_reg_1 == pytest.approx(wofz(20.0 + 1j * 0.001), 1e-4)
    assert wofz_approx_reg_2 == pytest.approx(wofz(2.0 + 1j * 0.001), 1e-4)
    assert wofz_approx_reg_3 == pytest.approx(wofz(1.0 + 1j * 0.001), 1e-4)


def test__wofz__regions_4_5_6():
    """See `test__wofz__regions_1_2_3`: these pin the JAX-path rational routine."""
    from scipy.special import wofz

    from autogalaxy.profiles.mass.abstract.mge import _wofz_rational as mp_wofz

    wofz_approx_reg_1 = mp_wofz(7.0 + 1j * 0.1)
    wofz_approx_reg_2 = mp_wofz(7.0 + 1j * 1e-11)
    wofz_approx_reg_3 = mp_wofz(2.0 + 1j * 1.0)

    assert wofz_approx_reg_1 == pytest.approx(wofz(7.0 + 1j * 0.1), 1e-4)
    assert wofz_approx_reg_2 == pytest.approx(wofz(7.0 + 1j * 1e-11), 1e-4)
    assert wofz_approx_reg_3 == pytest.approx(wofz(2.0 + 1j * 1.0), 1e-4)


def test__deflections_yx_2d_from__spherical_case__is_radial_and_matches_elliptical_limit(
    monkeypatch,
):
    """`Gaussian(ell_comps=(0, 0))` takes the exact radial closed form on the numpy path.

    Two checks: the deflection of a circular profile is purely radial (no cross-axis
    component on the axes, down to the rounding of the grid rotate-back, which is what
    the ~5e-18 tolerance below allows for), and it matches the elliptical Faddeeva form
    in its q -> 1 limit. That limit is only reachable with the 0.9999 clamp lifted --
    the clamp *is* the ~6e-5 bias this branch removes -- so it is monkeypatched away for
    the comparison only.
    """
    gaussian_sph = ag.mp.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=0.1, sigma=1.0
    )

    grid_axes = ag.Grid2DIrregular([[1.0, 0.0], [0.0, 2.5], [-1.7, 0.0], [0.0, 0.0]])

    deflections = gaussian_sph.deflections_yx_2d_from(grid=grid_axes)

    assert deflections[0, 1] == pytest.approx(0.0, abs=1.0e-17)
    assert deflections[1, 0] == pytest.approx(0.0, abs=1.0e-17)
    assert deflections[2, 1] == pytest.approx(0.0, abs=1.0e-17)
    assert deflections[3, 0] == 0.0 and deflections[3, 1] == 0.0

    monkeypatch.setattr(
        ag.mp.Gaussian,
        "axis_ratio",
        lambda self, xp=np: ag.convert.axis_ratio_from(ell_comps=self.ell_comps, xp=xp),
    )

    ell_comps = ag.convert.ell_comps_from(axis_ratio=0.999999, angle=0.0)

    gaussian_ell = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(float(ell_comps[0]), float(ell_comps[1])),
        intensity=0.1,
        sigma=1.0,
    )

    grid = ag.Grid2DIrregular([[0.5, 1.0], [-2.0, 0.3], [1.5, -1.5], [0.05, 0.05]])

    deflections_sph = np.asarray(gaussian_sph.deflections_yx_2d_from(grid=grid).array)
    deflections_ell = np.asarray(gaussian_ell.deflections_yx_2d_from(grid=grid).array)

    assert deflections_sph == pytest.approx(deflections_ell, rel=1.0e-5)
