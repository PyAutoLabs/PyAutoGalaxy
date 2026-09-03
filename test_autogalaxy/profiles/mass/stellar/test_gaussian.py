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


def test__wofz__weideman_matches_scipy():
    """The Weideman (1994) series is the JAX-path routine (numpy goes to
    `scipy.special.wofz`), so these pins target it directly with `xp=np` -- the suite
    never imports jax, and `MGEDecomposer.wofz` would dispatch away from it.

    The sample points are the six the old piecewise routine's region pins used (they
    straddle its `r2 = 2.5 / 30 / 62` and `y2 = 0.072 / 1e-13` boundaries, which the
    seam-free series does not have), plus |z| = 1e3 and z = 0. The reference values are
    `scipy.special.wofz`, not this routine's own output.
    """
    from scipy.special import wofz

    from autogalaxy.profiles.mass.abstract.mge import _wofz_weideman

    z_list = [
        20.0 + 1j * 0.001,
        7.0 + 1j * 0.1,
        7.0 + 1j * 1e-11,
        2.0 + 1j * 0.001,
        2.0 + 1j * 1.0,
        1.0 + 1j * 0.001,
        1.0e3 + 1j * 1.0e3,
        1.0e3 + 1j * 0.0,
        0.0 + 1j * 0.0,
    ]

    for z in z_list:
        assert _wofz_weideman(z, xp=np) == pytest.approx(wofz(z), rel=1.0e-10)


def test__is_circular__only_static_scalars_are_circular():
    """`_is_circular` is a branch predicate on the JAX path too, so a traced or array
    ellipticity must fall through to the elliptical path (return `False`) rather than
    raise. Detection is by the module the type is defined in, so no jax import here."""
    from autogalaxy.profiles.mass.abstract.mge import _is_circular

    class FakeTracer:
        pass

    FakeTracer.__module__ = "jax._src.core"

    assert _is_circular((0.0, 0.0)) is True
    assert _is_circular((0.1, 0.0)) is False
    assert _is_circular((np.zeros(1), np.zeros(1))) is False
    assert _is_circular((FakeTracer(), FakeTracer())) is False


def test__spherical_mge_deflections_from__numpy_path_is_unchanged():
    """The exact radial branch became `xp`-generic (the `np.divide(out=, where=)` guard
    became a `where`-safe denominator, so JAX can take the branch too). The numpy result
    must be bit-identical to what it was before: the literals below are the pre-change
    output, reproduced here to 17 significant digits."""
    from autogalaxy.profiles.mass.abstract.mge import _spherical_mge_deflections_from

    grid = ag.Grid2DIrregular(
        [[1.0, 0.0], [0.0, 2.5], [-1.7, 0.3], [0.0, 0.0], [0.05, -0.05]]
    )

    deflections = _spherical_mge_deflections_from(
        grid=grid,
        amps=np.array([0.3, 1.2, 0.05]),
        sigmas=np.array([0.5, 1.0, 4.0]),
        xp=np,
    )

    assert np.array_equal(
        np.asarray(deflections),
        np.array(
            [
                [1.1232529490420373, 0.0],
                [0.0, 1.0913706801871932],
                [-1.2270755548552907, 0.21654274497446305],
                [0.0, 0.0],
                [0.07735011653487707, -0.07735011653487707],
            ]
        ),
    )


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
