import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from__gnfw():
    mp = ag.mp.gNFW()

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflections_via_integral = mp.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_integral.array, 1.0e-4)


def test__deflections_yx_2d_from__gnfw_sph():
    mp = ag.mp.gNFWSph()

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflections_via_integral = mp.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_integral.array, 1.0e-4)


def test__deflections_yx_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.gNFW(
        centre=(0.1, 0.2),
        ell_comps=(0.0, 0.0),
        kappa_s=2.0,
        inner_slope=1.5,
        scale_radius=3.0,
    )
    spherical = ag.mp.gNFWSph(
        centre=(0.1, 0.2), kappa_s=2.0, inner_slope=1.5, scale_radius=3.0
    )

    assert elliptical.deflections_yx_2d_from(grid) == pytest.approx(
        spherical.deflections_yx_2d_from(grid).array, 1e-4
    )


def test__convergence_2d_from__gnfw_sph():
    mp = ag.mp.gNFWSph(
        centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5, scale_radius=1.0
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[2.0, 0.0]]))

    assert convergence == pytest.approx(0.30840, 1e-2)


def test__potential_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.gNFW(
        centre=(0.1, 0.2),
        ell_comps=(0.0, 0.0),
        kappa_s=2.0,
        inner_slope=1.5,
        scale_radius=3.0,
    )
    spherical = ag.mp.gNFWSph(
        centre=(0.1, 0.2), kappa_s=2.0, inner_slope=1.5, scale_radius=3.0
    )

    assert elliptical.potential_2d_from(grid) == pytest.approx(
        spherical.potential_2d_from(grid).array, 1e-4
    )


def test__compare_to_nfw():
    nfw = ag.mp.NFW(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.111111),
        kappa_s=1.0,
        scale_radius=20.0,
    )
    gnfw = ag.mp.gNFW(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.111111),
        kappa_s=1.0,
        inner_slope=1.0,
        scale_radius=20.0,
    )

    assert nfw.deflections_yx_2d_from(grid) == pytest.approx(
        gnfw.deflections_yx_2d_from(grid).array, 1e-3
    )
    assert nfw.deflections_yx_2d_from(grid) == pytest.approx(
        gnfw.deflections_yx_2d_from(grid).array, 1e-3
    )


def _convergence_per_point_quad_reference(inner_slope, kappa_s, eta):
    """
    Tight per-radius ``quad`` reference for the gNFW convergence, written in the cancellation-free
    form 1 - sqrt(1 - y^2) = y^2 / (1 + sqrt(1 - y^2)), with breakpoints resolving the peak of
    width ~eta at small eta.
    """
    import warnings
    from scipy.integrate import quad

    def integrand(y, eta_value):
        return (
            (y + eta_value) ** (inner_slope - 4)
            * y**2
            / (1.0 + np.sqrt(1.0 - y**2))
        )

    convergence = np.zeros(eta.shape)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for index, eta_value in enumerate(eta):
            points = [p for p in eta_value * 10.0 ** np.arange(4) if p < 1.0]
            integral = quad(
                integrand,
                0.0,
                1.0,
                args=(eta_value,),
                epsrel=1.0e-11,
                epsabs=0.0,
                limit=1000,
                points=points or None,
            )[0]
            convergence[index] = (
                2.0
                * kappa_s
                * eta_value ** (1 - inner_slope)
                * ((1 + eta_value) ** (inner_slope - 3) + (3 - inner_slope) * integral)
            )

    return convergence


def _convergence_func_per_point_loop(mp, grid_radius):
    """
    The original per-radius ``quad`` loop of ``gNFW.convergence_func``.
    """
    from scipy.integrate import quad

    def integral_y(y, eta):
        return (y + eta) ** (mp.inner_slope - 4) * (1 - np.sqrt(1 - y**2))

    grid_radius = np.array((1.0 / mp.scale_radius) * grid_radius.array)

    for index in range(grid_radius.shape[0]):
        integral_y_value = quad(
            integral_y, a=0.0, b=1.0, args=grid_radius[index], epsrel=ag.mp.gNFW.epsrel
        )[0]

        grid_radius[index] = (
            2.0
            * mp.kappa_s
            * (grid_radius[index] ** (1 - mp.inner_slope))
            * (
                (1 + grid_radius[index]) ** (mp.inner_slope - 3)
                + ((3 - mp.inner_slope) * integral_y_value)
            )
        )

    return grid_radius


@pytest.mark.parametrize("inner_slope", [0.0, 0.5, 1.0, 1.5, 1.99, 2.5])
def test__convergence_func__tabulated__matches_per_point_quad(inner_slope):
    mp = ag.mp.gNFWSph(kappa_s=0.3, inner_slope=inner_slope, scale_radius=2.0)

    eta = np.geomspace(1.0e-5, 1.0e3, 400)
    eta = np.concatenate([eta, np.random.default_rng(1).uniform(1.0e-5, 1.0e3, 200)])

    convergence = mp.convergence_func(
        grid_radius=ag.ArrayIrregular(eta * mp.scale_radius)
    )

    reference = _convergence_per_point_quad_reference(
        inner_slope=inner_slope, kappa_s=mp.kappa_s, eta=eta
    )

    assert isinstance(convergence, np.ndarray)
    assert np.max(np.abs(convergence / reference - 1.0)) < 2.0e-6


def test__convergence_func__tabulated__non_positive_and_non_finite_radii_as_per_point():
    mp = ag.mp.gNFWSph(kappa_s=0.3, inner_slope=0.5, scale_radius=2.0)

    radii = np.geomspace(1.0e-3, 1.0e2, 500)
    radii[[0, 10, 20]] = [0.0, np.nan, np.inf]

    convergence = mp.convergence_func(grid_radius=ag.ArrayIrregular(radii))
    convergence_loop = _convergence_func_per_point_loop(
        mp=mp, grid_radius=ag.ArrayIrregular(radii)
    )

    special = np.zeros(radii.shape, dtype=bool)
    special[[0, 10, 20]] = True

    np.testing.assert_array_equal(convergence[special], convergence_loop[special])
    assert convergence[~special] == pytest.approx(convergence_loop[~special], 1.0e-5)


@pytest.mark.parametrize("inner_slope", [0.0, 0.5, 1.0, 1.5, 1.99, 2.5])
def test__convergence_func__small_input__matches_reference_and_tabulated(inner_slope):
    mp = ag.mp.gNFWSph(kappa_s=0.3, inner_slope=inner_slope, scale_radius=2.0)

    eta = np.geomspace(1.0e-5, 1.0e3, 200)

    convergence_tabulated = mp.convergence_func(
        grid_radius=ag.ArrayIrregular(eta * mp.scale_radius)
    )
    convergence_per_point = np.concatenate(
        [
            mp.convergence_func(grid_radius=ag.ArrayIrregular(eta_chunk * mp.scale_radius))
            for eta_chunk in np.array_split(eta, 40)
        ]
    )

    reference = _convergence_per_point_quad_reference(
        inner_slope=inner_slope, kappa_s=mp.kappa_s, eta=eta
    )

    assert np.max(np.abs(convergence_per_point / reference - 1.0)) < 2.0e-6
    assert np.max(np.abs(convergence_tabulated / convergence_per_point - 1.0)) < 2.0e-6


def test__convergence_2d_from__elliptical_grid__tabulated_matches_per_point():
    mp = ag.mp.gNFW(
        centre=(0.05, -0.1),
        ell_comps=(0.2, -0.1),
        kappa_s=0.2,
        inner_slope=1.4,
        scale_radius=3.0,
    )

    grid_2d = ag.Grid2D.uniform(
        shape_native=(30, 30), pixel_scales=0.1, over_sample_size=1
    )

    convergence = mp.convergence_2d_from(grid=grid_2d)

    convergence_per_point = np.array(
        [
            mp.convergence_2d_from(grid=ag.Grid2DIrregular([coordinate])).array[0]
            for coordinate in grid_2d.array
        ]
    )

    assert isinstance(convergence, ag.Array2D)
    assert convergence.array == pytest.approx(convergence_per_point, 1.0e-5)
