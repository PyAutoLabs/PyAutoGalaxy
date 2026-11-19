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
        return (y + eta_value) ** (inner_slope - 4) * y**2 / (1.0 + np.sqrt(1.0 - y**2))

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
            mp.convergence_func(
                grid_radius=ag.ArrayIrregular(eta_chunk * mp.scale_radius)
            )
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


def _deflection_via_direct_integration(radius, inner_slope, scale_radius):
    """
    Spherical gNFW deflection (kappa_s = 1) from a direct numerical integration of the
    3D density: alpha(R) = m(<R) / (pi R), with m(<R) the 3D mass inside R plus the mass
    of the shells beyond R that the cylinder of radius R cuts, both integrated in ln t.
    """
    from scipy.integrate import quad

    x_cyl = radius / scale_radius
    g = inner_slope

    def shell(log_t):
        t = np.exp(log_t)
        return t ** (3.0 - g) * (1.0 + t) ** (g - 3.0)

    def cut_shell(log_t):
        return shell(log_t) * (1.0 - np.sqrt(1.0 - (x_cyl / np.exp(log_t)) ** 2))

    log_x = np.log(x_cyl)
    inner = quad(shell, -60.0, log_x, epsabs=0.0, epsrel=1e-11, limit=400)[0]
    outer = quad(
        cut_shell,
        log_x,
        log_x + 40.0,
        epsabs=0.0,
        epsrel=1e-11,
        limit=400,
        points=[log_x + 1e-6, log_x + 0.1, log_x + 1.0],
    )[0]

    return 4.0 * scale_radius**2 * (inner + outer) / radius


@pytest.mark.parametrize("inner_slope", [0.0, 1.0, 1.5, 2.0, 2.5])
@pytest.mark.parametrize("scale_radius", [5.0, 20.0])
def test__deflections_yx_2d_from__matches_direct_integration_for_steep_slopes(
    inner_slope, scale_radius
):
    # Without the inner-mass term the MGE deflections were 0.8% low at inner_slope 2 and
    # 10% low at 2.5 (PyAutoGalaxy#729).
    radii = np.geomspace(0.2, 3.0, 6)

    mp = ag.mp.gNFWSph(
        centre=(0.0, 0.0),
        kappa_s=1.0,
        inner_slope=inner_slope,
        scale_radius=scale_radius,
    )

    deflections = mp.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([(0.0, float(r)) for r in radii])
    )
    deflections = np.asarray(deflections.array)

    alpha_direct = np.array(
        [
            _deflection_via_direct_integration(r, inner_slope, scale_radius)
            for r in radii
        ]
    )

    assert np.hypot(deflections[:, 0], deflections[:, 1]) == pytest.approx(
        alpha_direct, rel=1e-3
    )


@pytest.mark.parametrize("inner_slope", [1.0, 2.5])
def test__deflections_yx_2d_from__elliptical__matches_wide_mge(inner_slope):
    # An 80-Gaussian MGE reaching down to scale_radius * 1e-13 captures the cusp
    # without the inner-mass term, so it checks the term's elliptical scaling.
    from autogalaxy.profiles.mass import MGEDecomposer

    mp = ag.mp.gNFW(
        centre=(0.05, -0.1),
        ell_comps=(0.15, 0.05),
        kappa_s=1.0,
        inner_slope=inner_slope,
        scale_radius=10.0,
    )

    grid = ag.Grid2DIrregular([(0.25, 0.1), (-0.8, 0.6), (1.5, -2.5)])

    deflections = np.asarray(mp.deflections_yx_2d_from(grid=grid).array)

    sigmas = np.exp(np.linspace(np.log(10.0 * 1e-13), np.log(10.0 * 200.0), 80))
    deflections_wide = MGEDecomposer(mass_profile=mp).deflections_2d_via_mge_from(
        grid=grid,
        sigma_log_list=sigmas,
        ellipticity_convention="major",
        three_D=True,
    )

    assert deflections == pytest.approx(np.asarray(deflections_wide.array), rel=1e-3)


def test__inner_mass_term__potential_gradient_matches_deflections():
    mp = ag.mp.gNFW(
        centre=(0.0, 0.0),
        ell_comps=(0.1, 0.05),
        kappa_s=1.0,
        inner_slope=2.4,
        scale_radius=10.0,
    )

    points = np.array([(0.3, 0.4), (-1.0, 1.5), (2.0, -0.5), (1e-6, 0.0)])

    def potential(p):
        return np.asarray(
            mp._potential_2d_inner_mass_from(
                grid=ag.Grid2DIrregular([tuple(x) for x in p])
            )
        )

    step = 1e-8
    grad_y = (potential(points + [step, 0.0]) - potential(points - [step, 0.0])) / (
        2.0 * step
    )
    grad_x = (potential(points + [0.0, step]) - potential(points - [0.0, step])) / (
        2.0 * step
    )

    deflections = np.asarray(
        mp._deflections_2d_inner_mass_from(
            grid=ag.Grid2DIrregular([tuple(x) for x in points])
        ).array
    )

    assert grad_y == pytest.approx(deflections[:, 0], rel=1e-5, abs=1e-8)
    assert grad_x == pytest.approx(deflections[:, 1], rel=1e-5, abs=1e-8)
    assert mp._potential_2d_inner_mass_from(grid=ag.Grid2DIrregular([(0.0, 0.0)]))[
        0
    ] == pytest.approx(0.0, abs=1e-12)
