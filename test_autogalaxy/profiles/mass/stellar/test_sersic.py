import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_2d_via_cse_from__config_1():
    mp = ag.mp.Sersic(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    deflections_via_integral = np.array([[1.1446, 0.79374]])
    deflections_via_cse = mp.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-4)


def test__deflections_2d_via_cse_from__config_2():
    mp = ag.mp.Sersic(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=10.0,
        effective_radius=0.2,
        sersic_index=3.0,
        mass_to_light_ratio=1.0,
    )

    deflections_via_integral = np.array([[2.6134, 1.80719]])
    deflections_via_cse = mp.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-3)


def test__deflections_2d_via_cse_from__config_3():
    mp = ag.mp.Sersic(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        sersic_index=1.5,
        mass_to_light_ratio=2.0,
    )

    deflections_via_integral = np.array([[2.0810915255, 1.4433668621]])
    deflections_via_cse = mp.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-3)


def test__deflections_yx_2d_from__sersic():
    mp = ag.mp.Sersic()

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflections_via_integral = mp.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_integral.array, 1.0e-4)


def test__deflections_yx_2d_from__sersic_sph():
    mp = ag.mp.SersicSph()

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflections_via_integral = mp.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_integral.array, 1.0e-4)


def test__deflections_yx_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=4.0,
        mass_to_light_ratio=1.0,
    )

    spherical = ag.mp.SersicSph(
        centre=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=4.0,
        mass_to_light_ratio=1.0,
    )

    elliptical_deflections = elliptical.deflections_yx_2d_from(grid=grid)
    spherical_deflections = spherical.deflections_yx_2d_from(grid=grid)

    assert elliptical_deflections == pytest.approx(spherical_deflections.array, 1.0e-4)


def test__convergence_2d_via_cse_from__sersic_intensity_3():
    mp = ag.mp.Sersic(
        centre=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    convergence = mp.convergence_2d_via_cse_from(grid=ag.Grid2DIrregular([[0.0, 1.5]]))

    assert convergence == pytest.approx(4.90657319276, 1e-3)


def test__convergence_2d_via_cse_from__sersic_intensity_6():
    mp = ag.mp.Sersic(
        centre=(0.0, 0.0),
        intensity=6.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    convergence = mp.convergence_2d_via_cse_from(grid=ag.Grid2DIrregular([[0.0, 1.5]]))

    assert convergence == pytest.approx(2.0 * 4.90657319276, 1e-3)


def test__convergence_2d_via_cse_from__sersic_mass_to_light_2():
    mp = ag.mp.Sersic(
        centre=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
    )

    convergence = mp.convergence_2d_via_cse_from(grid=ag.Grid2DIrregular([[0.0, 1.5]]))

    assert convergence == pytest.approx(2.0 * 4.90657319276, 1e-3)


def test__convergence_2d_via_cse_from__sersic_elliptical():
    mp = ag.mp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    convergence = mp.convergence_2d_via_cse_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert convergence == pytest.approx(5.38066670129, 1e-3)


def test__convergence_2d_from__sersic_intensity_3():
    mp = ag.mp.Sersic(
        centre=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.5]]))

    assert convergence == pytest.approx(4.90657319276, 1e-3)


def test__convergence_2d_from__sersic_intensity_6():
    mp = ag.mp.Sersic(
        centre=(0.0, 0.0),
        intensity=6.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.5]]))

    assert convergence == pytest.approx(2.0 * 4.90657319276, 1e-3)


def test__convergence_2d_from__sersic_mass_to_light_2():
    mp = ag.mp.Sersic(
        centre=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.5]]))

    assert convergence == pytest.approx(2.0 * 4.90657319276, 1e-3)


def test__convergence_2d_from__sersic_elliptical():
    mp = ag.mp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert convergence == pytest.approx(5.38066670129, 1e-3)


def test__convergence_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=4.0,
        mass_to_light_ratio=1.0,
    )

    spherical = ag.mp.SersicSph(
        centre=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=4.0,
        mass_to_light_ratio=1.0,
    )

    ell_convergence_2d = elliptical.convergence_2d_from(grid=grid)
    sph_convergence_2d = spherical.convergence_2d_from(grid=grid)

    assert ell_convergence_2d == pytest.approx(sph_convergence_2d, 1.0e-4)


def test__jax_grad__deflections_finite_and_match_finite_differences():
    # The CSE amplitudes used to come from `jnp.linalg.lstsq`, whose SVD-based reverse-mode
    # derivative is NaN on the ill-conditioned decomposition matrix, so every gradient was NaN.
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    grid = ag.Grid2DIrregular([(0.3, 0.4), (1.0, -0.2), (0.0, 1.0), (3.0, 2.0)])

    params = {
        "mass_to_light_ratio": 3.35,
        "intensity": 0.02,
        "effective_radius": 1.7,
        "sersic_index": 4.0,
    }

    with jax.enable_x64(True):

        @jax.jit
        def summed_deflections(parameters):
            mp = ag.mp.Sersic(centre=(0.0, 0.0), ell_comps=(0.12, 0.07), **parameters)
            return jnp.sum(mp.deflections_yx_2d_from(grid=grid, xp=jnp).array)

        parameters = {key: jnp.asarray(value) for key, value in params.items()}

        value = float(summed_deflections(parameters))
        gradient = jax.grad(summed_deflections)(parameters)

        for key, step_value in params.items():
            step = 1.0e-4 * step_value
            upper = dict(parameters, **{key: parameters[key] + step})
            lower = dict(parameters, **{key: parameters[key] - step})
            finite_difference = (
                float(summed_deflections(upper)) - float(summed_deflections(lower))
            ) / (2.0 * step)

            assert np.isfinite(float(gradient[key]))
            assert float(gradient[key]) == pytest.approx(finite_difference, rel=1.0e-4)

        # The deflections are exactly linear in the mass-to-light ratio.
        assert float(gradient["mass_to_light_ratio"]) == pytest.approx(
            value / params["mass_to_light_ratio"], rel=1.0e-8
        )
