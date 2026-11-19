import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


# def test__deflections_2d_via_mge_from():
#     mp = ag.mp.SersicGradient(
#         centre=(-0.4, -0.2),
#         ell_comps=(-0.07142, -0.085116),
#         intensity=5.0,
#         effective_radius=0.2,
#         sersic_index=2.0,
#         mass_to_light_ratio=1.0,
#         mass_to_light_gradient=-1.0,
#     )
#
#     deflections_via_integral = mp.deflections_2d_via_integral_from(
#         grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
#     )
#     deflections_via_mge = mp.deflections_2d_via_mge_from(
#         grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
#     )
#
#     assert deflections_via_integral == pytest.approx(deflections_via_mge.array, 1.0e-3)


def test__deflections_2d_via_cse_from__gradient_1():
    mp = ag.mp.SersicGradient(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=1.0,
    )

    deflections_via_integral = np.array([[2.882598979610964, 1.8911115687770221]])
    deflections_via_cse = mp.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-4)


def test__deflections_2d_via_cse_from__gradient_neg_1():
    mp = ag.mp.SersicGradient(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=-1.0,
    )

    deflections_via_integral = np.array([[1.222580447264419, 0.9068235185942723]])
    deflections_via_cse = mp.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-4)


def test__deflections_yx_2d_from__sersic_gradient():
    mp = ag.mp.SersicGradient()

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflections_via_integral = mp.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_integral.array, 1.0e-4)


def test__deflections_yx_2d_from__sersic_gradient_sph():
    mp = ag.mp.SersicGradientSph()

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflections_via_integral = mp.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_integral.array, 1.0e-4)


def test__deflections_yx_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.SersicGradient(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=4.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=1.0,
    )

    spherical = ag.mp.SersicGradient(
        centre=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=4.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=1.0,
    )

    ell_deflections_yx_2d = elliptical.deflections_yx_2d_from(grid=grid)
    sph_deflections_yx_2d = spherical.deflections_yx_2d_from(grid=grid)

    ell_deflections_yx_2d == pytest.approx(sph_deflections_yx_2d.array, 1.0e-4)


def test__convergence_2d_from__sersic_gradient_config_1():
    # ((radius/effective_radius)**-mass_to_light_gradient) = (1/0.6)**-1.0 = 0.6
    mp = ag.mp.SersicGradient(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=4.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.6 * 0.351797, 1e-3)


def test__convergence_2d_from__sersic_gradient_config_2():
    # ((radius/effective_radius)**-mass_to_light_gradient) = (1.5/2.0)**1.0 = 0.75

    mp = ag.mp.SersicGradient(
        ell_comps=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=-1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.5, 0.0]]))

    assert convergence == pytest.approx(0.75 * 4.90657319276, 1e-3)


def test__convergence_2d_from__sersic_gradient_intensity_6():
    mp = ag.mp.SersicGradient(
        ell_comps=(0.0, 0.0),
        intensity=6.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=-1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.5, 0.0]]))

    assert convergence == pytest.approx(2.0 * 0.75 * 4.90657319276, 1e-3)


def test__convergence_2d_from__sersic_gradient_mass_to_light_2():
    mp = ag.mp.SersicGradient(
        ell_comps=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
        mass_to_light_gradient=-1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.5, 0.0]]))

    assert convergence == pytest.approx(2.0 * 0.75 * 4.90657319276, 1e-3)


def test__convergence_2d_from__sersic_gradient_elliptical():
    # M/L = (radius / effective_radius) ** -mass_to_light_gradient, radius the eccentric radius.
    mp = ag.mp.SersicGradient(
        ell_comps=(0.0, 0.333333),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    sersic = ag.mp.Sersic(
        ell_comps=(0.0, 0.333333),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )
    grid_eval = ag.Grid2DIrregular([[1.0, 0.0]])
    radius = sersic.eccentric_radii_grid_from(
        grid=sersic.transformed_to_reference_frame_grid_from(grid=grid_eval)
    )[0]

    assert convergence == pytest.approx(
        (radius / 2.0) ** -1.0 * sersic.convergence_2d_from(grid=grid_eval)[0], 1e-6
    )
    assert convergence == pytest.approx(7.6097141, 1e-4)


def test__convergence_2d_from__elliptical_vs_spherical():
    elliptical = ag.mp.SersicGradient(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=4.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=1.0,
    )

    spherical = ag.mp.SersicGradient(
        centre=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=4.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=1.0,
    )

    ell_convergence_2d = elliptical.convergence_2d_from(grid=grid)
    sph_convergence_2d = spherical.convergence_2d_from(grid=grid)

    assert ell_convergence_2d == pytest.approx(sph_convergence_2d.array, 1.0e-4)


def test__compare_to_sersic__gradient_0_vs_exponential():
    mp = ag.mp.SersicGradient(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        sersic_index=1.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=0.0,
    )

    sersic_deflections = mp.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    exponential = ag.mp.Exponential(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        mass_to_light_ratio=1.0,
    )
    exponential_deflections = exponential.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert sersic_deflections[0, 0] == pytest.approx(
        exponential_deflections.array[0, 0], 1e-3
    )
    assert sersic_deflections[0, 0] == pytest.approx(0.90493, 1e-3)
    assert sersic_deflections[0, 1] == pytest.approx(
        exponential_deflections.array[0, 1], 1e-3
    )
    assert sersic_deflections[0, 1] == pytest.approx(0.62569, 1e-3)


def test__compare_to_sersic__gradient_0_vs_dev_vaucouleurs():
    mp = ag.mp.SersicGradient(
        centre=(0.4, 0.2),
        ell_comps=(0.0180010, 0.0494575),
        intensity=2.0,
        effective_radius=0.8,
        sersic_index=4.0,
        mass_to_light_ratio=3.0,
        mass_to_light_gradient=0.0,
    )
    sersic_deflections = mp.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    dev = ag.mp.DevVaucouleurs(
        centre=(0.4, 0.2),
        ell_comps=(0.0180010, 0.0494575),
        intensity=2.0,
        effective_radius=0.8,
        mass_to_light_ratio=3.0,
    )

    dev_deflections = dev.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    # assert sersic_deflections[0, 0] == pytest.approx(dev_deflections[0, 0], 1e-3)
    # assert sersic_deflections[0, 0] == pytest.approx(-24.528, 1e-3)
    # assert sersic_deflections[0, 1] == pytest.approx(dev_deflections[0, 1], 1e-3)
    # assert sersic_deflections[0, 1] == pytest.approx(-3.37605, 1e-3)


def test__compare_to_sersic__gradient_0_vs_sersic():
    sersic_grad = ag.mp.SersicGradient(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=0.0,
    )
    sersic_grad_deflections = sersic_grad.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    mp = ag.mp.Sersic(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )
    sersic_deflections = mp.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert sersic_deflections[0, 0] == pytest.approx(
        sersic_grad_deflections.array[0, 0], 1e-3
    )
    assert sersic_deflections[0, 0] == pytest.approx(1.1446, 1e-3)
    assert sersic_deflections[0, 1] == pytest.approx(
        sersic_grad_deflections.array[0, 1], 1e-3
    )
    assert sersic_deflections[0, 1] == pytest.approx(0.79374, 1e-3)


def test__mass_to_light_ratio__equals_input_on_effective_radius_ellipse():
    # The pivot of the gradient is the ellipse of the effective radius for any axis ratio, so the
    # mass-to-light ratio there equals `mass_to_light_ratio` (it used to be R_e / q).
    ell_comps = ag.convert.ell_comps_from(axis_ratio=0.6, angle=30.0)

    kwargs = dict(
        centre=(0.1, -0.2),
        ell_comps=ell_comps,
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
    )

    sersic = ag.mp.Sersic(**kwargs)

    # Points on the effective-radius ellipse: the eccentric radius is linear in the distance from
    # the centre, so scaling each offset by its eccentric radius puts it on the R_e = 1 ellipse.
    centre = np.array([0.1, -0.2])
    phi = np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False)
    offsets = np.stack([np.sin(phi), np.cos(phi)], axis=1)
    def eccentric_radii_from(grid):
        return np.asarray(
            sersic.eccentric_radii_grid_from(
                grid=sersic.transformed_to_reference_frame_grid_from(grid=grid)
            )
        )

    eccentric_radii = eccentric_radii_from(ag.Grid2DIrregular(centre + offsets))
    grid = ag.Grid2DIrregular(centre + offsets / eccentric_radii[:, None])

    assert eccentric_radii_from(grid) == pytest.approx(1.0, 1.0e-6)

    for mass_to_light_gradient in (-1.0, 0.5, 1.5):
        gradient = ag.mp.SersicGradient(
            **kwargs, mass_to_light_gradient=mass_to_light_gradient
        )

        ratio = np.asarray(gradient.convergence_2d_from(grid=grid)) / np.asarray(
            sersic.convergence_2d_from(grid=grid)
        )

        assert ratio == pytest.approx(1.0, 1.0e-6)


def test__mass_to_light_ratio__positive_gradient_falls_outward():
    # Positive gradient: M/L falls outward, M/L(R) = M/L * (R / R_e) ** -gradient.
    kwargs = dict(
        intensity=1.0, effective_radius=1.0, sersic_index=2.0, mass_to_light_ratio=1.0
    )

    gradient = ag.mp.SersicGradient(**kwargs, mass_to_light_gradient=1.0)
    sersic = ag.mp.Sersic(**kwargs)

    grid = ag.Grid2DIrregular([[0.0, 0.5], [0.0, 2.0]])

    ratio = np.asarray(gradient.convergence_2d_from(grid=grid)) / np.asarray(
        sersic.convergence_2d_from(grid=grid)
    )

    assert ratio == pytest.approx([2.0, 0.5], 1.0e-6)


@pytest.mark.parametrize("mass_to_light_gradient", [0.2, 1.0])
def test__jax_jit__free_gradient_likelihood_matches_numpy(mass_to_light_gradient):
    # A free (traced) gradient used to raise a tracer-boolean error in `cse_settings_from`.
    # Both values cover both sides of its `mass_to_light_gradient > 0.5` selection.
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    grid = ag.Grid2D.uniform(shape_native=(8, 8), pixel_scales=0.2)

    def make(gradient):
        return ag.mp.SersicGradient(
            centre=(0.05, -0.05),
            ell_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.8,
            sersic_index=2.5,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=gradient,
        )

    data = np.asarray(
        make(0.6).deflections_yx_2d_from(grid=grid).array
    )

    @jax.jit
    def log_likelihood(gradient):
        deflections = make(gradient).deflections_yx_2d_from(grid=grid, xp=jnp)
        return -0.5 * jnp.sum((deflections.array - data) ** 2)

    deflections_np = np.asarray(
        make(mass_to_light_gradient).deflections_yx_2d_from(grid=grid).array
    )
    log_likelihood_np = -0.5 * np.sum((deflections_np - data) ** 2)

    log_likelihood_jax = log_likelihood(jnp.array(mass_to_light_gradient))

    assert np.isfinite(float(log_likelihood_jax))
    assert float(log_likelihood_jax) == pytest.approx(log_likelihood_np, rel=1.0e-3)
