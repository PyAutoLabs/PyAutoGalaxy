import pytest

import autogalaxy as ag


def test__ell_comps_from():
    ell_comps = ag.convert.ell_comps_from(axis_ratio=0.00050025012, angle=0.0)

    assert ell_comps == pytest.approx((0.0, 0.999), abs=1.0e-4)


def test__axis_ratio_and_angle_from():
    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=(0.0, 1.0))

    assert axis_ratio == pytest.approx(0.00050025012, abs=1.0e-4)
    assert angle == pytest.approx(0.0, abs=1.0e-4)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=(1.0, 0.0))

    assert axis_ratio == pytest.approx(0.00050025012, abs=1.0e-4)
    assert angle == pytest.approx(45.0, abs=1.0e-4)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=(0.0, -1.0))

    assert axis_ratio == pytest.approx(0.00050025012, abs=1.0e-4)
    assert angle == pytest.approx(90.0, abs=1.0e-4)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=(-1.0, 0.0))

    assert axis_ratio == pytest.approx(0.00050025012, abs=1.0e-4)
    assert angle == pytest.approx(-45.0, abs=1.0e-4)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=(-1.0, -1.0))

    assert axis_ratio == pytest.approx(0.00050025012, abs=1.0e-4)
    assert angle == pytest.approx(112.5, abs=1.0e-4)


def test__shear_gamma_1_2_from():
    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=0.0)

    assert gamma_1 == pytest.approx(0.05, abs=1.0e-4)
    assert gamma_2 == pytest.approx(0.0, abs=1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=45.0)

    assert gamma_1 == pytest.approx(0.0, abs=1.0e-4)
    assert gamma_2 == pytest.approx(0.05, abs=1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=90.0)

    assert gamma_1 == pytest.approx(-0.05, abs=1.0e-4)
    assert gamma_2 == pytest.approx(0.0, abs=1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=135.0)

    assert gamma_1 == pytest.approx(0.0, abs=1.0e-4)
    assert gamma_2 == pytest.approx(-0.05, abs=1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=180.0)

    assert gamma_1 == pytest.approx(0.05, abs=1.0e-4)
    assert gamma_2 == pytest.approx(0.0, abs=1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=225.0)

    assert gamma_1 == pytest.approx(0.0, abs=1.0e-4)
    assert gamma_2 == pytest.approx(0.05, abs=1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(magnitude=0.05, angle=-45.0)

    assert gamma_1 == pytest.approx(0.0, abs=1.0e-4)
    assert gamma_2 == pytest.approx(-0.05, abs=1.0e-4)


def test__shear_magnitude_and_angle_from():
    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.05, gamma_2=0.0
    )

    assert magnitude == pytest.approx(0.05, abs=1.0e-4)
    assert angle == pytest.approx(0.0, abs=1.0e-4)

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.0, gamma_2=0.05
    )

    assert magnitude == pytest.approx(0.05, abs=1.0e-4)
    assert angle == pytest.approx(45.0, abs=1.0e-4)

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=-0.05, gamma_2=0.0
    )

    assert magnitude == pytest.approx(0.05, abs=1.0e-4)
    assert angle == pytest.approx(90.0, abs=1.0e-4)

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.0, gamma_2=-0.05
    )

    assert magnitude == pytest.approx(0.05, abs=1.0e-4)
    assert angle == pytest.approx(135.0, abs=1.0e-4)

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.05, gamma_2=0.0
    )

    assert magnitude == pytest.approx(0.05, abs=1.0e-4)
    assert angle == pytest.approx(0.0, abs=1.0e-4)

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.0, gamma_2=0.05
    )

    assert magnitude == pytest.approx(0.05, abs=1.0e-4)
    assert angle == pytest.approx(45.0, abs=1.0e-4)

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.05, gamma_2=0.0
    )

    assert magnitude == pytest.approx(0.05, abs=1.0e-4)

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.05, gamma_2=-0.05
    )

    assert magnitude == pytest.approx(0.07071067811865, abs=1.0e-4)
    assert angle == pytest.approx(-22.5, abs=1.0e-4)


def test__multipole_k_m_and_phi_m_from():
    k_m, phi = ag.convert.multipole_k_m_and_phi_m_from(multipole_comps=(0.1, 0.0), m=1)

    assert k_m == pytest.approx(0.1, abs=1e-3)
    assert phi == pytest.approx(90.0, abs=1e-3)

    k_m, phi = ag.convert.multipole_k_m_and_phi_m_from(multipole_comps=(0.0, 0.1), m=1)

    assert k_m == pytest.approx(0.1, abs=1e-3)
    assert phi == pytest.approx(0.0, abs=1e-3)

    k_m, phi = ag.convert.multipole_k_m_and_phi_m_from(multipole_comps=(0.1, 0.0), m=2)

    assert k_m == pytest.approx(0.1, abs=1e-3)
    assert phi == pytest.approx(45.0, abs=1e-3)

    k_m, phi = ag.convert.multipole_k_m_and_phi_m_from(
        multipole_comps=(-0.1, -0.1), m=2
    )

    assert k_m == pytest.approx(0.14142135, abs=1e-3)
    assert phi == pytest.approx(112.5, abs=1e-3)


def test__multipole_comps_from():
    multipole_comps = ag.convert.multipole_comps_from(k_m=0.1, phi_m=90.0, m=1)

    assert multipole_comps == pytest.approx((0.1, 0.0), abs=1e-3)

    multipole_comps = ag.convert.multipole_comps_from(k_m=0.1, phi_m=0.0, m=1)

    assert multipole_comps == pytest.approx((0.0, 0.1), abs=1e-3)

    multipole_comps = ag.convert.multipole_comps_from(k_m=0.1, phi_m=45.0, m=2)

    assert multipole_comps == pytest.approx((0.1, 0.0), abs=1e-3)

    multipole_comps = ag.convert.multipole_comps_from(k_m=0.14142135, phi_m=112.5, m=2)

    assert multipole_comps == pytest.approx((-0.1, -0.1), abs=1e-3)


def test__polar_conversions__numpy_origin_values_unchanged():
    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=(0.0, 0.0))

    assert axis_ratio == 1.0
    assert angle == 0.0

    magnitude, angle = ag.convert.shear_magnitude_and_angle_from(
        gamma_1=0.0, gamma_2=0.0
    )

    assert magnitude == 0.0
    assert angle == 0.0

    k_m, phi_m = ag.convert.multipole_k_m_and_phi_m_from(
        multipole_comps=(0.0, 0.0), m=4
    )

    assert k_m == 0.0
    assert phi_m == 0.0


def _deflection_objectives():
    """
    Scalar objectives (weighted sums of deflections on a tiny grid) of the profiles whose polar conversions take a
    square root of their components, as functions of those two components.
    """
    import jax.numpy as jnp
    import numpy as np

    grid = ag.Grid2D.uniform(shape_native=(3, 3), pixel_scales=0.3)
    weights = jnp.asarray(np.random.default_rng(1).normal(size=(9, 2)))

    def shear(c):
        mass = ag.mp.ExternalShear(gamma_1=c[0], gamma_2=c[1])
        return jnp.sum(mass.deflections_yx_2d_from(grid=grid, xp=jnp).array * weights)

    def multipole(c):
        mass = ag.mp.PowerLawMultipole(
            centre=(0.01, 0.02),
            einstein_radius=1.0,
            slope=2.0,
            m=4,
            multipole_comps=(c[0], c[1]),
        )
        return jnp.sum(mass.deflections_yx_2d_from(grid=grid, xp=jnp).array * weights)

    def isothermal(c):
        mass = ag.mp.Isothermal(
            centre=(0.01, 0.02), ell_comps=(c[0], c[1]), einstein_radius=1.0
        )
        return jnp.sum(mass.deflections_yx_2d_from(grid=grid, xp=jnp).array * weights)

    return {"shear": shear, "multipole": multipole, "isothermal": isothermal}


@pytest.mark.parametrize("name", ["shear", "multipole"])
def test__polar_conversions__jax_grad_finite_at_origin_fp64(name):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import numpy as np

    with jax.enable_x64(True):
        f = _deflection_objectives()[name]
        grad = jax.grad(f)(jnp.zeros(2))

        assert np.all(np.isfinite(np.asarray(grad)))


@pytest.mark.parametrize("name", ["shear", "multipole", "isothermal"])
def test__polar_conversions__jax_grad_finite_at_origin_fp32(name):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import numpy as np

    with jax.enable_x64(False):
        f = _deflection_objectives()[name]
        grad = jax.grad(f)(jnp.zeros(2, dtype=jnp.float32))

        assert np.all(np.isfinite(np.asarray(grad)))


@pytest.mark.parametrize("name", ["shear", "multipole"])
def test__polar_conversions__jax_grad_at_origin_matches_finite_difference(name):
    """
    Shear and multipole deflections are linear in their components, so the gradient at the origin is well defined
    and equal to the central finite difference there.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import numpy as np

    with jax.enable_x64(True):
        f = _deflection_objectives()[name]
        grad = np.asarray(jax.grad(f)(jnp.zeros(2)))

        h = 1.0e-5
        fd = np.array(
            [
                (f(jnp.array([h, 0.0])) - f(jnp.array([-h, 0.0]))) / (2 * h),
                (f(jnp.array([0.0, h])) - f(jnp.array([0.0, -h]))) / (2 * h),
            ]
        )

        assert grad == pytest.approx(fd, abs=1.0e-6)
