"""
Gradients through `MassProfile.vmapped_deflections_from`, the batched hook of the JAX
line-of-sight / subhalo ray-tracing scan.

A masked-out slot (callers pad these with zeros) must contribute exactly zero to the deflections
and to their gradients, and the cNFW deflections must have finite gradients everywhere.
"""

import importlib.util

import numpy as np
import pytest

import autogalaxy as ag

# The `vmapped_deflections_from` path is jax-backed; these tests need jax
# installed to run (it ships via the `[optional]` extras). The NumPy-only
# Python-version matrix has no jax, so skip there rather than fail.
requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="requires jax (installed via the [optional] extras; absent on the NumPy-only matrix env)",
)

PROFILES = [
    (
        ag.mp.NFWTruncatedSph,
        [
            [0.3, -0.2, 0.05, 0.8, 3.0],
            [-0.5, 0.4, 0.02, 1.3, 2.5],
        ],
    ),
    (
        ag.mp.cNFWSph,
        [
            [0.3, -0.2, 0.05, 1.0, 0.1],
            [-0.5, 0.4, 0.02, 1.5, 0.3],
        ],
    ),
]


def _grid():
    return np.random.default_rng(1).uniform(-2.0, 2.0, size=(25, 2))


@requires_jax
@pytest.mark.parametrize("profile_cls, real_params", PROFILES)
def test__masked_zero_slots__value_and_gradients_match_no_padding(
    profile_cls, real_params
):
    import jax
    import jax.numpy as jnp

    with jax.enable_x64(True):
        grid = jnp.array(_grid())
        real = jnp.array(real_params)
        padded = jnp.concatenate([real, jnp.zeros((3, real.shape[1]))])
        mask_padded = jnp.array([True, True, False, False, False])
        mask_real = jnp.array([True, True])

        def summed(grid, params, mask):
            return jnp.sum(profile_cls.vmapped_deflections_from(grid, params, mask))

        value_padded = summed(grid, padded, mask_padded)
        value_real = summed(grid, real, mask_real)
        d_grid_padded, d_params_padded = jax.grad(summed, argnums=(0, 1))(
            grid, padded, mask_padded
        )
        d_grid_real, d_params_real = jax.grad(summed, argnums=(0, 1))(
            grid, real, mask_real
        )

    assert np.isfinite(np.asarray(d_grid_padded)).all()
    assert np.isfinite(np.asarray(d_params_padded)).all()
    assert float(value_padded) == pytest.approx(float(value_real), abs=1e-12)
    np.testing.assert_allclose(d_grid_padded, d_grid_real, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(
        d_params_padded[:2], d_params_real, rtol=0.0, atol=1e-10
    )
    assert np.all(np.asarray(d_params_padded[2:]) == 0.0)


@requires_jax
@pytest.mark.parametrize("profile_cls, real_params", PROFILES)
def test__all_slots_masked__deflections_and_gradients_are_zero(
    profile_cls, real_params
):
    import jax
    import jax.numpy as jnp

    with jax.enable_x64(True):
        grid = jnp.array(_grid())
        params = jnp.zeros((3, len(real_params[0])))
        mask = jnp.array([False, False, False])

        def summed(grid, params):
            return jnp.sum(profile_cls.vmapped_deflections_from(grid, params, mask))

        d_grid, d_params = jax.grad(summed, argnums=(0, 1))(grid, params)

    assert float(summed(grid, params)) == 0.0
    assert np.all(np.asarray(d_grid) == 0.0)
    assert np.all(np.asarray(d_params) == 0.0)


@requires_jax
def test__cnfw_sph__vmapped_gradients_finite_and_match_finite_differences():
    import jax
    import jax.numpy as jnp

    params = np.array([[0.1, -0.2, 0.05, 1.0, 0.1]])
    # Points inside the core radius, between the core and scale radius, outside the scale
    # radius and exactly on the scale radius.
    grid = np.array(
        [[0.1 + 0.05, -0.2], [0.1 + 0.4, -0.2], [0.1, -0.2 + 2.5], [0.1 + 1.0, -0.2]]
    )
    mask = jnp.array([True])

    def summed(grid, params):
        return jnp.sum(ag.mp.cNFWSph.vmapped_deflections_from(grid, params, mask))

    with jax.enable_x64(True):
        d_grid, d_params = jax.grad(summed, argnums=(0, 1))(
            jnp.array(grid), jnp.array(params)
        )

        # Not smaller: the closed forms lose precision within ~1e-6 of the scale radius,
        # which a smaller central difference amplifies.
        step = 1.0e-4
        d_params_numerical = np.zeros_like(params)
        for i in range(params.shape[1]):
            up, down = params.copy(), params.copy()
            up[0, i] += step
            down[0, i] -= step
            d_params_numerical[0, i] = (
                float(summed(jnp.array(grid), jnp.array(up)))
                - float(summed(jnp.array(grid), jnp.array(down)))
            ) / (2.0 * step)

    assert np.isfinite(np.asarray(d_grid)).all()
    assert np.isfinite(np.asarray(d_params)).all()
    np.testing.assert_allclose(d_params, d_params_numerical, rtol=1e-5, atol=1e-8)


def test__cnfw_f_func__value_and_slope_continuous_at_the_radius():
    from autogalaxy.profiles.mass.dark.cnfw import F_func_from

    radius = 1.3
    step = 1.0e-7
    theta = np.array([radius - step, radius, radius + step])

    F = F_func_from(theta, radius)

    assert F[1] == pytest.approx(radius**2 * np.log(2.0), rel=1e-12)
    assert F[0] == pytest.approx(F[1], abs=1e-6)
    assert F[2] == pytest.approx(F[1], abs=1e-6)


def test__cnfw_dev_f_func__value_continuous_at_the_radius():
    from autogalaxy.profiles.mass.dark.cnfw import dev_F_func_from

    radius = 1.3
    step = 1.0e-7
    theta = np.array([radius - step, radius, radius + step])

    dev_F = dev_F_func_from(theta, radius)

    assert dev_F[1] == pytest.approx(2.0 * radius * (np.log(2.0) - 0.5), rel=1e-12)
    assert dev_F[0] == pytest.approx(dev_F[1], abs=1e-6)
    assert dev_F[2] == pytest.approx(dev_F[1], abs=1e-6)
