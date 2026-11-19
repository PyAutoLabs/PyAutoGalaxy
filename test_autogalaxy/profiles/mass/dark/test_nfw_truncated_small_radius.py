"""
Small-radius and r = 0 behaviour of the `NFWTruncatedSph` deflection.

The closed-form tNFW deflection subtracts large, nearly equal terms at small
radius and divides by the radius, so it lost precision there and was NaN at
r = 0. These tests check both deflection paths (the object path and the
`vmapped_deflections_from` hook) against a 50-digit mpmath evaluation of the
same closed form, in numpy float64, jax float64 and jax float32.

The reference values are pinned below (mpmath is not a test dependency). They
were generated with this 50-digit evaluation of Baltz, Marshall & Oguri (2009)
eq. 13, alpha = 4 kappa_s r_s m(x) / x with x = r / r_s and tau = r_t / r_s:

    import mpmath

    def reference(eta, kappa_s=0.1, scale_radius=0.5, truncation_radius=5.0):
        with mpmath.workdps(50):
            x = mpmath.mpf(eta)
            tau = mpmath.mpf(truncation_radius) / mpmath.mpf(scale_radius)
            if x < 1:
                f = mpmath.acosh(1 / x) / mpmath.sqrt(1 - x**2)
            elif x > 1:
                f = mpmath.acos(1 / x) / mpmath.sqrt(x**2 - 1)
            else:
                f = mpmath.mpf(1)
            root = mpmath.sqrt(x**2 + tau**2)
            l = mpmath.log(x / (root + tau))
            m = (tau**2 / (tau**2 + 1) ** 2) * (
                (tau**2 + 2 * x**2 - 1) * f
                + mpmath.pi * tau
                + (tau**2 - 1) * mpmath.log(tau)
                + root * ((tau**2 - 1) / tau * l - mpmath.pi)
            )
            return float(4 * mpmath.mpf(kappa_s) * mpmath.mpf(scale_radius) * m / x)
"""

import importlib.util

import numpy as np
import pytest

import autogalaxy as ag

requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="requires jax (installed via the [optional] extras; absent on the NumPy-only matrix env)",
)

KAPPA_S = 0.1
SCALE_RADIUS = 0.5
TRUNCATION_RADIUS = 5.0

# r / r_s from 1e-10 to 1e-2, plus r = 0.
ETA = np.concatenate([[0.0], np.logspace(-10.0, -2.0, 17)])
RADII = ETA * SCALE_RADIUS


# 50-digit reference deflections at ETA (the generator is in the module docstring).
REFERENCE = np.array(
    [
        0.0,
        2.320347298053841e-10,
        6.973511754457132e-10,
        2.0900887887544363e-09,
        6.2453704144359515e-09,
        1.859830279455032e-08,
        5.5172290744147745e-08,
        1.6295717701556391e-07,
        4.789087734393947e-07,
        1.3993132608572635e-06,
        4.060946394402611e-06,
        1.1690547516437151e-05,
        3.332805056827013e-05,
        9.387962492259342e-05,
        0.00026046639074797586,
        0.0007085382576922829,
        0.0018765362465362244,
        0.004783144881283246,
    ]
)


def _assert_matches_reference(deflection, rtol):
    deflection = np.asarray(deflection, dtype=np.float64)

    assert np.all(np.isfinite(deflection))
    assert deflection[0] == 0.0
    np.testing.assert_allclose(deflection[1:], REFERENCE[1:], rtol=rtol, atol=0.0)


def test__reference_is_positive_and_monotonic():
    assert np.all(REFERENCE[1:] > 0.0)
    assert np.all(np.diff(REFERENCE[1:]) > 0.0)


def test__deflections_yx_2d_from__numpy_float64():
    profile = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0),
        kappa_s=KAPPA_S,
        scale_radius=SCALE_RADIUS,
        truncation_radius=TRUNCATION_RADIUS,
    )

    grid = ag.Grid2DIrregular([(0.0, float(r)) for r in RADII])

    deflections = np.asarray(profile.deflections_yx_2d_from(grid=grid))

    assert np.all(deflections[:, 0] == 0.0)
    _assert_matches_reference(deflections[:, 1], rtol=1.0e-8)


def test__radial_deflection_from__numpy_float64():
    deflection = ag.mp.NFWTruncatedSph.radial_deflection_from(
        RADII,
        np.array([KAPPA_S, SCALE_RADIUS, TRUNCATION_RADIUS]),
        np,
    )

    _assert_matches_reference(deflection, rtol=1.0e-8)


@requires_jax
def test__radial_deflection_from__jax_float64():
    import jax
    import jax.numpy as jnp

    with jax.enable_x64(True):
        deflection = ag.mp.NFWTruncatedSph.radial_deflection_from(
            jnp.asarray(RADII, dtype=jnp.float64),
            jnp.asarray([KAPPA_S, SCALE_RADIUS, TRUNCATION_RADIUS], dtype=jnp.float64),
            jnp,
        )

        assert deflection.dtype == jnp.float64
        _assert_matches_reference(deflection, rtol=1.0e-8)


@requires_jax
def test__radial_deflection_from__jax_float32():
    import jax.numpy as jnp

    deflection = ag.mp.NFWTruncatedSph.radial_deflection_from(
        jnp.asarray(RADII, dtype=jnp.float32),
        jnp.asarray([KAPPA_S, SCALE_RADIUS, TRUNCATION_RADIUS], dtype=jnp.float32),
        jnp,
    )

    assert deflection.dtype == jnp.float32
    _assert_matches_reference(deflection, rtol=1.0e-4)


@requires_jax
@pytest.mark.parametrize("dtype_name, rtol", [("float64", 1.0e-8), ("float32", 1.0e-4)])
def test__vmapped_deflections_from__small_radius_and_centre(dtype_name, rtol):
    import jax
    import jax.numpy as jnp

    with jax.enable_x64(dtype_name == "float64"):
        dtype = getattr(jnp, dtype_name)

        grid = jnp.stack([jnp.zeros(len(RADII)), jnp.asarray(RADII)], axis=-1).astype(
            dtype
        )
        params = jnp.asarray(
            [[0.0, 0.0, KAPPA_S, SCALE_RADIUS, TRUNCATION_RADIUS]], dtype=dtype
        )
        mask = jnp.array([True])

        deflections = ag.mp.NFWTruncatedSph.vmapped_deflections_from(grid, params, mask)

        assert deflections.dtype == dtype
        assert np.all(np.asarray(deflections[:, 0]) == 0.0)
        _assert_matches_reference(deflections[:, 1], rtol=rtol)


@requires_jax
@pytest.mark.parametrize("dtype_name", ["float64", "float32"])
def test__vmapped_deflections_from__gradient_finite_at_centre(dtype_name):
    import jax
    import jax.numpy as jnp

    with jax.enable_x64(dtype_name == "float64"):
        dtype = getattr(jnp, dtype_name)

        # The first grid point sits exactly on the halo centre.
        grid = jnp.asarray([[0.1, -0.2], [0.1, -0.1999], [0.5, 0.3]], dtype=dtype)
        params = jnp.asarray(
            [[0.1, -0.2, KAPPA_S, SCALE_RADIUS, TRUNCATION_RADIUS]], dtype=dtype
        )
        mask = jnp.array([True])

        def summed_deflection(params):
            return ag.mp.NFWTruncatedSph.vmapped_deflections_from(
                grid, params, mask
            ).sum()

        gradient = jax.grad(summed_deflection)(params)

        assert np.all(np.isfinite(np.asarray(gradient)))

        deflections = ag.mp.NFWTruncatedSph.vmapped_deflections_from(grid, params, mask)
        assert np.all(np.asarray(deflections[0]) == 0.0)


def test__deflections_unchanged_at_ordinary_radii():
    """The small-radius form hands over to the closed form at r / r_s = 0.5."""
    profile = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0),
        kappa_s=KAPPA_S,
        scale_radius=SCALE_RADIUS,
        truncation_radius=TRUNCATION_RADIUS,
    )

    eta = np.array([0.05, 0.3, 0.4999, 0.5, 0.5001, 1.0, 2.0, 20.0])
    grid = ag.Grid2DIrregular([(0.0, float(e * SCALE_RADIUS)) for e in eta])

    deflections = np.asarray(profile.deflections_yx_2d_from(grid=grid))[:, 1]

    # 50-digit reference deflections (the generator is in the module docstring).
    reference = [
        0.01589587017633467,
        0.044313734570148366,
        0.05296578576342507,
        0.05296875971148687,
        0.052971732626911495,
        0.05978773599871623,
        0.05740693334032161,
        0.014806583378259849,
    ]

    np.testing.assert_allclose(deflections, reference, rtol=1.0e-12, atol=0.0)
