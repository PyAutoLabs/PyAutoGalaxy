"""
Behaviour of `NFWTruncatedSph` near its scale radius, r = r_s.

The NFW auxiliary functions f(x) and g(x) = (1 - f) / (x^2 - 1), with x = r / r_s,
are each ratios of two quantities that vanish at x = 1, so their closed forms cancel
there. The convergence (which reads f and g) and the deflection (which reads f) lost
precision for grid points within ~1e-6 r_s of the scale radius, and the gradient of the
deflection at exactly x = 1 was wrong. Near x = 1 both functions now use their power
series in u = 1 - x^2. These tests check the convergence and both deflection paths,
in numpy float64 and jax float32, and the jax gradient at x = 1, against a 50-digit
mpmath evaluation of the closed forms of Baltz, Marshall & Oguri (2009).

The reference values are pinned below (mpmath is not a test dependency). They were
generated with:

    import mpmath

    def references(eta, kappa_s=0.1, scale_radius=1.0, truncation_radius=5.0):
        with mpmath.workdps(50):
            x = mpmath.mpf(eta)
            tau = mpmath.mpf(truncation_radius) / mpmath.mpf(scale_radius)
            if x < 1:
                f = mpmath.acosh(1 / x) / mpmath.sqrt(1 - x**2)
            elif x > 1:
                f = mpmath.acos(1 / x) / mpmath.sqrt(x**2 - 1)
            else:
                f = mpmath.mpf(1)
            g = (1 - f) / (x**2 - 1) if x != 1 else mpmath.mpf(1) / 3
            root = mpmath.sqrt(x**2 + tau**2)
            l = mpmath.log(x / (root + tau))
            t = tau**2 / (tau**2 + 1) ** 2
            convergence = 2 * kappa_s * t * (
                (tau**2 + 1) * g + 2 * f - mpmath.pi / root
                + (tau**2 - 1) / (tau * root) * l
            )
            m = t * (
                (tau**2 + 2 * x**2 - 1) * f
                + mpmath.pi * tau
                + (tau**2 - 1) * mpmath.log(tau)
                + root * ((tau**2 - 1) / tau * l - mpmath.pi)
            )
            deflection = 4 * kappa_s * scale_radius * m / x
            return float(convergence), float(deflection)

and the derivative of the deflection at x = 1 with `mpmath.diff` at 50 digits.
"""

import importlib.util

import numpy as np
import pytest

import autogalaxy as ag
from autogalaxy.profiles.mass.dark.abstract import coord_func_f_from

requires_jax = pytest.mark.skipif(
    importlib.util.find_spec("jax") is None,
    reason="requires jax (installed via the [optional] extras; absent on the NumPy-only matrix env)",
)

KAPPA_S = 0.1
SCALE_RADIUS = 1.0
TRUNCATION_RADIUS = 5.0

# r / r_s = 1 exactly, then 1 + 1e-k and 1 - 1e-k for k = 3 ... 15.
ETA_NEAR = np.array(
    [1.0] + [1.0 + sign * 10.0**-k for k in range(3, 16) for sign in (1.0, -1.0)]
)

# 50-digit (convergence, deflection) at ETA_NEAR (the generator is in the docstring).
REFERENCE_NEAR = np.array(
    [
        [0.05823756361046192, 0.11416898574748188],
        [0.05815847543919252, 0.11417126867775279],
        [0.0583168015332641, 0.11416665635405912],
        [0.058229648060833666, 0.11416921612933399],
        [0.05824548065760446, 0.11416875490099855],
        [0.05823677198811727, 0.11416900880657348],
        [0.05823835524778171, 0.11416896268374398],
        [0.05823748444755359, 0.11416898805360012],
        [0.05823764277352002, 0.11416898344131718],
        [0.05823755569416434, 0.1141689859780958],
        [0.058237571526760984, 0.11416898551686751],
        [0.0582375628188321, 0.11416898577054330],
        [0.05823756440209176, 0.11416898572442047],
        [0.058237563531298935, 0.11416898574978802],
        [0.058237563689624905, 0.11416898574517574],
        [0.058237563602545624, 0.11416898574771249],
        [0.05823756361837822, 0.11416898574725126],
        [0.05823756360967029, 0.11416898574750495],
        [0.058237563611253554, 0.11416898574745882],
        [0.05823756361038275, 0.11416898574748419],
        [0.05823756361054108, 0.11416898574747958],
        [0.05823756361045401, 0.11416898574748212],
        [0.05823756361046984, 0.11416898574748165],
        [0.05823756361046113, 0.11416898574748191],
        [0.058237563610462714, 0.11416898574748185],
        [0.05823756361046183, 0.11416898574748188],
        [0.058237563610462, 0.11416898574748188],
    ]
)

# Radii either side of the series window |1 - x^2| = 0.1 (x = 0.9487, 1.0488) and
# ordinary radii, where the closed forms are still used.
ETA_EDGE = np.array([0.5, 0.9, 0.9486, 0.95, 1.048, 1.05, 1.1, 2.0])

REFERENCE_EDGE = np.array(
    [
        [0.1301984453911946, 0.10322120446710507],
        [0.0669722093312832, 0.11368408516606135],
        [0.0625134119331428, 0.11398619905636163],
        [0.06239122458760072, 0.11399295691660491],
        [0.05460346625556571, 0.1142283319317603],
        [0.05445894382740638, 0.11422867097074742],
        [0.05101108972901384, 0.1141862710089604],
        [0.01911526041154153, 0.1049167043998897],
    ]
)

# float32 radii near r_s (the float64 radii above round to 1 in float32), and the
# 50-digit deflection at each float32 value.
ETA_FLOAT32 = np.array(
    [1.0, 1.0004, 0.9999, 1.0001, 0.99995, 1.00005, 0.99999, 1.00001, 0.999999, 1.000001],
    dtype=np.float32,
)

REFERENCE_FLOAT32 = np.array(
    [
        0.11416898574748188,
        0.11416990436746469,
        0.11416875486265347,
        0.11416921616752485,
        0.1141688703631735,
        0.11416910087827556,
        0.1141689626524195,
        0.1141690088378853,
        0.1141689834106934,
        0.11416898794676865,
    ]
)

# 50-digit d(deflection)/dr at r = r_s.
REFERENCE_DERIVATIVE_AT_SCALE_RADIUS = 0.00230614147344196


def _profile():
    return ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0),
        kappa_s=KAPPA_S,
        scale_radius=SCALE_RADIUS,
        truncation_radius=TRUNCATION_RADIUS,
    )


def _grid(eta):
    return ag.Grid2DIrregular([(0.0, float(e * SCALE_RADIUS)) for e in eta])


def test__convergence_2d_from__near_scale_radius__matches_reference():
    convergence = np.asarray(_profile().convergence_2d_from(grid=_grid(ETA_NEAR)))

    assert np.all(np.isfinite(convergence))
    np.testing.assert_allclose(convergence, REFERENCE_NEAR[:, 0], rtol=1.0e-10, atol=0.0)


def test__convergence_2d_from__near_scale_radius__continuous_through_scale_radius():
    """
    Within 1e-7 r_s of the scale radius the convergence changes by under 1e-6 of its
    value (its logarithmic slope there is about -1.4), so every point must agree with
    the value at r = r_s to 1e-6.  Further out the profile's own slope moves it more.
    """
    convergence = np.asarray(_profile().convergence_2d_from(grid=_grid(ETA_NEAR)))

    close = np.abs(ETA_NEAR - 1.0) < 1.5e-7

    assert close.sum() == 19
    np.testing.assert_allclose(
        convergence[close], convergence[0], rtol=1.0e-6, atol=0.0
    )
    assert convergence[0] == pytest.approx(REFERENCE_NEAR[0, 0], rel=1.0e-12)


def test__deflections_yx_2d_from__near_scale_radius__numpy_float64():
    deflections = np.asarray(_profile().deflections_yx_2d_from(grid=_grid(ETA_NEAR)))

    assert np.all(deflections[:, 0] == 0.0)
    np.testing.assert_allclose(
        deflections[:, 1], REFERENCE_NEAR[:, 1], rtol=1.0e-10, atol=0.0
    )


def test__radial_deflection_from__near_scale_radius__numpy_float64():
    deflection = ag.mp.NFWTruncatedSph.radial_deflection_from(
        ETA_NEAR * SCALE_RADIUS,
        np.array([KAPPA_S, SCALE_RADIUS, TRUNCATION_RADIUS]),
        np,
    )

    np.testing.assert_allclose(deflection, REFERENCE_NEAR[:, 1], rtol=1.0e-10, atol=0.0)


def test__series_window_edges_and_ordinary_radii__match_reference():
    profile = _profile()

    convergence = np.asarray(profile.convergence_2d_from(grid=_grid(ETA_EDGE)))
    deflections = np.asarray(profile.deflections_yx_2d_from(grid=_grid(ETA_EDGE)))

    np.testing.assert_allclose(convergence, REFERENCE_EDGE[:, 0], rtol=1.0e-10, atol=0.0)
    np.testing.assert_allclose(
        deflections[:, 1], REFERENCE_EDGE[:, 1], rtol=1.0e-10, atol=0.0
    )


def test__shared_coord_funcs_f_and_g__near_scale_radius():
    """
    f(x) and g(x) are shared by the whole NFW family (`NFWSph`, gNFW).  With
    u = 1 - x^2, f = 1 + u/3 + u^2/5 + ... and g = 1/3 + u/5 + u^2/7 + ..., so within
    1e-8 of x = 1 the first two terms are exact to double precision.
    """
    profile = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=KAPPA_S, scale_radius=1.0)

    eta = np.array(
        [1.0] + [1.0 + sign * 10.0**-k for k in range(8, 16) for sign in (1.0, -1.0)]
    )
    u = (1.0 - eta) * (1.0 + eta)

    f = np.real(profile.coord_func_f(grid_radius=eta + 0j))
    g = np.real(profile.coord_func_g(grid_radius=eta + 0j))

    np.testing.assert_allclose(f, 1.0 + u / 3.0, rtol=1.0e-14, atol=0.0)
    np.testing.assert_allclose(g, 1.0 / 3.0 + u / 5.0, rtol=1.0e-14, atol=0.0)

    # Real input keeps a real result.
    assert np.isrealobj(coord_func_f_from(eta))


@requires_jax
def test__radial_deflection_from__near_scale_radius__jax_float32():
    import jax.numpy as jnp

    deflection = ag.mp.NFWTruncatedSph.radial_deflection_from(
        jnp.asarray(ETA_FLOAT32 * SCALE_RADIUS, dtype=jnp.float32),
        jnp.asarray([KAPPA_S, SCALE_RADIUS, TRUNCATION_RADIUS], dtype=jnp.float32),
        jnp,
    )

    assert deflection.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(deflection, dtype=np.float64),
        REFERENCE_FLOAT32,
        rtol=1.0e-4,
        atol=0.0,
    )


@requires_jax
@pytest.mark.parametrize("dtype_name, rtol", [("float64", 1.0e-6), ("float32", 1.0e-3)])
def test__radial_deflection_from__gradient_at_scale_radius(dtype_name, rtol):
    import jax
    import jax.numpy as jnp

    with jax.enable_x64(dtype_name == "float64"):
        dtype = getattr(jnp, dtype_name)
        params = jnp.asarray([KAPPA_S, SCALE_RADIUS, TRUNCATION_RADIUS], dtype=dtype)

        def deflection(r):
            return ag.mp.NFWTruncatedSph.radial_deflection_from(r, params, jnp)

        gradient = jax.grad(deflection)(jnp.asarray(SCALE_RADIUS, dtype=dtype))

        assert float(gradient) == pytest.approx(
            REFERENCE_DERIVATIVE_AT_SCALE_RADIUS, rel=rtol
        )
