"""
Regression tests for PyAutoGalaxy#440 — profile constructor validation (B9, B11,
B12) and the B10 Ell/Sph tolerance pin.

The guard tests are built from @rhayes777's own snippets in the issue body and
assert the *failure*: that the input is rejected at construction with a message
naming the offending parameter, rather than accepted and surfaced as an all-NaN
array or a bare ZeroDivisionError several calls later.

Each finding is paired with a control asserting the valid input still works, so a
guard cannot pass by rejecting everything.

Tests are numpy-only, per phase 1. Tracer-safety is asserted against the
concreteness gate the guards branch on (`autoarray.validate.is_concrete_scalar`)
rather than by importing JAX into the library unit tests.
"""

import numpy as np
import pytest

import autogalaxy as ag
from autogalaxy.profiles import validate


class _NotAConcreteScalar:
    """
    Stand-in for a JAX tracer: not a concrete Python/NumPy scalar, and raises if
    anything resolves it to a bool — exactly as a tracer does inside `jax.jit`.
    """

    def __bool__(self):
        raise AssertionError(
            "a guard compared a non-concrete value — the TracerBoolConversionError path"
        )

    def __lt__(self, other):
        return self

    def __le__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __add__(self, other):
        return self


# ======================================================================================
# B9 — scale_radius must be finite and positive
# ======================================================================================


@pytest.mark.parametrize("scale_radius", [0.0, -1.0, float("nan"), float("inf")])
def test__b9__nfw_rejects_non_positive_or_non_finite_scale_radius(scale_radius):
    with pytest.raises(ValueError, match="scale_radius"):
        ag.mp.NFW(scale_radius=scale_radius)


@pytest.mark.parametrize(
    "profile_cls", [ag.mp.NFW, ag.mp.NFWSph, ag.mp.gNFW, ag.mp.gNFWSph, ag.mp.cNFW]
)
def test__b9__every_nfw_family_profile_rejects_a_zero_scale_radius(profile_cls):
    """The reporter named `NFW`; the same hole was open across the halo family."""
    with pytest.raises(ValueError, match="scale_radius"):
        profile_cls(scale_radius=0.0)


def test__b9__control__a_positive_scale_radius_builds_and_returns_finite_deflections():
    grid = ag.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.1)

    deflections = ag.mp.NFW(scale_radius=1.0).deflections_yx_2d_from(grid=grid)

    assert np.isfinite(np.asarray(deflections)).all()


# ======================================================================================
# B11 — sersic_index must be finite and positive
# ======================================================================================


@pytest.mark.parametrize("sersic_index", [0.0, -1.0, float("nan"), float("inf")])
def test__b11__sersic_rejects_non_positive_or_non_finite_sersic_index(sersic_index):
    with pytest.raises(ValueError, match="sersic_index"):
        ag.lp.Sersic(sersic_index=sersic_index)


def test__b11__the_stellar_mass_sersic_is_guarded_too():
    with pytest.raises(ValueError, match="sersic_index"):
        ag.mp.Sersic(sersic_index=0.0)


def test__b11__control__a_normal_sersic_index_still_produces_a_finite_image():
    grid = ag.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.1)

    image = ag.lp.Sersic(sersic_index=1.0).image_2d_from(grid=grid)

    assert np.isfinite(np.asarray(image)).all()


# ======================================================================================
# B12 — ell_comps must lie inside the unit circle
# ======================================================================================


@pytest.mark.parametrize("ell_comps", [(2.0, 0.0), (0.0, 2.0), (0.9, 0.9), (1.0, 0.0)])
def test__b12__elliptical_profiles_reject_ell_comps_of_magnitude_one_or_above(
    ell_comps,
):
    """
    `q = (1 - f) / (1 + f)` is a valid axis ratio only for `f < 1`. At `f == 1` the
    ellipse degenerates to `q == 0`; beyond it `q` is negative and meaningless.
    """
    with pytest.raises(ValueError, match="ell_comps"):
        ag.lp.Sersic(ell_comps=ell_comps)


def test__b12__the_guard_is_on_the_shared_elliptical_base_not_one_subclass():
    """`EllProfile` is the single base every elliptical light and mass profile uses."""
    with pytest.raises(ValueError, match="ell_comps"):
        ag.mp.Isothermal(ell_comps=(2.0, 0.0))

    with pytest.raises(ValueError, match="ell_comps"):
        ag.lp.Gaussian(ell_comps=(2.0, 0.0))

    with pytest.raises(ValueError, match="ell_comps"):
        ag.mp.NFW(ell_comps=(2.0, 0.0), scale_radius=1.0)


def test__b12__the_message_reports_the_magnitude():
    with pytest.raises(ValueError) as error:
        ag.lp.Sersic(ell_comps=(3.0, 4.0))

    assert "5.0" in str(error.value)


@pytest.mark.parametrize("ell_comps", [(0.0, 0.0), (0.3, 0.4), (0.0, 0.9)])
def test__b12__control__ell_comps_inside_the_unit_circle_still_build(ell_comps):
    profile = ag.lp.Sersic(ell_comps=ell_comps)

    assert 0.0 < profile.axis_ratio() <= 1.0


# ======================================================================================
# Negative redshift (the PyAutoLens#532 half that lives in this repo)
# ======================================================================================
#
# Filed on PyAutoLens#532 because the reporter reached it through `al.Galaxy`, but
# `al.Galaxy` IS `ag.Galaxy` — the class and its `redshift` assignment live here, so
# the guard belongs here. The `Tracer(galaxies=...)` half of that issue is in
# PyAutoLens.


@pytest.mark.parametrize("redshift", [-0.5, -1.0, float("nan"), float("inf")])
def test__galaxy_rejects_a_negative_or_non_finite_redshift(redshift):
    with pytest.raises(ValueError, match="redshift"):
        ag.Galaxy(redshift=redshift)


def test__control__zero_and_tiny_redshifts_are_still_accepted():
    """
    Zero places a galaxy at the observer and is legitimate. `1e-12` was flagged by the
    reporter as degenerate, but it is not *invalid* — rejecting it would break
    single-plane work where the redshift is a label, so it stays accepted.
    """
    assert ag.Galaxy(redshift=0.0).redshift == 0.0
    assert ag.Galaxy(redshift=1e-12).redshift == 1e-12


def test__control__a_lens_redshift_above_the_source_redshift_still_constructs():
    """
    PHASE 4 GUARD-RAIL — deliberately pinning today's permissive behaviour.

    `z_lens > z_source` must NOT raise: multi-plane lensing genuinely supports
    geometries that look inverted under two-plane naming. Whether it should even
    *warn* is an open question put to @rhayes777 on PyAutoLens#532. This test exists
    so that phase 4 cannot quietly turn it into an error.
    """
    lens = ag.Galaxy(redshift=1.0, mass=ag.mp.IsothermalSph(einstein_radius=1.0))
    source = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))

    assert lens.redshift == 1.0
    assert source.redshift == 0.5


# ======================================================================================
# Tracer safety
# ======================================================================================


def test__guards_pass_through_non_concrete_values__never_compare_them():
    tracer_like = _NotAConcreteScalar()

    validate.validate_scale_radius(scale_radius=tracer_like)
    validate.validate_sersic_index(sersic_index=tracer_like)
    validate.validate_ell_comps(ell_comps=(tracer_like, tracer_like))


def test__profile_constructors_accept_tracer_like_parameters():
    """Profile parameters are free model parameters; under a trace they arrive traced."""
    tracer_like = _NotAConcreteScalar()

    assert ag.mp.NFW(scale_radius=tracer_like).scale_radius is tracer_like
    assert ag.lp.Sersic(sersic_index=tracer_like).sersic_index is tracer_like
    assert ag.lp.Sersic(ell_comps=(tracer_like, tracer_like)).ell_comps == (
        tracer_like,
        tracer_like,
    )


# ======================================================================================
# B10 — Isothermal(ell_comps=(0,0)) vs IsothermalSph agreement, pinned at a tolerance
# ======================================================================================
#
# These two are analytically identical: the elliptical form at zero ellipticity IS the
# spherical form. Numerically they differ, because the `Ell` form takes a different
# evaluation route even at the degenerate point.
#
# This is a TOLERANCE PIN, not a bug fix. Bit-identity is explicitly not the goal — the
# point is that a future refactor which makes the agreement materially worse gets
# caught. Measured values on `main` at the time of writing are in each test.


def _isothermal_pair():
    ell = ag.mp.Isothermal(ell_comps=(0.0, 0.0), einstein_radius=1.0)
    sph = ag.mp.IsothermalSph(einstein_radius=1.0)
    return ell, sph


def test__b10__deflections_agree_between_elliptical_and_spherical_isothermal():
    """Measured max|diff| = 2.357e-06 (relative 2.36e-06). Pinned an order looser."""
    grid = ag.Grid2D.uniform(shape_native=(40, 40), pixel_scales=0.1)
    ell, sph = _isothermal_pair()

    difference = np.max(
        np.abs(
            np.asarray(ell.deflections_yx_2d_from(grid=grid))
            - np.asarray(sph.deflections_yx_2d_from(grid=grid))
        )
    )

    assert difference < 1.0e-5


def test__b10__convergence_agrees_between_elliptical_and_spherical_isothermal():
    """Measured max|diff| = 1.207e-05 (relative 1.45e-06). Pinned an order looser."""
    grid = ag.Grid2D.uniform(shape_native=(40, 40), pixel_scales=0.1)
    ell, sph = _isothermal_pair()

    difference = np.max(
        np.abs(
            np.asarray(ell.convergence_2d_from(grid=grid))
            - np.asarray(sph.convergence_2d_from(grid=grid))
        )
    )

    assert difference < 1.0e-4


def test__b10__potential_agrees_between_elliptical_and_spherical_isothermal():
    """
    Measured max|diff| = 5.375e-03, which is a **relative** difference of 1.9e-03 —
    three orders of magnitude worse than the deflection and convergence agreement
    above, and NOT part of @rhayes777's original B10 report (he measured deflections
    only).

    This tolerance is therefore deliberately pinned at the currently-observed level
    rather than at a level anyone has argued is scientifically acceptable. It is a
    ratchet: it stops the agreement degrading further, and it is expected to be
    tightened when the potential discrepancy is investigated on its own.
    """
    grid = ag.Grid2D.uniform(shape_native=(40, 40), pixel_scales=0.1)
    ell, sph = _isothermal_pair()

    difference = np.max(
        np.abs(
            np.asarray(ell.potential_2d_from(grid=grid))
            - np.asarray(sph.potential_2d_from(grid=grid))
        )
    )

    assert difference < 1.0e-2
