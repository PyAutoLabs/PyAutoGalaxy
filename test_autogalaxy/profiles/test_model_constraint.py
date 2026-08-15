import numpy as np
import pytest

import autogalaxy as ag
from autogalaxy import convert
from autogalaxy.profiles.geometry_profiles import EllProfile


class TestDeclaration:
    def test_every_elliptical_profile_inherits_it(self):
        """`ell_comps` has exactly one assignment site, at `EllProfile`, so the
        declaration reaches every elliptical light and mass profile."""
        for cls in (ag.mp.Isothermal, ag.mp.PowerLaw, ag.lp.Sersic, ag.lp.Exponential):
            assert issubclass(cls, EllProfile)
            assert callable(getattr(cls, "__model_constraint__", None))

    def test_spherical_profiles_inherit_it_and_never_violate(self):
        """Spherical profiles subclass their elliptical counterpart
        (`IsothermalSph` -> `Isothermal` -> ... -> `EllProfile`), so they carry
        the declaration too. Their `ell_comps` are pinned at (0, 0), so it is
        always satisfied — correct, if a few wasted ops."""
        profile = ag.mp.IsothermalSph(einstein_radius=1.0)
        assert profile.ell_comps == (0.0, 0.0)
        assert profile.__model_constraint__() == 0.0


class TestViolation:
    def test_zero_well_inside(self):
        profile = ag.mp.Isothermal(ell_comps=(0.3, 0.2))
        assert profile.__model_constraint__() == 0.0

    def test_zero_just_inside_the_clamp(self):
        profile = ag.mp.Isothermal(ell_comps=(0.99, 0.0))
        assert profile.__model_constraint__() == 0.0

    def test_positive_beyond_the_clamp(self):
        """Constructed inside the guard's valid region (magnitude < 1) but past
        the clamp — the annulus keyed to 0.999 rather than 1.0 exists for."""
        profile = ag.mp.Isothermal(ell_comps=(0.9995, 0.0))
        assert profile.__model_constraint__() > 0.0

    def test_grows_with_distance(self):
        near = ag.mp.Isothermal(ell_comps=(0.9995, 0.0)).__model_constraint__()
        far = ag.mp.Isothermal(ell_comps=(0.99999, 0.0)).__model_constraint__()
        assert far > near

    def test_matches_the_clamp_threshold(self):
        profile = ag.mp.Isothermal(ell_comps=(0.9995, 0.0))
        assert profile.__model_constraint__() == pytest.approx(
            0.9995 - convert.ELL_COMPS_MAGNITUDE_CLAMP
        )


class TestAgreesWithTheGuard:
    """The constraint and `validate_ell_comps` state the same geometry; the
    constraint simply engages earlier, at the clamp rather than at validity."""

    def test_guard_rejects_what_the_constraint_flags_beyond_one(self):
        with pytest.raises(Exception):
            ag.mp.Isothermal(ell_comps=(1.2, 0.0))

    def test_corner_region_is_flagged_and_rejected(self):
        """Both components inside (-1, 1), magnitude above 1 — invisible to any
        per-parameter limit check."""
        with pytest.raises(Exception):
            ag.mp.Isothermal(ell_comps=(0.8, 0.8))

    def test_constraint_covers_the_annulus_the_guard_permits(self):
        """0.999 <= magnitude < 1.0: guard-valid, clamp-saturated."""
        profile = ag.mp.Isothermal(ell_comps=(0.9995, 0.0))
        magnitude_squared = 0.9995**2
        assert magnitude_squared < 1.0  # the guard is satisfied
        assert profile.__model_constraint__() > 0.0  # the constraint is not


class TestClampConstant:
    def test_conversion_saturates_at_the_constant(self):
        at = convert.axis_ratio_from(
            ell_comps=(convert.ELL_COMPS_MAGNITUDE_CLAMP, 0.0)
        )
        beyond = convert.axis_ratio_from(ell_comps=(0.99999, 0.0))
        assert at == pytest.approx(beyond)

    def test_value_is_unchanged_by_the_refactor(self):
        assert convert.ELL_COMPS_MAGNITUDE_CLAMP == 0.999
