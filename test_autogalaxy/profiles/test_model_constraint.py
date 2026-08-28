import inspect

import numpy as np
import pytest

import autofit as af
import autogalaxy as ag
from autogalaxy import convert
from autogalaxy.profiles import validate
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


def _ell_profile_subclasses():
    """Every `EllProfile` subclass reachable from the public `ag.lp` / `ag.mp`
    namespaces, so a profile added later is covered without editing this list."""
    classes = set()
    for namespace in (ag.lp, ag.mp, ag.lmp):
        for _, obj in inspect.getmembers(namespace, inspect.isclass):
            if issubclass(obj, EllProfile):
                classes.add(obj)
    return sorted(classes, key=lambda cls: cls.__name__)


class TestBallDeclaration:
    def test_there_are_profiles_to_check(self):
        """Guards the sweep below: an empty namespace scan would pass vacuously."""
        assert len(_ell_profile_subclasses()) > 20

    def test_every_elliptical_profile_declares_the_ball(self):
        """`ell_comps` has one assignment site, at `EllProfile`, so the ball
        declaration reaches every elliptical light and mass profile — including
        the spherical ones, whose components are pinned and therefore never
        projected."""
        for cls in _ell_profile_subclasses():
            assert cls.__model_ball_constraints__ == (
                (("ell_comps",), convert.ELL_COMPS_MAGNITUDE_CLAMP),
            ), cls.__name__

    def test_the_radius_is_the_clamp_not_the_validity_boundary(self):
        """Between 0.999 and 1.0 the conversion to an axis ratio saturates, so the
        likelihood is flat radially. Projecting onto `1 - margin` would move a
        lane from a region the model rejects into one the optimizer cannot leave;
        projecting onto the clamp puts it where the gradient is alive again."""
        ((_, radius),) = EllProfile.__model_ball_constraints__

        assert radius == convert.ELL_COMPS_MAGNITUDE_CLAMP
        assert radius == 0.999
        assert radius < 1.0

    def test_pyautofit_resolves_the_declaration_to_a_parameter_pair(self):
        """The declaration is only useful if PyAutoFit can turn it into indices
        into the vector a search steps."""
        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(ag.Galaxy, redshift=0.5, mass=ag.mp.Isothermal),
            )
        )

        ((index_0, index_1, radius),) = model.ball_constraint_index_pairs()

        names = [tuple_.name for tuple_ in model.prior_tuples_ordered_by_id]
        assert names[index_0] == "ell_comps_0"
        assert names[index_1] == "ell_comps_1"
        assert radius == convert.ELL_COMPS_MAGNITUDE_CLAMP

    def test_a_spherical_profile_contributes_no_pair(self):
        """`IsothermalSph` inherits the declaration but pins `ell_comps` to
        `(0, 0)`, so there is no free pair to project."""
        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(ag.Galaxy, redshift=0.5, mass=ag.mp.IsothermalSph),
            )
        )

        assert model.ball_constraint_index_pairs() == ()

    def test_the_joint_clipper_projects_a_real_lens_model(self):
        """End to end, with a real profile and PyAutoFit's opt-in clipper: a lane
        at `|e| = 1.4` — inside both `ell_comps` prior boxes, outside the disk —
        comes back inside it."""
        model = af.Collection(
            galaxies=af.Collection(
                lens=af.Model(ag.Galaxy, redshift=0.5, mass=ag.mp.Isothermal),
            )
        )

        ((index_0, index_1, radius),) = model.ball_constraint_index_pairs()

        vector = np.array(model.physical_values_from_prior_medians)
        vector[index_0] = 1.4 / np.sqrt(2.0)
        vector[index_1] = 1.4 / np.sqrt(2.0)

        projected, mask = af.ClipperPriorBoxJoint(margin=0.0).project(
            vector=vector, model=model
        )

        assert np.hypot(projected[index_0], projected[index_1]) == pytest.approx(radius)
        assert mask[index_0]
        assert mask[index_1]


class TestGuardIsUntouched:
    """The ball is a *search-side* projection. `validate_ell_comps`'s
    standalone-construction behaviour is deliberately unchanged: making it fire on
    the traced path would turn a 20%-of-lanes condition into a 20%-of-lanes crash
    in the middle of a multi-hour fit."""

    def test_it_still_rejects_a_magnitude_of_one_on_construction(self):
        with pytest.raises(Exception):
            ag.mp.Isothermal(ell_comps=(1.2, 0.0))

    def test_it_still_rejects_the_corner_the_box_permits(self):
        with pytest.raises(Exception):
            ag.mp.Isothermal(ell_comps=(0.8, 0.8))

    def test_it_still_accepts_the_saturating_annulus(self):
        """0.999 <= magnitude < 1.0 remains constructible. The constraint flags it
        and the clipper projects out of it; the guard does not raise on it, and
        that has not changed."""
        assert ag.mp.Isothermal(ell_comps=(0.9995, 0.0)) is not None

    def test_it_still_returns_early_for_a_non_concrete_magnitude(self):
        """The escape hatch that makes the guard a no-op under a trace, which is
        why the search needed a projection in the first place."""

        class Tracer:
            def __mul__(self, other):
                return self

            __rmul__ = __mul__
            __add__ = __mul__
            __radd__ = __mul__

            def __float__(self):
                raise TypeError("tracer")

        validate.validate_ell_comps(ell_comps=(Tracer(), Tracer()))
