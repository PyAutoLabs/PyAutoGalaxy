import numpy as np
import pytest
from matplotlib.colors import LogNorm, Normalize

import autogalaxy as ag
from autogalaxy.util.plot_utils import norm_from


class TestNormFrom:
    def test__no_arguments__returns_none(self):
        assert norm_from(array=np.array([1.0, 2.0, 3.0])) is None

    def test__vmin_vmax__returns_linear_norm(self):
        norm = norm_from(array=np.array([1.0, 2.0, 3.0]), vmin=0.5, vmax=2.5)

        assert isinstance(norm, Normalize)
        assert not isinstance(norm, LogNorm)
        assert norm.vmin == pytest.approx(0.5)
        assert norm.vmax == pytest.approx(2.5)

    def test__use_log10__returns_log_norm_with_explicit_limits(self):
        norm = norm_from(
            array=np.array([1.0, 2.0, 3.0]), use_log10=True, vmin=1.0e-3, vmax=1.0
        )

        assert isinstance(norm, LogNorm)
        assert norm.vmin == pytest.approx(1.0e-3)
        assert norm.vmax == pytest.approx(1.0)

    def test__use_log10__vmax_derived_from_array(self):
        norm = norm_from(array=np.array([1.0, 2.0, 7.0]), use_log10=True, vmin=1.0e-3)

        assert isinstance(norm, LogNorm)
        assert norm.vmax == pytest.approx(7.0)

    def test__use_log10__degenerate_range_is_widened(self):
        """vmax <= vmin would make LogNorm unusable, so it is pushed up a decade."""
        norm = norm_from(
            array=np.array([1.0, 2.0]), use_log10=True, vmin=10.0, vmax=1.0
        )

        assert norm.vmin == pytest.approx(10.0)
        assert norm.vmax == pytest.approx(100.0)

    @pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
    def test__use_log10__all_nan_array_still_yields_finite_limits(self):
        norm = norm_from(array=np.array([np.nan, np.nan]), use_log10=True)

        assert np.isfinite(norm.vmin)
        assert np.isfinite(norm.vmax)
        assert norm.vmax > norm.vmin


class TestPublicNamespace:
    def test__gui_classes_are_exported_once(self):
        assert hasattr(ag, "Scribbler")
        assert hasattr(ag, "Clicker")

    def test__scribbler_accepts_flat_colour_arguments(self):
        """The removed `Cmap` object was the only way to colour these GUIs.

        `Scribbler` must therefore take the colour scale as plain values, since
        no public plot namespace exports a `Cmap` to hand it any more.
        """
        import inspect

        params = inspect.signature(ag.Scribbler.__init__).parameters

        for name in ("cmap", "norm", "vmin", "vmax"):
            assert name in params, f"Scribbler.__init__ is missing `{name}`"

    def test__cmap_is_absent_from_every_public_plot_namespace(self):
        """Regression guard for the defect this module was added for.

        If `Cmap` ever returns to a public namespace, the GUIs' flat arguments
        are no longer the only option and this test should be revisited.
        """
        import autoarray.plot as aaplt
        import autogalaxy.plot as agplt

        assert not hasattr(aaplt, "Cmap")
        assert not hasattr(agplt, "Cmap")
