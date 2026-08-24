import numpy as np
import pytest

import autogalaxy as ag
import autogalaxy.plot as aplt
from autogalaxy.util.plot_utils import _mask_edge


@pytest.fixture
def array():
    return ag.Array2D.no_mask(
        values=np.abs(np.random.rand(30, 30)) + 1.0e-3, pixel_scales=0.1
    )


@pytest.fixture
def mask(array):
    return ag.Mask2D.circular(
        shape_native=array.shape_native, pixel_scales=array.pixel_scales, radius=1.0
    )


class TestMaskEdge:
    def test__none_returns_none(self):
        assert _mask_edge(None) is None

    def test__fully_unmasked_returns_none(self, array):
        """Nothing to outline, matching `auto_mask_edge`'s contract."""
        assert _mask_edge(array.mask) is None

    def test__mask2d_returns_edge_coordinates(self, mask):
        edge = _mask_edge(mask)

        assert isinstance(edge, np.ndarray)
        assert edge.ndim == 2 and edge.shape[1] == 2
        assert len(edge) > 0

    def test__raw_coordinates_pass_through(self):
        coords = np.array([[1.0, 2.0], [3.0, 4.0]])

        assert _mask_edge(coords) == pytest.approx(coords)


class TestPlotArrayMaskOverlay:
    def test__unmasked_array_derives_no_overlay(self, array):
        """The reason an explicit `mask=` is needed at all.

        `plot_array` auto-derives the outline from the array's own mask, which
        yields nothing for an unmasked array — so a caller wanting a *different*
        mask outlined must pass it.
        """
        from autoarray.plot.utils import auto_mask_edge

        assert auto_mask_edge(array) is None

    def test__explicit_mask_changes_the_rendered_figure(self, array, mask, tmp_path):
        import matplotlib.image as mpimg

        aplt.plot_array(
            array=array,
            output_path=str(tmp_path),
            output_filename="without",
            output_format="png",
        )
        aplt.plot_array(
            array=array,
            mask=mask,
            output_path=str(tmp_path),
            output_filename="with",
            output_format="png",
        )

        without = mpimg.imread(tmp_path / "without.png")
        with_mask = mpimg.imread(tmp_path / "with.png")

        assert without.shape == with_mask.shape
        differing = np.any(np.abs(without - with_mask) > 1.0e-6, axis=-1).sum()
        assert differing > 0, "the mask outline was not drawn"

    def test__mask_is_optional(self, array, tmp_path):
        aplt.plot_array(
            array=array,
            output_path=str(tmp_path),
            output_filename="no_mask",
            output_format="png",
        )

        assert (tmp_path / "no_mask.png").exists()
