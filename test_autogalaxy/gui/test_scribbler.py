"""
Headless tests of the `Scribbler` GUI's non-interactive logic.

The GUI is built with the Agg backend and `block=False`, so no window opens and no event
loop runs; the callbacks are then driven directly with synthetic events.
"""

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import autoarray as aa
import autogalaxy as ag


def _scribbler(shape=(20, 20), **kwargs):
    image = aa.Array2D.no_mask(values=np.zeros(shape), pixel_scales=0.1)
    return ag.Scribbler(image=image.native, backend="Agg", block=False, **kwargs)


def _event(scribbler, x, y, inaxes=True, key=None):
    return SimpleNamespace(
        inaxes=scribbler.ax if inaxes else None, xdata=x, ydata=y, key=key
    )


class TestScribbles:
    def test__circle_rasterises_into_add_segment(self):
        s = _scribbler(brush_width=0.1)  # 2 px brush on a 20 px image

        s.add_circle_to_scribble((10.0, 5.0))  # (x=col, y=row)

        masks = s.get_scribble_masks()
        assert masks["1"][5, 10]
        assert masks["1"][5, 12] and masks["1"][7, 10]
        assert not masks["1"][5, 13]
        assert not masks["2"].any()
        assert masks["1"].shape == (20, 20)

    def test__circle_at_zero_coordinate_is_not_dropped(self):
        s = _scribbler(brush_width=0.1)

        s.add_circle_to_scribble((0.0, 0.0))

        assert s.get_scribble_masks()["1"][0, 0]

    def test__undo_removes_last_circle(self):
        s = _scribbler(brush_width=0.1)
        s.add_circle_to_scribble((10.0, 5.0))
        s.add_circle_to_scribble((15.0, 15.0))

        s.on_keypress(_event(s, None, None, key="z"))

        masks = s.get_scribble_masks()
        assert masks["1"][5, 10]
        assert not masks["1"][15, 15]
        assert s.num_patches == 1

    def test__mouse_down_and_drag_paint__outside_axes_ignored(self):
        s = _scribbler(brush_width=0.1)

        s.on_mouse_down(_event(s, 10.0, 10.0))
        s.on_mouse_motion(_event(s, 14.0, 10.0))
        s.on_mouse_up(_event(s, 14.0, 10.0))
        s.on_mouse_down(_event(s, 3.0, 3.0, inaxes=False))
        s.on_mouse_up(_event(s, 3.0, 3.0, inaxes=False))

        masks = s.get_scribble_masks()
        assert masks["1"][10, 10] and masks["1"][10, 14]
        assert not masks["1"][3, 3]
        assert not s.mouse_is_down


class TestEraseAndProposal:
    def test__mask_from__is_add_minus_erase(self):
        s = _scribbler(brush_width=0.1)
        s.add_circle_to_scribble((10.0, 10.0))
        s.on_keypress(_event(s, None, None, key="2"))
        s.add_circle_to_scribble((12.0, 10.0))

        masks = s.get_scribble_masks()
        mask = s.mask_from()

        assert masks["1"][10, 10] and masks["2"][10, 12]
        assert mask[10, 8]  # added, not erased
        assert not mask[10, 12]  # erased
        assert np.array_equal(mask, masks["1"] & ~masks["2"])
        assert s.show_mask() is not None
        assert np.array_equal(s.show_mask(), masks["1"])

    def test__proposal_is_refined_by_both_brushes(self):
        proposal = np.zeros((20, 20), dtype=bool)
        proposal[2:6, 2:6] = True
        s = _scribbler(brush_width=0.1, proposal=proposal)

        s.add_circle_to_scribble((15.0, 15.0))  # add far from the proposal
        s.set_active_segment(1)
        s.add_circle_to_scribble((3.0, 3.0))  # erase inside it

        mask = s.mask_from()

        assert mask[15, 15]
        assert not mask[3, 3]
        assert mask[5, 5]  # proposal pixel outside the erase circle survives
        masks = s.get_scribble_masks()
        assert np.array_equal(mask, (proposal | masks["1"]) & ~masks["2"])

    def test__mask_from__explicit_proposal_overrides_constructor(self):
        s = _scribbler()
        other = np.ones((20, 20), dtype=bool)

        assert s.mask_from(proposal=other).all()
        assert not s.mask_from().any()

    def test__proposal_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            _scribbler(proposal=np.zeros((10, 10), dtype=bool))


class TestBrush:
    def test__resize_is_multiplicative_with_floor(self):
        s = _scribbler(shape=(100, 100), brush_width=0.1, brush_resize_factor=1.4)
        assert s.brush_radius == 10

        s.on_keypress(_event(s, None, None, key="="))
        assert s.brush_radius == 14
        s.on_keypress(_event(s, None, None, key="-"))
        assert s.brush_radius == 10

        for _ in range(20):
            s.shrink_brush()
        assert s.brush_radius == 1
        s.enlarge_brush()
        assert s.brush_radius == 2  # at least +1 px even when the factor rounds to 0

    def test__min_radius_is_respected_from_construction(self):
        s = _scribbler(shape=(10, 10), brush_width=0.01, min_radius=3)

        assert s.brush_radius == 3

    def test__motion_outside_axes_does_not_raise_and_brush_tracks_cursor(self):
        s = _scribbler()

        s.on_mouse_motion(_event(s, None, None, inaxes=False))
        assert s.brush is None

        s.on_mouse_motion(_event(s, 4.0, 6.0))
        assert s.brush is not None
        assert s.brush.center == (4.0, 6.0)

        s.on_mouse_motion(_event(s, 8.0, 9.0))
        assert s.brush.center == (8.0, 9.0)
