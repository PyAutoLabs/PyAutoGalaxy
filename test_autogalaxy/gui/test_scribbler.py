"""
Headless tests of the `Scribbler` GUI's non-interactive logic.

The GUI is built with the Agg backend and `block=False`, so no window opens and no event
loop runs; the callbacks are then driven directly with synthetic events.
"""

from types import SimpleNamespace

from unittest import mock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np
import pytest

import autoarray as aa
import autogalaxy as ag


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _scribbler(shape=(20, 20), values=None, **kwargs):
    values = np.zeros(shape) if values is None else np.asarray(values, dtype=float)
    image = aa.Array2D.no_mask(values=values, pixel_scales=0.1)
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

    def test__mask_from__explicit_proposal_shape_mismatch_raises(self):
        s = _scribbler()

        with pytest.raises(ValueError):
            s.mask_from(proposal=np.zeros((1, 20), dtype=bool))

    def test__proposal_outline_aligns_with_imshow_pixels(self):
        # An asymmetric proposal (rows / cols 2..5) must be outlined over those pixels,
        # i.e. at data coordinates 1.5..5.5 in both x and y, whatever the imshow origin.
        proposal = np.zeros((20, 20), dtype=bool)
        proposal[2:6, 2:6] = True
        s = _scribbler(proposal=proposal)

        vertices = np.concatenate(
            [path.vertices for path in s._proposal_contour.get_paths() if len(path)]
        )
        x, y = vertices[:, 0], vertices[:, 1]

        assert x.min() == pytest.approx(1.5, abs=1e-6)
        assert x.max() == pytest.approx(5.5, abs=1e-6)
        assert y.min() == pytest.approx(1.5, abs=1e-6)
        assert y.max() == pytest.approx(5.5, abs=1e-6)


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

    def test__motion_inside_axes_requests_redraw(self):
        s = _scribbler()

        with mock.patch.object(s.figure.canvas, "draw_idle") as draw_idle:
            s.on_mouse_motion(_event(s, None, None, inaxes=False))
            draw_idle.assert_not_called()

            s.on_mouse_motion(_event(s, 4.0, 6.0))
            draw_idle.assert_called_once()


class TestSideBySidePanels:
    def test__display_holds_two_panels_and_strokes_fold_onto_image_grid(self):
        s = _scribbler(
            shape=(20, 20), brush_width=0.1, subtract_radial=True, panel_gap=6
        )

        assert s.n_panels == 2
        assert s.panel_names == ["radial-subtracted", "as-observed"]
        assert s.display.shape == (20, 46)
        assert np.isnan(s.display[:, 20:26]).all()  # gutter is unpainted
        panels = np.hstack([s.display[:, :20], s.display[:, 26:]])
        assert float(panels.min()) >= 0.0 and float(panels.max()) <= 1.0

        s.add_circle_to_scribble((5.0, 10.0))  # left panel
        s.add_circle_to_scribble((26.0 + 15.0, 3.0))  # right panel, column 15

        masks = s.get_scribble_masks()
        assert masks["1"].shape == (20, 20)
        assert masks["1"][10, 5] and masks["1"][3, 15]
        assert s.mask_from().shape == (20, 20)

    def test__proposal_and_erase_work_on_the_right_panel(self):
        proposal = np.zeros((20, 20), dtype=bool)
        proposal[8:12, 8:12] = True
        s = _scribbler(
            shape=(20, 20), brush_width=0.1, subtract_radial=True, proposal=proposal
        )

        s.set_active_segment(1)
        s.add_circle_to_scribble((26.0 + 10.0, 10.0))  # erase the centre, right panel

        mask = s.mask_from()
        assert not mask[10, 10]
        assert mask[8, 8]

    def test__proposal_outline_aligns_with_imshow_pixels_on_both_panels(self):
        # The asymmetric proposal (rows / cols 2..5) is outlined over those pixels in the
        # left panel and again 20 + 6 = 26 columns to the right in the right panel.
        proposal = np.zeros((20, 20), dtype=bool)
        proposal[2:6, 2:6] = True
        s = _scribbler(subtract_radial=True, panel_gap=6, proposal=proposal)

        vertices = np.concatenate(
            [path.vertices for path in s._proposal_contour.get_paths() if len(path)]
        )
        x, y = vertices[:, 0], vertices[:, 1]
        left, right = x < 20, x >= 26

        assert left.any() and right.any() and (left | right).all()
        assert x[left].min() == pytest.approx(1.5, abs=1e-6)
        assert x[left].max() == pytest.approx(5.5, abs=1e-6)
        assert x[right].min() == pytest.approx(27.5, abs=1e-6)
        assert x[right].max() == pytest.approx(31.5, abs=1e-6)
        assert y.min() == pytest.approx(1.5, abs=1e-6)
        assert y.max() == pytest.approx(5.5, abs=1e-6)

    def test__stroke_at_left_panel_inner_edge_does_not_wrap_onto_column_0(self):
        # 40 px panels, 6 px gutter: the right panel starts at display column 46. An
        # 8 px brush at column 38 reaches column 46, which used to fold back onto
        # image column 0.
        s = _scribbler(shape=(40, 40), subtract_radial=True)
        s._set_brush_radius(8)

        s.add_circle_to_scribble((38.0, 20.0))

        masked_cols = np.flatnonzero(s.mask_from().any(axis=0))
        assert list(masked_cols) == list(range(30, 40))
        assert not s.mask_from()[:, 0].any()

    def test__stroke_at_right_panel_inner_edge_does_not_wrap_onto_last_column(self):
        s = _scribbler(shape=(40, 40), subtract_radial=True)
        s._set_brush_radius(8)

        s.add_circle_to_scribble((46.0 + 1.0, 20.0))  # right panel, image column 1

        masked_cols = np.flatnonzero(s.mask_from().any(axis=0))
        assert list(masked_cols) == list(range(0, 10))
        assert not s.mask_from()[:, 39].any()

    def test__stroke_centred_in_the_gutter_goes_to_the_nearest_panel(self):
        s = _scribbler(shape=(40, 40), subtract_radial=True)
        s._set_brush_radius(8)

        s.add_circle_to_scribble((41.0, 20.0))  # gutter, nearer the left panel
        assert list(np.flatnonzero(s.mask_from().any(axis=0))) == list(range(33, 40))

        s.remove_circle_from_scribble()
        s.add_circle_to_scribble((44.0, 20.0))  # gutter, nearer the right panel
        assert list(np.flatnonzero(s.mask_from().any(axis=0))) == list(range(0, 7))

    def test__subtract_radial_alone_gives_one_panel(self):
        s = _scribbler(shape=(20, 20), subtract_radial=True, side_by_side=False)

        assert s.n_panels == 1
        assert s.panel_names == ["radial-subtracted"]
        assert s.display.shape == (20, 20)

    def test__single_panel_display_is_the_image(self):
        s = _scribbler(shape=(12, 14))

        assert s.n_panels == 1
        assert s.display.shape == (12, 14)


class TestPositionMarkers:
    def test__positions_convert_to_pixels_and_draw_a_closed_cross_each(self):
        # 20 px at 0.1"/px: (y, x) = (+0.5", -0.5") -> row 4.5, column 4.5
        s = _scribbler(shape=(20, 20), positions=[(0.5, -0.5), (0.0, 0.0)])

        pixels = s.positions_pixels()
        assert pixels[0] == pytest.approx([4.5, 4.5])
        assert pixels[1] == pytest.approx([9.5, 9.5])
        assert len(s.position_markers) == 4  # two lines per cross

        # default arm: 1.5% of the shorter side, never below 3 px
        assert s.position_marker_size == 3
        xs, ys = s.position_markers[
            0
        ].get_data()  # vertical arm through the first position
        assert list(xs) == pytest.approx([4.5, 4.5])
        assert list(ys) == pytest.approx([4.5 - 3, 4.5 + 3])
        xs, ys = s.position_markers[1].get_data()  # horizontal arm
        assert list(xs) == pytest.approx([4.5 - 3, 4.5 + 3])
        assert list(ys) == pytest.approx([4.5, 4.5])

    def test__marker_size_scales_with_the_image_or_is_explicit(self):
        assert (
            _scribbler(shape=(400, 400), positions=[(0.0, 0.0)]).position_marker_size
            == 6
        )
        s = _scribbler(shape=(20, 20), positions=[(0.0, 0.0)], position_marker_size=5)
        _, ys = s.position_markers[0].get_data()
        assert list(ys) == pytest.approx([9.5 - 5, 9.5 + 5])

    def test__markers_repeat_on_every_panel_and_never_enter_the_mask(self):
        s = _scribbler(shape=(20, 20), positions=[(0.0, 0.0)], subtract_radial=True)

        assert len(s.position_markers) == 4
        xs, _ = s.position_markers[2].get_data()  # vertical arm of the right-hand copy
        assert list(xs) == pytest.approx([9.5 + 26, 9.5 + 26])
        assert not s.mask_from().any()

    def test__grid_irregular_and_origin_are_honoured(self):
        image = aa.Array2D.no_mask(
            values=np.zeros((10, 10)), pixel_scales=0.2, origin=(1.0, -1.0)
        )
        s = ag.Scribbler(
            image=image.native,
            backend="Agg",
            block=False,
            positions=aa.Grid2DIrregular(values=[(1.0, -1.0)]),
        )

        assert s.positions_pixels()[0] == pytest.approx([4.5, 4.5])

    def test__no_positions_draws_nothing(self):
        assert _scribbler().position_markers == []
        assert _scribbler(positions=[]).position_markers == []


class TestPanelStretch:
    def test__arcsinh_default_keeps_faint_structure_visible_beside_a_bright_core(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0.0, 1.0, (40, 40))
        values[20, 20] = 1e4  # a core thousands of sigma above the sky
        values[5, 5] = 4.0  # a faint companion

        s = _scribbler(shape=(40, 40), values=values, subtract_radial=True)
        right = s.display[:, 46:]  # the as-observed panel

        assert s.stretch == "arcsinh"
        assert right[20, 20] == 1.0
        # the companion is clearly separated from the sky, not crushed onto it
        assert right[5, 5] - np.median(right) > 0.1

    def test__linear_stretch_reproduces_the_old_scaling(self):
        values = np.zeros((40, 40))
        values[20, 20] = 1e4
        values[5, 5] = 4.0

        s = _scribbler(
            shape=(40, 40), values=values, subtract_radial=True, stretch="linear"
        )
        right = s.display[:, 46:]

        assert right[20, 20] == pytest.approx(1.0)
        assert right[5, 5] == pytest.approx(4e-4, abs=1e-6)

    def test__single_panel_is_stretched_too_and_limits_are_honoured(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0.0, 1.0, (40, 40))
        values[20, 20] = 1e4
        values[5, 5] = 4.0

        s = _scribbler(shape=(40, 40), values=values)
        assert s.display.shape == (40, 40)
        assert s.display[20, 20] == 1.0
        assert s.display[5, 5] - np.median(s.display) > 0.1

        # vmax below the core: the core saturates and the companion sits at the top too
        s = _scribbler(shape=(40, 40), values=values, vmin=0.0, vmax=4.0)
        assert s.display[20, 20] == 1.0
        assert s.display[5, 5] == pytest.approx(1.0)

        # linear keeps the raw image as the display, as before
        s = _scribbler(shape=(40, 40), values=values, stretch="linear")
        assert s.display[20, 20] == 1e4

    def test__unknown_stretch_is_rejected(self):
        with pytest.raises(ValueError):
            _scribbler(subtract_radial=True, stretch="sqrt")


class TestBrushColour:
    def test__brush_ring_and_halo_follow_the_active_brush(self):
        s = _scribbler(shape=(20, 20))
        s.on_mouse_motion(_event(s, 4.0, 6.0))

        assert s.brush_color == "w" and s.brush_halo_color == "k"
        assert s.brush.get_edgecolor()[:3] == pytest.approx((1.0, 1.0, 1.0))
        assert s.brush_halo.get_edgecolor()[:3] == pytest.approx((0.0, 0.0, 0.0))
        assert s.brush_halo.center == s.brush.center

        s.set_active_segment(1)
        assert s.brush_color == "k" and s.brush_halo_color == "w"
        assert s.brush.get_edgecolor()[:3] == pytest.approx((0.0, 0.0, 0.0))
        assert s.brush_halo.get_edgecolor()[:3] == pytest.approx((1.0, 1.0, 1.0))

        s.on_keypress(_event(s, 4.0, 6.0, key="="))
        assert s.brush_halo.radius == s.brush.radius == s.brush_radius

    def test__strokes_are_painted_white_to_add_and_black_to_erase(self):
        s = _scribbler(shape=(20, 20))
        s.add_circle_to_scribble((5.0, 5.0))
        s.set_active_segment(1)
        s.add_circle_to_scribble((15.0, 15.0))

        add, erase = (list(v) for v in s.scribbles.values())
        assert add[0].get_facecolor()[:3] == pytest.approx((1.0, 1.0, 1.0))
        assert erase[0].get_facecolor()[:3] == pytest.approx((0.0, 0.0, 0.0))

    def test__explicit_brush_colour_is_kept(self):
        s = _scribbler(shape=(20, 20), brush_color="w")
        s.set_active_segment(1)
        assert s.brush_color == "w"


class TestTitle:
    def test__title_lists_what_is_shown(self):
        s = _scribbler()
        assert s.title_text == s.KEY_LEGEND
        assert s.ax.get_title() == s.KEY_LEGEND

        proposal = np.zeros((20, 20), dtype=bool)
        proposal[5:8, 5:8] = True
        overlay = aa.Mask2D.circular(
            shape_native=(20, 20), pixel_scales=0.1, radius=0.5
        )
        s = _scribbler(
            title="Mask extra galaxies: F277W, proposing from F150W",
            subtract_radial=True,
            proposal=proposal,
            mask_overlay=overlay,
            positions=[(0.0, 0.0)],
        )
        lines = s.title_text.split("\n")
        assert lines[0] == "Mask extra galaxies: F277W, proposing from F150W"
        assert lines[1].startswith("LEFT: radial-subtracted")
        assert "outline = the mask being refined" in lines[2]
        assert "black x = edge of the overlaid mask" in lines[2]
        assert "+ = marked positions" in lines[2]
        assert lines[3] == s.KEY_LEGEND
        assert s.ax.get_title() == s.title_text
