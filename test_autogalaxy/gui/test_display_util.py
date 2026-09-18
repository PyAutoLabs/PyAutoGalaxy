import numpy as np
import pytest

import autoarray as aa
from autogalaxy.gui import display_util as du


def _radius_grid(shape):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.hypot(yy - (shape[0] - 1) / 2.0, xx - (shape[1] - 1) / 2.0)


class TestRadialMedianSubtract:
    def test__pure_radial_profile_is_removed(self):
        r = _radius_grid((41, 41))
        image = np.exp(-r.astype(int) / 5.0)  # constant within each integer radius bin

        residual = du.radial_median_subtract(image)

        assert residual == pytest.approx(np.zeros_like(image), abs=1e-12)

    def test__off_centre_source_survives_and_nans_are_ignored(self):
        r = _radius_grid((41, 41))
        image = np.exp(-r.astype(int) / 5.0)
        image[5, 30] += 3.0
        image[0, 0] = np.nan

        residual = du.radial_median_subtract(image)

        assert residual[5, 30] == pytest.approx(3.0, abs=1e-3)
        assert np.isnan(residual[0, 0])

    def test__exclude_keeps_contaminant_out_of_ring_median(self):
        r = _radius_grid((21, 21))
        image = np.exp(-r.astype(int) / 5.0)
        ring = r.astype(int) == 8
        # a contaminant covering most of one ring drags that ring's median up...
        contaminant = ring & (np.mgrid[0:21, 0:21][1] > 6)
        image[contaminant] += 10.0

        biased = du.radial_median_subtract(image)
        clean = du.radial_median_subtract(image, exclude=contaminant)

        untouched = ring & ~contaminant
        assert np.abs(biased[untouched]).max() > 1.0  # ...stamping the whole annulus
        assert np.abs(clean[untouched]).max() == pytest.approx(0.0, abs=1e-12)
        assert clean[contaminant].min() > 9.0


class TestMaskBoundary:
    def test__one_pixel_inner_outline(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:7, 2:7] = True

        edge = du.mask_boundary(mask)

        assert edge.sum() == 16
        assert edge[2, 2] and edge[6, 6] and edge[2, 4]
        assert not edge[3, 3] and not edge[4, 4]
        assert not edge[1, 1]  # outline lies on masked pixels only

    def test__mask_at_array_edge_is_not_outlined_along_edge(self):
        mask = np.zeros((6, 6), dtype=bool)
        mask[0:3, :] = True

        edge = du.mask_boundary(mask)

        assert edge[2, :].all()
        assert not edge[0, :].any()

    def test__empty_mask_gives_empty_outline(self):
        assert not du.mask_boundary(np.zeros((4, 4), dtype=bool)).any()


class TestPanels:
    def test__composite_then_fold_round_trips(self):
        a = np.zeros((5, 8))
        b = np.ones((5, 8))

        composite = du.composite_panels([a, b], gap=3)

        assert composite.shape == (5, 19)
        assert composite[:, 8:11].sum() == 0.0
        assert composite[:, 11:].all()

        scribbled = np.zeros((5, 19), dtype=bool)
        scribbled[1, 2] = True  # left panel
        scribbled[3, 11 + 6] = True  # right panel, column 6
        scribbled[4, 9] = True  # gutter: ignored

        folded, per_panel = du.fold_panels(scribbled, n_panels=2, panel_n_x=8, gap=3)

        assert folded.shape == (5, 8)
        assert folded[1, 2] and folded[3, 6]
        assert folded.sum() == 2
        assert per_panel == [1, 1]

    def test__single_panel_is_identity(self):
        a = np.arange(6.0).reshape(2, 3)

        assert du.composite_panels([a]) is a
        folded, per_panel = du.fold_panels(a > 2, n_panels=1, panel_n_x=3)
        assert np.array_equal(folded, a > 2)
        assert per_panel == [3]


class TestMaskRegrid:
    def test__centred_disc_between_grids_of_different_pixel_scale(self):
        # cosmos_web_ring-like: 0.03"/px 419 px  <->  0.06"/px 209 px
        r_src = _radius_grid((419, 419)) * 0.03
        src = r_src < 2.0

        dst = du.mask_regridded_from(
            src,
            src_pixel_scales=0.03,
            dst_shape_native=(209, 209),
            dst_pixel_scales=0.06,
        )

        r_dst = _radius_grid((209, 209)) * 0.06
        assert dst.shape == (209, 209)
        assert dst[104, 104]
        assert np.array_equal(dst, r_dst < 2.0)
        assert dst.sum() * 0.06**2 == pytest.approx(src.sum() * 0.03**2, rel=0.01)

    def test__upsampling_and_origin_shift(self):
        src = np.zeros((4, 4), dtype=bool)
        src[0, 3] = True  # top-right pixel: y in (+1, +2), x in (+1, +2) at 1"/px

        dst = du.mask_regridded_from(
            src, src_pixel_scales=1.0, dst_shape_native=(8, 8), dst_pixel_scales=0.5
        )

        assert dst.sum() == 4
        assert dst[0:2, 6:8].all()

        # shift the destination origin by (+1, +1): the masked region moves down/left
        shifted = du.mask_regridded_from(
            src,
            src_pixel_scales=1.0,
            dst_shape_native=(8, 8),
            dst_pixel_scales=0.5,
            dst_origin=(1.0, 1.0),
        )

        assert shifted.sum() == 4
        assert shifted[2:4, 4:6].all()

    def test__outside_source_footprint_is_unmasked(self):
        src = np.ones((4, 4), dtype=bool)

        dst = du.mask_regridded_from(
            src, src_pixel_scales=1.0, dst_shape_native=(8, 8), dst_pixel_scales=1.0
        )

        assert dst[2:6, 2:6].all()
        assert dst.sum() == 16

    def test__mask_2d_wrapper(self):
        src = aa.Mask2D.circular(shape_native=(40, 40), pixel_scales=0.1, radius=1.0)

        dst = du.mask_2d_regridded_from(src, shape_native=(20, 20), pixel_scales=0.2)

        expected = aa.Mask2D.circular(
            shape_native=(20, 20), pixel_scales=0.2, radius=1.0
        )
        assert isinstance(dst, aa.Mask2D)
        assert dst.pixel_scales == (0.2, 0.2)
        # Every 0.2" pixel centre sits exactly on a 0.1" pixel edge, so pixels on the
        # circle's boundary may resolve either way; the interior and exterior must agree.
        differ = np.asarray(dst) != np.asarray(expected)
        assert differ.sum() <= 8
        r = _radius_grid((20, 20)) * 0.2
        assert not differ[np.abs(r - 1.0) > 0.15].any()
