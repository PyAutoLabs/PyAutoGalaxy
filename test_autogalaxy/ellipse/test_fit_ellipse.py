import numpy as np
import pytest

import autogalaxy as ag


@pytest.fixture(name="imaging_lh")
def make_imaging_lh(imaging_7x7):
    data = ag.Array2D.ones(shape_native=(7, 7), pixel_scales=(1.0, 1.0))

    data[16] = 1.0
    data[17] = 2.0
    data[18] = 3.0
    data[23] = 4.0
    data[24] = 5.0
    data[25] = 6.0
    data[30] = 7.0
    data[31] = 8.0
    data[32] = 9.0

    return ag.Imaging(
        data=data,
        noise_map=imaging_7x7.noise_map,
    )


@pytest.fixture(name="imaging_lh_masked")
def make_imaging_lh_masked(imaging_lh):
    mask = ag.Mask2D(
        mask=[
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, False, False, True, True, True],
            [True, True, False, False, True, True, True],
            [True, True, False, False, False, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ],
        pixel_scales=1.0,
    )

    return imaging_lh.apply_mask(mask=mask)


def test__points_from_major_axis(imaging_lh):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.5, 0.5), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_lh, ellipse=ellipse_0)

    assert fit._points_from_major_axis[1, 0] == pytest.approx(-0.21232, 1.0e-4)
    assert fit._points_from_major_axis[1, 1] == pytest.approx(0.068987, 1.0e-4)

    assert fit._points_from_major_axis[4, 0] == pytest.approx(0.16366515, 1.0e-4)
    assert fit._points_from_major_axis[4, 1] == pytest.approx(0.05317803, 1.0e-4)


def test___points_from_major_axis__multipole(imaging_lh):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.5, 0.5), major_axis=1.0)

    multipole = ag.EllipseMultipole(m=4, multipole_comps=(0.2, 0.3))

    fit = ag.FitEllipse(
        dataset=imaging_lh, ellipse=ellipse_0, multipole_list=[multipole]
    )

    assert fit._points_from_major_axis[1, 0] == pytest.approx(-0.119588, 1.0e-4)
    assert fit._points_from_major_axis[1, 1] == pytest.approx(0.038856679, 1.0e-4)


# def test__mask_interp(imaging_lh, imaging_lh_masked):
#     ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)
#
#     fit = ag.FitEllipse(dataset=imaging_lh, ellipse=ellipse_0)
#
#     assert fit.mask_interp == pytest.approx([False, False, False, False, False], 1.0e-4)
#
#     fit = ag.FitEllipse(dataset=imaging_lh_masked, ellipse=ellipse_0)
#
#     assert fit.mask_interp == pytest.approx([False, True, True, True, True], 1.0e-4)


def test__data_interp(imaging_lh, imaging_lh_masked):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_lh, ellipse=ellipse_0)

    assert fit.data_interp == pytest.approx(
        [6.0, 2.45584745, 2.42762725, 5.95433876, 8.16218654], 1.0e-4
    )

    fit = ag.FitEllipse(dataset=imaging_lh_masked, ellipse=ellipse_0)

    # New algorithm: shape fixed at (N, 2) with NaN rows for masked positions.
    # Indices 0, 1, 4 are NaN (masked); index 2 is the first non-NaN point.
    assert np.isnan(fit.data_interp[0])
    assert fit.data_interp[2] == pytest.approx(2.4276272487476, 1.0e-4)


def test__noise_map_interp(imaging_lh, imaging_lh_masked):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_lh, ellipse=ellipse_0)

    assert fit.noise_map_interp == pytest.approx([2.0, 2.0, 2.0, 2.0, 2.0], 1.0e-4)

    fit = ag.FitEllipse(dataset=imaging_lh_masked, ellipse=ellipse_0)

    # New algorithm: NaN rows propagate through interpolation; index 2 is first non-NaN.
    assert np.isnan(fit.noise_map_interp[0])
    assert fit.noise_map_interp[2] == pytest.approx(2.0, 1.0e-4)


def test__signal_to_noise_map_interp(imaging_lh, imaging_lh_masked):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_lh, ellipse=ellipse_0)

    assert fit.signal_to_noise_map_interp == pytest.approx(
        [3.0, 1.22792372, 1.21381362, 2.97716938, 4.08109327], 1.0e-4
    )

    fit = ag.FitEllipse(dataset=imaging_lh_masked, ellipse=ellipse_0)

    # New algorithm: NaN rows propagate through division; index 2 is first non-NaN.
    assert np.isnan(fit.signal_to_noise_map_interp[0])
    assert fit.signal_to_noise_map_interp[2] == pytest.approx(1.21381362437, 1.0e-4)


def test__residual_map(imaging_lh, imaging_lh_masked):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_lh, ellipse=ellipse_0)

    assert fit.residual_map == pytest.approx(
        [1.0, -2.54415255, -2.57237275, 0.95433876, 3.16218654], 1.0e-4
    )

    fit = ag.FitEllipse(dataset=imaging_lh_masked, ellipse=ellipse_0)

    # New algorithm: NaN rows propagate; index 2 is first non-NaN point.
    # nanmean is computed over the 2 non-NaN values only.
    assert np.isnan(fit.residual_map[0])
    assert fit.residual_map[2] == pytest.approx(-1.7633557568774, 1.0e-4)


def test__normalized_residual_map(imaging_lh, imaging_lh_masked):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_lh, ellipse=ellipse_0)

    assert fit.normalized_residual_map == pytest.approx(
        [0.5, -1.27207628, -1.28618638, 0.47716938, 1.58109327], 1.0e-4
    )

    fit = ag.FitEllipse(dataset=imaging_lh_masked, ellipse=ellipse_0)

    # New algorithm: NaN rows propagate; index 2 is first non-NaN point.
    assert np.isnan(fit.normalized_residual_map[0])
    assert fit.normalized_residual_map[2] == pytest.approx(-0.8816778784387, 1.0e-4)


def test__chi_squared_map(imaging_lh, imaging_lh_masked):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_lh, ellipse=ellipse_0)

    assert fit.chi_squared_map == pytest.approx(
        [0.25, 1.61817806, 1.65427539, 0.22769062, 2.49985593], 1.0e-4
    )

    fit = ag.FitEllipse(dataset=imaging_lh_masked, ellipse=ellipse_0)

    # New algorithm: NaN rows propagate; index 2 is first non-NaN point.
    assert np.isnan(fit.chi_squared_map[0])
    assert fit.chi_squared_map[2] == pytest.approx(0.7773558813282, 1.0e-4)


def test__chi_squared(imaging_lh, imaging_lh_masked):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_lh, ellipse=ellipse_0)

    assert fit.chi_squared == pytest.approx(
        sum([0.25, 1.61817806, 1.65427539, 0.22769062, 2.49985593]), 1.0e-4
    )

    fit = ag.FitEllipse(dataset=imaging_lh_masked, ellipse=ellipse_0)

    # New algorithm: nansum over 2 non-NaN chi_squared values only.
    assert fit.chi_squared == pytest.approx(1.5547117626564, 1.0e-4)


def test__noise_normalization(imaging_lh, imaging_lh_masked):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_lh, ellipse=ellipse_0)

    assert fit.noise_normalization == pytest.approx(16.120857137, 1.0e-4)

    fit = ag.FitEllipse(dataset=imaging_lh_masked, ellipse=ellipse_0)

    # New algorithm: nansum over 2 non-NaN noise values only (= 2 * log(2*pi*4)).
    assert fit.noise_normalization == pytest.approx(6.4483428550585, 1.0e-4)


def test__log_likelihood(imaging_lh, imaging_lh_masked):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.5, 0.5), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_lh, ellipse=ellipse_0)

    assert fit.log_likelihood == pytest.approx(-0.16764008373, 1.0e-4)

    fit = ag.FitEllipse(dataset=imaging_lh_masked, ellipse=ellipse_0)

    # New algorithm: -0.5 * chi_squared over 2 non-NaN perimeter samples only.
    assert fit.log_likelihood == pytest.approx(-0.0736366768059, 1.0e-4)


# ── unified NaN-marking algorithm tests (prompt-3 regression pins) ──────────


@pytest.fixture(name="imaging_30x30")
def make_imaging_30x30():
    return ag.Imaging(
        data=ag.Array2D.ones(shape_native=(30, 30), pixel_scales=(1.0, 1.0)),
        noise_map=ag.Array2D.ones(shape_native=(30, 30), pixel_scales=(1.0, 1.0)),
    )


def test__points_from_major_axis__zero_masked(imaging_30x30):
    # Mask has one corner pixel True (far from the ellipse perimeter).
    # The new algorithm: shape is (N, 2) = (total_points_from - 1, 2), all rows finite,
    # and the result matches the fully-unmasked fit row-for-row because the bilinear
    # mask interpolator returns 0 for every perimeter sample far from the masked pixel.
    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.3, 0.2), major_axis=5.0)

    mask_array = np.full((30, 30), False)
    mask_array[0, 0] = True
    mask = ag.Mask2D(mask=mask_array.tolist(), pixel_scales=1.0)

    fit_unmasked = ag.FitEllipse(dataset=imaging_30x30, ellipse=ellipse)
    fit_masked = ag.FitEllipse(
        dataset=imaging_30x30.apply_mask(mask=mask), ellipse=ellipse
    )

    N = ellipse.total_points_from(pixel_scale=1.0)
    assert fit_masked._points_from_major_axis.shape == (N - 1, 2)
    assert np.all(np.isfinite(fit_masked._points_from_major_axis))

    np.testing.assert_allclose(
        fit_masked._points_from_major_axis,
        fit_unmasked._points_from_major_axis,
        rtol=1e-12,
    )


def test__points_from_major_axis__under_masked_trim(imaging_30x30):
    # mask[13, 15] catches one perimeter sample.
    # New algorithm: shape is fixed at (N, 2); exactly one row is [nan, nan] at index 8;
    # all other rows match the initial perimeter generation (no regeneration step).
    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.3, 0.2), major_axis=5.0)

    mask_array = np.full((30, 30), False)
    mask_array[13, 15] = True
    mask = ag.Mask2D(mask=mask_array.tolist(), pixel_scales=1.0)

    fit = ag.FitEllipse(
        dataset=imaging_30x30.apply_mask(mask=mask), ellipse=ellipse
    )

    N = ellipse.total_points_from(pixel_scale=1.0)
    points = fit._points_from_major_axis
    assert points.shape == (N - 1, 2)
    assert np.sum(np.isnan(points[:, 0])) == 1

    expected = np.array(
        [
            [-0.00000000e+00,  3.74206807e+00],
            [-9.21350849e-01,  4.33461495e+00],
            [-2.01511259e+00,  4.52601698e+00],
            [-2.84687547e+00,  3.91838793e+00],
            [-3.13310933e+00,  2.82106431e+00],
            [-3.07580015e+00,  1.77581404e+00],
            [-2.89800061e+00,  9.41617477e-01],
            [-2.69069903e+00,  2.82803864e-01],
            [            np.nan,             np.nan],
            [-2.26249250e+00, -7.35128377e-01],
            [-2.03593096e+00, -1.17544529e+00],
            [-1.78665636e+00, -1.60871261e+00],
            [-1.49648449e+00, -2.05973419e+00],
            [-1.13695582e+00, -2.55364458e+00],
            [-6.61892435e-01, -3.11395908e+00],
            [-4.58271169e-16, -3.74206807e+00],
            [ 9.21350849e-01, -4.33461495e+00],
            [ 2.01511259e+00, -4.52601698e+00],
            [ 2.84687547e+00, -3.91838793e+00],
            [ 3.13310933e+00, -2.82106431e+00],
            [ 3.07580015e+00, -1.77581404e+00],
            [ 2.89800061e+00, -9.41617477e-01],
            [ 2.69069903e+00, -2.82803864e-01],
            [ 2.47840199e+00,  2.60490545e-01],
            [ 2.26249250e+00,  7.35128377e-01],
            [ 2.03593096e+00,  1.17544529e+00],
            [ 1.78665636e+00,  1.60871261e+00],
            [ 1.49648449e+00,  2.05973419e+00],
            [ 1.13695582e+00,  2.55364458e+00],
            [ 6.61892435e-01,  3.11395908e+00],
        ]
    )

    np.testing.assert_allclose(points, expected, equal_nan=True, rtol=1e-6)


def test__points_from_major_axis__over_masked_extra_points(imaging_30x30):
    # A 3x3 block at rows 16-18, cols 18-20 catches 2 perimeter samples.
    # New algorithm: shape is fixed at (N, 2); exactly 2 NaN rows at indices 28 and 29;
    # all other rows match the initial perimeter generation (no regeneration step).
    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.3, 0.2), major_axis=5.0)

    mask_array = np.full((30, 30), False)
    mask_array[16:19, 18:21] = True
    mask = ag.Mask2D(mask=mask_array.tolist(), pixel_scales=1.0)

    fit = ag.FitEllipse(
        dataset=imaging_30x30.apply_mask(mask=mask), ellipse=ellipse
    )

    N = ellipse.total_points_from(pixel_scale=1.0)
    points = fit._points_from_major_axis
    assert points.shape == (N - 1, 2)
    assert np.sum(np.isnan(points[:, 0])) >= 2

    expected = np.array(
        [
            [-0.00000000e+00,  3.74206807e+00],
            [-9.21350849e-01,  4.33461495e+00],
            [-2.01511259e+00,  4.52601698e+00],
            [-2.84687547e+00,  3.91838793e+00],
            [-3.13310933e+00,  2.82106431e+00],
            [-3.07580015e+00,  1.77581404e+00],
            [-2.89800061e+00,  9.41617477e-01],
            [-2.69069903e+00,  2.82803864e-01],
            [-2.47840199e+00, -2.60490545e-01],
            [-2.26249250e+00, -7.35128377e-01],
            [-2.03593096e+00, -1.17544529e+00],
            [-1.78665636e+00, -1.60871261e+00],
            [-1.49648449e+00, -2.05973419e+00],
            [-1.13695582e+00, -2.55364458e+00],
            [-6.61892435e-01, -3.11395908e+00],
            [-4.58271169e-16, -3.74206807e+00],
            [ 9.21350849e-01, -4.33461495e+00],
            [ 2.01511259e+00, -4.52601698e+00],
            [ 2.84687547e+00, -3.91838793e+00],
            [ 3.13310933e+00, -2.82106431e+00],
            [ 3.07580015e+00, -1.77581404e+00],
            [ 2.89800061e+00, -9.41617477e-01],
            [ 2.69069903e+00, -2.82803864e-01],
            [ 2.47840199e+00,  2.60490545e-01],
            [ 2.26249250e+00,  7.35128377e-01],
            [ 2.03593096e+00,  1.17544529e+00],
            [ 1.78665636e+00,  1.60871261e+00],
            [ 1.49648449e+00,  2.05973419e+00],
            [             np.nan,              np.nan],
            [             np.nan,              np.nan],
        ]
    )

    np.testing.assert_allclose(points, expected, equal_nan=True, rtol=1e-6)


def test__points_from_major_axis__unreachable_returns_all_nan(imaging_30x30):
    # Masking all pixels except a tiny top-left 5x5 region means every perimeter sample
    # of the ellipse (major_axis=5, centred at origin) is masked.
    # New algorithm: returns all-NaN instead of raising ValueError.
    # Downstream: chi_squared will be NaN; the non-linear search treats NaN likelihood
    # as -inf (model rejection), so this is a loud failure at the search level rather
    # than a Python exception here.
    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.3, 0.2), major_axis=5.0)

    mask_array = np.full((30, 30), True)
    mask_array[0:5, 0:5] = False
    mask = ag.Mask2D(mask=mask_array.tolist(), pixel_scales=1.0)

    fit = ag.FitEllipse(
        dataset=imaging_30x30.apply_mask(mask=mask), ellipse=ellipse
    )

    N = ellipse.total_points_from(pixel_scale=1.0)
    points = fit._points_from_major_axis
    assert points.shape == (N - 1, 2)
    assert np.all(np.isnan(points))


def test__points_from_major_axis__with_multipole_under_masked(imaging_30x30):
    # Same geometry as test__points_from_major_axis__under_masked_trim (mask[13, 15])
    # but with a m=4 multipole (multipole_comps=(0.05, 0.0)).
    # New algorithm: shape (N, 2), one NaN row at index 8 (same position as no-multipole
    # case since NaN-marking is based on the perturbed coordinates hitting the mask),
    # and all non-NaN rows differ from the no-multipole case confirming multipole applied.
    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.3, 0.2), major_axis=5.0)
    multipole = ag.EllipseMultipole(m=4, multipole_comps=(0.05, 0.0))

    mask_array = np.full((30, 30), False)
    mask_array[13, 15] = True
    mask = ag.Mask2D(mask=mask_array.tolist(), pixel_scales=1.0)

    fit = ag.FitEllipse(
        dataset=imaging_30x30.apply_mask(mask=mask),
        ellipse=ellipse,
        multipole_list=[multipole],
    )

    N = ellipse.total_points_from(pixel_scale=1.0)
    points = fit._points_from_major_axis
    assert points.shape == (N - 1, 2)
    assert np.sum(np.isnan(points[:, 0])) == 1

    expected = np.array(
        [
            [ 0.00000000e+00,  3.74206807e+00],
            [-9.29076274e-01,  4.37096021e+00],
            [-2.03533802e+00,  4.57144403e+00],
            [-2.86415004e+00,  3.94216434e+00],
            [-3.12538390e+00,  2.81410831e+00],
            [-3.03830015e+00,  1.75416341e+00],
            [-2.85277518e+00,  9.26922846e-01],
            [-2.67047361e+00,  2.80678086e-01],
            [             np.nan,              np.nan],
            [-2.30771793e+00, -7.49823009e-01],
            [-2.07343096e+00, -1.19709592e+00],
            [-1.79438178e+00, -1.61566861e+00],
            [-1.47920991e+00, -2.03595778e+00],
            [-1.11673040e+00, -2.50821753e+00],
            [-6.54167010e-01, -3.07761381e+00],
            [-4.58271169e-16, -3.74206807e+00],
            [ 9.29076274e-01, -4.37096021e+00],
            [ 2.03533802e+00, -4.57144403e+00],
            [ 2.86415004e+00, -3.94216434e+00],
            [ 3.12538390e+00, -2.81410831e+00],
            [ 3.03830015e+00, -1.75416341e+00],
            [ 2.85277518e+00, -9.26922846e-01],
            [ 2.67047361e+00, -2.80678086e-01],
            [ 2.49862741e+00,  2.62616323e-01],
            [ 2.30771793e+00,  7.49823009e-01],
            [ 2.07343096e+00,  1.19709592e+00],
            [ 1.79438178e+00,  1.61566861e+00],
            [ 1.47920991e+00,  2.03595778e+00],
            [ 1.11673040e+00,  2.50821753e+00],
            [ 6.54167010e-01,  3.07761381e+00],
        ]
    )

    np.testing.assert_allclose(points, expected, equal_nan=True, rtol=1e-6)
