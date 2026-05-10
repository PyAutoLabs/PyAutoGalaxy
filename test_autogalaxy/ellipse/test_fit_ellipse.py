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

    assert fit.data_interp[0] == pytest.approx(1.8378134567395, 1.0e-4)


def test__noise_map_interp(imaging_lh, imaging_lh_masked):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_lh, ellipse=ellipse_0)

    assert fit.noise_map_interp == pytest.approx([2.0, 2.0, 2.0, 2.0, 2.0], 1.0e-4)

    fit = ag.FitEllipse(dataset=imaging_lh_masked, ellipse=ellipse_0)

    assert fit.noise_map_interp[0] == pytest.approx(2.0, 1.0e-4)


def test__signal_to_noise_map_interp(imaging_lh, imaging_lh_masked):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_lh, ellipse=ellipse_0)

    assert fit.signal_to_noise_map_interp == pytest.approx(
        [3.0, 1.22792372, 1.21381362, 2.97716938, 4.08109327], 1.0e-4
    )

    fit = ag.FitEllipse(dataset=imaging_lh_masked, ellipse=ellipse_0)

    assert fit.signal_to_noise_map_interp[0] == pytest.approx(0.91890672836, 1.0e-4)


def test__residual_map(imaging_lh, imaging_lh_masked):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_lh, ellipse=ellipse_0)

    assert fit.residual_map == pytest.approx(
        [1.0, -2.54415255, -2.57237275, 0.95433876, 3.16218654], 1.0e-4
    )

    fit = ag.FitEllipse(dataset=imaging_lh_masked, ellipse=ellipse_0)

    assert fit.residual_map[0] == pytest.approx(-2.514972947, 1.0e-4)


def test__normalized_residual_map(imaging_lh, imaging_lh_masked):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_lh, ellipse=ellipse_0)

    assert fit.normalized_residual_map == pytest.approx(
        [0.5, -1.27207628, -1.28618638, 0.47716938, 1.58109327], 1.0e-4
    )

    fit = ag.FitEllipse(dataset=imaging_lh_masked, ellipse=ellipse_0)

    assert fit.normalized_residual_map[0] == pytest.approx(-1.25748647, 1.0e-4)


def test__chi_squared_map(imaging_lh, imaging_lh_masked):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_lh, ellipse=ellipse_0)

    assert fit.chi_squared_map == pytest.approx(
        [0.25, 1.61817806, 1.65427539, 0.22769062, 2.49985593], 1.0e-4
    )

    fit = ag.FitEllipse(dataset=imaging_lh_masked, ellipse=ellipse_0)

    assert fit.chi_squared_map[0] == pytest.approx(1.58127223199, 1.0e-4)


def test__chi_squared(imaging_lh, imaging_lh_masked):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_lh, ellipse=ellipse_0)

    assert fit.chi_squared == pytest.approx(
        sum([0.25, 1.61817806, 1.65427539, 0.22769062, 2.49985593]), 1.0e-4
    )

    fit = ag.FitEllipse(dataset=imaging_lh_masked, ellipse=ellipse_0)

    assert fit.chi_squared == pytest.approx(5.72639320225, 1.0e-4)


def test__noise_normalization(imaging_lh, imaging_lh_masked):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_lh, ellipse=ellipse_0)

    assert fit.noise_normalization == pytest.approx(16.120857137, 1.0e-4)

    fit = ag.FitEllipse(dataset=imaging_lh_masked, ellipse=ellipse_0)

    assert fit.noise_normalization == pytest.approx(16.120857137646, 1.0e-4)


def test__log_likelihood(imaging_lh, imaging_lh_masked):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.5, 0.5), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_lh, ellipse=ellipse_0)

    assert fit.log_likelihood == pytest.approx(-0.16764008373, 1.0e-4)

    fit = ag.FitEllipse(dataset=imaging_lh_masked, ellipse=ellipse_0)

    assert fit.log_likelihood == pytest.approx(-0.169821080058, 1.0e-4)


# ── mask-rejection loop tests (pinned for JAX-rewrite regression) ──────────


@pytest.fixture(name="imaging_30x30")
def make_imaging_30x30():
    return ag.Imaging(
        data=ag.Array2D.ones(shape_native=(30, 30), pixel_scales=(1.0, 1.0)),
        noise_map=ag.Array2D.ones(shape_native=(30, 30), pixel_scales=(1.0, 1.0)),
    )


def test__points_from_major_axis__zero_masked(imaging_30x30):
    # Mask has one corner pixel True so interp.mask_interp is constructed and the loop
    # fires, but the masked pixel is far from the ellipse perimeter.  The equals-branch
    # fires every iteration and the returned points must be identical to the unmasked fit.
    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.3, 0.2), major_axis=5.0)

    mask_array = np.full((30, 30), False)
    mask_array[0, 0] = True
    mask = ag.Mask2D(mask=mask_array.tolist(), pixel_scales=1.0)

    fit_unmasked = ag.FitEllipse(dataset=imaging_30x30, ellipse=ellipse)
    fit_masked = ag.FitEllipse(
        dataset=imaging_30x30.apply_mask(mask=mask), ellipse=ellipse
    )

    assert fit_masked._points_from_major_axis.shape[0] == ellipse.total_points_from(
        pixel_scale=1.0
    ) - 1

    np.testing.assert_allclose(
        fit_masked._points_from_major_axis,
        fit_unmasked._points_from_major_axis,
        rtol=1e-12,
    )


def test__points_from_major_axis__under_masked_trim(imaging_30x30):
    # mask[13, 15] causes: at i=1 the extra-points branch regenerates to n_i=1 (31 pts);
    # at i=2 the 31-point set has 0 masked points (unmasked=31 > required=30) so the
    # trim branch fires and removes 1 extra point; subsequent iterations hit equals-branch.
    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.3, 0.2), major_axis=5.0)

    mask_array = np.full((30, 30), False)
    mask_array[13, 15] = True
    mask = ag.Mask2D(mask=mask_array.tolist(), pixel_scales=1.0)

    fit = ag.FitEllipse(
        dataset=imaging_30x30.apply_mask(mask=mask), ellipse=ellipse
    )

    assert fit._points_from_major_axis.shape[0] == ellipse.total_points_from(
        pixel_scale=1.0
    ) - 1

    expected = np.array(
        [
            [-0.8875687769622342, 4.318959680242291],
            [-1.9465967000974331, 4.536106717915154],
            [-2.7904545348693404, 4.00915543006672],
            [-3.121759540619221, 2.9674537201425717],
            [-3.0971817353867803, 1.9304881368053626],
            [-2.9362004594598003, 1.0874492395351023],
            [-2.7382783396152313, 0.4194888120858164],
            [-2.533386213036308, -0.12847880766605385],
            [-2.325860264652024, -0.6022069457624895],
            [-2.110815651174979, -1.0354045082443406],
            [-1.8786866984151742, -1.4542117489414634],
            [-1.6151780479744147, -1.881457224557489],
            [-1.2986189827210775, -2.3396631393811322],
            [-0.894688693486922, -2.8515790329629698],
            [-0.34900318176777473, -3.4320284500721225],
            [0.41165293239048256, -4.048113740292],
            [1.4128512563813995, -4.503082523252524],
            [2.4255072367965793, -4.3699267851023045],
            [3.0174521816485678, -3.5149110737600795],
            [3.137367386840201, -2.428503123238989],
            [3.0251208310290503, -1.4838926102749905],
            [2.83904957529029, -0.7350808643096405],
            [2.6361114386419064, -0.13368844148940043],
            [2.430126819911951, 0.3722817356273538],
            [2.2197796316635774, 0.8221161006262405],
            [1.9976499645857964, 1.2451447437071157],
            [1.7519886861942746, 1.6653894308155623],
            [1.4652965048117415, 2.105248935438702],
            [1.1104174658849209, 2.5875786835756953],
            [0.6439087789280391, 3.1332964003786445],
        ]
    )

    np.testing.assert_allclose(fit._points_from_major_axis, expected, rtol=1e-12)


def test__points_from_major_axis__over_masked_extra_points(imaging_30x30):
    # A 3x3 block at rows 16-18, cols 18-20 causes 2 points to be masked on the initial
    # 30-point set (unmasked=28 < required=30).  The extra-points branch fires at i=1
    # (n_i=1, 31 pts, still 2 masked -> unmasked=29 < 30) and again at i=2 (n_i=2,
    # 32 pts, still 2 masked -> unmasked=30 == 30).  From i=3 the equals-branch fires.
    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.3, 0.2), major_axis=5.0)

    mask_array = np.full((30, 30), False)
    mask_array[16:19, 18:21] = True
    mask = ag.Mask2D(mask=mask_array.tolist(), pixel_scales=1.0)

    fit = ag.FitEllipse(
        dataset=imaging_30x30.apply_mask(mask=mask), ellipse=ellipse
    )

    assert fit._points_from_major_axis.shape[0] == ellipse.total_points_from(
        pixel_scale=1.0
    ) - 1

    # Spot-check first and last points with full-precision reference values.
    np.testing.assert_allclose(
        fit._points_from_major_axis[0],
        np.array([-0.0, 3.7420680720326427]),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        fit._points_from_major_axis[-1],
        np.array([1.4354511411249111, 2.1483044498322945]),
        rtol=1e-12,
    )


def test__points_from_major_axis__unreachable_raises(imaging_30x30):
    # Masking all pixels except a tiny top-left 5x5 region means the ellipse (major_axis=5,
    # centred at origin) cannot accumulate the required number of unmasked points regardless
    # of how many extra angles are added.  The loop must reach i=300 and raise.
    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.3, 0.2), major_axis=5.0)

    mask_array = np.full((30, 30), True)
    mask_array[0:5, 0:5] = False
    mask = ag.Mask2D(mask=mask_array.tolist(), pixel_scales=1.0)

    fit = ag.FitEllipse(
        dataset=imaging_30x30.apply_mask(mask=mask), ellipse=ellipse
    )

    with pytest.raises(ValueError, match="attempted to add over 300 extra points"):
        _ = fit._points_from_major_axis


def test__points_from_major_axis__with_multipole_under_masked(imaging_30x30):
    # Same geometry as test__points_from_major_axis__under_masked_trim (mask[13,15],
    # EXTRA->TRIM->EQUAL path) but with a m=4 multipole that perturbs the points inside
    # the inner loop block (fit_ellipse.py lines 112-120).  The output must differ from
    # the no-multipole case, confirming that multipole perturbation was applied.
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

    assert fit._points_from_major_axis.shape[0] == ellipse.total_points_from(
        pixel_scale=1.0
    ) - 1

    expected = np.array(
        [
            [-0.8948637627342421, 4.35445749205703],
            [-1.9662891852074944, 4.5819956347080995],
            [-2.8090599546476698, 4.0358865660880605],
            [-3.118093237115593, 2.963968638787185],
            [-3.0636273709808624, 1.9095735415513715],
            [-2.8898535455248733, 1.0702842274696087],
            [-2.7100443128493983, 0.4151635182772758],
            [-2.54343822078039, -0.12898858780260153],
            [-2.366937919038917, -0.6128426873688835],
            [-2.154272852773611, -1.0567212833146127],
            [-1.8978749492680655, -1.4690645606718333],
            [-1.605428202942083, -1.8701000144979532],
            [-1.2768336674699277, -2.300413521324486],
            [-0.8806522731441982, -2.806841726860251],
            [-0.34700836576138805, -3.4124118229345113],
            [0.4136477483968692, -4.067730367429611],
            [1.4268876767241232, -4.547819829355243],
            [2.447292552047729, -4.409176403158951],
            [3.0272020266808997, -3.5262682838196153],
            [3.11817913598731, -2.413650311508619],
            [2.981663629430418, -1.4625758352047185],
            [2.797971920903396, -0.7244451227032465],
            [2.626059430897824, -0.13317866135285275],
            [2.4583608466777838, 0.3766070294358943],
            [2.2661265455985045, 0.839281112691734],
            [2.031204328991714, 1.266059338961107],
            [1.7556549896979028, 1.6688745121709494],
            [1.446691085033412, 2.0785177994173614],
            [1.0907249807748596, 2.5416897667827496],
            [0.6366137931560313, 3.0977985885639057],
        ]
    )

    np.testing.assert_allclose(fit._points_from_major_axis, expected, rtol=1e-12)
