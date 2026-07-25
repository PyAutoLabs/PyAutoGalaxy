from autofit.non_linear.samples.pdf import quantile
import autogalaxy as ag
import numpy as np


def test__quantile_1d_profile():
    profile_1d_0 = np.array([1.0, 2.0, 3.0])
    profile_1d_1 = np.array([1.0, 2.0, 3.0])

    profile_1d_list = [profile_1d_0, profile_1d_1]

    median_profile_1d = ag.util.error.quantile_profile_1d(
        profile_1d_list=profile_1d_list, q=0.5
    )

    assert (median_profile_1d == np.array([1.0, 2.0, 3.0])).all()

    profile_1d_0 = np.array([1.0, 2.0, 3.0])
    profile_1d_1 = np.array([2.0, 4.0, 6.0])

    profile_1d_list = [profile_1d_0, profile_1d_1]

    median_profile_1d = ag.util.error.quantile_profile_1d(
        profile_1d_list=profile_1d_list, q=0.5
    )

    assert (median_profile_1d == np.array([1.5, 3.0, 4.5])).all()

    profile_1d_list = [
        profile_1d_0,
        profile_1d_0,
        profile_1d_0,
        profile_1d_1,
        profile_1d_1,
        profile_1d_1,
        profile_1d_1,
    ]

    weights = np.array([9.9996, 9.9996, 9.9996, 1e-4, 1e-4, 1e-4, 1e-4])

    median_profile_1d = ag.util.error.quantile_profile_1d(
        profile_1d_list=profile_1d_list, q=0.5, weights=weights
    )

    assert (median_profile_1d == np.array([1.0, 2.0, 3.0])).all()

    radial_values = [1.0, 2.0, 3.0, 4.0, 5.0]
    weights = [0.1, 0.3, 0.2, 0.05, 0.35]

    quantile_result = quantile(x=radial_values, q=0.23, weights=weights)

    profile_1d_0 = np.array([1.0])
    profile_1d_1 = np.array([2.0])
    profile_1d_2 = np.array([3.0])
    profile_1d_3 = np.array([4.0])
    profile_1d_4 = np.array([5.0])

    profile_1d_list = [
        profile_1d_0,
        profile_1d_1,
        profile_1d_2,
        profile_1d_3,
        profile_1d_4,
    ]

    profile_1d_via_error_util = ag.util.error.quantile_profile_1d(
        profile_1d_list=profile_1d_list, q=0.23, weights=weights
    )

    assert quantile_result == profile_1d_via_error_util[0]
