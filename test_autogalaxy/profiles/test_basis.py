import numpy as np
import pytest

import autoarray as aa
import autogalaxy as ag


def test__light_profile_list():
    lp_0 = ag.lp.Sersic(intensity=0.1)
    lp_1 = ag.lp.Sersic(intensity=0.2)

    mp_0 = ag.mp.Isothermal()
    mp_1 = ag.mp.NFW()

    basis = ag.lp_basis.Basis(profile_list=[lp_0, mp_0, lp_1, mp_1])

    assert basis.light_profile_list == [lp_0, lp_1]


def test__mass_profile_list():
    lp_0 = ag.lp.Sersic(intensity=0.1)
    lp_1 = ag.lp.Sersic(intensity=0.2)

    mp_0 = ag.mp.Isothermal()
    mp_1 = ag.mp.NFW()

    basis = ag.lp_basis.Basis(profile_list=[lp_0, mp_0, lp_1, mp_1])

    assert basis.mass_profile_list == [mp_0, mp_1]


def test__image_2d_from__does_not_include_linear_light_profiles(grid_2d_7x7):
    lp = ag.lp.Sersic(intensity=0.1)

    lp_linear = ag.lp_linear.Sersic(effective_radius=2.0, sersic_index=2.0)

    lp_image = lp.image_2d_from(grid=grid_2d_7x7)

    basis = ag.lp_basis.Basis(profile_list=[lp, lp_linear])

    image = basis.image_2d_from(grid=grid_2d_7x7)

    assert (image == lp_image).all()


def test__image_2d_from__returns_array2d_for_linear_only_basis(grid_2d_7x7):
    # Regression test for the case where every constituent is a LightProfileLinear:
    # the placeholder for the linear-intensity-unknown image used to be a raw
    # ndarray of zeros, which made `sum(...)` return a raw ndarray rather than an
    # `Array2D`.  After the fix the placeholder is itself an `Array2D` so the
    # summed return type is uniform regardless of the constituent mix.
    basis = ag.lp_basis.Basis(
        profile_list=[
            ag.lp_linear.Gaussian(sigma=0.5),
            ag.lp_linear.Gaussian(sigma=1.0),
        ]
    )

    image = basis.image_2d_from(grid=grid_2d_7x7)

    assert isinstance(image, aa.Array2D)


def test__image_2d_from__operated_only_false__returns_only_non_operated_profile_image(
    grid_2d_7x7, lp_0, lp_operated_0
):
    image_2d_not_operated = lp_0.image_2d_from(grid=grid_2d_7x7)

    basis = ag.lp_basis.Basis(profile_list=[lp_0, lp_operated_0])

    image_2d = basis.image_2d_from(grid=grid_2d_7x7, operated_only=False)
    assert (image_2d == image_2d_not_operated).all()


def test__image_2d_from__operated_only_true__returns_only_operated_profile_image(
    grid_2d_7x7, lp_0, lp_operated_0
):
    image_2d_operated = lp_operated_0.image_2d_from(grid=grid_2d_7x7)

    basis = ag.lp_basis.Basis(profile_list=[lp_0, lp_operated_0])

    image_2d = basis.image_2d_from(grid=grid_2d_7x7, operated_only=True)
    assert (image_2d == image_2d_operated).all()


def test__image_2d_from__operated_only_none__returns_sum_of_all_images(
    grid_2d_7x7, lp_0, lp_operated_0
):
    image_2d_not_operated = lp_0.image_2d_from(grid=grid_2d_7x7)
    image_2d_operated = lp_operated_0.image_2d_from(grid=grid_2d_7x7)

    basis = ag.lp_basis.Basis(profile_list=[lp_0, lp_operated_0])

    image_2d = basis.image_2d_from(grid=grid_2d_7x7, operated_only=None)
    assert (image_2d == image_2d_not_operated + image_2d_operated).all()


def test__image_2d_list_from__operated_only_false__non_operated_image_returned_operated_image_zeros(
    grid_2d_7x7, lp_0, lp_operated_0
):
    image_2d_not_operated = lp_0.image_2d_from(grid=grid_2d_7x7)

    basis = ag.lp_basis.Basis(profile_list=[lp_0, lp_operated_0])

    image_2d_list = basis.image_2d_list_from(grid=grid_2d_7x7, operated_only=False)
    assert (image_2d_list[0] == image_2d_not_operated).all()
    assert (image_2d_list[1] == np.zeros((9))).all()


def test__image_2d_list_from__operated_only_true__non_operated_image_zeros_operated_image_returned(
    grid_2d_7x7, lp_0, lp_operated_0
):
    image_2d_operated = lp_operated_0.image_2d_from(grid=grid_2d_7x7)

    basis = ag.lp_basis.Basis(profile_list=[lp_0, lp_operated_0])

    image_2d_list = basis.image_2d_list_from(grid=grid_2d_7x7, operated_only=True)
    assert (image_2d_list[0] == np.zeros((9))).all()
    assert (image_2d_list[1] == image_2d_operated).all()


def test__image_2d_list_from__operated_only_none__list_sums_to_total_image(
    grid_2d_7x7, lp_0, lp_operated_0
):
    image_2d_not_operated = lp_0.image_2d_from(grid=grid_2d_7x7)
    image_2d_operated = lp_operated_0.image_2d_from(grid=grid_2d_7x7)

    basis = ag.lp_basis.Basis(profile_list=[lp_0, lp_operated_0])

    image_2d_list = basis.image_2d_list_from(grid=grid_2d_7x7, operated_only=None)
    assert (
        image_2d_list[0] + image_2d_list[1] == image_2d_not_operated + image_2d_operated
    ).all()


def test__mass_profile_quantities(grid_2d_7x7, mp_0, mp_1):
    convergence_2d_0 = mp_0.convergence_2d_from(grid=grid_2d_7x7)
    convergence_2d_1 = mp_1.convergence_2d_from(grid=grid_2d_7x7)

    basis = ag.lp_basis.Basis(profile_list=[mp_0, mp_1])

    convergence_2d = basis.convergence_2d_from(grid=grid_2d_7x7)

    assert convergence_2d.array == pytest.approx(
        convergence_2d_0.array + convergence_2d_1.array, 1.0e-4
    )
