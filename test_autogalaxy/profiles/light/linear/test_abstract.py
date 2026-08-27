import numpy as np
import pytest

import autoarray as aa
import autogalaxy as ag

from autogalaxy.profiles.light.linear import LightProfileLinear
from autogalaxy.profiles.light.linear import (
    LightProfileLinearObjFuncList,
)


def test__params__two_light_profiles__equals_two(
    grid_2d_7x7, blurring_grid_2d_7x7, psf_3x3
):
    lp_0 = ag.lp_linear.Sersic(effective_radius=1.0)
    lp_1 = ag.lp_linear.Sersic(effective_radius=2.0)

    lp_linear_obj_func_list = LightProfileLinearObjFuncList(
        grid=grid_2d_7x7,
        blurring_grid=blurring_grid_2d_7x7,
        psf=psf_3x3,
        light_profile_list=[lp_0, lp_1],
    )

    assert lp_linear_obj_func_list.params == 2


def test__mapping_matrix__columns_match_individual_profile_images(
    grid_2d_7x7, blurring_grid_2d_7x7, psf_3x3
):
    lp_0 = ag.lp_linear.Sersic(effective_radius=1.0)
    lp_1 = ag.lp_linear.Sersic(effective_radius=2.0)

    lp_linear_obj_func_list = LightProfileLinearObjFuncList(
        grid=grid_2d_7x7,
        blurring_grid=blurring_grid_2d_7x7,
        psf=psf_3x3,
        light_profile_list=[lp_0, lp_1],
    )

    lp_0_image = lp_0.image_2d_from(grid=grid_2d_7x7)
    lp_1_image = lp_1.image_2d_from(grid=grid_2d_7x7)

    assert lp_linear_obj_func_list.mapping_matrix[:, 0] == pytest.approx(
        lp_0_image.array, 1.0e-4
    )
    assert lp_linear_obj_func_list.mapping_matrix[:, 1] == pytest.approx(
        lp_1_image.array, 1.0e-4
    )


def test__operated_mapping_matrix__columns_match_individual_blurred_images(
    grid_2d_7x7, blurring_grid_2d_7x7, psf_3x3
):
    lp_0 = ag.lp_linear.Sersic(effective_radius=1.0)
    lp_1 = ag.lp_linear.Sersic(effective_radius=2.0)

    lp_linear_obj_func_list = LightProfileLinearObjFuncList(
        grid=grid_2d_7x7,
        blurring_grid=blurring_grid_2d_7x7,
        psf=psf_3x3,
        light_profile_list=[lp_0, lp_1],
    )

    lp_0_blurred_image = lp_0.blurred_image_2d_from(
        grid=grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7, psf=psf_3x3
    )

    lp_1_blurred_image = lp_1.blurred_image_2d_from(
        grid=grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7, psf=psf_3x3
    )

    assert lp_linear_obj_func_list.operated_mapping_matrix_override[
        :, 0
    ] == pytest.approx(lp_0_blurred_image.array, 1.0e-4)
    assert lp_linear_obj_func_list.operated_mapping_matrix_override[
        :, 1
    ] == pytest.approx(lp_1_blurred_image.array, 1.0e-4)


def test__operated_mapping_matrix_override__psf_none_returns_none(
    grid_2d_7x7, blurring_grid_2d_7x7
):
    lp_linear_obj_func_list = LightProfileLinearObjFuncList(
        grid=grid_2d_7x7,
        blurring_grid=blurring_grid_2d_7x7,
        psf=None,
        light_profile_list=[ag.lp_linear.Sersic(effective_radius=1.0)],
    )

    assert lp_linear_obj_func_list.operated_mapping_matrix_override is None

    lp_linear_obj_func_list_operated = LightProfileLinearObjFuncList(
        grid=grid_2d_7x7,
        blurring_grid=blurring_grid_2d_7x7,
        psf=None,
        light_profile_list=[ag.lp_linear_operated.Gaussian()],
    )

    assert lp_linear_obj_func_list_operated.operated_mapping_matrix_override is None


def test__lp_instance_from__returns_non_linear_instance_with_correct_type_and_centre():
    lp_linear = ag.lp_linear.Sersic(centre=(1.0, 2.0))

    lp_non_linear = lp_linear.lp_instance_from(
        linear_light_profile_intensity_dict={lp_linear: 3.0}
    )

    assert not isinstance(lp_non_linear, LightProfileLinear)
    assert type(lp_non_linear) is ag.lp.Sersic
    assert lp_non_linear.centre == (1.0, 2.0)


def test__lp_instance_from__returns_instance_with_correct_intensity():
    lp_linear = ag.lp_linear.Sersic(centre=(1.0, 2.0))

    lp_non_linear = lp_linear.lp_instance_from(
        linear_light_profile_intensity_dict={lp_linear: 3.0}
    )

    assert lp_non_linear.intensity == 3.0


def test__point_source_lp_instance_from__returns_standard_point_source():
    lp_linear = ag.lp_linear.PointSource(centre=(1.0, 2.0))

    lp_non_linear = lp_linear.lp_instance_from(
        linear_light_profile_intensity_dict={lp_linear: 3.0}
    )

    assert type(lp_non_linear) is ag.lp.PointSource
    assert lp_non_linear.centre == (1.0, 2.0)
    assert lp_non_linear.intensity == 3.0


def test__point_source_operated_mapping_matrix__oversampled_psf_conserves_flux():
    mask = ag.Mask2D.all_false(shape_native=(11, 11), pixel_scales=1.0)

    over_sample_size = 2
    kernel = aa.Array2D.no_mask(
        values=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        pixel_scales=1.0 / over_sample_size,
    )
    psf = aa.Convolver(kernel=kernel, convolve_over_sample_size=over_sample_size)

    grid = ag.Grid2D.from_mask(mask=mask, over_sample_size=over_sample_size)
    blurring_mask = mask.derive_mask.blurring_from(
        kernel_shape_native=psf.kernel_shape_image_resolution,
        allow_padding=True,
    )
    blurring_grid = ag.Grid2D.from_mask(
        mask=blurring_mask, over_sample_size=over_sample_size
    )

    lp = ag.lp_linear.PointSource(centre=(0.3, 0.3))
    func_list = LightProfileLinearObjFuncList(
        grid=grid,
        blurring_grid=blurring_grid,
        psf=psf,
        light_profile_list=[lp],
        regularization=None,
    )

    operated_mapping_matrix = func_list.operated_mapping_matrix_override

    assert operated_mapping_matrix.shape == (mask.pixels_in_mask, 1)
    assert np.sum(operated_mapping_matrix[:, 0]) == pytest.approx(1.0)


def test__pytree_token_is_int_and_unique():
    lp_0 = ag.lp_linear.Sersic()
    lp_1 = ag.lp_linear.Sersic()

    assert isinstance(lp_0.pytree_token, int)
    assert isinstance(lp_1.pytree_token, int)
    assert lp_0.pytree_token != lp_1.pytree_token

    assert isinstance(hash(lp_0), int)
    assert hash(lp_0) == hash(lp_0)
    assert hash(lp_0) != hash(lp_1)


def test__getstate__omits_pytree_token():
    lp = ag.lp_linear.Sersic()
    state = lp.__getstate__()

    assert "pytree_token" not in state
    assert "effective_radius" in state


def test__setstate__assigns_fresh_pytree_token_when_missing():
    lp = ag.lp_linear.Sersic()
    state = lp.__getstate__()

    restored = ag.lp_linear.Sersic.__new__(ag.lp_linear.Sersic)
    restored.__setstate__(state)

    assert isinstance(restored.pytree_token, int)
    assert isinstance(hash(restored), int)


def test__pickle_roundtrip_preserves_int_hash():
    import pickle

    lp = ag.lp_linear.Sersic()
    restored = pickle.loads(pickle.dumps(lp))

    assert isinstance(hash(restored), int)
    assert isinstance(restored.pytree_token, int)
    assert restored.effective_radius == lp.effective_radius


def test__setstate__preserves_pytree_token_when_present():
    lp = ag.lp_linear.Sersic()
    state_with_token = dict(lp.__dict__)

    restored = ag.lp_linear.Sersic.__new__(ag.lp_linear.Sersic)
    restored.__setstate__(state_with_token)

    assert restored.pytree_token == lp.pytree_token


def test__operated_mapping_matrix_override__oversampled_psf__matches_direct_convolver():
    # With an oversampled PSF each linear light profile is evaluated on the
    # over-sampled coordinates and convolved at the fine resolution — the column
    # must equal calling the (phase-2a tested) oversampled Convolver directly.
    import numpy as np
    import autoarray as aa
    from autogalaxy.profiles.light.linear.abstract import (
        LightProfileLinearObjFuncList,
    )

    mask = aa.Mask2D.circular(shape_native=(11, 11), pixel_scales=1.0, radius=3.5)

    s = 2
    n = 9
    c = (np.arange(n) - (n - 1) / 2.0) * (1.0 / s)
    yy, xx = np.meshgrid(-c, c, indexing="ij")
    kernel = np.exp(-0.5 * (yy**2 + xx**2) / 0.8**2)
    kernel = aa.Array2D.no_mask(values=kernel / kernel.sum(), pixel_scales=1.0 / s)
    psf = aa.Convolver(kernel=kernel, convolve_over_sample_size=s)

    grid = aa.Grid2D.from_mask(mask=mask, over_sample_size=s)
    blurring_mask = mask.derive_mask.blurring_from(
        kernel_shape_native=psf.kernel_shape_image_resolution, allow_padding=True
    )
    blurring_grid = aa.Grid2D.from_mask(mask=blurring_mask, over_sample_size=s)

    lp_0 = ag.lp_linear.Sersic(
        centre=(0.3, -0.4), effective_radius=1.0, sersic_index=2.0
    )
    lp_1 = ag.lp_linear.Gaussian(centre=(-0.5, 0.2), sigma=0.7)

    func_list = LightProfileLinearObjFuncList(
        grid=grid,
        blurring_grid=blurring_grid,
        psf=psf,
        light_profile_list=[lp_0, lp_1],
        regularization=None,
    )

    override = np.array(func_list.operated_mapping_matrix_override)

    assert override.shape == (mask.pixels_in_mask, 2)

    for i, lp in enumerate([lp_0, lp_1]):
        image_sub = lp.image_2d_from(grid=grid.over_sampled)
        blurring_sub = lp.image_2d_from(grid=blurring_grid.over_sampled)
        direct = psf.convolved_image_from(
            image=image_sub, blurring_image=blurring_sub, mask=mask
        )
        assert override[:, i] == pytest.approx(np.array(direct), abs=1.0e-14)


def test__operated_mapping_matrix_override__batched_numpy_path__matches_per_profile_convolution():
    # The numpy fast path convolves every linear light profile in one batched call
    # instead of looping `psf.convolved_image_from` once per profile. Each column
    # must still equal the per-profile convolution exactly, including the flux
    # blurred in from outside the mask (a bright Gaussian sits at the mask edge)
    # and for a non-symmetric kernel.
    mask = aa.Mask2D.circular(shape_native=(25, 25), pixel_scales=0.1, radius=0.9)

    kernel_native = np.random.default_rng(7).random((5, 7)) + 0.05
    kernel = aa.Array2D.no_mask(
        values=kernel_native / kernel_native.sum(), pixel_scales=0.1
    )
    psf = aa.Convolver(kernel=kernel)

    assert psf.convolve_over_sample_size == 1

    grid = aa.Grid2D.from_mask(mask=mask)
    blurring_mask = mask.derive_mask.blurring_from(
        kernel_shape_native=psf.kernel_shape_image_resolution, allow_padding=True
    )
    blurring_grid = aa.Grid2D.from_mask(mask=blurring_mask)

    light_profile_list = [
        # Bright and narrow, sat on the mask edge so its flux is dominated by the
        # blurring region -- this is what breaks if the blurring mapping matrix is
        # scattered in the wrong order.
        ag.lp_linear.Gaussian(centre=(0.85, 0.0), sigma=0.05),
        ag.lp_linear.Gaussian(centre=(-0.8, 0.6), sigma=0.08),
        ag.lp_linear.Sersic(centre=(0.1, -0.2), effective_radius=0.4, sersic_index=3.0),
        ag.lp_linear.Gaussian(centre=(0.0, 0.0), sigma=0.3),
    ]

    func_list = LightProfileLinearObjFuncList(
        grid=grid,
        blurring_grid=blurring_grid,
        psf=psf,
        light_profile_list=light_profile_list,
        regularization=None,
    )

    override = np.array(func_list.operated_mapping_matrix_override)

    assert override.shape == (mask.pixels_in_mask, len(light_profile_list))

    for i, light_profile in enumerate(light_profile_list):
        image = light_profile.image_2d_from(grid=grid, xp=np)
        blurring_image = light_profile.image_2d_from(grid=blurring_grid, xp=np)

        direct = psf.convolved_image_from(
            image=image, blurring_image=blurring_image, xp=np
        )

        assert override[:, i] == pytest.approx(np.array(direct), abs=1.0e-13)


def test__operated_mapping_matrix_override__blurring_mask_ordering_matches_convolver_state():
    # The batched call scatters the blurring mapping matrix using the blurring mask
    # derived inside the `ConvolverState` (from the resized FFT-frame mask), whereas
    # the blurring grid is built upstream from the unresized mask. The two masks live
    # on different frames, so this asserts their slim orderings are the same
    # permutation of the same pixels (a pure translation of the native indices).
    mask = aa.Mask2D.circular(shape_native=(21, 21), pixel_scales=0.1, radius=0.7)

    kernel_native = np.random.default_rng(3).random((5, 7)) + 0.05
    kernel = aa.Array2D.no_mask(
        values=kernel_native / kernel_native.sum(), pixel_scales=0.1
    )
    psf = aa.Convolver(kernel=kernel)

    upstream_blurring_mask = mask.derive_mask.blurring_from(
        kernel_shape_native=psf.kernel_shape_image_resolution, allow_padding=True
    )
    state_blurring_mask = psf.state_from(mask=mask).blurring_mask

    y_up, x_up = (np.asarray(a) for a in upstream_blurring_mask.slim_to_native_tuple)
    y_st, x_st = (np.asarray(a) for a in state_blurring_mask.slim_to_native_tuple)

    assert y_up.shape == y_st.shape

    assert (y_st - (y_st[0] - y_up[0]) == y_up).all()
    assert (x_st - (x_st[0] - x_up[0]) == x_up).all()
