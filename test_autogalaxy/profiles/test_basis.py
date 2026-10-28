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


def test__image_2d_from__grid_2d_irregular__linear_profile_placeholder_has_no_mask():
    # Regression test: the linear-intensity-unknown placeholder used to be built as
    # `Array2D(..., mask=grid.mask)`, which raised `AttributeError: Grid2DIrregular
    # does not have attribute mask` whenever a basis containing a linear light
    # profile was evaluated on an irregular grid (as JIT-traced likelihood paths do).
    grid = aa.Grid2DIrregular(values=[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])

    lp = ag.lp.Sersic(intensity=0.1)
    lp_linear = ag.lp_linear.Sersic(effective_radius=2.0, sersic_index=2.0)

    basis = ag.lp_basis.Basis(profile_list=[lp, lp_linear])

    image = basis.image_2d_from(grid=grid)

    assert isinstance(image, aa.ArrayIrregular)
    assert image.shape == (3,)
    assert image == pytest.approx(np.asarray(lp.image_2d_from(grid=grid)), 1.0e-8)


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


def test__mass_basis_from__copies_gaussian_fields_with_shared_mass_to_light_ratio():
    light_basis = ag.lp_basis.Basis(
        profile_list=[
            ag.lp.Gaussian(
                centre=(0.1, -0.2), ell_comps=(0.05, 0.1), intensity=1.5, sigma=0.3
            ),
            ag.lp.Gaussian(
                centre=(0.1, -0.2), ell_comps=(0.05, 0.1), intensity=0.5, sigma=1.2
            ),
            ag.lmp.Gaussian(
                centre=(0.0, 0.1),
                ell_comps=(-0.1, 0.0),
                intensity=0.2,
                sigma=2.0,
                mass_to_light_ratio=9.0,
            ),
        ],
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    mass_basis = light_basis.mass_basis_from(mass_to_light_ratio=3.5)

    # The fields `chaining_util.mass_light_dark_basis_from` copies onto each `lmp` model
    # (centre, ell_comps, intensity, sigma) plus one shared mass_to_light_ratio, built by hand.
    expected_list = [
        ag.lmp.Gaussian(
            centre=(0.1, -0.2),
            ell_comps=(0.05, 0.1),
            intensity=1.5,
            sigma=0.3,
            mass_to_light_ratio=3.5,
        ),
        ag.lmp.Gaussian(
            centre=(0.1, -0.2),
            ell_comps=(0.05, 0.1),
            intensity=0.5,
            sigma=1.2,
            mass_to_light_ratio=3.5,
        ),
        ag.lmp.Gaussian(
            centre=(0.0, 0.1),
            ell_comps=(-0.1, 0.0),
            intensity=0.2,
            sigma=2.0,
            mass_to_light_ratio=3.5,
        ),
    ]

    assert isinstance(mass_basis, ag.lp_basis.Basis)
    assert mass_basis is not light_basis
    assert mass_basis.regularization is light_basis.regularization
    assert len(mass_basis.profile_list) == len(expected_list)

    for mass_profile, expected in zip(mass_basis.profile_list, expected_list):
        assert type(mass_profile) is ag.lmp.Gaussian
        assert mass_profile.centre == expected.centre
        assert mass_profile.ell_comps == expected.ell_comps
        assert mass_profile.intensity == expected.intensity
        assert mass_profile.sigma == expected.sigma
        assert mass_profile.mass_to_light_ratio == expected.mass_to_light_ratio

    # The input basis is unchanged.
    assert light_basis.profile_list[2].mass_to_light_ratio == 9.0


def test__mass_basis_from__convergence_is_mass_to_light_ratio_times_image():
    # No over-sampling, so the image and the convergence are evaluated at the same points.
    grid = ag.Grid2D.uniform(shape_native=(7, 7), pixel_scales=0.3, over_sample_size=1)

    light_basis = ag.lp_basis.Basis(
        profile_list=[
            ag.lp.Gaussian(
                centre=(0.1, -0.2), ell_comps=(0.05, 0.1), intensity=1.5, sigma=0.3
            ),
            ag.lp.Gaussian(
                centre=(0.0, 0.0), ell_comps=(-0.1, 0.2), intensity=0.5, sigma=1.2
            ),
        ]
    )

    mass_basis = light_basis.mass_basis_from(mass_to_light_ratio=2.5)

    convergence = mass_basis.convergence_2d_from(grid=grid)
    image = light_basis.image_2d_from(grid=grid)

    assert np.allclose(
        np.asarray(convergence.array), 2.5 * np.asarray(image.array), rtol=1.0e-10
    )

    # `mp.Gaussian` floors its axis ratio at 0.9999 (the analytic deflections diverge at q = 1),
    # so for circular members the identity holds only to that floor's ~1e-4 level.
    light_basis = ag.lp_basis.Basis(
        profile_list=[ag.lp.GaussianSph(centre=(0.0, 0.0), intensity=0.5, sigma=1.2)]
    )

    mass_basis = light_basis.mass_basis_from(mass_to_light_ratio=2.5)

    convergence = mass_basis.convergence_2d_from(grid=grid)
    image = light_basis.image_2d_from(grid=grid)

    assert np.allclose(
        np.asarray(convergence.array), 2.5 * np.asarray(image.array), rtol=1.0e-3
    )


@pytest.mark.parametrize(
    "profile",
    [
        ag.lp.Sersic(intensity=0.1),
        ag.lp_linear.Gaussian(sigma=0.5),
        ag.mp.Isothermal(),
    ],
)
def test__mass_basis_from__member_without_lmp_gaussian_counterpart__raises(profile):
    basis = ag.lp_basis.Basis(
        profile_list=[ag.lp.Gaussian(intensity=1.0, sigma=0.5), profile]
    )

    with pytest.raises(ag.exc.ProfileException, match=type(profile).__name__):
        basis.mass_basis_from(mass_to_light_ratio=1.0)
