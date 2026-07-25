import pytest
import numpy as np

import autofit as af
import autogalaxy as ag


def test__mge_model_from__single_basis_elliptical():
    model = ag.model_util.mge_model_from(
        mask_radius=1.0, total_gaussians=5, gaussian_per_basis=1
    )
    assert model.prior_count == 4


def test__mge_model_from__single_basis_spherical():
    model = ag.model_util.mge_model_from(
        mask_radius=1.0, total_gaussians=5, gaussian_per_basis=1, use_spherical=True
    )
    assert model.prior_count == 2


def test__mge_model_from__two_bases_shared_centre():
    model = ag.model_util.mge_model_from(
        mask_radius=1.0, total_gaussians=5, gaussian_per_basis=2
    )
    # 2 shared centre + 2 ell_comps per basis * 2 = 6
    assert model.prior_count == 6


def test__mge_model_from__two_bases_centre_per_basis():
    model = ag.model_util.mge_model_from(
        mask_radius=1.0, total_gaussians=5, gaussian_per_basis=2, centre_per_basis=True
    )
    # 2 centre per basis * 2 + 2 ell_comps per basis * 2 = 8
    assert model.prior_count == 8


def test__mge_model_from__three_bases_shared_centre():
    model = ag.model_util.mge_model_from(
        mask_radius=1.0, total_gaussians=5, gaussian_per_basis=3
    )
    # 2 shared centre + 2 ell_comps * 3 = 8
    assert model.prior_count == 8


def test__mge_model_from__two_bases_spherical_shared_centre():
    model = ag.model_util.mge_model_from(
        mask_radius=1.0, total_gaussians=5, gaussian_per_basis=2, use_spherical=True
    )
    # Spherical: no ell_comps, shared centre = 2
    assert model.prior_count == 2


def test__mge_model_from__two_bases_spherical_centre_per_basis():
    model = ag.model_util.mge_model_from(
        mask_radius=1.0,
        total_gaussians=5,
        gaussian_per_basis=2,
        use_spherical=True,
        centre_per_basis=True,
    )
    # Spherical + centre_per_basis: 2 centre * 2 = 4
    assert model.prior_count == 4


def test__mge_model_from__centre_fixed_with_two_bases():
    model = ag.model_util.mge_model_from(
        mask_radius=1.0,
        total_gaussians=5,
        gaussian_per_basis=2,
        centre_fixed=(0.0, 0.0),
    )
    # Centres fixed (0 free), 2 ell_comps per basis * 2 = 4
    assert model.prior_count == 4


def test__mge_model_from__centre_fixed_overrides_centre_per_basis():
    model = ag.model_util.mge_model_from(
        mask_radius=1.0,
        total_gaussians=5,
        gaussian_per_basis=2,
        centre_fixed=(0.0, 0.0),
        centre_per_basis=True,
    )
    # centre_fixed takes precedence: 0 centre + 4 ell_comps = 4
    assert model.prior_count == 4


def test__mge_model_from__backward_compat_single_basis():
    """gaussian_per_basis=1 must produce identical output to the old code."""
    model = ag.model_util.mge_model_from(mask_radius=3.0, total_gaussians=10)
    assert model.prior_count == 4

    instance = model.instance_from_prior_medians()
    assert isinstance(instance, ag.lp_basis.Basis)
    assert len(instance.profile_list) == 10


def test__mge_model_from__total_gaussians_per_basis():
    """With gaussian_per_basis=2 and total_gaussians=5, should produce 10 Gaussians."""
    model = ag.model_util.mge_model_from(
        mask_radius=1.0, total_gaussians=5, gaussian_per_basis=2
    )
    instance = model.instance_from_prior_medians()
    assert len(instance.profile_list) == 10


def test__mge_point_model_from__returns_basis_model_with_correct_gaussians():
    """
    mge_point_model_from should return an af.Model wrapping a Basis whose
    profile_list contains the requested number of linear Gaussian components.
    """
    model = ag.model_util.mge_point_model_from(pixel_scales=0.1, total_gaussians=2)

    instance = model.instance_from_prior_medians()

    assert isinstance(instance, ag.lp_basis.Basis)
    assert len(instance.profile_list) == 2


def test__mge_point_model_from__sigma_values_span_correct_range():
    """
    Sigma values should run from 10^-2 = 0.01 arcseconds up to 2 * pixel_scales,
    logarithmically spaced.
    """
    pixel_scales = 0.1
    total_gaussians = 3

    model = ag.model_util.mge_point_model_from(
        pixel_scales=pixel_scales, total_gaussians=total_gaussians
    )

    gaussian_list = list(model.profile_list)

    assert gaussian_list[0].sigma == pytest.approx(0.01, rel=1.0e-4)
    assert gaussian_list[-1].sigma == pytest.approx(pixel_scales * 2.0, rel=1.0e-4)


def test__mge_point_model_from__shared_centre_and_ell_comps():
    """
    All Gaussians must share exactly the same centre prior objects and ell_comps
    prior objects so the model has only 4 free parameters total.
    """
    model = ag.model_util.mge_point_model_from(pixel_scales=0.1, total_gaussians=2)

    gaussian_list = list(model.profile_list)

    # Centres are all the same prior objects
    for gaussian in gaussian_list[1:]:
        assert gaussian.centre.centre_0 is gaussian_list[0].centre.centre_0
        assert gaussian.centre.centre_1 is gaussian_list[0].centre.centre_1

    # Ell_comps are all the same prior objects
    for gaussian in gaussian_list[1:]:
        assert gaussian.ell_comps is gaussian_list[0].ell_comps

    # Only 4 free parameters: centre_0, centre_1, ell_comps_0, ell_comps_1
    assert model.prior_count == 4


def test__hilbert_pixels_from_pixel_scale__above_006():
    assert ag.model_util.hilbert_pixels_from_pixel_scale(0.07) == 1000
    assert ag.model_util.hilbert_pixels_from_pixel_scale(0.1) == 1000


def test__hilbert_pixels_from_pixel_scale__between_004_and_006():
    assert ag.model_util.hilbert_pixels_from_pixel_scale(0.05) == 1250
    assert ag.model_util.hilbert_pixels_from_pixel_scale(0.061) == 1000
    assert ag.model_util.hilbert_pixels_from_pixel_scale(0.041) == 1250


def test__hilbert_pixels_from_pixel_scale__between_003_and_004():
    assert ag.model_util.hilbert_pixels_from_pixel_scale(0.03) == 1500
    assert ag.model_util.hilbert_pixels_from_pixel_scale(0.035) == 1500


def test__hilbert_pixels_from_pixel_scale__below_003():
    assert ag.model_util.hilbert_pixels_from_pixel_scale(0.02) == 1750
    assert ag.model_util.hilbert_pixels_from_pixel_scale(0.01) == 1750


def test__hilbert_pixels_from_pixel_scale__raises_for_non_positive():
    with pytest.raises(ValueError):
        ag.model_util.hilbert_pixels_from_pixel_scale(0.0)
    with pytest.raises(ValueError):
        ag.model_util.hilbert_pixels_from_pixel_scale(-0.05)


def test__hilbert_pixels_from_pixel_scale__raises_for_non_finite():
    with pytest.raises(ValueError):
        ag.model_util.hilbert_pixels_from_pixel_scale(float("nan"))
    with pytest.raises(ValueError):
        ag.model_util.hilbert_pixels_from_pixel_scale(float("inf"))


def test__hilbert_pixels_from_pixel_scale__boundary_values():
    # Exactly 0.06 is NOT > 0.06, so falls to next branch (> 0.04 → 1250)
    assert ag.model_util.hilbert_pixels_from_pixel_scale(0.06) == 1250
    # Exactly 0.04 is NOT > 0.04, but IS >= 0.03 → 1500
    assert ag.model_util.hilbert_pixels_from_pixel_scale(0.04) == 1500


def test__mge_point_model_from__centre_prior_bounds():
    """
    When a custom centre is supplied the UniformPrior limits shift by ±0.1
    arcseconds around that centre.
    """
    centre = (0.3, -0.2)
    model = ag.model_util.mge_point_model_from(
        pixel_scales=0.1, total_gaussians=3, centre=centre
    )

    gaussian_list = list(model.profile_list)
    centre_0_prior = gaussian_list[0].centre.centre_0
    centre_1_prior = gaussian_list[0].centre.centre_1

    assert isinstance(centre_0_prior, af.UniformPrior)
    assert centre_0_prior.lower_limit == pytest.approx(centre[0] - 0.1, rel=1.0e-6)
    assert centre_0_prior.upper_limit == pytest.approx(centre[0] + 0.1, rel=1.0e-6)

    assert isinstance(centre_1_prior, af.UniformPrior)
    assert centre_1_prior.lower_limit == pytest.approx(centre[1] - 0.1, rel=1.0e-6)
    assert centre_1_prior.upper_limit == pytest.approx(centre[1] + 0.1, rel=1.0e-6)
