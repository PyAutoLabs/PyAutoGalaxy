"""
Area magnification (LEGGOS II, arXiv:2606.20804, Eq. 6-7) on `LensCalc`.

The singular isothermal sphere is the analytic witness: its magnification is
`mu = theta / (theta - theta_E)`, so an image-plane annulus `r1 <= theta <= r2`
outside the Einstein radius maps from the source-plane annulus
`r1 - theta_E <= beta <= r2 - theta_E` and

    mu_area = (r2^2 - r1^2) / ((r2 - theta_E)^2 - (r1 - theta_E)^2)

exactly; inside the Einstein radius the image is parity-flipped (mu < 0).
"""

import numpy as np
import pytest

import autoarray as aa
import autogalaxy as ag

EINSTEIN_RADIUS = 1.0


def _lens_calc():
    mass = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=EINSTEIN_RADIUS)
    return ag.LensCalc.from_mass_obj(mass)


def _annulus_mask(inner, outer, pixel_scale=0.02):
    shape = int(2 * (outer + 0.2) / pixel_scale)
    return aa.Mask2D.circular_annular(
        shape_native=(shape, shape),
        pixel_scales=pixel_scale,
        inner_radius=inner,
        outer_radius=outer,
    )


def _analytic(r1, r2):
    return (r2**2 - r1**2) / ((r2 - EINSTEIN_RADIUS) ** 2 - (r1 - EINSTEIN_RADIUS) ** 2)


def test__sis_annulus__matches_the_analytic_area_magnification():
    r1, r2 = 1.5, 2.0

    mu_area = _lens_calc().area_magnification_from(
        mask=_annulus_mask(inner=r1, outer=r2), sub_size=4
    )

    # measured -4.7e-4; the pixelized annulus edge, not the method, sets it
    assert mu_area == pytest.approx(_analytic(r1, r2), rel=2.0e-3)


def test__source_plane_area__matches_the_analytic_source_annulus():
    r1, r2 = 1.5, 2.0

    source_plane_area = _lens_calc().source_plane_area_from(
        mask=_annulus_mask(inner=r1, outer=r2), sub_size=4
    )

    analytic = np.pi * ((r2 - EINSTEIN_RADIUS) ** 2 - (r1 - EINSTEIN_RADIUS) ** 2)

    # measured -1.5e-4
    assert source_plane_area == pytest.approx(analytic, rel=1.0e-3)


def test__sub_size__converges():
    """
    On a coarse mask the per-pixel area integral is evaluated at sub-pixel
    centres; refining the sub-pixels must converge on the same answer.
    """
    mask = _annulus_mask(inner=1.2, outer=1.8, pixel_scale=0.1)
    lens_calc = _lens_calc()

    reference = lens_calc.source_plane_area_from(mask=mask, sub_size=32)

    errors = [
        abs(lens_calc.source_plane_area_from(mask=mask, sub_size=s) / reference - 1)
        for s in (1, 2, 4, 8)
    ]

    # measured 3.9e-4, 9.7e-5, 2.4e-5, 5.7e-6: second order in the sub-pixel size
    assert all(a > b for a, b in zip(errors, errors[1:]))
    assert errors[2] < 1.0e-4


def test__image_masks__split_a_mask_across_the_critical_curve_by_parity():
    """
    An annulus straddling the Einstein ring contains the positive-parity outer
    image ring and the negative-parity inner ring; they are returned separately,
    and the outer one's area magnification matches the analytic value.
    """
    lens_calc = _lens_calc()

    mask = _annulus_mask(inner=0.6, outer=1.6)

    image_masks = lens_calc.image_masks_from(mask=mask)

    assert len(image_masks) == 2

    total = sum(np.sum(~np.asarray(image_mask)) for image_mask in image_masks)
    assert total == np.sum(~np.asarray(mask))

    signs = []
    for image_mask in image_masks:
        grid = aa.Grid2D.from_mask(mask=image_mask, over_sample_size=1)
        mu = np.asarray(
            lens_calc.magnification_2d_via_hessian_from(
                grid=aa.Grid2DIrregular(values=np.asarray(grid.array))
            )
        )
        assert np.all(np.sign(mu) == np.sign(mu[0]))
        signs.append(np.sign(mu[0]))

    assert sorted(signs) == [-1.0, 1.0]

    mu_list = lens_calc.area_magnification_list_from(mask=mask, sub_size=4)

    assert len(mu_list) == 2
    assert all(mu > 0.0 for mu in mu_list)

    # the positive-parity image is the outer ring, theta_E <= theta <= 1.6
    # (measured -6e-4; the Einstein-ring boundary is pixelized)
    outer = mu_list[signs.index(1.0)]
    assert outer == pytest.approx(_analytic(EINSTEIN_RADIUS, 1.6), rel=3.0e-3)


def test__a_mask_spanning_two_images_differs_from_each_image():
    """
    The documented trap: summing A_src over two images counts the source twice,
    so the whole-mask ratio is not either image's magnification.
    """
    lens_calc = _lens_calc()

    mask = _annulus_mask(inner=0.6, outer=1.6)

    whole = lens_calc.area_magnification_from(mask=mask, sub_size=4)
    per_image = lens_calc.area_magnification_list_from(mask=mask, sub_size=4)

    assert all(abs(whole / mu - 1.0) > 0.01 for mu in per_image)


def test__source_plane_area__correct_when_over_sampling_is_capped(monkeypatch):
    """
    `PYAUTO_SMALL_DATASETS=1` caps over-sampling at 2 whatever `sub_size` asks for;
    the sub-pixel area must follow the sub-pixels actually generated.
    """
    mask = _annulus_mask(inner=1.5, outer=2.0, pixel_scale=0.05)
    lens_calc = _lens_calc()

    uncapped = lens_calc.source_plane_area_from(mask=mask, sub_size=4)

    monkeypatch.setenv("PYAUTO_SMALL_DATASETS", "1")

    capped = lens_calc.source_plane_area_from(mask=mask, sub_size=4)

    assert capped == pytest.approx(uncapped, rel=1.0e-3)
