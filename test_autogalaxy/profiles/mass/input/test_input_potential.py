import numpy as np
import pytest

import autoarray as aa
import autogalaxy as ag


def masked_setup(shape=(16, 16), pixel_scale=0.5, radius=3.4):
    mask = aa.Mask2D.circular(
        shape_native=shape, pixel_scales=pixel_scale, radius=radius
    )
    cleaned, _ = aa.util.derivative.cleaned_mask_from(np.asarray(mask))
    mask = aa.Mask2D(mask=cleaned, pixel_scales=pixel_scale)
    grid = aa.Grid2D.from_mask(mask=mask)
    return mask, grid


def test__linear_potential__deflections_exact_and_convergence_zero():
    mask, grid = masked_setup()
    grid_arr = np.asarray(grid)

    # psi = 2 y + 3 x: alpha = (2, 3) everywhere, kappa = 0. All the mask's
    # finite-difference schemes are exact on a linear potential.
    potential = 2.0 * grid_arr[:, 0] + 3.0 * grid_arr[:, 1]

    profile = ag.mp.InputPotential(
        lensing_potential=potential, image_plane_grid=grid_arr, mask=mask
    )

    deflections = np.asarray(profile.deflections_yx_2d_from(grid=grid))
    assert deflections[:, 0] == pytest.approx(2.0, abs=1.0e-8)
    assert deflections[:, 1] == pytest.approx(3.0, abs=1.0e-8)

    convergence = np.asarray(profile.convergence_2d_from(grid=grid))
    assert convergence == pytest.approx(0.0, abs=1.0e-8)


def test__quadratic_potential__convergence_exact_and_potential_interp_at_nodes():
    mask, grid = masked_setup()
    grid_arr = np.asarray(grid)

    # psi = a y^2 + b x^2: kappa = 0.5 (2a + 2b) = a + b. Second-difference
    # schemes (central and one-sided) are exact on quadratics.
    a, b = 0.75, 0.25
    potential = a * grid_arr[:, 0] ** 2 + b * grid_arr[:, 1] ** 2

    profile = ag.mp.InputPotential(
        lensing_potential=potential, image_plane_grid=grid_arr, mask=mask
    )

    convergence = np.asarray(profile.convergence_2d_from(grid=grid))
    assert convergence == pytest.approx(a + b, abs=1.0e-8)

    # linear interpolation at the triangulation nodes returns the inputs
    potential_interp = np.asarray(profile.potential_2d_from(grid=grid))
    assert potential_interp == pytest.approx(potential, abs=1.0e-8)


def test__operators_can_be_preloaded():
    mask, grid = masked_setup()
    grid_arr = np.asarray(grid)
    potential = np.asarray(grid_arr[:, 0])

    Hy, Hx = aa.util.derivative.derivative_1st_operators_from(
        np.asarray(mask), pixel_scale=mask.pixel_scale
    )
    Hyy, Hxx = aa.util.derivative.derivative_2nd_operators_from(
        np.asarray(mask), pixel_scale=mask.pixel_scale
    )

    profile = ag.mp.InputPotential(
        lensing_potential=potential,
        image_plane_grid=grid_arr,
        mask=mask,
        Hy=Hy,
        Hx=Hx,
        Hyy=Hyy,
        Hxx=Hxx,
    )

    deflections = np.asarray(profile.deflections_yx_2d_from(grid=grid))
    assert deflections[:, 0] == pytest.approx(1.0, abs=1.0e-8)


def test__zero_extrapolation__deflections_vanish_outside_mesh():
    mask, grid = masked_setup()
    grid_arr = np.asarray(grid)
    potential = 2.0 * grid_arr[:, 0] + 3.0 * grid_arr[:, 1]

    profile_nearest = ag.mp.InputPotential(
        lensing_potential=potential, image_plane_grid=grid_arr, mask=mask
    )
    profile_zero = ag.mp.InputPotential(
        lensing_potential=potential,
        image_plane_grid=grid_arr,
        mask=mask,
        extrapolate="zero",
    )

    far_outside = aa.Grid2DIrregular(values=[(6.0, 6.0), (-7.0, 5.0)])

    deflections_nearest = np.asarray(
        profile_nearest.deflections_yx_2d_from(grid=far_outside)
    )
    deflections_zero = np.asarray(profile_zero.deflections_yx_2d_from(grid=far_outside))

    # nearest extrapolation smears constant non-zero deflections outward;
    # zero extrapolation vanishes
    assert not np.allclose(deflections_nearest, 0.0)
    assert deflections_zero == pytest.approx(0.0, abs=1.0e-12)

    # inside the mesh the two modes agree exactly
    inside = aa.Grid2DIrregular(values=[(0.1, -0.2), (0.7, 0.4)])
    assert np.asarray(profile_zero.deflections_yx_2d_from(grid=inside)) == pytest.approx(
        np.asarray(profile_nearest.deflections_yx_2d_from(grid=inside)), abs=1.0e-12
    )


def test__invalid_extrapolate_raises():
    mask, grid = masked_setup()
    grid_arr = np.asarray(grid)
    with pytest.raises(ValueError):
        ag.mp.InputPotential(
            lensing_potential=np.ones(grid_arr.shape[0]),
            image_plane_grid=grid_arr,
            mask=mask,
            extrapolate="taper",
        )
