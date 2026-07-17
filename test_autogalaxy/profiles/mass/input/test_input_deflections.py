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


def test__linear_deflections__interp_exact_at_nodes_and_convergence_exact():
    mask, grid = masked_setup()
    grid_arr = np.asarray(grid)

    # alpha_y = a y, alpha_x = b x (the deflections of psi = a y^2/2 + b x^2/2):
    # kappa = 0.5 (a + b), exact for every first-difference scheme since the
    # deflections are linear in the coordinates.
    a, b = 1.5, 0.5
    deflections_y = a * grid_arr[:, 0]
    deflections_x = b * grid_arr[:, 1]

    profile = ag.mp.InputDeflections(
        deflections_y=deflections_y,
        deflections_x=deflections_x,
        image_plane_grid=grid_arr,
        mask=mask,
    )

    deflections = np.asarray(profile.deflections_yx_2d_from(grid=grid))
    assert deflections[:, 0] == pytest.approx(deflections_y, abs=1.0e-8)
    assert deflections[:, 1] == pytest.approx(deflections_x, abs=1.0e-8)

    convergence = np.asarray(profile.convergence_2d_from(grid=grid))
    assert convergence == pytest.approx(0.5 * (a + b), abs=1.0e-8)


def test__potential_returns_zeros():
    mask, grid = masked_setup()
    grid_arr = np.asarray(grid)

    profile = ag.mp.InputDeflections(
        deflections_y=np.ones(grid_arr.shape[0]),
        deflections_x=np.ones(grid_arr.shape[0]),
        image_plane_grid=grid_arr,
        mask=mask,
    )

    potential = np.asarray(profile.potential_2d_from(grid=grid))
    assert potential == pytest.approx(0.0, abs=1.0e-12)


def test__interpolation_off_nodes_is_linear():
    mask, grid = masked_setup()
    grid_arr = np.asarray(grid)

    a, b = 1.5, 0.5
    profile = ag.mp.InputDeflections(
        deflections_y=a * grid_arr[:, 0],
        deflections_x=b * grid_arr[:, 1],
        image_plane_grid=grid_arr,
        mask=mask,
    )

    # linear interpolation reproduces a linear deflection field exactly at
    # positions inside the convex hull, including off-node positions.
    off_node = aa.Grid2DIrregular(values=[(0.13, -0.41), (1.07, 0.66)])
    deflections = np.asarray(profile.deflections_yx_2d_from(grid=off_node))
    assert deflections[:, 0] == pytest.approx(
        a * np.asarray(off_node)[:, 0], abs=1.0e-8
    )
    assert deflections[:, 1] == pytest.approx(
        b * np.asarray(off_node)[:, 1], abs=1.0e-8
    )
