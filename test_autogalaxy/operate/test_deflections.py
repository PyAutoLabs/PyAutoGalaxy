import numpy as np
import pytest

from skimage import measure

import autogalaxy as ag

from autogalaxy.operate.lens_calc import (
    grid_scaled_2d_for_marching_squares_from,
    LensCalc,
)


def critical_curve_via_magnification_from(mass_profile, grid):
    magnification = LensCalc.from_mass_obj(
        mass_profile
    ).magnification_2d_from(grid=grid)

    inverse_magnification = 1 / magnification

    critical_curves_indices = measure.find_contours(
        np.array(inverse_magnification.native._array), 0
    )

    no_critical_curves = len(critical_curves_indices)
    contours = []
    critical_curves = []

    for jj in np.arange(no_critical_curves):
        contours.append(critical_curves_indices[jj])
        contour_x, contour_y = contours[jj].T
        pixel_coord = np.stack((contour_x, contour_y), axis=-1)

        critical_curve = grid_scaled_2d_for_marching_squares_from(
            grid_pixels_2d=pixel_coord,
            shape_native=magnification.shape_native,
            mask=grid.mask,
        )

        critical_curves.append(critical_curve)

    return critical_curves


def caustics_via_magnification_from(mass_profile, grid):
    caustics = []

    critical_curves = critical_curve_via_magnification_from(
        mass_profile=mass_profile, grid=grid
    )

    for i in range(len(critical_curves)):
        critical_curve = critical_curves[i]

        deflections_1d = mass_profile.deflections_yx_2d_from(grid=critical_curve)

        caustic = critical_curve - deflections_1d

        caustics.append(caustic)

    return caustics


def test__time_delay_geometry_term_from():

    grid = ag.Grid2DIrregular(values=[(0.7, 0.5), (1.0, 1.0)])

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.0, -0.111111), einstein_radius=2.0
    )

    od = LensCalc.from_mass_obj(mp)
    time_delay_geometry_term = od.time_delay_geometry_term_from(grid=grid)

    assert time_delay_geometry_term == pytest.approx(
        np.array([1.92815688, 1.97625436]), 1.0e-4
    )


def test__fermat_potential_from():

    grid = ag.Grid2DIrregular(values=[(0.7, 0.5), (1.0, 1.0)])

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.0, -0.111111), einstein_radius=2.0
    )

    fermat_potential = LensCalc.from_mass_obj(mp).fermat_potential_from(grid=grid)

    assert fermat_potential == pytest.approx(
        np.array([0.24329033, -0.82766592]), 1.0e-4
    )


def test__hessian_from__diagonal_grid__correct_values():
    grid = ag.Grid2DIrregular(values=[(0.5, 0.5), (1.0, 1.0)])

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.0, -0.111111), einstein_radius=2.0
    )

    od = LensCalc.from_mass_obj(mp)
    hessian_yy, hessian_xy, hessian_yx, hessian_xx = od.hessian_from(grid=grid)

    assert hessian_yy == pytest.approx(np.array([1.3882113, 0.6941056]), 1.0e-4)
    assert hessian_xy == pytest.approx(np.array([-1.3882113, -0.6941056]), 1.0e-4)
    assert hessian_yx == pytest.approx(np.array([-1.3882113, -0.6941056]), 1.0e-4)
    assert hessian_xx == pytest.approx(np.array([1.3882113, 0.6941056]), 1.0e-4)


def test__hessian_from__axis_aligned_grid__correct_values():
    grid = ag.Grid2DIrregular(values=[(1.0, 0.0), (0.0, 1.0)])

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.0, -0.111111), einstein_radius=2.0
    )

    od = LensCalc.from_mass_obj(mp)
    hessian_yy, hessian_xy, hessian_yx, hessian_xx = od.hessian_from(grid=grid)

    assert hessian_yy == pytest.approx(np.array([0.0, 1.777699]), 1.0e-4)
    assert hessian_xy == pytest.approx(np.array([0.0, 0.0]), 1.0e-4)
    assert hessian_yx == pytest.approx(np.array([0.0, 0.0]), 1.0e-4)
    assert hessian_xx == pytest.approx(np.array([2.22209, 0.0]), 1.0e-4)


def test__convergence_2d_via_hessian_from():
    grid = ag.Grid2DIrregular(
        values=[(1.075, -0.125), (-0.875, -0.075), (-0.925, -0.075), (0.075, 0.925)]
    )

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.001, 0.001), einstein_radius=1.0
    )

    od = LensCalc.from_mass_obj(mp)
    convergence = od.convergence_2d_via_hessian_from(grid=grid)

    assert convergence.in_list[0] == pytest.approx(0.46208, 1.0e-1)
    assert convergence.in_list[1] == pytest.approx(0.56840, 1.0e-1)
    assert convergence.in_list[2] == pytest.approx(0.53815, 1.0e-1)
    assert convergence.in_list[3] == pytest.approx(0.53927, 1.0e-1)


def test__magnification_2d_via_hessian_from():
    grid = ag.Grid2DIrregular(values=[(0.5, 0.5), (1.0, 1.0)])

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.0, -0.111111), einstein_radius=2.0
    )

    od = LensCalc.from_mass_obj(mp)
    magnification = od.magnification_2d_via_hessian_from(grid=grid)

    assert magnification.in_list[0] == pytest.approx(-0.5629291, 1.0e-4)
    assert magnification.in_list[1] == pytest.approx(-2.575917, 1.0e-4)


def test__tangential_critical_curve_list_from__radius_matches_einstein_radius():
    grid = ag.Grid2D.uniform(shape_native=(15, 15), pixel_scales=0.3)

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    od = LensCalc.from_mass_obj(mp)
    tangential_critical_curve_list = od.tangential_critical_curve_list_from(grid=grid)

    x_critical_tangential, y_critical_tangential = (
        tangential_critical_curve_list[0][:, 1],
        tangential_critical_curve_list[0][:, 0],
    )

    assert np.mean(
        x_critical_tangential**2 + y_critical_tangential**2
    ) == pytest.approx(mp.einstein_radius**2, 5e-1)


def test__tangential_critical_curve_list_from__centre_at_origin__curve_centred_on_origin():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    od = LensCalc.from_mass_obj(mp)
    tangential_critical_curve_list = od.tangential_critical_curve_list_from(grid=grid)

    y_centre = np.mean(tangential_critical_curve_list[0][:, 0])
    x_centre = np.mean(tangential_critical_curve_list[0][:, 1])

    assert -0.03 < y_centre < 0.03
    assert -0.03 < x_centre < 0.03


def test__tangential_critical_curve_list_from__offset_centre__curve_centred_on_offset():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.IsothermalSph(centre=(0.5, 1.0), einstein_radius=2.0)

    od = LensCalc.from_mass_obj(mp)
    tangential_critical_curve_list = od.tangential_critical_curve_list_from(grid=grid)

    y_centre = np.mean(tangential_critical_curve_list[0][:, 0])
    x_centre = np.mean(tangential_critical_curve_list[0][:, 1])

    assert 0.47 < y_centre < 0.53
    assert 0.97 < x_centre < 1.03


# TODO : reinstate one JAX deflections in.

# def test__tangential_critical_curve_list_from__compare_via_magnification():
#     grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)
#
#     mp = ag.mp.Isothermal(
#         centre=(0.0, 0.0), einstein_radius=2, ell_comps=(0.109423, -0.019294)
#     )
#
#     tangential_critical_curve_via_magnification = critical_curve_via_magnification_from(
#         mass_profile=mp, grid=grid
#     )[0]
#
#     tangential_critical_curve_list = mp.tangential_critical_curve_list_from(
#         grid=grid,
#     )
#
#     assert tangential_critical_curve_list[0] == pytest.approx(
#         tangential_critical_curve_via_magnification, 5e-1
#     )
#
#     tangential_critical_curve_via_magnification = critical_curve_via_magnification_from(
#         mass_profile=mp, grid=grid
#     )[0]
#
#     tangential_critical_curve_list = mp.tangential_critical_curve_list_from(
#         grid=grid,
#     )
#
#     assert tangential_critical_curve_list[0] == pytest.approx(
#         tangential_critical_curve_via_magnification, 5e-1
#     )


def test__radial_critical_curve_list_from__centre_at_origin__curve_centred_on_origin():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.PowerLawSph(centre=(0.0, 0.0), einstein_radius=2.0, slope=1.5)

    od = LensCalc.from_mass_obj(mp)
    radial_critical_curve_list = od.radial_critical_curve_list_from(grid=grid)

    y_centre = np.mean(radial_critical_curve_list[0][:, 0])
    x_centre = np.mean(radial_critical_curve_list[0][:, 1])

    assert -0.05 < y_centre < 0.05
    assert -0.05 < x_centre < 0.05


def test__radial_critical_curve_list_from__offset_centre__curve_centred_on_offset():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.PowerLawSph(centre=(0.5, 1.0), einstein_radius=2.0, slope=1.5)

    od = LensCalc.from_mass_obj(mp)
    radial_critical_curve_list = od.radial_critical_curve_list_from(grid=grid)

    y_centre = np.mean(radial_critical_curve_list[0][:, 0])
    x_centre = np.mean(radial_critical_curve_list[0][:, 1])

    assert 0.45 < y_centre < 0.55
    assert 0.95 < x_centre < 1.05


def test__radial_critical_curve_list_from__compare_via_magnification():

    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.PowerLaw(
        centre=(0.0, 0.0), einstein_radius=2, ell_comps=(0.109423, -0.019294), slope=1.5
    )

    critical_curve_radial_via_magnification = critical_curve_via_magnification_from(
        mass_profile=mp, grid=grid
    )[1]

    od = LensCalc.from_mass_obj(mp)
    radial_critical_curve_list = od.radial_critical_curve_list_from(grid=grid)

    assert sum(critical_curve_radial_via_magnification) == pytest.approx(
        sum(radial_critical_curve_list[0]), abs=0.7
    )


def test__tangential_caustic_list_from__centre_at_origin__caustic_centred_on_origin():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    od = LensCalc.from_mass_obj(mp)
    tangential_caustic_list = od.tangential_caustic_list_from(grid=grid)

    y_centre = np.mean(tangential_caustic_list[0][:, 0])
    x_centre = np.mean(tangential_caustic_list[0][:, 1])

    assert -0.03 < y_centre < 0.03
    assert -0.03 < x_centre < 0.03


def test__tangential_caustic_list_from__offset_centre__caustic_centred_on_offset():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.IsothermalSph(centre=(0.5, 1.0), einstein_radius=2.0)

    od = LensCalc.from_mass_obj(mp)
    tangential_caustic_list = od.tangential_caustic_list_from(grid=grid)

    y_centre = np.mean(tangential_caustic_list[0][:, 0])
    x_centre = np.mean(tangential_caustic_list[0][:, 1])

    assert 0.47 < y_centre < 0.53
    assert 0.97 < x_centre < 1.03


# TODO : Reinstate one JAX defleciton sin.

# def test__tangential_caustic_list_from___compare_via_magnification():
#     grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)
#
#     mp = ag.mp.Isothermal(
#         centre=(0.0, 0.0), einstein_radius=2, ell_comps=(0.109423, -0.019294)
#     )
#
#     tangential_caustic_via_magnification = caustics_via_magnification_from(
#         mass_profile=mp, grid=grid
#     )[0]
#
#     tangential_caustic_list = mp.tangential_caustic_list_from(
#         grid=grid,
#     )
#
#     assert sum(tangential_caustic_list[0]) == pytest.approx(
#         sum(tangential_caustic_via_magnification), 5e-1
#     )


def test__radial_caustic_list_from__radius_check__correct_mean_radius():
    grid = ag.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.2)

    mp = ag.mp.PowerLawSph(centre=(0.0, 0.0), einstein_radius=2.0, slope=1.5)

    od = LensCalc.from_mass_obj(mp)
    radial_caustic_list = od.radial_caustic_list_from(grid=grid)

    x_caustic_radial, y_caustic_radial = (
        radial_caustic_list[0][:, 1],
        radial_caustic_list[0][:, 0],
    )

    assert np.mean(x_caustic_radial**2 + y_caustic_radial**2) == pytest.approx(
        0.25, 5e-1
    )


def test__radial_caustic_list_from__centre_at_origin__caustic_centred_on_origin():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.PowerLawSph(centre=(0.0, 0.0), einstein_radius=2.0, slope=1.5)

    od = LensCalc.from_mass_obj(mp)
    radial_caustic_list = od.radial_caustic_list_from(grid=grid)

    y_centre = np.mean(radial_caustic_list[0][:, 0])
    x_centre = np.mean(radial_caustic_list[0][:, 1])

    assert -0.2 < y_centre < 0.2
    assert -0.35 < x_centre < 0.35


def test__radial_caustic_list_from__offset_centre__caustic_centred_near_offset():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.PowerLawSph(centre=(0.5, 1.0), einstein_radius=2.0, slope=1.5)

    od = LensCalc.from_mass_obj(mp)
    radial_caustic_list = od.radial_caustic_list_from(grid=grid)

    y_centre = np.mean(radial_caustic_list[0][:, 0])
    x_centre = np.mean(radial_caustic_list[0][:, 1])

    assert 0.3 < y_centre < 0.7
    assert 0.7 < x_centre < 1.2


def test__radial_caustic_list_from___compare_via_magnification():
    grid = ag.Grid2D.uniform(shape_native=(180, 180), pixel_scales=0.02)

    mp = ag.mp.PowerLaw(
        centre=(0.0, 0.0), einstein_radius=2, ell_comps=(0.109423, -0.019294), slope=1.5
    )

    caustic_radial_via_magnification = caustics_via_magnification_from(
        mass_profile=mp, grid=grid
    )[1]

    od = LensCalc.from_mass_obj(mp)
    radial_caustic_list = od.radial_caustic_list_from(grid=grid)

    assert sum(radial_caustic_list[0]) == pytest.approx(
        sum(caustic_radial_via_magnification), 7e-1
    )


def test__radial_critical_curve_area_list_from():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.PowerLawSph(centre=(0.0, 0.0), einstein_radius=2.0, slope=1.5)

    od = LensCalc.from_mass_obj(mp)
    area_within_radial_critical_curve_list = od.radial_critical_curve_area_list_from(
        grid=grid
    )

    assert area_within_radial_critical_curve_list[0] == pytest.approx(0.78293, 1e-1)


def test__tangential_critical_curve_area_list_from():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    area_calc = np.pi * mp.einstein_radius**2

    od = LensCalc.from_mass_obj(mp)
    area_within_tangential_critical_curve_list = (
        od.tangential_critical_curve_area_list_from(grid=grid)
    )

    assert area_within_tangential_critical_curve_list[0] == pytest.approx(
        area_calc, 1e-1
    )


def test__einstein_radius_list_from__isothermal_sph__correct_einstein_radius():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    od = LensCalc.from_mass_obj(mp)
    einstein_radius_list = od.einstein_radius_list_from(grid=grid)

    assert einstein_radius_list[0] == pytest.approx(2.0, 1e-1)


def test__einstein_radius_list_from__isothermal_elliptical__correct_einstein_radius():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), einstein_radius=2.0, ell_comps=(0.0, -0.25)
    )

    od = LensCalc.from_mass_obj(mp)
    einstein_radius_list = od.einstein_radius_list_from(grid=grid)

    assert einstein_radius_list[0] == pytest.approx(1.9360, 1e-1)


def test__einstein_radius_from__isothermal_sph__correct_einstein_radius():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    od = LensCalc.from_mass_obj(mp)
    einstein_radius = od.einstein_radius_from(grid=grid)

    assert einstein_radius == pytest.approx(2.0, 1e-1)


def test__einstein_radius_from__isothermal_elliptical__correct_einstein_radius():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), einstein_radius=2.0, ell_comps=(0.0, -0.25)
    )

    od = LensCalc.from_mass_obj(mp)
    einstein_radius = od.einstein_radius_from(grid=grid)

    assert einstein_radius == pytest.approx(1.9360, 1e-1)


def test__einstein_mass_angular_list_from():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    od = LensCalc.from_mass_obj(mp)
    einstein_mass_angular_list = od.einstein_mass_angular_list_from(grid=grid)

    assert einstein_mass_angular_list[0] == pytest.approx(np.pi * 2.0**2.0, 1e-1)


def test__einstein_mass_angular_from():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    od = LensCalc.from_mass_obj(mp)
    einstein_mass_angular = od.einstein_mass_angular_from(grid=grid)

    assert einstein_mass_angular == pytest.approx(np.pi * 2.0**2.0, 1e-1)


def test__jacobian_from():
    """
    The Jacobian is A = I - H, where H is the Hessian of the deflection angles.

    This test verifies the structure and values of `jacobian_from` by checking that:
    - it returns a 2x2 list of lists;
    - the convergence derived from its diagonal matches `convergence_2d_via_hessian_from`;
    - the magnification derived from its determinant matches `magnification_2d_via_hessian_from`.
    """
    grid = ag.Grid2DIrregular(values=[(1.0, 1.0), (2.0, 0.5)])

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.0, -0.111111), einstein_radius=2.0
    )

    od = LensCalc.from_mass_obj(mp)
    jacobian = od.jacobian_from(grid=grid)

    assert len(jacobian) == 2
    assert len(jacobian[0]) == 2 and len(jacobian[1]) == 2

    # convergence = 1 - 0.5 * (a11 + a22) should match convergence_2d_via_hessian_from
    convergence_via_jacobian = 1 - 0.5 * (jacobian[0][0] + jacobian[1][1])
    convergence_via_hessian = od.convergence_2d_via_hessian_from(grid=grid)

    assert convergence_via_jacobian == pytest.approx(
        np.array(convergence_via_hessian), rel=1e-6
    )

    # magnification = 1 / det(A) = 1 / (a11*a22 - a12*a21)
    det_A = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]
    magnification_via_jacobian = 1 / det_A
    magnification_via_hessian = od.magnification_2d_via_hessian_from(grid=grid)

    assert magnification_via_jacobian == pytest.approx(
        np.array(magnification_via_hessian), rel=1e-6
    )
