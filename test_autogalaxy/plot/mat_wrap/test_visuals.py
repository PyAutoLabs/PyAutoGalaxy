from os import path
import pytest

import autogalaxy.plot as aplt
from autogalaxy.operate.lens_calc import LensCalc

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_profile_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "profiles"
    )


def test__1d__add_half_light_radius(lp_0):

    visuals_1d_via = aplt.Visuals1D().add_half_light_radius(light_obj=lp_0)

    assert visuals_1d_via.half_light_radius == lp_0.half_light_radius


def test__1d__add_half_light_radius_errors(lp_0):

    visuals_1d_via = aplt.Visuals1D().add_half_light_radius_errors(
        light_obj_list=[lp_0, lp_0], low_limit=1.0
    )

    assert visuals_1d_via.half_light_radius == lp_0.half_light_radius
    assert visuals_1d_via.half_light_radius_errors[0][0] == lp_0.half_light_radius


def test__1d__add_einstein_radius(mp_0, grid_2d_7x7):

    visuals_1d_via = aplt.Visuals1D().add_einstein_radius(
        mass_obj=mp_0, grid=grid_2d_7x7
    )

    assert visuals_1d_via.einstein_radius == LensCalc.from_mass_obj(
        mp_0
    ).einstein_radius_from(grid=grid_2d_7x7)


def test__1d__add_einstein_radius_errors(mp_0, grid_2d_7x7):

    visuals_1d_via = aplt.Visuals1D().add_einstein_radius_errors(
        mass_obj_list=[mp_0, mp_0], grid=grid_2d_7x7, low_limit=1.0
    )

    od = LensCalc.from_mass_obj(mp_0)
    assert visuals_1d_via.einstein_radius == od.einstein_radius_from(grid=grid_2d_7x7)
    assert visuals_1d_via.einstein_radius_errors[0][0] == od.einstein_radius_from(
        grid=grid_2d_7x7
    )


def test__2d__add_critical_curve(gal_x1_mp, grid_2d_7x7):

    visuals_2d_via = aplt.Visuals2D().add_critical_curves_or_caustics(
        mass_obj=gal_x1_mp, grid=grid_2d_7x7, plane_index=0
    )

    od = LensCalc.from_mass_obj(gal_x1_mp)
    assert (
        visuals_2d_via.tangential_critical_curves[0]
        == od.tangential_critical_curve_list_from(grid=grid_2d_7x7)[0]
    ).all()


def test__2d__add_caustic(gal_x1_mp, grid_2d_7x7):

    visuals_2d_via = aplt.Visuals2D().add_critical_curves_or_caustics(
        mass_obj=gal_x1_mp, grid=grid_2d_7x7, plane_index=1
    )

    od = LensCalc.from_mass_obj(gal_x1_mp)
    assert (
        visuals_2d_via.tangential_caustics[0]
        == od.tangential_caustic_list_from(grid=grid_2d_7x7)[0]
    ).all()
    assert (
        visuals_2d_via.radial_caustics[0]
        == od.radial_caustic_list_from(grid=grid_2d_7x7)[0]
    ).all()
