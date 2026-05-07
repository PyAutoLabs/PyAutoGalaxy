from pathlib import Path
import pytest

from autogalaxy.operate.lens_calc import LensCalc

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_profile_plotter_setup():
    return Path(__file__).resolve().parent / "files" / "plots" / "profiles"


def test__1d__half_light_radius_from_light_profile(lp_0):
    assert lp_0.half_light_radius is not None


def test__1d__einstein_radius_from_mass_profile(mp_0, grid_2d_7x7):
    od = LensCalc.from_mass_obj(mp_0)
    einstein_radius = od.einstein_radius_from(grid=grid_2d_7x7)
    assert einstein_radius is not None


def test__2d__critical_curves_from_mass_obj(gal_x1_mp, grid_2d_7x7):
    od = LensCalc.from_mass_obj(gal_x1_mp)
    tc = od.tangential_critical_curve_list_from(grid=grid_2d_7x7)
    assert tc is not None
    assert len(tc) > 0


def test__2d__caustics_from_mass_obj(gal_x1_mp, grid_2d_7x7):
    od = LensCalc.from_mass_obj(gal_x1_mp)
    tc = od.tangential_caustic_list_from(grid=grid_2d_7x7)
    rc = od.radial_caustic_list_from(grid=grid_2d_7x7)
    assert tc is not None
    assert rc is not None


def test__mass_plotter__tangential_critical_curves(gal_x1_mp, grid_2d_7x7):
    import numpy as np
    from autogalaxy.plot.plot_utils import _critical_curves_from

    tc, rc = _critical_curves_from(gal_x1_mp, grid_2d_7x7)

    assert tc is not None
    assert len(tc) > 0

    # The default method is zero_contour, which traces the same locus as
    # marching squares but with a different point density.  Verify geometric
    # consistency: mean radius must agree to within 5%.
    od = LensCalc.from_mass_obj(gal_x1_mp)
    expected_tc = od.tangential_critical_curve_list_from(grid=grid_2d_7x7)

    def _mean_radius(curve):
        pts = np.array(curve)
        return float(np.mean(np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)))

    assert abs(_mean_radius(tc[0]) - _mean_radius(expected_tc[0])) / _mean_radius(expected_tc[0]) < 0.05


def test__mass_plotter__caustics_via_lens_calc(gal_x1_mp, grid_2d_7x7):
    od = LensCalc.from_mass_obj(gal_x1_mp)
    tc = od.tangential_caustic_list_from(grid=grid_2d_7x7)
    rc = od.radial_caustic_list_from(grid=grid_2d_7x7)

    assert tc is not None
    assert len(tc) > 0
    assert rc is not None
