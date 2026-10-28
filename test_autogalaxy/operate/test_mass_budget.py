import numpy as np
import pytest

import autogalaxy as ag
from autogalaxy.operate.mass_budget import Aperture


@pytest.fixture(name="grid")
def make_grid():
    # 0.05" pixels, extent +/- 1.1" so a 1" aperture about the origin fits.
    return ag.Grid2D.uniform(shape_native=(44, 44), pixel_scales=0.05)


@pytest.fixture(name="grid_fine")
def make_grid_fine():
    # 0.025" pixels, extent +/- 1.1", used where an isothermal cusp must integrate to 1e-4.
    return ag.Grid2D.uniform(shape_native=(88, 88), pixel_scales=0.025)


@pytest.fixture(name="grid_coarse")
def make_grid_coarse():
    # A coarse grid for the gNFW, whose convergence is a per-point quadrature.
    return ag.Grid2D.uniform(shape_native=(12, 12), pixel_scales=0.2)


@pytest.fixture(name="light_basis")
def make_light_basis():
    return ag.lp_basis.Basis(
        profile_list=[
            ag.lp.GaussianSph(centre=(0.0, 0.0), intensity=1.0, sigma=0.5),
            ag.lp.GaussianSph(centre=(0.0, 0.0), intensity=0.3, sigma=1.0),
        ]
    )


def _gaussian_basis_mass_within_circle(basis, mass_to_light_ratio, radius):
    return sum(
        2.0
        * np.pi
        * mass_to_light_ratio
        * gaussian.intensity
        * gaussian.sigma**2
        * (1.0 - np.exp(-(radius**2) / (2.0 * gaussian.sigma**2)))
        for gaussian in basis.profile_list
    )


def _nfw_sph_mass_within_circle(kappa_s, scale_radius, radius):
    # Closed-form projected mass of a spherical NFW inside x = R / r_s < 1 (Bartelmann 1996).
    x = radius / scale_radius
    h = np.log(x / 2.0) + np.arccosh(1.0 / x) / np.sqrt(1.0 - x**2)
    return 4.0 * np.pi * kappa_s * scale_radius**2 * h


def test__circular_aperture__mass_from_equals_grid_integrator(grid):
    aperture = ag.CircularAperture(radius=0.8, centre=(0.1, -0.1))
    mass = ag.mp.IsothermalSph(centre=(0.1, -0.1), einstein_radius=1.0)

    assert isinstance(aperture, Aperture)
    assert aperture.mass_from(mass_obj=mass, grid=grid) == pytest.approx(
        ag.mass_angular_within_circle_via_grid_from(
            mass_obj=mass, radius=0.8, centre=(0.1, -0.1), grid=grid
        ),
        rel=1.0e-12,
    )


def test__mass_to_light_ratio_for_mass_from__matches_gaussian_closed_form(
    grid, light_basis
):
    aperture = ag.CircularAperture(radius=1.0, centre=(0.0, 0.0))

    mass_to_light_ratio = ag.mass_to_light_ratio_for_mass_from(
        basis=light_basis, mass=2.0, aperture=aperture, grid=grid
    )

    mass_basis = light_basis.mass_basis_from(mass_to_light_ratio=mass_to_light_ratio)

    assert aperture.mass_from(mass_obj=mass_basis, grid=grid) == pytest.approx(
        2.0, rel=1.0e-10
    )
    assert _gaussian_basis_mass_within_circle(
        basis=light_basis, mass_to_light_ratio=mass_to_light_ratio, radius=1.0
    ) == pytest.approx(2.0, rel=1.0e-4)


def test__mass_to_light_ratio_for_mass_from__negative_mass__raises(grid, light_basis):
    with pytest.raises(ValueError):
        ag.mass_to_light_ratio_for_mass_from(
            basis=light_basis,
            mass=-1.0,
            aperture=ag.CircularAperture(radius=1.0),
            grid=grid,
        )


def test__mass_to_light_ratio_for_mass_from__zero_light__raises(grid):
    basis = ag.lp_basis.Basis(
        profile_list=[ag.lp.GaussianSph(centre=(0.0, 0.0), intensity=0.0, sigma=0.5)]
    )

    with pytest.raises(ValueError):
        ag.mass_to_light_ratio_for_mass_from(
            basis=basis, mass=1.0, aperture=ag.CircularAperture(radius=1.0), grid=grid
        )


def test__kappa_s_for_mass_from__nfw_sph__hits_target_and_matches_closed_form(grid):
    aperture = ag.CircularAperture(radius=1.0, centre=(0.0, 0.0))

    kappa_s = ag.kappa_s_for_mass_from(
        dark_cls=ag.mp.NFWSph,
        mass=1.5,
        aperture=aperture,
        scale_radius=5.0,
        centre=(0.0, 0.0),
        grid=grid,
    )

    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=kappa_s, scale_radius=5.0)

    assert aperture.mass_from(mass_obj=nfw, grid=grid) == pytest.approx(1.5, rel=1.0e-4)
    assert nfw.mass_angular_within_circle_from(radius=1.0) == pytest.approx(
        1.5, rel=1.0e-3
    )
    assert _nfw_sph_mass_within_circle(
        kappa_s=kappa_s, scale_radius=5.0, radius=1.0
    ) == pytest.approx(1.5, rel=1.0e-3)


def test__kappa_s_for_mass_from__nfw_sph__aperture_mass_linear_in_kappa_s(grid):
    aperture = ag.CircularAperture(radius=1.0, centre=(0.0, 0.0))

    masses = [
        aperture.mass_from(
            mass_obj=ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=kappa_s, scale_radius=5.0),
            grid=grid,
        )
        for kappa_s in (0.1, 0.2, 0.4)
    ]

    assert masses[1] == pytest.approx(2.0 * masses[0], rel=1.0e-10)
    assert masses[2] == pytest.approx(4.0 * masses[0], rel=1.0e-10)

    kappa_s_list = [
        ag.kappa_s_for_mass_from(
            dark_cls=ag.mp.NFWSph,
            mass=mass,
            aperture=aperture,
            scale_radius=5.0,
            centre=(0.0, 0.0),
            grid=grid,
        )
        for mass in (0.5, 1.0, 2.0)
    ]

    assert kappa_s_list[0] < kappa_s_list[1] < kappa_s_list[2]
    assert kappa_s_list[1] == pytest.approx(2.0 * kappa_s_list[0], rel=1.0e-10)
    assert kappa_s_list[2] == pytest.approx(4.0 * kappa_s_list[0], rel=1.0e-10)


def test__kappa_s_for_mass_from__gnfw_sph_inner_slope_1__matches_nfw_sph(
    grid_coarse,
):
    aperture = ag.CircularAperture(
        radius=1.0,
        centre=(0.0, 0.0),
        sub_size_interior=2,
        sub_size_edge=4,
        sub_size_centre=8,
    )

    kappa_s_nfw = ag.kappa_s_for_mass_from(
        dark_cls=ag.mp.NFWSph,
        mass=1.0,
        aperture=aperture,
        scale_radius=5.0,
        centre=(0.0, 0.0),
        grid=grid_coarse,
    )
    kappa_s_gnfw = ag.kappa_s_for_mass_from(
        dark_cls=ag.mp.gNFWSph,
        mass=1.0,
        aperture=aperture,
        scale_radius=5.0,
        centre=(0.0, 0.0),
        grid=grid_coarse,
        inner_slope=1.0,
    )

    assert kappa_s_gnfw == pytest.approx(kappa_s_nfw, rel=1.0e-4)


def test__kappa_s_for_mass_from__negative_mass__raises(grid):
    with pytest.raises(ValueError):
        ag.kappa_s_for_mass_from(
            dark_cls=ag.mp.NFWSph,
            mass=-1.0,
            aperture=ag.CircularAperture(radius=1.0),
            scale_radius=5.0,
            centre=(0.0, 0.0),
            grid=grid,
        )


def test__mass_budget__50_50_isothermal_reference(grid_fine, light_basis):
    # The isothermal cusp at the aperture centre dominates the grid-integration error (~2e-4 with
    # the integrator defaults on a 0.05" grid, ~1e-4 on a 0.025" grid); a 0.025" grid with a
    # 128 x 128 centre sub-grid brings it to ~5e-5.
    reference_mass = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0)

    budget = ag.MassBudget(
        aperture=ag.CircularAperture(
            radius=reference_mass.einstein_radius,
            centre=reference_mass.centre,
            sub_size_centre=128,
        ),
        fractions={"lens": {"stellar": 0.5, "dark": 0.5}},
    )

    result = budget.solve(
        reference_mass=reference_mass,
        light_basis=light_basis,
        dark_cls=ag.mp.NFWSph,
        scale_radius=5.0,
        centre=(0.0, 0.0),
        grid=grid_fine,
    )

    mass_reference = result.derived["mass_reference"]

    assert mass_reference == pytest.approx(np.pi, rel=1.0e-4)

    mass_stellar = budget.aperture.mass_from(mass_obj=result.stellar, grid=grid_fine)
    mass_dark = budget.aperture.mass_from(mass_obj=result.dark, grid=grid_fine)

    assert mass_stellar + mass_dark == pytest.approx(mass_reference, rel=1.0e-4)
    assert mass_dark / mass_reference == pytest.approx(0.5, rel=1.0e-4)
    assert result.derived["f_dark"] == 0.5

    assert isinstance(result.stellar, ag.lp_basis.Basis)
    assert all(isinstance(p, ag.lmp.Gaussian) for p in result.stellar.profile_list)
    assert result.stellar.profile_list[0].mass_to_light_ratio == pytest.approx(
        result.derived["mass_to_light_ratio"], rel=1.0e-12
    )
    assert isinstance(result.dark, ag.mp.NFWSph)
    assert result.dark.kappa_s == pytest.approx(result.derived["kappa_s"], rel=1.0e-12)
    assert result.dark.scale_radius == 5.0

    assert set(result.derived) == {
        "aperture_radius",
        "aperture_centre",
        "mass_reference",
        "mass_light",
        "mass_to_light_ratio",
        "scale_radius",
        "kappa_s",
        "f_dark",
        "fixed",
    }
    assert result.derived["aperture_radius"] == 1.0
    assert result.derived["aperture_centre"] == (0.0, 0.0)
    assert result.derived["scale_radius"] == 5.0
    assert result.derived["fixed"] == {}
    assert result.derived["mass_to_light_ratio"] == pytest.approx(
        0.5 * mass_reference / result.derived["mass_light"], rel=1.0e-12
    )


def test__mass_budget__fixed_parameters_pass_through(grid_coarse, light_basis):
    budget = ag.MassBudget(
        aperture=ag.CircularAperture(
            radius=1.0, sub_size_interior=2, sub_size_edge=4, sub_size_centre=8
        ),
        fractions={"lens": {"stellar": 0.3, "dark": 0.7}},
    )

    result = budget.solve(
        reference_mass=ag.mp.IsothermalSph(einstein_radius=1.0),
        light_basis=light_basis,
        dark_cls=ag.mp.gNFWSph,
        scale_radius=5.0,
        centre=(0.0, 0.0),
        grid=grid_coarse,
        inner_slope=1.5,
    )

    assert isinstance(result.dark, ag.mp.gNFWSph)
    assert result.dark.inner_slope == 1.5
    assert result.derived["fixed"] == {"inner_slope": 1.5}


def test__mass_budget__zero_fraction__component_is_none(grid, light_basis):
    reference_mass = ag.mp.IsothermalSph(einstein_radius=1.0)

    all_dark = ag.MassBudget(
        aperture=ag.CircularAperture(radius=1.0),
        fractions={"lens": {"stellar": 0.0, "dark": 1.0}},
    ).solve(
        reference_mass=reference_mass,
        light_basis=light_basis,
        dark_cls=ag.mp.NFWSph,
        scale_radius=5.0,
        centre=(0.0, 0.0),
        grid=grid,
    )

    assert all_dark.stellar is None
    assert all_dark.derived["mass_to_light_ratio"] == 0.0
    assert isinstance(all_dark.dark, ag.mp.NFWSph)

    all_stellar = ag.MassBudget(
        aperture=ag.CircularAperture(radius=1.0),
        fractions={"lens": {"stellar": 1.0, "dark": 0.0}},
    ).solve(
        reference_mass=reference_mass,
        light_basis=light_basis,
        dark_cls=ag.mp.NFWSph,
        scale_radius=5.0,
        centre=(0.0, 0.0),
        grid=grid,
    )

    assert all_stellar.dark is None
    assert all_stellar.derived["kappa_s"] == 0.0
    assert all_stellar.derived["f_dark"] == 0.0
    assert isinstance(all_stellar.stellar, ag.lp_basis.Basis)


@pytest.mark.parametrize(
    "fractions",
    [
        {"lens": {"stellar": 0.5, "dark": 0.6}},
        {"lens": {"stellar": 0.5, "dark": 0.5 - 1.0e-6}},
        {"lens": {"stellar": -0.1, "dark": 1.1}},
        {"lens": {"stellar": 1.0}},
        {"lens": {"stellar": 0.5, "dark": 0.5, "gas": 0.0}},
        {},
    ],
)
def test__mass_budget__invalid_fractions__raise(fractions):
    with pytest.raises(ValueError):
        ag.MassBudget(aperture=ag.CircularAperture(radius=1.0), fractions=fractions)


def test__mass_budget__two_galaxies__raises_not_implemented():
    with pytest.raises(NotImplementedError, match="multi-galaxy"):
        ag.MassBudget(
            aperture=ag.CircularAperture(radius=1.0),
            fractions={
                "lens": {"stellar": 0.5, "dark": 0.5},
                "dwarf": {"stellar": 0.5, "dark": 0.5},
            },
        )
