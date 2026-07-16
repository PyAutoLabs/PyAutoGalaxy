import numpy as np
import pytest

import autoarray as aa
import autogalaxy as ag
from autogalaxy import exc
from autogalaxy.interop import coolest


def grid_yx():
    return np.array(
        [
            [1.0, 0.5],
            [-0.3, 0.7],
            [0.2, -1.1],
            [0.8, 0.9],
        ]
    )


def coolest_sersic_image(parameters, sersic_constant, grid):
    """
    The COOLEST Sersic surface brightness evaluated directly from
    COOLEST-convention parameters:

    I(r) = I_eff * exp(-b_n * ((r / theta_eff)^(1/n) - 1)),
    r = sqrt(q x'^2 + y'^2 / q),

    with (x', y') the coordinates rotated into the profile frame whose major
    axis lies at position angle phi counter-clockwise from the +y axis.
    """
    x = grid[:, 1] - parameters["center_x"]
    y = grid[:, 0] - parameters["center_y"]

    alpha = np.radians(parameters["phi"] + 90.0)
    x_prime = np.cos(alpha) * x + np.sin(alpha) * y
    y_prime = -np.sin(alpha) * x + np.cos(alpha) * y

    q = parameters["q"]
    r = np.sqrt(q * x_prime**2 + y_prime**2 / q)

    return parameters["I_eff"] * np.exp(
        -sersic_constant
        * ((r / parameters["theta_eff"]) ** (1.0 / parameters["n"]) - 1.0)
    )


def test__sersic__image_matches_coolest_analytic_form():
    profile = ag.lp.Sersic(
        centre=(0.1, -0.2),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.6, angle=40.0),
        intensity=2.0,
        effective_radius=0.8,
        sersic_index=3.0,
    )

    profile_dict = coolest.coolest_dict_from_light(profile=profile)

    assert profile_dict["type"] == "Sersic"
    assert profile_dict["parameters"]["I_eff"] == pytest.approx(2.0)
    assert profile_dict["parameters"]["theta_eff"] == pytest.approx(0.8)
    assert profile_dict["parameters"]["n"] == pytest.approx(3.0)
    assert profile_dict["parameters"]["q"] == pytest.approx(0.6)
    assert profile_dict["parameters"]["phi"] == pytest.approx(-50.0)

    image = profile.image_2d_from(grid=aa.Grid2DIrregular(grid_yx()))
    image_coolest = coolest_sersic_image(
        parameters=profile_dict["parameters"],
        sersic_constant=profile.sersic_constant,
        grid=grid_yx(),
    )

    assert np.asarray(image) == pytest.approx(image_coolest, rel=1e-8)


def test__sersic__round_trip_is_numerically_identical():
    profile = ag.lp.Sersic(
        centre=(-0.05, 0.15),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.75, angle=100.0),
        intensity=1.3,
        effective_radius=1.1,
        sersic_index=2.2,
    )

    profile_dict = coolest.coolest_dict_from_light(profile=profile)
    profile_back = coolest.light_profile_from(
        profile_type=profile_dict["type"], parameters=profile_dict["parameters"]
    )

    grid = aa.Grid2DIrregular(grid_yx())

    assert np.asarray(profile_back.image_2d_from(grid=grid)) == pytest.approx(
        np.asarray(profile.image_2d_from(grid=grid)), rel=1e-8
    )


def test__sersic_sph__round_trip():
    profile = ag.lp.SersicSph(
        centre=(0.1, 0.2), intensity=0.5, effective_radius=0.7, sersic_index=1.5
    )

    profile_dict = coolest.coolest_dict_from_light(profile=profile)

    assert profile_dict["parameters"]["q"] == pytest.approx(1.0)

    profile_back = coolest.light_profile_from(
        profile_type=profile_dict["type"], parameters=profile_dict["parameters"]
    )

    grid = aa.Grid2DIrregular(grid_yx())

    assert np.asarray(profile_back.image_2d_from(grid=grid)) == pytest.approx(
        np.asarray(profile.image_2d_from(grid=grid)), rel=1e-8
    )


def test__major_axis_along_y__coolest_phi_is_zero():
    profile = ag.lp.Sersic(
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.5, angle=90.0)
    )

    profile_dict = coolest.coolest_dict_from_light(profile=profile)

    assert profile_dict["parameters"]["phi"] == pytest.approx(0.0)


def test__unsupported_profiles__raise_with_named_error():
    # Subclasses of Sersic (e.g. Exponential) must not silently convert as if
    # they were plain Sersic profiles.
    with pytest.raises(exc.ProfileException):
        coolest.coolest_dict_from_light(profile=ag.lp.Exponential())

    with pytest.raises(exc.ProfileException):
        coolest.light_profile_from(profile_type="Shapelets", parameters={})
