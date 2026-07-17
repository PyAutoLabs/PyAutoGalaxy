import numpy as np
import pytest

import autoarray as aa
import autogalaxy as ag


def grid_yx():
    return np.array(
        [
            [1.0, 0.5],
            [-0.3, 0.7],
            [0.2, -1.1],
            [0.8, 0.9],
            [-1.2, -0.4],
        ]
    )


def theta_intermediate_from(einstein_radius, axis_ratio, slope):
    return (
        np.sqrt(axis_ratio)
        * (2.0 / (1.0 + axis_ratio)) ** (1.0 / (slope - 1.0))
        * einstein_radius
    )


@pytest.mark.parametrize(
    "axis_ratio,angle,slope",
    [(0.7, 45.0, 2.0), (0.6, 40.0, 2.3), (0.9, -20.0, 1.7)],
)
def test__equivalent_to_power_law_under_theta_rescale(axis_ratio, angle, slope):
    """
    A `PowerLawIntermediate` whose einstein_radius is the intermediate-axis
    conversion of a `PowerLaw`'s einstein_radius is the identical mass
    distribution.
    """
    ell_comps = ag.convert.ell_comps_from(axis_ratio=axis_ratio, angle=angle)

    power_law = ag.mp.PowerLaw(
        centre=(0.1, -0.2),
        ell_comps=ell_comps,
        einstein_radius=1.4,
        slope=slope,
    )
    intermediate = ag.mp.PowerLawIntermediate(
        centre=(0.1, -0.2),
        ell_comps=ell_comps,
        einstein_radius=theta_intermediate_from(
            einstein_radius=1.4, axis_ratio=axis_ratio, slope=slope
        ),
        slope=slope,
    )

    grid = aa.Grid2DIrregular(grid_yx())

    assert np.asarray(intermediate.convergence_2d_from(grid=grid)) == pytest.approx(
        np.asarray(power_law.convergence_2d_from(grid=grid)), rel=1e-10
    )
    assert np.asarray(
        intermediate.deflections_yx_2d_from(grid=grid)
    ) == pytest.approx(
        np.asarray(power_law.deflections_yx_2d_from(grid=grid)), rel=1e-8
    )
    assert np.asarray(intermediate.potential_2d_from(grid=grid)) == pytest.approx(
        np.asarray(power_law.potential_2d_from(grid=grid)), rel=1e-8
    )


def test__convergence_matches_coolest_form_with_identity_theta():
    """
    The COOLEST / lenstronomy PEMD convergence evaluated directly with the
    profile's own einstein_radius (no conversion factor) matches the profile —
    the defining property of the intermediate-axis convention.
    """
    axis_ratio, angle, slope = 0.6, 40.0, 2.3

    profile = ag.mp.PowerLawIntermediate(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=axis_ratio, angle=angle),
        einstein_radius=1.2,
        slope=slope,
    )

    grid = grid_yx()
    x = grid[:, 1]
    y = grid[:, 0]

    alpha = np.radians(angle)
    x_prime = np.cos(alpha) * x + np.sin(alpha) * y
    y_prime = -np.sin(alpha) * x + np.cos(alpha) * y

    r = np.sqrt(axis_ratio * x_prime**2 + y_prime**2 / axis_ratio)
    convergence_coolest = (3.0 - slope) / 2.0 * (1.2 / r) ** (slope - 1.0)

    convergence = profile.convergence_2d_from(grid=aa.Grid2DIrregular(grid))

    assert np.asarray(convergence) == pytest.approx(convergence_coolest, rel=1e-8)


def test__power_law_regression__unchanged_by_hook_refactor():
    """
    Pin PowerLaw numerics: the einstein_radius_major_from hook must be an
    exact no-op for the existing profile.
    """
    profile = ag.mp.PowerLaw(
        centre=(0.0, 0.0),
        ell_comps=(0.05, 0.05),
        einstein_radius=1.0,
        slope=2.2,
    )

    assert profile.einstein_radius_major_from() == 1.0

    convergence = profile.convergence_2d_from(
        grid=aa.Grid2DIrregular(np.array([[0.5, 0.5]]))
    )

    # (3 - 2.2) / (1 + q) * (1 / xi)^(1.2) evaluated independently.
    q, phi = profile.axis_ratio(), profile.angle()
    alpha = np.radians(phi)
    x_p = np.cos(alpha) * 0.5 + np.sin(alpha) * 0.5
    y_p = -np.sin(alpha) * 0.5 + np.cos(alpha) * 0.5
    xi = np.sqrt(x_p**2 + (y_p / q) ** 2)

    assert np.asarray(convergence)[0] == pytest.approx(
        (3 - 2.2) / (1 + q) * xi ** (-1.2), rel=1e-8
    )
