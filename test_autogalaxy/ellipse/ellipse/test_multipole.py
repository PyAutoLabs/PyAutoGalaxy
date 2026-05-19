import numpy as np
import pytest

import autogalaxy as ag


def test__points_perturbed_from():
    pixel_scale = 1.0

    ellipse = ag.Ellipse(major_axis=1.0)

    points = ellipse.points_from_major_axis_from(pixel_scale=pixel_scale)

    multipole = ag.EllipseMultipole(m=4, multipole_comps=(0.0, 0.0))

    points_perturbed = multipole.points_perturbed_from(
        pixel_scale=pixel_scale, points=points, ellipse=ellipse
    )

    assert points_perturbed == pytest.approx(points, 1.0e-4)

    multipole = ag.EllipseMultipole(m=4, multipole_comps=(0.1, 0.2))

    points_perturbed = multipole.points_perturbed_from(
        pixel_scale=pixel_scale, points=points, ellipse=ellipse
    )

    assert points_perturbed[1, 0] == pytest.approx(-0.919384, 1.0e-4)
    assert points_perturbed[1, 1] == pytest.approx(0.298726, 1.0e-4)

    ellipse = ag.Ellipse(major_axis=1.5)

    points = ellipse.points_from_major_axis_from(pixel_scale=pixel_scale)

    multipole_scaled = ag.EllipseMultipoleScaled(
        m=4, scaled_multipole_comps=(0.1, 0.2), major_axis=1.5
    )

    points_perturbed = multipole_scaled.points_perturbed_from(
        pixel_scale=pixel_scale, points=points, ellipse=ellipse
    )

    assert points_perturbed[1, 0] == pytest.approx(-0.848528, 1.0e-4)
    assert points_perturbed[1, 1] == pytest.approx(0.848528, 1.0e-4)


def test__ellipse_multipole_scaled__points_perturbed_from__pinning():
    """
    Pinning test for EllipseMultipoleScaled.points_perturbed_from.

    Reference output was captured from the pre-refactor implementation (where
    specific_multipole_comps was derived at __init__ time). Post-refactor the
    derivation happens inside points_perturbed_from with xp threaded through;
    the algebraic result is identical to rtol=1e-10.
    """
    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.1, 0.05), major_axis=1.0)
    ms = ag.EllipseMultipoleScaled(m=4, scaled_multipole_comps=(0.05, 0.0), major_axis=1.0)
    points = np.array([[0.5, 0.5], [-0.3, 0.7]])

    out = ms.points_perturbed_from(pixel_scale=0.1, points=points, ellipse=ellipse, xp=np)

    expected = np.array(
        [[ 0.5                ,  0.5               ],
         [-0.3196725452322533,  0.7459026055419243]]
    )
    np.testing.assert_allclose(out, expected, rtol=1e-10)
