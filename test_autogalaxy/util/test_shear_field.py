from matplotlib.patches import Ellipse
import pytest
import numpy as np

import autogalaxy as ag


def test__elliptical_properties_and_patches():
    vectors = ag.ShearYX2D.no_mask(
        values=[(0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)],
        shape_native=(2, 2),
        pixel_scales=1.0,
    )

    assert isinstance(vectors.ellipticities, ag.ArrayIrregular)
    assert vectors.ellipticities.in_list == [1.0, 1.0, np.sqrt(2.0), 0.0]

    assert isinstance(vectors.semi_major_axes, ag.ArrayIrregular)
    assert vectors.semi_major_axes.in_list == pytest.approx(
        [6.0, 6.0, 7.242640, 3.0], 1.0e-4
    )

    assert isinstance(vectors.semi_minor_axes, ag.ArrayIrregular)
    assert vectors.semi_minor_axes.in_list == pytest.approx(
        [0.0, 0.0, -1.242640, 3.0], 1.0e-4
    )

    assert isinstance(vectors.phis, ag.ArrayIrregular)
    assert vectors.phis.in_list == pytest.approx([0.0, 45.0, 22.5, 0.0], 1.0e-4)

    assert isinstance(vectors.elliptical_patches[0], Ellipse)
    assert vectors.elliptical_patches[1].center == pytest.approx((0.5, 0.5), 1.0e-4)
    assert vectors.elliptical_patches[1].width == pytest.approx(6.0, 1.0e-4)
    assert vectors.elliptical_patches[1].height == pytest.approx(0.0, 1.0e-4)
    assert vectors.elliptical_patches[1].angle == pytest.approx(45.0, 1.0e-4)

    vectors = ag.ShearYX2DIrregular(
        values=[(0.0, 1.0), (1.0, 0.0), (1.0, 1.0)],
        grid=[[1.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
    )

    assert isinstance(vectors.ellipticities, ag.ArrayIrregular)
    assert vectors.ellipticities.in_list == [1.0, 1.0, np.sqrt(2.0)]

    assert isinstance(vectors.semi_major_axes, ag.ArrayIrregular)
    assert vectors.semi_major_axes.in_list == pytest.approx(
        [6.0, 6.0, 7.242640], 1.0e-4
    )

    assert isinstance(vectors.semi_minor_axes, ag.ArrayIrregular)
    assert vectors.semi_minor_axes.in_list == pytest.approx(
        [0.0, 0.0, -1.242640], 1.0e-4
    )

    assert isinstance(vectors.phis, ag.ArrayIrregular)
    assert vectors.phis.in_list == pytest.approx([0.0, 45.0, 22.5], 1.0e-4)

    assert isinstance(vectors.elliptical_patches[0], Ellipse)
    assert vectors.elliptical_patches[1].center == pytest.approx((1.0, 1.0), 1.0e-4)
    assert vectors.elliptical_patches[1].width == pytest.approx(6.0, 1.0e-4)
    assert vectors.elliptical_patches[1].height == pytest.approx(0.0, 1.0e-4)
    assert vectors.elliptical_patches[1].angle == pytest.approx(45.0, 1.0e-4)
