from __future__ import division, print_function
import numpy as np

import autogalaxy as ag

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


class TestPointFlux:
    def test__constructor(self):
        point_source = ag.ps.PointFlux(centre=(0.0, 0.0), flux=0.1)

        assert point_source.centre == (0.0, 0.0)
        assert point_source.flux == 0.1
