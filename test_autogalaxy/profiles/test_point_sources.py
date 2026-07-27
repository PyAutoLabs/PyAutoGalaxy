from __future__ import division, print_function
import numpy as np

import autofit as af
import autogalaxy as ag

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


class TestPointFlux:
    def test__constructor(self):
        point_source = ag.ps.PointFlux(centre=(0.0, 0.0), flux=0.1)

        assert point_source.centre == (0.0, 0.0)
        assert point_source.flux == 0.1


class TestPointSolved:
    def test__constructor__no_parameters(self):
        point_source = ag.ps.PointSolved()

        assert not hasattr(point_source, "centre")
        assert not hasattr(point_source, "flux")

    def test__model_composition__contributes_zero_free_parameters(self):
        # A zero-parameter class must compose fine under `af.Model` without a config
        # entry (its priors dict is derived from `__init__`, which has no arguments) and
        # must not add to the model's overall prior count.
        model = af.Model(ag.Galaxy, redshift=1.0, point_0=ag.ps.PointSolved)

        assert model.prior_count == 0

        instance = model.instance_from_prior_medians()

        assert isinstance(instance.point_0, ag.ps.PointSolved)

    def test__model_composition__prior_count_unchanged_by_other_free_parameters(self):
        # A `PointSolved` component contributes 0 to the total, regardless of what else
        # is on the galaxy (regression guard for the component-count summing).
        model = af.Model(
            ag.Galaxy,
            redshift=1.0,
            point_0=ag.ps.PointSolved,
            mass=ag.mp.IsothermalSph,
        )

        mass_only_model = af.Model(ag.Galaxy, redshift=1.0, mass=ag.mp.IsothermalSph)

        assert model.prior_count == mass_only_model.prior_count
