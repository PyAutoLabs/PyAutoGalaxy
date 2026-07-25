import numpy as np

import autogalaxy as ag


class MockMassProfile(ag.mp.MassProfile):
    def __init__(
        self,
        convergence_2d=None,
        potential_2d=None,
        deflections_yx_2d=None,
        value=None,
        value1=None,
    ):
        super().__init__()

        self.convergence_2d = convergence_2d
        self.potential_2d = potential_2d
        self.deflections_2d = deflections_yx_2d

        self.value = value
        self.value1 = value1

    def convergence_2d_from(self, grid, xp=np):
        return self.convergence_2d

    def potential_2d_from(self, grid, xp=np):
        return self.potential_2d

    def deflections_yx_2d_from(self, grid, xp=np):
        return self.deflections_2d
