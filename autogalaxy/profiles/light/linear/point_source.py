from typing import Tuple

from autogalaxy.profiles.light.linear.abstract import LightProfileLinear
from autogalaxy.profiles.light import standard as lp


class PointSource(lp.PointSource, LightProfileLinear):
    """A point source whose total flux is solved for by linear inversion."""

    def __init__(self, centre: Tuple[float, float] = (0.0, 0.0)):
        super().__init__(centre=centre, intensity=1.0)
