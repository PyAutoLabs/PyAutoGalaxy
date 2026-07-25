from typing import Tuple


class Point:
    def __init__(self, centre: Tuple[float, float] = (0.0, 0.0)):
        self.centre = centre


class PointFlux(Point):
    def __init__(self, centre: Tuple[float, float] = (0.0, 0.0), flux: float = 0.1):
        super().__init__(centre=centre)

        self.flux = flux
