from typing import Tuple


class Point:
    def __init__(self, centre: Tuple[float, float] = (0.0, 0.0)):
        self.centre = centre


class PointFlux(Point):
    def __init__(self, centre: Tuple[float, float] = (0.0, 0.0), flux: float = 0.1):
        super().__init__(centre=centre)

        self.flux = flux


class PointSolved:
    """
    A point source profile with no free parameters.

    Unlike `Point` and `PointFlux`, `PointSolved` has no `centre` (and no `flux`)
    attribute at all, so it contributes zero dimensions to a model's prior count. It is
    intended to be paired, via name pairing, with one of the `*Solved` fit classes in
    `autolens.point.fit` (e.g. `FitPositionsSourceSolved`, `FitPositionsImagePairAllSolved`,
    `FitPositionsImagePairRepeatSolved`, `FitFluxesSolved`), which analytically solve for
    the source-plane centre (and, if the dataset has fluxes, the source flux) that
    maximizes the likelihood given the current tracer, rather than sampling these
    quantities as free parameters.

    Using `PointSolved` with a fit class that is not one of the `*Solved` variants, or
    using `Point` / `PointFlux` with a `*Solved` fit class, is an invalid combination and
    raises an informative `PointExtractionException` (see `autolens.point.fit.abstract`
    and `autolens.point.fit.fluxes`).
    """

    pass
