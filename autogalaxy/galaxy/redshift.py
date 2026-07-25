"""
Provides the `Redshift` class, a thin float subclass used when the redshift of a `Galaxy` is treated as a
free parameter in a model fit.

In standard use, galaxy redshifts are fixed scalars passed directly to `Galaxy(redshift=0.5, ...)`.
When the redshift itself needs to be inferred by the non-linear search, **PyAutoFit** requires every model
parameter to be wrapped in a Python class. The `Redshift` class satisfies this requirement while behaving
identically to a plain Python `float` in all arithmetic and comparison contexts.
"""


class Redshift(float):
    """
    Class used when assigning a redshift to a `Galaxy` object.

    This object is only required when making the `Redshift` of the `Galaxy` a free parameter in a model that
    is fitted.

    This is because **PyAutoFit** (which handles model-fitting), requires all parameters to be a Python class.

    The `Redshift` object does not need to be used for general **PyAutoGalaxy** use.

    Examples
    --------

    import autofit as af
    import autogalaxy as ag

    bulge = af.Model(ag.lp.Sersic)

    redshift = af.Model(ag.Redshift)
    redshift.redshift = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

    galaxy = af.Model(ag.Galaxy, redshift=redshift, bulge=bulge)

    model = af.Collection(galaxies=af.Collection(galaxy=galaxy))
    """

    def __new__(cls, redshift):
        # noinspection PyArgumentList
        return float.__new__(cls, redshift)

    def __init__(self, redshift):
        float.__init__(redshift)
