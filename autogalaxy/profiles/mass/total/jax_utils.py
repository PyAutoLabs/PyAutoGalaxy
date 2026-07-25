import numpy as np


def body_fun(carry, n, factor, ei2phi, slope):
    omega_nm1, partial_sum = carry
    two_n = 2 * n
    two_minus_slope = 2 - slope
    ratio = (two_n - two_minus_slope) / (two_n + two_minus_slope)
    omega_n = -factor * ratio * ei2phi * omega_nm1
    partial_sum = partial_sum + omega_n
    return (omega_n, partial_sum), None


def omega(eiphi, slope, factor, n_terms=20, xp=np):
    """JAX implementation of the numerical evaluation of the angular component of
    the complex deflection angle for the elliptical power law profile as given as
    given by Tessore and Metcalf 2015.  Based on equation 29, and gives
    omega (e.g. can be used as a drop in replacement for the exp(i * phi) * special.hyp2f1
    term in equation 13).

    Parameters
    ----------
    eiphi:
        `exp(i * phi)` where `phi` is the elliptical angle of the profile
    slope:
        The density slope of the power-law (lower value -> shallower profile, higher value
        -> steeper profile).
    factor:
        The second flattening of and ellipse with axis ration q give by `f = (1 - q) / (1 + q)`
    n_terms:
        The number of terms to calculate for the series expansion, defaults to 20 (this should
        be sufficient most of the time)
    """

    from jax.tree_util import Partial as partial
    import jax

    scan = jax.jit(jax.lax.scan, static_argnames=("length", "reverse", "unroll"))

    # use modified scan with a partial'ed function to avoid re-compile
    (_, partial_sum), _ = scan(
        partial(body_fun, factor=factor, ei2phi=eiphi**2, slope=slope),
        (eiphi, eiphi),
        xp.arange(1, n_terms),
    )
    return partial_sum
