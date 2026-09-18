"""Lazy compatibility boundary for the optional zero-contour solver."""

from functools import lru_cache


@lru_cache(maxsize=1)
def zero_solver_type():
    """Return a solver class whose Newton loop accepts JAX 0.11 callables.

    jax-zero-contour 2.0 carries the function supplied by ``custom_root`` in
    its ``while_loop`` state. JAX 0.11 can supply a plain functools.partial,
    which is not a valid JAX value. A pytree Partial keeps the callable static
    while exposing any bound array arguments to JAX's transformations.

    Import only when a contour is requested, and cache the class to preserve
    solver type identity. The upstream class and numerical algorithm remain
    untouched.
    """
    from jax.tree_util import Partial
    from jax_zero_contour import ZeroSolver

    class CompatibleZeroSolver(ZeroSolver):
        def step_parallel_tol(self, f, init_guess, factor=1):
            return super().step_parallel_tol(
                Partial(f), init_guess, factor=factor
            )

    return CompatibleZeroSolver
