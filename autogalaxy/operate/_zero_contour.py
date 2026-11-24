"""Lazy compatibility boundary for the optional zero-contour solver."""

from functools import lru_cache


@lru_cache(maxsize=1)
def zero_solver_type():
    """Return a solver class compatible with JAX transformations.

    jax-zero-contour 2.0 carries the function supplied by ``custom_root`` in
    its ``while_loop`` state. JAX 0.11 can supply a plain functools.partial,
    which is not a valid JAX value. A pytree Partial keeps the callable static
    while exposing any bound array arguments to JAX's transformations.

    Import only when a contour is requested, and cache the class to preserve
    solver type identity. The upstream class and numerical algorithm remain
    untouched.
    """
    import jax
    import jax.numpy as jnp
    from jax.tree_util import Partial
    from jax_zero_contour import ZeroSolver

    class CompatibleZeroSolver(ZeroSolver):
        @Partial(jax.vmap, in_axes=(None, None, 0, None))
        def newton(self, f, init_guess, factor):
            # custom_root traces the tangent RHS with the shape of its guess
            # on JAX 0.12. Embed the scalar contour equation in R² so the
            # traced and actual RHS shapes agree under batched transposition.
            def residual(point):
                value = f(point)
                return jnp.stack((value, jnp.zeros_like(value)))

            def solve(equations, guess):
                return self.step_parallel_tol(
                    lambda point: equations(point)[0], guess, factor=factor
                )

            def tangent_solve(linearized, rhs):
                # The second equation is identically zero. Use the upstream
                # minimum-norm solution along the first equation's normal.
                jacobian = (
                    jax.jacfwd if self.forward_mode_differentiation else jax.jacrev
                )
                normal = jacobian(lambda point: linearized(point)[0])(rhs)
                return rhs[0] * normal / jnp.sum(normal**2)

            return jax.lax.custom_root(
                residual, init_guess, solve, tangent_solve, has_aux=True
            )

        def step_parallel_tol(self, f, init_guess, factor=1):
            return super().step_parallel_tol(
                Partial(f), init_guess, factor=factor
            )

    return CompatibleZeroSolver
