"""
Benchmark the JAX prototype against colossus, and exercise grad + vmap
to confirm the JAX path supports differentiation and vectorisation —
the two capabilities the pure_callback rules out.

Run with:
    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
        JAX_ENABLE_X64=1 python docs/research/nfw_ludlow16_jax/bench.py
"""

import os
import time
import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from colossus.cosmology import cosmology as col_cosmology
from colossus.halo.concentration import concentration as col_concentration

import sys
sys.path.insert(0, os.path.dirname(__file__))
from ludlow16_jax import ludlow16_concentration_jax  # noqa: E402


PLANCK15 = dict(
    h=0.6774,
    Om0=0.3089,
    Ob0=0.0486,
    Tcmb0=2.7255,
    sigma8=0.8159,
    ns=0.9667,
)


def main():
    col_cosmology.setCosmology("planck15")

    # ----------------------------------------------------------------
    # 1) grad — does jax.grad work through the concentration solver?
    # ----------------------------------------------------------------

    @jax.jit
    def c_of_M(M):
        return ludlow16_concentration_jax(M, 0.5, **PLANCK15)

    print("grad test:")
    M0 = jnp.float64(1.0e12)
    c0 = c_of_M(M0)
    g = jax.grad(c_of_M)(M0)
    print(f"  c(1e12, z=0.5) = {float(c0):.4f}")
    print(f"  dc/dM         = {float(g):.4e}")

    # Finite-difference check using colossus
    eps = 1.0e9
    c_low = float(col_concentration(1.0e12 - eps, "200c", 0.5, model="ludlow16"))
    c_high = float(col_concentration(1.0e12 + eps, "200c", 0.5, model="ludlow16"))
    g_fd = (c_high - c_low) / (2 * eps)
    print(f"  dc/dM (fd)    = {g_fd:.4e}")
    print(f"  agreement: rel err = {abs(g - g_fd) / abs(g_fd):.2e}")
    print()

    # ----------------------------------------------------------------
    # 2) vmap — batch many (M, z) pairs in one JIT trace
    # ----------------------------------------------------------------

    @jax.jit
    def c_single(M, z):
        return ludlow16_concentration_jax(M, z, **PLANCK15)

    batch_size = 32
    M_batch = jnp.geomspace(1.0e10, 1.0e14, batch_size)
    z_batch = jnp.linspace(0.1, 2.5, batch_size)

    c_batched = jax.jit(jax.vmap(c_single))
    # warm up
    _ = c_batched(M_batch, z_batch).block_until_ready()

    n_iter = 20
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = c_batched(M_batch, z_batch).block_until_ready()
    t_jax_batch = (time.perf_counter() - t0) / n_iter

    t0 = time.perf_counter()
    for _ in range(n_iter):
        for M, z in zip(np.array(M_batch), np.array(z_batch)):
            _ = col_concentration(float(M), "200c", float(z), model="ludlow16")
    t_col_batch = (time.perf_counter() - t0) / n_iter

    print(f"Batch of {batch_size} concentration calls:")
    print(f"  colossus (serial)         : {t_col_batch*1e3:.2f} ms")
    print(f"  jax vmap (post-jit)       : {t_jax_batch*1e3:.2f} ms")
    print(f"  speedup                   : {t_col_batch / t_jax_batch:.2f}x")
    print()

    # ----------------------------------------------------------------
    # 3) Single-call wall time, including JIT compile vs post-compile
    # ----------------------------------------------------------------

    fresh_jit = jax.jit(
        lambda M, z: ludlow16_concentration_jax(M, z, **PLANCK15)
    )
    t0 = time.perf_counter()
    _ = fresh_jit(1.0e12, 0.5).block_until_ready()
    t_compile = time.perf_counter() - t0
    print(f"JIT compile + first call: {t_compile*1e3:.1f} ms")

    n_iter = 100
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = fresh_jit(1.0e12, 0.5).block_until_ready()
    t_post = (time.perf_counter() - t0) / n_iter
    print(f"Post-compile single call: {t_post*1e3:.3f} ms")


if __name__ == "__main__":
    main()
