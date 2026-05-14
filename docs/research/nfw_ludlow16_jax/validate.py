"""
Validate the JAX prototype `ludlow16_concentration_jax` against colossus'
`modelLudlow16` over the strong-lensing parameter regime.

Run with:
    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
        python docs/research/nfw_ludlow16_jax/validate.py
"""

import os
import time
import numpy as np

# Force float64 in JAX
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from colossus.cosmology import cosmology as col_cosmology
from colossus.halo.concentration import concentration as col_concentration

import sys
sys.path.insert(0, os.path.dirname(__file__))
from ludlow16_jax import ludlow16_concentration_jax  # noqa: E402


# Planck15 parameters (matches colossus' built-in 'planck15')
PLANCK15 = dict(
    h=0.6774,
    Om0=0.3089,
    Ob0=0.0486,
    Tcmb0=2.7255,
    sigma8=0.8159,
    ns=0.9667,
)


def colossus_c200c(M_Msun_per_h, z):
    """Reference concentration via colossus."""
    col_cosmology.setCosmology("planck15")
    c = col_concentration(M_Msun_per_h, "200c", z, model="ludlow16")
    return float(c)


def main():
    col_cosmology.setCosmology("planck15")

    # Grid: log M_200c in Msun/h ∈ [10, 14], z ∈ [0.1, 2.5]
    log_M_grid = np.linspace(10.0, 14.0, 9)
    z_grid = np.linspace(0.1, 2.5, 7)

    print("=" * 72)
    print("Ludlow16 concentration: JAX prototype vs colossus")
    print("(M in Msun/h, c200c dimensionless)")
    print("=" * 72)
    print(f"{'log M':>8} {'z':>6} {'c_colossus':>12} {'c_jax':>12} "
          f"{'rel_err':>10}")
    print("-" * 72)

    max_rel = 0.0
    rows = []
    for log_M in log_M_grid:
        for z in z_grid:
            M = 10.0 ** log_M
            c_col = colossus_c200c(M, z)
            c_jax = float(
                ludlow16_concentration_jax(M, z, **PLANCK15)
            )
            rel = abs(c_jax - c_col) / c_col
            max_rel = max(max_rel, rel)
            rows.append((log_M, z, c_col, c_jax, rel))
            print(f"{log_M:8.2f} {z:6.2f} {c_col:12.4f} {c_jax:12.4f} "
                  f"{rel:10.2e}")

    print("-" * 72)
    print(f"Max relative error across grid: {max_rel:.3e}")
    print()

    # Speed comparison (cached compilation)
    jit_fn = jax.jit(
        lambda M, z: ludlow16_concentration_jax(M, z, **PLANCK15)
    )
    # warm up
    _ = jit_fn(1.0e12, 0.5).block_until_ready()

    M_test, z_test = 1.0e12, 0.5
    n_iter = 50

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = colossus_c200c(M_test, z_test)
    t_col = (time.perf_counter() - t0) / n_iter

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = jit_fn(M_test, z_test).block_until_ready()
    t_jax = (time.perf_counter() - t0) / n_iter

    print(f"Single-call wall time at (M=1e12, z=0.5):")
    print(f"  colossus      : {t_col*1e3:.3f} ms")
    print(f"  jax (post-jit): {t_jax*1e3:.3f} ms")
    print(f"  speedup       : {t_col / t_jax:.2f}x")


if __name__ == "__main__":
    main()
