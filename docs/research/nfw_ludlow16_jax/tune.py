"""
Sweep nk (σ(R) k-grid) and nz (growth-factor z-grid) to find the smallest
grids that hold ≤ 0.1% relative error vs colossus.
"""

import os
import time
import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
jax.config.update("jax_enable_x64", True)

from colossus.cosmology import cosmology as col_cosmology
from colossus.halo.concentration import concentration as col_concentration

import sys
sys.path.insert(0, os.path.dirname(__file__))
from ludlow16_jax import ludlow16_concentration_jax  # noqa: E402


PLANCK15 = dict(
    h=0.6774, Om0=0.3089, Ob0=0.0486, Tcmb0=2.7255,
    sigma8=0.8159, ns=0.9667,
)


def max_rel_err(nk, nz):
    col_cosmology.setCosmology("planck15")
    fn = jax.jit(lambda M, z: ludlow16_concentration_jax(
        M, z, **PLANCK15, sigma_nk=nk, growth_nz=nz,
    ))
    _ = fn(1.0e12, 0.5).block_until_ready()  # warm up

    max_err = 0.0
    for log_M in np.linspace(10.0, 14.0, 9):
        for z in np.linspace(0.1, 2.5, 7):
            M = 10.0 ** log_M
            c_col = float(col_concentration(M, "200c", z, model="ludlow16"))
            c_jax = float(fn(M, z))
            err = abs(c_jax - c_col) / c_col
            max_err = max(max_err, err)
    # timing
    n_iter = 30
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = fn(1.0e12, 0.5).block_until_ready()
    t = (time.perf_counter() - t0) / n_iter
    return max_err, t


print(f"{'nk':>6} {'nz':>6} {'max_rel_err':>14} {'wall (ms)':>10}")
print("-" * 42)
for nk in [256, 512, 1024, 2048, 4096]:
    for nz in [128, 256, 512]:
        e, t = max_rel_err(nk, nz)
        print(f"{nk:>6d} {nz:>6d} {e:>14.3e} {t*1e3:>10.3f}")
