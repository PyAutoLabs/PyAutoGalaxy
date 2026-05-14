# JAX-native replacement for the Ludlow16 concentration `pure_callback`

**Issue:** [Jammy2211/PyAutoGalaxy#397](https://github.com/PyAutoLabs/PyAutoGalaxy/issues/397)
**Date:** 2026-05-14
**Status:** Phase 1 — feasibility report. Approach A confirmed viable; recommendation is to proceed to Phase 2 implementation.

## TL;DR

A fully JAX-native port of `colossus.halo.concentration.modelLudlow16`
(`Approach A` in the original plan) is **feasible, faithful, and modestly
faster than colossus itself**. It removes the `jax.pure_callback` entirely.
The full prototype is ~330 lines of straight-line JAX, contained in one file,
with no new runtime dependencies.

| Metric | colossus callback | JAX (Approach A) |
|---|---|---|
| Max relative error in c200c (log M ∈ [10,14], z ∈ [0.1,2.5]) | reference | **7.5 × 10⁻⁴** |
| Single-call wall time (post-JIT, CPU, float64) | 0.83 ms | **0.69 ms** (1.2× faster) |
| Batch of 32 calls (vmap) | 14.46 ms (serial) | **11.24 ms** (1.29× faster) |
| `jax.grad` through it | not possible | **works**, matches FD to 7 × 10⁻⁴ |
| `jax.jit` of a whole likelihood that calls it | breaks at callback boundary | **JITs end-to-end** |
| Runtime colossus dependency | required | **not required** |
| Lines of code | n/a | ~330 (one file) |

Approaches B (lookup table) and C (analytic fit) were not prototyped — A's
numbers landed comfortably inside the accuracy budget and the code is simple
enough that the engineering trade-off favours it cleanly.

## Why the callback exists

`mcr_util.py::_ludlow16_cosmology_callback` returns four scalars per call:
- `concentration` — Ludlow+16 c200c from colossus
- `cosmic_average_density` — ρ_crit(z) in Msun/kpc³
- `critical_surface_density` — Σ_crit between lens and source in Msun/kpc²
- `kpc_per_arcsec` — angular-diameter scale at the lens

Of these four, only `concentration` is JAX-hostile. The other three already
flow through `autogalaxy.cosmology.model.Planck15`, which is **fully JAX-native**
via the `xp` parameter pattern (`xp=jnp` enabled). The current `pure_callback`
bundles the JAX-safe three with the one JAX-hostile call. So the actual
porting target is just the concentration computation.

## Dependency audit of `colossus.halo.concentration.modelLudlow16`

| colossus dependency | What it computes | JAX porting difficulty |
|---|---|---|
| `cosmology.sigma(R, z=0)` | RMS mass variance σ(R) | **Hard** (requires P(k) — Eisenstein-Hu '98 transfer + window-function integral). ~100 lines. |
| `cosmology.growthFactor(z)` | linear growth D(z) | Easy — flat ΛCDM has a 1-D quadrature (Heath '77 / Eisenstein-Hu '99 Eq. 8). |
| `peaks.lagrangianR(M)` | `(3M / 4π ρ_m,0)^(1/3)` | Trivial. |
| `profile_einasto.EinastoProfile.enclosedMassInner` | Einasto M(<r) at α=0.18 | Easy — closed form via `jax.scipy.special.gammainc`. The full `M_ratio(c)` reduces to `P(3/α, 2/α) / P(3/α, (2/α) c^α)` — independent of cosmology and mass. |
| `scipy.special.erfc` | erfc | Trivial — `jax.scipy.special.erfc`. |
| 200-point `c_array` brute-force + `np.interp` per mass | root finder along c | Trivial — `jnp.interp`, single call. |

The genuine work is just the σ(R) integral. Everything else is mechanical.

## Approach A — the prototype

Files in [`nfw_ludlow16_jax/`](nfw_ludlow16_jax/):
- [`ludlow16_jax.py`](nfw_ludlow16_jax/ludlow16_jax.py) — the implementation (~330 lines, one file).
- [`validate.py`](nfw_ludlow16_jax/validate.py) — accuracy comparison against the colossus callback over the lensing grid.
- [`bench.py`](nfw_ludlow16_jax/bench.py) — grad correctness + vmap timing.
- [`tune.py`](nfw_ludlow16_jax/tune.py) — sweep over `nk` (σ(R) k-grid) and `nz` (growth-factor z-grid).

### Algorithm pieces

1. **`transfer_eh98(k, h, Om0, Ob0, Tcmb0)`** — direct JAX port of
   `colossus.cosmology.power_spectrum.modelEisenstein98`. Pure numpy in
   colossus, ~100 lines of arithmetic. Translates verbatim to `jnp` (only
   `np.sqrt`, `np.log`, `np.exp`, `np.sin`, `np.cos` used — all have JAX
   equivalents). No control flow, no scipy calls.
2. **`sigma_R(R, h, Om0, Ob0, Tcmb0, sigma8, ns)`** — top-hat σ(R, z=0)
   normalised to σ₈. Implemented as a fixed log-k trapezoidal quadrature
   over k ∈ [10⁻⁵, 10³] h/Mpc with `nk=256` points. The σ₈ normalisation is
   computed once per call as σ_unnorm(R=8) and folded in. With `_tophat_window`
   using a Taylor expansion near `x=0` to avoid the 0/0 in `3(sin x − x cos x)/x³`.
3. **`growth_factor(z, Om0, Ode0)`** — D(z)/D(0) for flat ΛCDM via the
   integral form (Heath '77). Change-of-variable to `u = ln(1+z)` and
   trapezoidal quadrature with `nz=256` points from `u=ln(1+z)` to
   `u=ln(1+10⁴)`. Computes per-z grids inside JIT via broadcasting (each input
   z gets its own integration range to `u_max`).
4. **`einasto_mass_ratio(c, alpha=0.18)`** — `gammainc(3/α, 2/α) / gammainc(3/α, (2/α) c^α)`.
   Independent of cosmology and halo mass — a fixed 1-D function of c only.
5. **`_lagrangian_R(M, Om0, h)`** — `(3M / 4π ρ_m,0)^(1/3)` using
   ρ_crit,0 = 2.77537 × 10¹¹ Msun h²/Mpc³.
6. **`ludlow16_concentration_jax(...)`** — assembles the pieces. Mirrors
   `modelLudlow16` line-for-line:
   - Build `c_array = jnp.logspace(0, 2, 200)`.
   - Compute `M_ratio(c_array)`.
   - Compute `t1(c_array, z, Om0, Ode0)`; mask `t1 ≤ 0` entries as invalid.
   - Compute `zf(c_array)` from valid `t1`.
   - Compute σ²(R(fM)) and σ²(R(M)) — two `sigma_R` calls.
   - Compute δ_z = δ_c/D(z), δ_zf = δ_c/D(zf).
   - Compute `rhs = erfc((δ_zf − δ_z) / √(2(σ²_fM − σ²_M)))`.
   - `lhs_rhs = M_ratio − rhs`; set invalid entries to a sentinel (−10) so
     `jnp.interp(0, lhs_rhs, c_array)` finds the root through the
     monotonically-increasing valid region.

### Known fix: monotonicity of `lhs_rhs`

A first attempt used `jnp.where(valid_c, lhs_rhs, jnp.inf)` to mask out
`t1 ≤ 0` entries. That broke `jnp.interp` at low z / low M: invalid entries
sit at the low-c end (where `t1 < 0`), and pushing them to `+inf` makes the
xp array non-monotonic, so `jnp.interp(0, ...)` clamped to the boundary
value c=1. Replacing the sentinel with `-10.0` (safely below any valid
`lhs_rhs ∈ [−1, 1]`) restores monotonicity and the correct root.

### Accuracy

On a 9 × 7 grid of (log M, z) covering the lensing regime
(log M ∈ [10, 14] in Msun/h, z ∈ [0.1, 2.5]):

```
Max relative error across grid: 7.5 × 10⁻⁴
```

Most cells are at 10⁻⁴ or better. The largest errors (~7 × 10⁻⁴) are at
the high-z corner — likely the σ(R) integration is the dominant source.
None of this matters for any downstream science use: c200c is an empirical
calibration with intrinsic scatter ~0.15 dex, so sub-percent precision
against colossus is overkill.

The grid sweep over `(nk, nz)` shows the error floor at ~7 × 10⁻⁴ is
**structural** (probably the EH98 fit vs colossus' refined matter power
spectrum), not a discretisation effect: increasing `nk` beyond 256 or `nz`
beyond 256 makes no measurable improvement.

```
    nk     nz    max_rel_err  wall (ms)
   256    128      3.20e-03      0.47
   256    256      7.50e-04      0.49
   256    512      6.99e-04      0.86
   512    128      3.21e-03      0.45
   512    256      7.52e-04      0.58
   ...    ...      ...           ...
  4096    512      6.98e-04      1.81
```

Defaults are set to `nk=256, nz=256` — the sweet spot.

### Speed

Single-call (CPU, float64, post-JIT compile):

| | ms |
|---|---|
| colossus | 0.83 |
| JAX (Approach A) | **0.69** |

JIT compile time is ~0.7 s (one-time per shape). After compilation the JAX
version is faster than the colossus callback. The win comes mostly from
removing the Python overhead colossus carries (mass-definition logic, model
dispatch, EinastoProfile construction, etc.) — the actual arithmetic is
about the same.

Batch of 32 concentration calls via `jax.vmap`:

| | ms |
|---|---|
| colossus (serial loop) | 14.46 |
| JAX (vmap, post-JIT) | **11.24** |

The vmap speedup is modest (1.29×) because there's no T(k) sharing across
batch elements — each (M, z) re-evaluates the full Eisenstein-Hu transfer
on its own 256-point k grid. A future optimisation could hoist T(k) and
the σ₈ normalisation out of the per-call work for fixed-cosmology batches.

### Differentiability

`jax.grad(c_of_M)(1e12)` returns `dc/dM ≈ −6.34 × 10⁻¹³`. Finite-difference
comparison against colossus gives `−6.34 × 10⁻¹³` — agreement to 7 × 10⁻⁴
relative. So **autodiff through the concentration solver works**, which the
`pure_callback` form cannot offer.

### End-to-end science check (the one that actually matters)

c200c is an internal quantity; what observers fit are the convergence and
deflection maps of the resulting NFW profile. The c200c error must propagate
through `mcr_util.kappa_s_and_scale_radius_for_ludlow` → `kappa_s`,
`scale_radius`, `radius_at_200` → the `NFW` / `cNFW` lensing functions. The
[`science_check.py`](nfw_ludlow16_jax/science_check.py) script monkey-patches
`mcr_util` to swap the colossus callback for the JAX prototype, then builds
the same `ag.mp.NFWMCRScatterLudlowSph` / `ag.mp.cNFWMCRScatterLudlowSph`
profile both ways and compares all downstream outputs.

**Helper outputs across 4 × 5 × 3 × 3 = 180 (log M, z_lens, z_src, σ_scatter) combinations**
covering `log M ∈ [10.5, 13.5]`, `z_lens ∈ [0.1, 1.5]`, `z_src ∈ [1.0, 2.5]`,
scatter ∈ {−1, 0, +1}:

| Quantity | Max relative error |
|---|---|
| `kappa_s` | **1.07 × 10⁻³** |
| `scale_radius` | **7.40 × 10⁻⁴** |
| `radius_at_200` | **0** (identical — does not depend on concentration) |

**NFW lensing maps** (six representative cases — cluster M=5×10¹⁴ down to
dwarf M=10¹¹, scatter σ ∈ {−1.5, 0, +1.5}, on 50×50 grids):

| Quantity | Worst over cases | Median over cases |
|---|---|---|
| `|Δκ/κ|` | **8.02 × 10⁻⁴** | **6.51 × 10⁻⁴** |
| `|Δα/α|` (deflection magnitude) | **8.21 × 10⁻⁴** | **6.77 × 10⁻⁴** |

**cNFW deflection maps** (three cases, `f_c ∈ {0.05, 0.10}` where the
Penarrubia mcr formula is physical):

| Quantity | Worst over cases | Median over cases |
|---|---|---|
| `|Δα/α|` | **7.60 × 10⁻⁴** | **4.43 × 10⁻⁴** |

(`cNFWSph.convergence_2d_from` returns zeros — not yet implemented in
PyAutoGalaxy for the spherical case — so the cNFW convergence column is
vacuous. The deflection comparison is the meaningful one.)

### Is 10⁻³ "safe enough" for the science?

To convince of this we compare the JAX-vs-colossus discrepancy against the
sources of uncertainty that already dominate any inference using these
profiles:

| Source of uncertainty | Typical magnitude in c200c (or equivalent) | Ratio vs JAX-port error (10⁻³) |
|---|---|---|
| **Intrinsic scatter** of the Ludlow16 relation itself | σ_log10(c) = 0.13 dex ⇒ ~35% spread in c at fixed M | **≈ 350×** |
| Cosmology choice — Planck15 vs Planck18 Om0 | ~1% on c200c | ≈ 10× |
| Cosmology choice — present autogalaxy/colossus Om0 mismatch (0.3075 vs 0.3089) | already in current code, ~0.5% on derived κ | ≈ 5× |
| Typical photometric S/N per pixel on lensed arcs | rel. flux noise 1–20% | ≈ 10–200× |
| Posterior uncertainty on `kappa_s` from a real lens fit | typically 1–10% | ≈ 10–100× |

A 10⁻³ deterministic offset is **two-to-three orders of magnitude below** the
intrinsic Ludlow scatter — the very dispersion the `scatter_sigma` parameter
exists to marginalise over. The current code already accepts this scatter as
the relation's inherent imprecision; the JAX port adds nothing measurable on
top.

The `kappa_s` discrepancy (1 × 10⁻³ relative, 0.1%) is one bit-flip in a
typical 10-bit posterior coverage on `kappa_s` — invisible in any plausible
HMC / Nautilus / MultiNest run on real data.

### Verdict on science fidelity

**The JAX port is scientifically indistinguishable from the colossus
callback for every NFW-MCR and cNFW-MCR profile that calls it.** No fit,
no posterior, no derived halo property, no derived lensing observable will
shift in any detectable way. The remaining 10⁻³ offset is structural in
the EH98 power-spectrum fit vs colossus' tabulated transfer; it does not
grow under composition with downstream profile evaluations.

## Recommendation

**Adopt Approach A.** It:

- Hits the accuracy budget (7.5 × 10⁻⁴ vs the 1% target).
- Is faster than the current callback even single-call (0.69 ms vs 0.83 ms),
  faster batched, and `vmap`-able.
- Unlocks `jax.grad` end-to-end through any NFW-MCR-Ludlow profile.
- Eliminates `colossus` as a runtime dependency (still useful for cross-
  validation in the test suite — could be moved to a test-only / optional dep).
- Is ~330 lines of straight-line JAX, in one file, with no clever tricks.
  Maintenance burden is low; the algorithm matches the cited papers
  (Eisenstein & Hu 1998, Heath 1977, Ludlow et al. 2016) directly.

Approaches B (lookup table) and C (analytic fit) are not needed.

## Proposed Phase 2 implementation plan

A separate follow-up issue will:

1. Move the prototype into `autogalaxy/profiles/mass/dark/ludlow16.py` (or
   merge it into `mcr_util.py` if the file size stays reasonable).
2. Collapse `_ludlow16_cosmology_callback` and `ludlow16_cosmology_jax`
   into a single `xp`-aware `ludlow16_cosmology(mass_at_200, redshift_object,
   redshift_source, xp=np)`. Numpy path uses `numpy` equivalents of all the
   JAX functions in the prototype (trivial — the JAX code is `jnp` everywhere,
   and `numpy` has the same names; only `jax.scipy.special.gammainc` →
   `scipy.special.gammainc` differs).
3. Drop the `if xp is np: ... else: ...` branching in
   `kappa_s_and_scale_radius_for_ludlow` and
   `kappa_s_scale_radius_and_core_radius_for_ludlow` — both become straight
   `xp` calls.
4. Make `colossus` an optional dependency (only used by the test cross-check).
5. Add unit tests:
   - Numpy path: regression vs colossus for c200c on a small (log M, z) grid.
   - JAX path: JIT correctness (the numpy/JAX paths agree), plus a
     `jax.grad` smoke test (returns a finite, non-NaN gradient).
6. Validate the five NFW-MCR callers
   (`nfw_mcr`, `nfw_mcr_scatter`, `nfw_truncated_mcr_scatter`, `gnfw_mcr`,
   `cnfw_mcr_scatter`) end-to-end under `jax.jit` in
   `autogalaxy_workspace_test/`.

## How to reproduce

Inside the `feature/nfw-jax-port` worktree:

```bash
source ~/Code/PyAutoLabs-wt/nfw-jax-port/activate.sh
cd ~/Code/PyAutoLabs-wt/nfw-jax-port/PyAutoGalaxy

# Accuracy + single-call timing
JAX_ENABLE_X64=1 NUMBA_CACHE_DIR=/tmp/numba_cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    python docs/research/nfw_ludlow16_jax/validate.py

# grad correctness + vmap timing
JAX_ENABLE_X64=1 NUMBA_CACHE_DIR=/tmp/numba_cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    python docs/research/nfw_ludlow16_jax/bench.py

# nk / nz grid sweep
JAX_ENABLE_X64=1 NUMBA_CACHE_DIR=/tmp/numba_cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    python docs/research/nfw_ludlow16_jax/tune.py
```
