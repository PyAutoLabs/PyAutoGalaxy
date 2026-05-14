"""
End-to-end science validation of the JAX Ludlow16 prototype.

The c200c relative error vs colossus is at most 7.5e-4 in isolation. This
script propagates that into the *downstream* lensing quantities that drive
likelihoods:

  - kappa_s, scale_radius, radius_at_200      (from mcr_util helpers)
  - convergence map of the resulting NFW profile, on a 2D grid
  - deflection map of the resulting NFW profile, on a 2D grid
  - same for the cNFW (core) helper

It compares the *current* pipeline (colossus callback inside mcr_util) to a
parallel pipeline that swaps in `ludlow16_concentration_jax`. Everything else
about the helpers — the autogalaxy Planck15 cosmology used for rho_crit,
Sigma_crit, kpc/arcsec — is identical between the two paths.

Run:
    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
        JAX_ENABLE_X64=1 python docs/research/nfw_ludlow16_jax/science_check.py
"""

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import autogalaxy as ag
from autogalaxy.profiles.mass.dark import mcr_util
from autogalaxy.cosmology.model import Planck15

sys.path.insert(0, os.path.dirname(__file__))
from ludlow16_jax import ludlow16_concentration_jax  # noqa: E402


# Colossus 'planck15' preset — what _ludlow16_cosmology_callback uses
# internally for the concentration. autogalaxy's Planck15 (Om0=0.3075) is used
# for everything else; we match that pattern.
COLOSSUS_PLANCK15 = dict(
    h=0.6774,
    Om0=0.3089,
    Ob0=0.0486,
    Tcmb0=2.7255,
    sigma8=0.8159,
    ns=0.9667,
)


# ---------------------------------------------------------------------------
# JAX-native replacement for _ludlow16_cosmology_callback
# ---------------------------------------------------------------------------


def ludlow16_cosmology_jax_native(mass_at_200, redshift_object, redshift_source):
    """
    Drop-in replacement for `mcr_util._ludlow16_cosmology_callback`.

    Same four returns as the current callback, but with the colossus
    concentration call replaced by the JAX-native prototype. The other three
    values flow through autogalaxy's Planck15 unchanged.
    """
    cosmology = Planck15()
    h = COLOSSUS_PLANCK15["h"]

    # JAX path: convert M_sun → M_sun/h, call the JAX concentration
    concentration = float(
        ludlow16_concentration_jax(
            mass_at_200 * h,
            redshift_object,
            **COLOSSUS_PLANCK15,
        )
    )

    cosmic_average_density = cosmology.critical_density(redshift_object, xp=np)
    critical_surface_density = (
        cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
            redshift_0=redshift_object,
            redshift_1=redshift_source,
            xp=np,
        )
    )
    kpc_per_arcsec = cosmology.kpc_per_arcsec_from(
        redshift=redshift_object, xp=np
    )

    return (
        np.float64(concentration),
        np.float64(cosmic_average_density),
        np.float64(critical_surface_density),
        np.float64(kpc_per_arcsec),
    )


def kappa_s_scale_radius_ludlow_jax(
    mass_at_200, scatter_sigma, redshift_object, redshift_source
):
    """
    Mirror of `mcr_util.kappa_s_and_scale_radius_for_ludlow` numpy branch
    but with the JAX concentration replacement.
    """
    (
        concentration,
        cosmic_average_density,
        critical_surface_density,
        kpc_per_arcsec,
    ) = ludlow16_cosmology_jax_native(
        mass_at_200, redshift_object, redshift_source
    )

    concentration = 10.0 ** (np.log10(concentration) + scatter_sigma * 0.15)

    radius_at_200 = (
        mass_at_200 / (200.0 * cosmic_average_density * (4.0 * np.pi / 3.0))
    ) ** (1.0 / 3.0)

    de_c = (
        200.0
        / 3.0
        * (
            concentration ** 3
            / (np.log(1.0 + concentration) - concentration / (1.0 + concentration))
        )
    )

    scale_radius_kpc = radius_at_200 / concentration
    rho_s = cosmic_average_density * de_c
    kappa_s = rho_s * scale_radius_kpc / critical_surface_density
    scale_radius = scale_radius_kpc / kpc_per_arcsec

    return kappa_s, scale_radius, radius_at_200


def kappa_s_scale_radius_core_radius_ludlow_jax(
    mass_at_200, scatter_sigma, f_c, redshift_object, redshift_source
):
    """Mirror of `mcr_util.kappa_s_scale_radius_and_core_radius_for_ludlow`."""
    (
        concentration,
        cosmic_average_density,
        critical_surface_density,
        kpc_per_arcsec,
    ) = ludlow16_cosmology_jax_native(
        mass_at_200, redshift_object, redshift_source
    )

    concentration = 10.0 ** (np.log10(concentration) + scatter_sigma * 0.15)

    radius_at_200 = (
        mass_at_200 / (200.0 * cosmic_average_density * (4.0 * np.pi / 3.0))
    ) ** (1.0 / 3.0)

    mcr_penarrubia = (
        f_c ** 2 * np.log(1 + concentration / f_c)
        + (1 - 2 * f_c) * np.log(1 + concentration)
    ) / (1 + f_c) ** 2 - concentration / ((1 + concentration) * (1 - f_c))

    scale_radius_kpc = radius_at_200 / concentration
    rho_0 = mass_at_200 / (4 * np.pi * scale_radius_kpc ** 3 * mcr_penarrubia)
    kappa_s = rho_0 * scale_radius_kpc / critical_surface_density
    scale_radius = scale_radius_kpc / kpc_per_arcsec
    core_radius = f_c * scale_radius

    return kappa_s, scale_radius, core_radius, radius_at_200


# ---------------------------------------------------------------------------
# Check 1: helper outputs (kappa_s, scale_radius, radius_at_200) across grid
# ---------------------------------------------------------------------------


def helper_grid_check():
    print("=" * 86)
    print("Check 1: NFWMCRLudlow helper outputs — kappa_s, scale_radius, "
          "radius_at_200")
    print("(mass M_200 in Msun physical; scatter_sigma sweep too)")
    print("=" * 86)

    log_M_grid = [10.5, 11.5, 12.5, 13.5]
    z_obj_grid = [0.1, 0.3, 0.5, 1.0, 1.5]
    z_src_grid = [1.0, 1.5, 2.5]
    scatter_grid = [-1.0, 0.0, 1.0]

    print(f"{'log M':>8} {'z_obj':>6} {'z_src':>6} {'σ_scat':>7} "
          f"{'kappa_s':>10} {'sc.rad':>10} {'r_200':>10} "
          f"{'Δκ/κ':>10} {'Δs/s':>10} {'Δr/r':>10}")
    print("-" * 110)

    max_rel = {"kappa_s": 0.0, "scale_radius": 0.0, "radius_at_200": 0.0}

    for log_M in log_M_grid:
        for z_obj in z_obj_grid:
            for z_src in z_src_grid:
                if z_src <= z_obj:
                    continue
                for scatter in scatter_grid:
                    M = 10.0 ** log_M
                    k_col, s_col, r_col = (
                        mcr_util.kappa_s_and_scale_radius_for_ludlow(
                            M, scatter, z_obj, z_src
                        )
                    )
                    k_jax, s_jax, r_jax = kappa_s_scale_radius_ludlow_jax(
                        M, scatter, z_obj, z_src
                    )
                    rk = abs(k_jax - k_col) / abs(k_col)
                    rs = abs(s_jax - s_col) / abs(s_col)
                    rr = abs(r_jax - r_col) / abs(r_col)
                    max_rel["kappa_s"] = max(max_rel["kappa_s"], rk)
                    max_rel["scale_radius"] = max(max_rel["scale_radius"], rs)
                    max_rel["radius_at_200"] = max(max_rel["radius_at_200"], rr)
                    print(
                        f"{log_M:8.2f} {z_obj:6.2f} {z_src:6.2f} {scatter:7.1f} "
                        f"{k_col:10.4e} {s_col:10.4e} {r_col:10.4e} "
                        f"{rk:10.2e} {rs:10.2e} {rr:10.2e}"
                    )

    print("-" * 110)
    print("Max relative errors across the helper grid:")
    for k, v in max_rel.items():
        print(f"  {k:>16s}: {v:.3e}")
    print()
    return max_rel


# ---------------------------------------------------------------------------
# Check 2: convergence + deflection maps for representative NFW model
# ---------------------------------------------------------------------------


def lensing_map_check(label, build_profile_col, build_profile_jax,
                     case_descs, grid_shape=(50, 50), pixel_scales=0.1):
    print("=" * 86)
    print(f"Check 2 ({label}): convergence + deflection maps on a 2D grid")
    print(f"(grid {grid_shape[0]}x{grid_shape[1]} at {pixel_scales} arcsec/pix)")
    print("=" * 86)

    grid = ag.Grid2D.uniform(shape_native=grid_shape, pixel_scales=pixel_scales)

    print(f"{'case':>32} "
          f"{'|Δκ|_max':>10} {'|Δκ/κ|_max':>12} {'|Δκ/κ|_med':>12} "
          f"{'|Δα|_max':>10} {'|Δα/α|_max':>12} {'|Δα/α|_med':>12}")
    print("-" * 110)

    overall = {
        "abs_kappa": 0.0,
        "rel_kappa_max": 0.0,
        "rel_kappa_med": 0.0,
        "abs_defl": 0.0,
        "rel_defl_max": 0.0,
        "rel_defl_med": 0.0,
    }

    for desc, params in case_descs:
        prof_col = build_profile_col(**params)
        prof_jax = build_profile_jax(**params)

        k_col = np.asarray(prof_col.convergence_2d_from(grid=grid))
        k_jax = np.asarray(prof_jax.convergence_2d_from(grid=grid))
        d_col = np.asarray(prof_col.deflections_yx_2d_from(grid=grid))
        d_jax = np.asarray(prof_jax.deflections_yx_2d_from(grid=grid))

        abs_k = np.abs(k_jax - k_col)
        rel_k = abs_k / np.maximum(np.abs(k_col), 1e-30)
        # deflections are (y, x) vectors; use magnitude
        d_col_mag = np.linalg.norm(d_col.reshape(-1, 2), axis=-1)
        d_jax_mag = np.linalg.norm(d_jax.reshape(-1, 2), axis=-1)
        abs_d = np.abs(d_jax_mag - d_col_mag)
        rel_d = abs_d / np.maximum(np.abs(d_col_mag), 1e-30)

        overall["abs_kappa"] = max(overall["abs_kappa"], abs_k.max())
        overall["rel_kappa_max"] = max(overall["rel_kappa_max"], rel_k.max())
        overall["rel_kappa_med"] = max(overall["rel_kappa_med"], np.median(rel_k))
        overall["abs_defl"] = max(overall["abs_defl"], abs_d.max())
        overall["rel_defl_max"] = max(overall["rel_defl_max"], rel_d.max())
        overall["rel_defl_med"] = max(overall["rel_defl_med"], np.median(rel_d))

        print(
            f"{desc:>32} "
            f"{abs_k.max():10.2e} {rel_k.max():12.2e} {np.median(rel_k):12.2e} "
            f"{abs_d.max():10.2e} {rel_d.max():12.2e} {np.median(rel_d):12.2e}"
        )

    print("-" * 110)
    print("Worst across cases:")
    for k, v in overall.items():
        print(f"  {k:>16s}: {v:.3e}")
    print()
    return overall


# ---------------------------------------------------------------------------
# Profile builders — we instantiate the existing autogalaxy profiles for the
# colossus path. For the JAX path we instantiate the same profile class but
# monkey-patch the helper to call our JAX replacement.
# ---------------------------------------------------------------------------

from contextlib import contextmanager


@contextmanager
def patch_mcr_util_with_jax():
    """Within this context, mcr_util.kappa_s_and_scale_radius_for_ludlow and
    mcr_util.kappa_s_scale_radius_and_core_radius_for_ludlow use the JAX
    concentration instead of the colossus callback."""
    orig_ludlow = mcr_util.kappa_s_and_scale_radius_for_ludlow
    orig_cnfw = mcr_util.kappa_s_scale_radius_and_core_radius_for_ludlow
    mcr_util.kappa_s_and_scale_radius_for_ludlow = (
        kappa_s_scale_radius_ludlow_jax
    )
    mcr_util.kappa_s_scale_radius_and_core_radius_for_ludlow = (
        kappa_s_scale_radius_core_radius_ludlow_jax
    )
    try:
        yield
    finally:
        mcr_util.kappa_s_and_scale_radius_for_ludlow = orig_ludlow
        mcr_util.kappa_s_scale_radius_and_core_radius_for_ludlow = orig_cnfw


def build_nfwmcrludlow_col(M, z_obj, z_src, scatter):
    return ag.mp.NFWMCRScatterLudlowSph(
        centre=(0.0, 0.0),
        mass_at_200=M,
        scatter_sigma=scatter,
        redshift_object=z_obj,
        redshift_source=z_src,
    )


def build_nfwmcrludlow_jax(M, z_obj, z_src, scatter):
    with patch_mcr_util_with_jax():
        return ag.mp.NFWMCRScatterLudlowSph(
            centre=(0.0, 0.0),
            mass_at_200=M,
            scatter_sigma=scatter,
            redshift_object=z_obj,
            redshift_source=z_src,
        )


def build_cnfwmcrludlow_col(M, z_obj, z_src, scatter, f_c):
    return ag.mp.cNFWMCRScatterLudlowSph(
        centre=(0.0, 0.0),
        mass_at_200=M,
        scatter_sigma=scatter,
        f_c=f_c,
        redshift_object=z_obj,
        redshift_source=z_src,
    )


def build_cnfwmcrludlow_jax(M, z_obj, z_src, scatter, f_c):
    with patch_mcr_util_with_jax():
        return ag.mp.cNFWMCRScatterLudlowSph(
            centre=(0.0, 0.0),
            mass_at_200=M,
            scatter_sigma=scatter,
            f_c=f_c,
            redshift_object=z_obj,
            redshift_source=z_src,
        )


def main():
    helper_grid_check()

    # Representative NFW lensing cases (cluster, group, galaxy, dwarf)
    nfw_cases = [
        ("M=5e14 cluster, z=(0.3,1.5), σ=0",
         dict(M=5.0e14, z_obj=0.3, z_src=1.5, scatter=0.0)),
        ("M=1e13 group,  z=(0.5,2.0), σ=0",
         dict(M=1.0e13, z_obj=0.5, z_src=2.0, scatter=0.0)),
        ("M=1e12 galaxy, z=(0.2,1.0), σ=0",
         dict(M=1.0e12, z_obj=0.2, z_src=1.0, scatter=0.0)),
        ("M=1e11 dwarf,  z=(0.1,1.0), σ=0",
         dict(M=1.0e11, z_obj=0.1, z_src=1.0, scatter=0.0)),
        ("M=1e12, σ=+1.5 (high scatter)",
         dict(M=1.0e12, z_obj=0.3, z_src=2.0, scatter=1.5)),
        ("M=1e12, σ=-1.5 (low scatter)",
         dict(M=1.0e12, z_obj=0.3, z_src=2.0, scatter=-1.5)),
    ]
    lensing_map_check(
        "NFWMCRScatterLudlow", build_nfwmcrludlow_col,
        build_nfwmcrludlow_jax, nfw_cases,
    )

    # cNFW: f_c values where the Penarrubia mcr is physical (positive)
    # and grids large enough to sample the convergence outside the core.
    cnfw_cases = [
        ("M=1e13, f_c=0.05, σ=0  (grid 50x50 @ 0.5\")",
         dict(M=1.0e13, z_obj=0.3, z_src=1.5, scatter=0.0, f_c=0.05),
         (50, 50), 0.5),
        ("M=1e12, f_c=0.05, σ=0  (grid 50x50 @ 0.1\")",
         dict(M=1.0e12, z_obj=0.3, z_src=1.5, scatter=0.0, f_c=0.05),
         (50, 50), 0.1),
        ("M=1e12, f_c=0.10, σ=+1.0 (grid 50x50 @ 0.1\")",
         dict(M=1.0e12, z_obj=0.5, z_src=2.0, scatter=1.0, f_c=0.10),
         (50, 50), 0.1),
    ]

    # need per-case grid override → small helper inline
    print("=" * 86)
    print("Check 2 (cNFWMCRScatterLudlow): convergence + deflection maps")
    print("=" * 86)
    print(f"{'case':>45} "
          f"{'|Δκ|_max':>10} {'|Δκ/κ|_max':>12} {'|Δκ/κ|_med':>12} "
          f"{'|Δα|_max':>10} {'|Δα/α|_max':>12} {'|Δα/α|_med':>12}")
    print("-" * 130)

    overall = {"abs_kappa": 0.0, "rel_kappa_max": 0.0, "rel_kappa_med": 0.0,
               "abs_defl": 0.0, "rel_defl_max": 0.0, "rel_defl_med": 0.0}
    for desc, params, gshape, gscale in cnfw_cases:
        grid = ag.Grid2D.uniform(shape_native=gshape, pixel_scales=gscale)
        p_col = build_cnfwmcrludlow_col(**params)
        p_jax = build_cnfwmcrludlow_jax(**params)
        k_col = np.asarray(p_col.convergence_2d_from(grid=grid))
        k_jax = np.asarray(p_jax.convergence_2d_from(grid=grid))
        d_col = np.asarray(p_col.deflections_yx_2d_from(grid=grid))
        d_jax = np.asarray(p_jax.deflections_yx_2d_from(grid=grid))

        # mask out nans (cNFW formulas hit invalid-sqrt branches at theta≈radius)
        valid = ~(np.isnan(k_col) | np.isnan(k_jax))
        abs_k = np.abs(k_jax[valid] - k_col[valid])
        rel_k = abs_k / np.maximum(np.abs(k_col[valid]), 1e-30)
        d_col_m = np.linalg.norm(d_col.reshape(-1, 2), axis=-1)
        d_jax_m = np.linalg.norm(d_jax.reshape(-1, 2), axis=-1)
        valid_d = ~(np.isnan(d_col_m) | np.isnan(d_jax_m))
        abs_d = np.abs(d_jax_m[valid_d] - d_col_m[valid_d])
        rel_d = abs_d / np.maximum(np.abs(d_col_m[valid_d]), 1e-30)

        overall["abs_kappa"] = max(overall["abs_kappa"], abs_k.max() if abs_k.size else 0)
        overall["rel_kappa_max"] = max(overall["rel_kappa_max"], rel_k.max() if rel_k.size else 0)
        overall["rel_kappa_med"] = max(overall["rel_kappa_med"], np.median(rel_k) if rel_k.size else 0)
        overall["abs_defl"] = max(overall["abs_defl"], abs_d.max() if abs_d.size else 0)
        overall["rel_defl_max"] = max(overall["rel_defl_max"], rel_d.max() if rel_d.size else 0)
        overall["rel_defl_med"] = max(overall["rel_defl_med"], np.median(rel_d) if rel_d.size else 0)

        print(f"{desc:>45} "
              f"{(abs_k.max() if abs_k.size else 0):10.2e} "
              f"{(rel_k.max() if rel_k.size else 0):12.2e} "
              f"{(np.median(rel_k) if rel_k.size else 0):12.2e} "
              f"{(abs_d.max() if abs_d.size else 0):10.2e} "
              f"{(rel_d.max() if rel_d.size else 0):12.2e} "
              f"{(np.median(rel_d) if rel_d.size else 0):12.2e}")
    print("-" * 130)
    print("Worst across cNFW cases:")
    for k, v in overall.items():
        print(f"  {k:>16s}: {v:.3e}")
    print()


if __name__ == "__main__":
    main()
