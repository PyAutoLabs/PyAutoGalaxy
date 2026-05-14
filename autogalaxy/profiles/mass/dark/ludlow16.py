"""
JAX-native Ludlow et al. 2016 mass-concentration relation.

Replaces the previous ``jax.pure_callback``-wrapped call to
``colossus.halo.concentration`` (formerly in ``mcr_util.py``). The
algorithm follows colossus' ``modelLudlow16`` (``concentration.py``
lines 1104-1192) and ``modelEisenstein98`` (``power_spectrum.py``
lines 476-608) line-for-line, but every operation is done in pure
``xp.*`` arithmetic so the same function runs under both numpy
(``xp=np``) and JAX (``xp=jnp``).

Validated in PR #402 (Phase 1 feasibility): max relative error in
c200c vs colossus over the lensing parameter grid
(log M ∈ [10, 14] Msun/h, z ∈ [0.1, 2.5]) is **7.5 × 10⁻⁴**, with
end-to-end downstream errors in convergence/deflection of ≤ 8 × 10⁻⁴.
The 0.13 dex intrinsic Ludlow16 scatter is ~350× larger.

Units (matching colossus throughout):
    M200c is in Msun / h
    R is in Mpc / h
    z is dimensionless redshift
"""

import numpy as np


# Colossus' 'planck15' preset — the cosmology the previous callback used
# internally to compute concentration. autogalaxy's own Planck15 (Om0=0.3075)
# is used for everything else in mcr_util.ludlow16_cosmology; this constant is
# only for the concentration call. Keeping the split is an apples-to-apples
# swap with the previous behaviour; unifying the two is a separate decision.
PLANCK15_COSMOLOGY = dict(
    h=0.6774,
    Om0=0.3089,
    Ob0=0.0486,
    Tcmb0=2.7255,
    sigma8=0.8159,
    ns=0.9667,
)


def _gammainc(a, x, xp):
    if xp is np:
        from scipy.special import gammainc
    else:
        from jax.scipy.special import gammainc
    return gammainc(a, x)


def _erfc(arg, xp):
    if xp is np:
        from scipy.special import erfc
    else:
        from jax.scipy.special import erfc
    return erfc(arg)


def _trapezoid_last_axis(y, x, xp):
    """Trapezoidal-rule integration along the last axis of ``y``.

    Equivalent to ``xp.trapezoid(y, x, axis=-1)`` but works on numpy
    versions before 1.26 (which only have the now-deprecated ``trapz``).
    Broadcasting follows the same rule: ``x`` may be 1-D (shared across
    the leading axes of ``y``) or fully-shaped to match ``y``.
    """
    dx = x[..., 1:] - x[..., :-1]
    y_avg = 0.5 * (y[..., :-1] + y[..., 1:])
    return xp.sum(y_avg * dx, axis=-1)


# ---------------------------------------------------------------------------
# Eisenstein & Hu 1998 transfer function — direct port of
# colossus.cosmology.power_spectrum.modelEisenstein98.
# ---------------------------------------------------------------------------


def transfer_eh98(k, h, Om0, Ob0, Tcmb0, xp=np):
    """EH98 transfer function T(k) including baryon acoustic features."""
    omc = Om0 - Ob0
    ombom0 = Ob0 / Om0
    h2 = h ** 2
    om0h2 = Om0 * h2
    ombh2 = Ob0 * h2
    theta2p7 = Tcmb0 / 2.7
    theta2p72 = theta2p7 ** 2
    theta2p74 = theta2p72 ** 2

    kh = k * h

    zeq = 2.50e4 * om0h2 / theta2p74
    keq = 7.46e-2 * om0h2 / theta2p72

    b1d = 0.313 * om0h2 ** -0.419 * (1.0 + 0.607 * om0h2 ** 0.674)
    b2d = 0.238 * om0h2 ** 0.223
    zd = 1291.0 * om0h2 ** 0.251 / (1.0 + 0.659 * om0h2 ** 0.828) * (
        1.0 + b1d * ombh2 ** b2d
    )

    Rd = 31.5 * ombh2 / theta2p74 / (zd / 1e3)
    Req = 31.5 * ombh2 / theta2p74 / (zeq / 1e3)

    s = (
        2.0
        / 3.0
        / keq
        * xp.sqrt(6.0 / Req)
        * xp.log((xp.sqrt(1.0 + Rd) + xp.sqrt(Rd + Req)) / (1.0 + xp.sqrt(Req)))
    )

    ksilk = 1.6 * ombh2 ** 0.52 * om0h2 ** 0.73 * (1.0 + (10.4 * om0h2) ** -0.95)

    q = kh / 13.41 / keq

    a1 = (46.9 * om0h2) ** 0.670 * (1.0 + (32.1 * om0h2) ** -0.532)
    a2 = (12.0 * om0h2) ** 0.424 * (1.0 + (45.0 * om0h2) ** -0.582)
    ac = a1 ** (-ombom0) * a2 ** (-(ombom0 ** 3))

    b1 = 0.944 / (1.0 + (458.0 * om0h2) ** -0.708)
    b2 = (0.395 * om0h2) ** -0.0266
    bc = 1.0 / (1.0 + b1 * ((omc / Om0) ** b2 - 1.0))

    y = (1.0 + zeq) / (1.0 + zd)
    Gy = y * (
        -6.0 * xp.sqrt(1.0 + y)
        + (2.0 + 3.0 * y)
        * xp.log((xp.sqrt(1.0 + y) + 1.0) / (xp.sqrt(1.0 + y) - 1.0))
    )

    ab = 2.07 * keq * s * (1.0 + Rd) ** (-3.0 / 4.0) * Gy

    f = 1.0 / (1.0 + (kh * s / 5.4) ** 4)

    C = 14.2 / ac + 386.0 / (1.0 + 69.9 * q ** 1.08)
    T0t = xp.log(xp.e + 1.8 * bc * q) / (xp.log(xp.e + 1.8 * bc * q) + C * q * q)

    C1bc = 14.2 + 386.0 / (1.0 + 69.9 * q ** 1.08)
    T0t1bc = xp.log(xp.e + 1.8 * bc * q) / (
        xp.log(xp.e + 1.8 * bc * q) + C1bc * q * q
    )
    Tc = f * T0t1bc + (1.0 - f) * T0t

    bb = (
        0.5
        + ombom0
        + (3.0 - 2.0 * ombom0)
        * xp.sqrt((17.2 * om0h2) * (17.2 * om0h2) + 1.0)
    )

    bnode = 8.41 * om0h2 ** 0.435

    st = s / (1.0 + (bnode / kh / s) * (bnode / kh / s) * (bnode / kh / s)) ** (
        1.0 / 3.0
    )

    C11 = 14.2 + 386.0 / (1.0 + 69.9 * q ** 1.08)
    T0t11 = xp.log(xp.e + 1.8 * q) / (xp.log(xp.e + 1.8 * q) + C11 * q * q)
    Tb = (
        T0t11 / (1.0 + (kh * s / 5.2) ** 2)
        + ab / (1.0 + (bb / kh / s) ** 3) * xp.exp(-((kh / ksilk) ** 1.4))
    ) * xp.sin(kh * st) / (kh * st)

    return ombom0 * Tb + omc / Om0 * Tc


# ---------------------------------------------------------------------------
# sigma(R, z=0): RMS of mass within top-hat radius R, normalised to sigma8.
# ---------------------------------------------------------------------------


def _tophat_window(x, xp=np):
    """W(x) = 3 (sin x - x cos x) / x^3, with a safe small-x expansion."""
    small = x < 1.0e-3
    x2 = x * x
    safe_small = 1.0 - x2 / 10.0 + x2 * x2 / 280.0
    safe_large = 3.0 * (xp.sin(x) - x * xp.cos(x)) / xp.where(small, 1.0, x ** 3)
    return xp.where(small, safe_small, safe_large)


def _sigma2_unnormalised(
    R, h, Om0, Ob0, Tcmb0, ns,
    xp=np,
    k_log_min=-5.0, k_log_max=3.0, nk=256,
):
    """sigma^2(R) at z=0 for an unnormalised power spectrum P(k) = k^ns T(k)^2."""
    ln_k = xp.linspace(
        k_log_min * xp.log(xp.asarray(10.0)),
        k_log_max * xp.log(xp.asarray(10.0)),
        nk,
    )
    k = xp.exp(ln_k)

    Tk = transfer_eh98(k, h, Om0, Ob0, Tcmb0, xp=xp)
    Pk_unnorm = k ** ns * Tk ** 2

    R = xp.atleast_1d(R)
    kR = k[None, :] * R[:, None]
    W = _tophat_window(kR, xp=xp)

    integrand = k[None, :] ** 3 * Pk_unnorm[None, :] * W ** 2
    integrand = integrand / (2.0 * xp.pi ** 2)

    sigma2 = _trapezoid_last_axis(integrand, ln_k, xp=xp)
    if sigma2.shape == (1,):
        return sigma2[0]
    return sigma2


def sigma_R(
    R, h, Om0, Ob0, Tcmb0, sigma8, ns,
    xp=np,
    k_log_min=-5.0, k_log_max=3.0, nk=256,
):
    """sigma(R, z=0), normalised so sigma(R=8 Mpc/h) = sigma8."""
    sigma2_unnorm = _sigma2_unnormalised(
        R, h, Om0, Ob0, Tcmb0, ns,
        xp=xp, k_log_min=k_log_min, k_log_max=k_log_max, nk=nk,
    )
    sigma2_8_unnorm = _sigma2_unnormalised(
        xp.asarray(8.0), h, Om0, Ob0, Tcmb0, ns,
        xp=xp, k_log_min=k_log_min, k_log_max=k_log_max, nk=nk,
    )
    norm = sigma8 ** 2 / sigma2_8_unnorm
    return xp.sqrt(norm * sigma2_unnorm)


# ---------------------------------------------------------------------------
# Linear growth factor D(z), flat LCDM (no relspecies).
# Integral form (Eisenstein & Hu 1999 Eq. 8 / Heath 1977).
# ---------------------------------------------------------------------------


def _E_lcdm(z, Om0, Ode0, xp=np):
    return xp.sqrt(Om0 * (1.0 + z) ** 3 + Ode0)


def _growth_unnormalised(z, Om0, Ode0, xp=np, nz=256, z_max=1.0e4):
    """D_+(z) un-normalised. Integrate (1+z') / E(z')^3 from z to z_max via u=ln(1+z')."""
    z_arr = xp.atleast_1d(z).astype(xp.float64)

    u_max = xp.log(xp.asarray(1.0 + z_max))
    u_low = xp.log(1.0 + z_arr)
    u_grid = (
        u_low[:, None]
        + (u_max - u_low)[:, None] * xp.linspace(0.0, 1.0, nz)[None, :]
    )
    zp = xp.exp(u_grid) - 1.0
    Ep = _E_lcdm(zp, Om0, Ode0, xp=xp)
    integrand = (1.0 + zp) ** 2 / Ep ** 3
    integral = _trapezoid_last_axis(integrand, u_grid, xp=xp)

    D = _E_lcdm(z_arr, Om0, Ode0, xp=xp) * integral
    if z_arr.shape == ():
        return D[0]
    return D


def growth_factor(z, Om0, Ode0, xp=np, nz=256, z_max=1.0e4):
    """D(z) / D(0), normalised growth factor for flat LCDM."""
    D_z = _growth_unnormalised(z, Om0, Ode0, xp=xp, nz=nz, z_max=z_max)
    D_0 = _growth_unnormalised(xp.asarray(0.0), Om0, Ode0, xp=xp, nz=nz, z_max=z_max)
    return D_z / D_0


# ---------------------------------------------------------------------------
# Einasto enclosed-mass ratio. For alpha = 0.18 (the value colossus uses
# internally in modelLudlow16), M(<r_s) / M(<c r_s) = P(3/alpha, 2/alpha) /
# P(3/alpha, (2/alpha) c^alpha), where P is the regularised lower incomplete
# gamma function. Independent of cosmology and halo mass.
# ---------------------------------------------------------------------------


_EINASTO_ALPHA = 0.18


def einasto_mass_ratio(c, xp=np, alpha=_EINASTO_ALPHA):
    """M(<r_s) / M(<c r_s) for an Einasto profile, dimensionless."""
    s = 3.0 / alpha
    x_inner = 2.0 / alpha
    x_outer = 2.0 / alpha * c ** alpha
    return _gammainc(s, x_inner, xp=xp) / _gammainc(s, x_outer, xp=xp)


# ---------------------------------------------------------------------------
# Concentration solver — vectorised port of modelLudlow16.
# ---------------------------------------------------------------------------


_C_LUDLOW = 650.0
_F_LUDLOW = 0.02
_DELTA_COLLAPSE = 1.68647019984  # matches colossus.utils.constants.DELTA_COLLAPSE


def _lagrangian_R(M, Om0, h, xp=np):
    """Lagrangian radius for mass M (Msun/h) → R (Mpc/h)."""
    # Critical density today: 2.77536627e11 Msun h^2 / Mpc^3.
    rho_crit_0 = 2.77536627e11
    rho_m_0 = Om0 * rho_crit_0
    return (3.0 * M / (4.0 * xp.pi * rho_m_0)) ** (1.0 / 3.0)


def ludlow16_concentration(
    M200c_Msun_per_h,
    z,
    h,
    Om0,
    Ob0,
    Tcmb0,
    sigma8,
    ns,
    xp=np,
    Ode0=None,
    c_array_size=200,
    sigma_nk=256,
    growth_nz=256,
):
    """
    JAX-native port of ``colossus.halo.concentration.modelLudlow16``.

    Assumes flat LCDM (``Ode0 = 1 - Om0`` if not supplied) and ignores
    relativistic species, matching the analytic LCDM branch in colossus.

    Parameters
    ----------
    M200c_Msun_per_h : float or scalar xp array
        Halo mass in Msun/h.
    z : float or scalar xp array
        Redshift.
    h, Om0, Ob0, Tcmb0, sigma8, ns : float
        Cosmology parameters. See ``PLANCK15_COSMOLOGY`` for the values
        matching colossus' built-in ``planck15`` preset.
    xp : module
        Numerical backend — ``numpy`` or ``jax.numpy``.

    Returns
    -------
    c200c : scalar xp array
    """
    if Ode0 is None:
        Ode0 = 1.0 - Om0

    M = xp.asarray(M200c_Msun_per_h, dtype=xp.float64)
    z = xp.asarray(z, dtype=xp.float64)

    c_array = xp.logspace(0.0, 2.0, c_array_size)

    M_ratio = einasto_mass_ratio(c_array, xp=xp)
    rho_f_rho_c = 200.0 * c_array ** 3 * M_ratio / _C_LUDLOW

    # Formation redshift (closed-form LCDM); entries with t1 <= 0 are invalid
    # (low-c, where the formation redshift becomes < -1) and are masked below.
    t1 = (rho_f_rho_c * (Om0 * (1.0 + z) ** 3 + Ode0) - Ode0) / Om0
    valid_c = t1 > 0.0
    t1_safe = xp.where(valid_c, t1, 1.0)
    zf = t1_safe ** (1.0 / 3.0) - 1.0

    R_fM = _lagrangian_R(_F_LUDLOW * M, Om0, h, xp=xp)
    R_M = _lagrangian_R(M, Om0, h, xp=xp)

    sigma_fM = sigma_R(R_fM, h, Om0, Ob0, Tcmb0, sigma8, ns, xp=xp, nk=sigma_nk)
    sigma_M = sigma_R(R_M, h, Om0, Ob0, Tcmb0, sigma8, ns, xp=xp, nk=sigma_nk)
    sigma2_fM = sigma_fM ** 2
    sigma2_M = sigma_M ** 2

    D_z = growth_factor(z, Om0, Ode0, xp=xp, nz=growth_nz)
    delta_z = _DELTA_COLLAPSE / D_z
    D_zf = growth_factor(zf, Om0, Ode0, xp=xp, nz=growth_nz)
    delta_zf = _DELTA_COLLAPSE / D_zf

    arg = (delta_zf - delta_z) / xp.sqrt(2.0 * (sigma2_fM - sigma2_M))
    rhs = _erfc(arg, xp=xp)

    # Solve M_ratio - rhs == 0 along c. Colossus trims c_array to entries
    # with t1 > 0 then np.interp; the un-trimmed array must remain monotonic
    # increasing in lhs_rhs for xp.interp. Pin invalid entries (low c) below
    # the lowest valid lhs_rhs (∈ [-1, 1]) so they sit at the bottom of xp.
    lhs_rhs = M_ratio - rhs
    lhs_rhs = xp.where(valid_c, lhs_rhs, -10.0)

    return xp.interp(0.0, lhs_rhs, c_array)
