"""
Approach A prototype: a fully JAX-native port of the Ludlow et al. 2016
mass-concentration relation, replacing the `colossus.halo.concentration.modelLudlow16`
call currently wrapped by `jax.pure_callback` in
`autogalaxy/profiles/mass/dark/mcr_util.py`.

Algorithm structure follows colossus' `modelLudlow16` (concentration.py, lines
1104-1192) and `modelEisenstein98` (power_spectrum.py, lines 476-608) exactly.
Re-derivations inline below where the JAX form differs from the numpy original.

Inputs/outputs match colossus:
    M200c is in Msun / h
    R is in Mpc / h
    z is dimensionless redshift
    Cosmology parameters: h, Om0, Ob0, Tcmb0, sigma8, ns
"""

import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, erfc


# ---------------------------------------------------------------------------
# Eisenstein & Hu 1998 transfer function
# Direct port of colossus.cosmology.power_spectrum.modelEisenstein98
# ---------------------------------------------------------------------------


def transfer_eh98(k, h, Om0, Ob0, Tcmb0):
    """
    EH98 transfer function T(k) including baryon acoustic features.

    Parameters
    ----------
    k : array_like
        Wavenumber in comoving h/Mpc.
    h, Om0, Ob0 : floats
        Cosmology (Hubble parameter / 100; matter, baryon densities).
    Tcmb0 : float
        CMB temperature today in Kelvin.

    Returns
    -------
    Tk : jnp.ndarray
        Transfer function value(s); same shape as k.
    """
    omc = Om0 - Ob0
    ombom0 = Ob0 / Om0
    h2 = h ** 2
    om0h2 = Om0 * h2
    ombh2 = Ob0 * h2
    theta2p7 = Tcmb0 / 2.7
    theta2p72 = theta2p7 ** 2
    theta2p74 = theta2p72 ** 2

    # k in colossus is comoving h/Mpc; internal kh is in 1/Mpc.
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
        * jnp.sqrt(6.0 / Req)
        * jnp.log(
            (jnp.sqrt(1.0 + Rd) + jnp.sqrt(Rd + Req)) / (1.0 + jnp.sqrt(Req))
        )
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
        -6.0 * jnp.sqrt(1.0 + y)
        + (2.0 + 3.0 * y)
        * jnp.log((jnp.sqrt(1.0 + y) + 1.0) / (jnp.sqrt(1.0 + y) - 1.0))
    )

    ab = 2.07 * keq * s * (1.0 + Rd) ** (-3.0 / 4.0) * Gy

    f = 1.0 / (1.0 + (kh * s / 5.4) ** 4)

    C = 14.2 / ac + 386.0 / (1.0 + 69.9 * q ** 1.08)
    T0t = jnp.log(jnp.e + 1.8 * bc * q) / (
        jnp.log(jnp.e + 1.8 * bc * q) + C * q * q
    )

    C1bc = 14.2 + 386.0 / (1.0 + 69.9 * q ** 1.08)
    T0t1bc = jnp.log(jnp.e + 1.8 * bc * q) / (
        jnp.log(jnp.e + 1.8 * bc * q) + C1bc * q * q
    )
    Tc = f * T0t1bc + (1.0 - f) * T0t

    bb = (
        0.5
        + ombom0
        + (3.0 - 2.0 * ombom0)
        * jnp.sqrt((17.2 * om0h2) * (17.2 * om0h2) + 1.0)
    )

    bnode = 8.41 * om0h2 ** 0.435

    st = s / (1.0 + (bnode / kh / s) * (bnode / kh / s) * (bnode / kh / s)) ** (
        1.0 / 3.0
    )

    C11 = 14.2 + 386.0 / (1.0 + 69.9 * q ** 1.08)
    T0t11 = jnp.log(jnp.e + 1.8 * q) / (jnp.log(jnp.e + 1.8 * q) + C11 * q * q)
    Tb = (
        T0t11 / (1.0 + (kh * s / 5.2) ** 2)
        + ab / (1.0 + (bb / kh / s) ** 3) * jnp.exp(-((kh / ksilk) ** 1.4))
    ) * jnp.sin(kh * st) / (kh * st)

    return ombom0 * Tb + omc / Om0 * Tc


# ---------------------------------------------------------------------------
# sigma(R, z=0): RMS of mass within top-hat radius R, normalised to sigma8
# ---------------------------------------------------------------------------


def _tophat_window(x):
    """W(x) = 3 (sin x - x cos x) / x^3, with a safe small-x expansion."""
    small = x < 1.0e-3
    # Taylor expansion to avoid 0/0 at x=0: W(x) ≈ 1 - x²/10 + x⁴/280
    x2 = x * x
    safe_small = 1.0 - x2 / 10.0 + x2 * x2 / 280.0
    safe_large = 3.0 * (jnp.sin(x) - x * jnp.cos(x)) / jnp.where(small, 1.0, x ** 3)
    return jnp.where(small, safe_small, safe_large)


def _sigma2_unnormalised(R, h, Om0, Ob0, Tcmb0, ns,
                         k_log_min=-5.0, k_log_max=3.0, nk=256):
    """
    sigma^2(R) at z=0 for an *unnormalised* power spectrum P(k) = k^ns T(k)^2.

    The actual sigma^2 is then this value times the normalisation A fixed by
    sigma(R=8) = sigma8.

    Uses fixed log-k Gauss quadrature (trapezoidal in log-k) over a wide range.
    """
    ln_k = jnp.linspace(k_log_min * jnp.log(10.0),
                        k_log_max * jnp.log(10.0), nk)
    k = jnp.exp(ln_k)  # h/Mpc

    Tk = transfer_eh98(k, h, Om0, Ob0, Tcmb0)
    Pk_unnorm = k ** ns * Tk ** 2  # power spectrum modulo normalisation

    # Broadcast: R is scalar or 1D; k is 1D
    R = jnp.atleast_1d(R)
    kR = k[None, :] * R[:, None]
    W = _tophat_window(kR)

    # integrand in log-k:  k^3 P(k) W^2(kR) / (2 pi^2)
    integrand = k[None, :] ** 3 * Pk_unnorm[None, :] * W ** 2
    integrand = integrand / (2.0 * jnp.pi ** 2)

    sigma2 = jnp.trapezoid(integrand, ln_k, axis=-1)
    if sigma2.shape == (1,):
        return sigma2[0]
    return sigma2


def sigma_R(R, h, Om0, Ob0, Tcmb0, sigma8, ns,
            k_log_min=-5.0, k_log_max=3.0, nk=256):
    """sigma(R, z=0), normalised so sigma(R=8 Mpc/h) = sigma8."""
    sigma2_unnorm = _sigma2_unnormalised(
        R, h, Om0, Ob0, Tcmb0, ns,
        k_log_min=k_log_min, k_log_max=k_log_max, nk=nk,
    )
    sigma2_8_unnorm = _sigma2_unnormalised(
        jnp.array(8.0), h, Om0, Ob0, Tcmb0, ns,
        k_log_min=k_log_min, k_log_max=k_log_max, nk=nk,
    )
    norm = sigma8 ** 2 / sigma2_8_unnorm
    return jnp.sqrt(norm * sigma2_unnorm)


# ---------------------------------------------------------------------------
# Linear growth factor D(z), flat LCDM (no relspecies)
# Integral form (Eisenstein & Hu 1999 Eq. 8 / Heath 1977), JAX-jittable.
# ---------------------------------------------------------------------------


def _E_lcdm(z, Om0, Ode0):
    return jnp.sqrt(Om0 * (1.0 + z) ** 3 + Ode0)


def _growth_unnormalised(z, Om0, Ode0, nz=512, z_max=1.0e4):
    """
    D_+(z) un-normalised, computed as

        D_+(z) ∝ H(z)/H0 * integral from z to ∞ of (1+z') / E(z')^3 dz'

    We integrate via change of variable u = ln(1+z') from z to ln(1+z_max).
    """
    z_arr = jnp.atleast_1d(z).astype(jnp.float64)

    # Fixed grid in u = ln(1+z'), high upper limit so the integral converges
    u_max = jnp.log(1.0 + z_max)
    # Build a per-z grid: u goes from ln(1+z_i) to u_max in nz steps.
    u_low = jnp.log(1.0 + z_arr)  # shape (M,)
    u_grid = u_low[:, None] + (u_max - u_low)[:, None] * jnp.linspace(0.0, 1.0, nz)[None, :]
    zp = jnp.exp(u_grid) - 1.0
    Ep = _E_lcdm(zp, Om0, Ode0)
    integrand = (1.0 + zp) ** 2 / Ep ** 3  # extra (1+zp) from du = dz'/(1+z')
    integral = jnp.trapezoid(integrand, u_grid, axis=-1)

    D = _E_lcdm(z_arr, Om0, Ode0) * integral
    if z_arr.shape == ():
        return D[0]
    return D


def growth_factor(z, Om0, Ode0, nz=512, z_max=1.0e4):
    """D(z) / D(0), normalised growth factor for flat LCDM."""
    D_z = _growth_unnormalised(z, Om0, Ode0, nz=nz, z_max=z_max)
    D_0 = _growth_unnormalised(jnp.array(0.0), Om0, Ode0, nz=nz, z_max=z_max)
    return D_z / D_0


# ---------------------------------------------------------------------------
# Einasto enclosed mass ratio (alpha = 0.18, dimensionless)
#
# For an Einasto profile ρ(r) = ρ_s exp(-2/α ((r/r_s)^α - 1)),
#
#     M(<r) ∝ γ_lower(3/α, 2/α (r/r_s)^α)        [unnormalised]
#
# Colossus calls this with c=1 (so r_s == R_200c of the reference halo) and
# evaluates M(<r_s) / M(<r_s * c_array). The ratio simplifies to
#
#     M_ratio(c) = γ_lower(3/α, 2/α) / γ_lower(3/α, 2/α * c^α)
#                = P(3/α, 2/α)        / P(3/α, 2/α * c^α)
#
# where P(s,x) = γ_lower(s,x)/Γ(s) is the regularised lower incomplete gamma
# function. jax.scipy.special.gammainc(a, x) is exactly P(a, x).
# ---------------------------------------------------------------------------


_EINASTO_ALPHA = 0.18


def einasto_mass_ratio(c, alpha=_EINASTO_ALPHA):
    """M(<r_s) / M(<c*r_s) for an Einasto profile, dimensionless."""
    s = 3.0 / alpha
    x_inner = 2.0 / alpha
    x_outer = 2.0 / alpha * c ** alpha
    return gammainc(s, x_inner) / gammainc(s, x_outer)


# ---------------------------------------------------------------------------
# Concentration solver (vectorised port of modelLudlow16)
# ---------------------------------------------------------------------------


# Hard-coded constants from Ludlow+16
_C_LUDLOW = 650.0
_F_LUDLOW = 0.02
_DELTA_COLLAPSE = 1.68647019984  # colossus constants.DELTA_COLLAPSE


def _lagrangian_R(M, Om0, h):
    """
    Lagrangian radius of a halo of mass M.

    M in Msun/h, returns R in Mpc/h.

    R = (3 M / (4 π ρ_m,0))^(1/3) with ρ_m,0 in Msun*h^2 / Mpc^3.
    """
    # Critical density today: rho_crit = 2.77536627e11 Msun h^2 / Mpc^3
    # (= 3 H0^2 / (8 pi G) with H0 in km/s/Mpc and standard G).
    rho_crit_0 = 2.77536627e11
    rho_m_0 = Om0 * rho_crit_0
    return (3.0 * M / (4.0 * jnp.pi * rho_m_0)) ** (1.0 / 3.0)


def _interp_zero(c_array, lhs_rhs_array):
    """
    Find c where lhs - rhs crosses zero, using linear interpolation.

    Mirrors `np.interp(0.0, lhs_rhs, c_array)` from colossus, which requires
    lhs_rhs to be monotonically increasing along c.
    """
    return jnp.interp(0.0, lhs_rhs_array, c_array)


def ludlow16_concentration_jax(
    M200c_Msun_per_h,
    z,
    h,
    Om0,
    Ob0,
    Tcmb0,
    sigma8,
    ns,
    Ode0=None,
    c_array_size=200,
    sigma_nk=256,
    growth_nz=256,
):
    """
    JAX-native port of `colossus.halo.concentration.modelLudlow16`.

    Returns c200c for a single (M200c, z) pair. Assumes flat LCDM
    (Ode0 = 1 - Om0 if not supplied) and ignores relativistic species,
    matching the analytic LCDM branch in colossus.

    Parameters
    ----------
    M200c_Msun_per_h : float or scalar jax array
        Halo mass in Msun/h.
    z : float
        Redshift.
    h, Om0, Ob0, Tcmb0, sigma8, ns : float
        Cosmology parameters.
    Ode0 : float, optional
        Dark energy density today. Defaults to 1 - Om0 (flat universe).

    Returns
    -------
    c200c : jnp.ndarray (scalar)
    """
    if Ode0 is None:
        Ode0 = 1.0 - Om0

    M = jnp.asarray(M200c_Msun_per_h, dtype=jnp.float64)
    z = jnp.asarray(z, dtype=jnp.float64)

    # Brute-force c grid (colossus uses np.logspace(0, 2, 200))
    c_array = jnp.logspace(0.0, 2.0, c_array_size)

    # M_ratio(c) — depends only on c, alpha (fixed=0.18). Independent of cosmo.
    M_ratio = einasto_mass_ratio(c_array)

    rho_f_rho_c = 200.0 * c_array ** 3 * M_ratio / _C_LUDLOW

    # Formation redshift zf (closed-form LCDM expression from colossus).
    t1 = (rho_f_rho_c * (Om0 * (1.0 + z) ** 3 + Ode0) - Ode0) / Om0
    # Mask out c values where t1 <= 0 (no valid zf). For these we will
    # let zf = 0 and rely on the lhs_rhs sign flip to keep them outside
    # the interpolation root.
    valid_c = t1 > 0.0
    t1_safe = jnp.where(valid_c, t1, 1.0)
    zf = t1_safe ** (1.0 / 3.0) - 1.0

    # sigma^2 at f*M and at M (z=0 normalisation, growth factor applied later)
    R_fM = _lagrangian_R(_F_LUDLOW * M, Om0, h)
    R_M = _lagrangian_R(M, Om0, h)

    sigma_fM = sigma_R(
        R_fM, h, Om0, Ob0, Tcmb0, sigma8, ns, nk=sigma_nk
    )
    sigma_M = sigma_R(
        R_M, h, Om0, Ob0, Tcmb0, sigma8, ns, nk=sigma_nk
    )
    sigma2_fM = sigma_fM ** 2
    sigma2_M = sigma_M ** 2

    # delta_z, delta_zf
    D_z = growth_factor(z, Om0, Ode0, nz=growth_nz)
    delta_z = _DELTA_COLLAPSE / D_z
    D_zf = growth_factor(zf, Om0, Ode0, nz=growth_nz)
    delta_zf = _DELTA_COLLAPSE / D_zf

    # right-hand side of Ludlow16 Eq. 7
    arg = (delta_zf - delta_z) / jnp.sqrt(
        2.0 * (sigma2_fM - sigma2_M)
    )
    rhs = erfc(arg)

    # Solve M_ratio - rhs == 0 along c. colossus trims c_array to entries with
    # t1 > 0 then calls np.interp; the un-trimmed array must remain monotonic
    # increasing in lhs_rhs for jnp.interp. Pin invalid entries (low c) below
    # the lowest valid lhs_rhs so they sit at the bottom of the monotonic xp.
    # Valid lhs_rhs ∈ [-1, 1] (both M_ratio, rhs ∈ [0, 1]); -10 is safely below.
    lhs_rhs = M_ratio - rhs
    lhs_rhs = jnp.where(valid_c, lhs_rhs, -10.0)

    c200c = _interp_zero(c_array, lhs_rhs)
    return c200c
