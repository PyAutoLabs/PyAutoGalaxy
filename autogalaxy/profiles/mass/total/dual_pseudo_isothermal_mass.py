from typing import Tuple
import numpy as np

import autoarray as aa
from autogalaxy import convert
from autogalaxy.cosmology.model import LensingCosmology
from autogalaxy.profiles.mass.abstract.abstract import MassProfile


def _b0_from_lenstool_sigma(
    sigma: float,
    redshift_object: float,
    redshift_source: float,
    cosmology: LensingCosmology,
) -> float:
    """
    Convert Lenstool's fiducial velocity dispersion ``v_disp`` (sigma_LT, in km/s) to the
    dPIE lens strength ``b0`` in arcseconds.

    Lenstool stores ``b0 = 6 * pia_c2 * sigma_LT^2`` with ``pia_c2 = 648000 / c^2``
    (``constant.h``; c in km/s) and applies the distance ratio D_LS / D_S separately when
    computing deflections (``e_grad.c``). PyAutoGalaxy's ``b0`` is fully normalized, so the
    ratio is folded in here:

        b0 [arcsec] = 6 * 648000 * (sigma_LT / c)^2 * (D_LS / D_S)

    This is identical to ``4 * pi * (sigma_0 / c)^2 * D_LS / D_S`` (in radians) for the
    central velocity dispersion ``sigma_0 = sqrt(3/2) * sigma_LT`` — the fiducial-vs-central
    distinction of Eliasdottir et al. (2007) App. A / Bergamini et al. (2019).
    """
    c_km_s = 299792.458

    d_s = cosmology.angular_diameter_distance_to_earth_in_kpc_from(
        redshift=redshift_source
    )
    d_ls = cosmology.angular_diameter_distance_between_redshifts_in_kpc_from(
        redshift_0=redshift_object, redshift_1=redshift_source
    )

    return 6.0 * 648000.0 * (sigma / c_km_s) ** 2 * (d_ls / d_s)


# Within this profile family, PIEMass, dPIEMassB0, and dPIEMassB0Sph are directly ported from Lenstool's C code, and have been thoroughly annotated and adapted for PyAutoLens.
# dPIEMass and dPIEMassSph (the default profiles) expose the same physics in Lenstool's native parameterization.
# The dPIEPotential and dPIEPotentialSph profiles are modified from the original `dPIEPotential` and `dPIEPotentialSph`, which were implemented to PyAutoLens by Jackson O'Donnell.


def _ci05(x, y, eps, rcore, xp=np):
    """
    Returns the first derivatives of the lens potential as complex number I'* = (∂ψ/∂x + i ∂ψ/∂y) / E0 for PIEMass at any positions (x,y),
    see Kassiola & Kovner(1993) Eq. 4.1.2, which is the integral of Eq. 2.3.8.
    Note here b0(or called E0) is out of the `_ci05`.

    Parameters
    ----------
    eps
        The ellipticity of the corresponding profiles.
    rcore
        The inner core radius.
    Returns
    -------
    complex
        The value of the I'* term.
    """
    sqe = xp.sqrt(eps)
    axis_ratio = (1.0 - eps) / (1.0 + eps)
    cxro = (1.0 + eps) * (1.0 + eps)
    cyro = (1.0 - eps) * (1.0 - eps)
    rem2 = x * x / cxro + y * y / cyro
    ##### I'* = zres = zci * ln(zis) = zci * ln(znum / zden), see Eq. 4.1.2 #####

    # Define intermediate complex variables
    zci = xp.array(0.0 + 1j * (-0.5 * (1.0 - eps * eps) / sqe), dtype=xp.complex128)
    znum = xp.complex128(
        axis_ratio * x
        + 1j * (2.0 * sqe * xp.sqrt(rcore * rcore + rem2) - y / axis_ratio)
    )
    zden = xp.complex128(x + 1j * (2.0 * rcore * sqe - y))

    # zis = znum / zden = (a+bi)/(c+di) = [(ac+bd)+(bc-ad i)] / (c^2+d^2)
    norm = zden.real * zden.real + zden.imag * zden.imag  # |zden|^2
    zis_re = (znum.real * zden.real + znum.imag * zden.imag) / norm
    zis_im = (znum.imag * zden.real - znum.real * zden.imag) / norm
    zis = xp.complex128(zis_re + 1j * zis_im)

    # ln(zis) = ln(|zis|) + i*Arg(zis)
    zis_mag = xp.abs(zis)
    zis_re = xp.log(zis_mag)
    zis_im = xp.angle(zis)
    zis = xp.complex128(zis_re + 1j * zis_im)

    # I'* = zres = zci * ln(zis)
    zres = zci * zis

    return zres


def _ci05f(x, y, eps, rcore, rcut, xp=np):
    """
    Returns the first derivatives of the lens potential as complex number I'* = (∂ψ/∂x + i ∂ψ/∂y) / (b0 * ra / (rs - ra)) for dPIEMass at any positions (x,y),
    which is the integral of Eq. 2.3.8 in  Kassiola & Kovner(1993).

    Note here (b0 * ra / (rs - ra)) is out of the `_ci05f`. The only difference of integral of Eq. 2.3.8 between dPIEMass and PIEMass is the \\kappa:
    \\kappa(r_{em})_{dPIEMass} = rs / (rs - ra) * (\\kappa_{PIEMass,ra} - \\kappa_{PIEMass,rs}).
    I*_{dPIEMass} = ra / (rs - ra) * (I*_{PIEMass}(ra) - I*_{PIEMass}(ra))

    Parameters
    ----------
    eps
        The ellipticity of the corresponding profiles.
    rcore
        The inner core radius.
    rcut
        The outer cut radius.
    Returns
    -------
    complex
        The value of the I'* term.
    """
    sqe = xp.sqrt(eps)
    axis_ratio = (1.0 - eps) / (1.0 + eps)
    cxro = (1.0 + eps) * (1.0 + eps)
    cyro = (1.0 - eps) * (1.0 - eps)
    rem2 = x * x / cxro + y * y / cyro

    ##### I'* = zres_rc - zres_rcut = zci * ln(zis_rc / zis_rcut) = zci * ln((znum_rc / zden_rc) / (znum_rcut / zden_rcut)) #####

    # Define intermediate complex variables
    zci = xp.array(0.0 + 1j * (-0.5 * (1.0 - eps * eps) / sqe), dtype=xp.complex128)
    znum_rc = xp.complex128(
        axis_ratio * x
        + 1j * (2.0 * sqe * xp.sqrt(rcore * rcore + rem2) - y / axis_ratio)
    )  # a + bi
    zden_rc = xp.complex128(x + 1j * (2.0 * rcore * sqe - y))  # c + di
    znum_rcut = xp.complex128(
        axis_ratio * x + 1j * (2.0 * sqe * xp.sqrt(rcut * rcut + rem2) - y / axis_ratio)
    )  # a + ei
    zden_rcut = xp.complex128(x + 1j * (2.0 * rcut * sqe - y))  # c + fi

    # zis_rc = znum_rc / zden_rc = (a+bi)/(c+di)
    # zis_rcut = znum_rcut / zden_rcut = (a+ei)/(c+fi)
    # zis_tot = zis_rc / zis_rcut = (znum_rc / zden_rc) / (znum_rcut / zden_rcut)
    #                             = [(ac - bf) + (af + bc)i] / [(ac - de) + (ad + ce)i]
    #                             = (aa + bb*i) / (cc + dd*i)
    #                             = (aa + bb*i) * (cc -dd*i) / (cc^2 + dd^2)
    #                             = [(aa*cc + bb*dd) / (cc^ + dd^2)] + [(bb*cc - aa*dd) / (cc^2 + dd^2)]*i
    #                             =                 aaa              +                 bbb*i
    aa = znum_rc.real * zden_rc.real - znum_rc.imag * zden_rcut.imag  # ac - bf
    bb = znum_rc.real * zden_rcut.imag + znum_rc.imag * zden_rc.real  # af + bc
    cc = znum_rc.real * zden_rc.real - zden_rc.imag * znum_rcut.imag  # ac - de
    dd = znum_rc.real * zden_rc.imag + zden_rc.real * znum_rcut.imag  # ad + ce
    norm = cc * cc + dd * dd
    aaa = (aa * cc + bb * dd) / norm
    bbb = (bb * cc - aa * dd) / norm
    zis_tot = xp.complex128(aaa + 1j * bbb)

    # ln(zis_tot) = ln(|zis_tot|) + i*Arg(zis_tot)
    zis_tot_mag = xp.abs(zis_tot)
    zr_re = xp.log(zis_tot_mag)
    zr_im = xp.angle(zis_tot)
    zr = xp.complex128(zr_re + 1j * zr_im)

    # I'* = zci * ln(zis_tot)
    zres = zci * zr

    return zres


def _mdci05(x, y, eps, rcore, b0, xp=np):
    """
    Returns the second derivatives (Hessian matrix) of the lens potential as complex number for PIEMass at any positions (x,y):
    ∂²ψ/∂x² = Re(∂I*/∂x), ∂²ψ/∂y² = Im(∂I*/∂y), ∂²ψ/∂x∂y = ∂²ψ/∂y∂x = Im(∂I*/∂x) = Re(∂I*/∂y)
    see Kassiola & Kovner(1993) Eq. 4.1.4.

    Parameters
    ----------
    eps
        The ellipticity of the corresponding profiles.
    rcore
        The inner core radius.
    Returns
    -------
    complex
        The value of the I'* term.
    """

    # Calculate intermediate values
    # I*(x,y) = b0 * ci * (-i) * (ln{ q * x + (2.0 * sqe * wrem - y * 1/q )*i} - ln{ x + (2.0 * rcore * sqe - y)*i})
    #         = b0 * ci * (-i) * (ln{ q * x + num1*i} - ln{ x + num2*i})
    #         = b0 * ci * (-i) * (ln{u(x,y)} - ln{v(x,y)})
    sqe = xp.sqrt(eps)
    axis_ratio = (1.0 - eps) / (1.0 + eps)
    axis_ratio_inv = 1.0 / axis_ratio
    cxro = (1.0 + eps) * (1.0 + eps)
    cyro = (1.0 - eps) * (1.0 - eps)
    ci = 0.5 * (1.0 - eps * eps) / sqe
    wrem = xp.sqrt(rcore * rcore + x * x / cxro + y * y / cyro)  # √(w(x,y))
    num1 = 2.0 * sqe * wrem - y * axis_ratio_inv
    den1 = axis_ratio * axis_ratio * x * x + num1 * num1  # |q * x + num1*i|^2
    num2 = 2.0 * rcore * sqe - y
    den2 = x * x + num2 * num2  # |x + num2*i|^2

    # eg.
    # ∂²ψ/∂x² = Re(∂I*/∂x) = b0 * didxre
    # ∂I*/∂x = b0 * ci * (-i) * ∂(ln{u(x,y)} - ln{v(x,y)})∂x
    #        = b0 * ci * (-i) * (1/u * ∂u/∂x - 1/v * ∂v/∂x)
    # ∂u/∂x = q + ∂(num1)/∂x * i
    #       = q + [2.0 * sqe * ∂(wrem)/∂x] * i
    #       = q + [2.0 * sqe * ∂(√(w(x,y)))/∂x] * i
    #       = q + [2.0 * sqe * x / cxro / wrem] * i
    # 1/u * ∂u/∂x = {q + [2.0 * sqe * x / cxro / wrem] * i}  /  {q * x + num1*i}
    #             = {q + [2.0 * sqe * x / cxro / wrem] * i} * {q * x - num1*i}  /  |q * x + num1*i|^2
    #             = {q + [2.0 * sqe * x / cxro / wrem] * i} * {q * x - (2.0 * sqe * wrem - y / q)*i}  /  den1
    #             = {q^2 * x + 4.0 * sqe^2 * x - y / q * 2.0 * sqe * x / cxro / wrem} / den1 + q * { (2.0 * sqe * x^2 / cxro / wrem) - (2.0 * sqe * wrem - y / q)} / den1 * i
    #             = {x - 2.0 * sqe * x * y * q / cyro / wrem} / den1 + q * { (2.0 * sqe * x^2 / cxro / wrem) - (2.0 * sqe * wrem - y / q)} / den1 * i
    # (-i) * (1/u * ∂u/∂x) = (2.0 * sqe * x * y * q / cyro / wrem - x) / den1 * i
    #                      + q * { (2.0 * sqe * x^2 / cxro / wrem) - (2.0 * sqe * wrem - y / q)} / den1
    # ∂v/∂x = 1 + ∂(num2)/∂x * i
    #       = 1
    # 1/v * ∂v/∂x = 1 / (x + num2*i)
    #             = (x - num2*i) / |x + num2*i|^2
    #             = (x - num2*i) / den2
    # -(-i) * (1/v * ∂v/∂x) = (x*i + num2) / den2

    # ∂I*/∂x = b0 * ci * {(-i) * (1/u * ∂u/∂x) - (-i) * (1/v * ∂v/∂x)}

    # Compute second derivatives
    didxre = ci * (
        axis_ratio
        * (2.0 * sqe * x * x / cxro / wrem - 2.0 * sqe * wrem + y * axis_ratio_inv)
        / den1
        + num2 / den2
    )
    didyre = ci * ((2.0 * sqe * x * y * axis_ratio / cyro / wrem - x) / den1 + x / den2)
    didyim = ci * (
        (
            2.0 * sqe * wrem * axis_ratio_inv
            - y * axis_ratio_inv * axis_ratio_inv
            - 4.0 * eps * y / cyro
            + 2.0 * sqe * y * y / cyro / wrem * axis_ratio_inv
        )
        / den1
        - num2 / den2
    )

    # Construct Hessian matrix components
    a = b0 * didxre  # ∂²ψ/∂x²
    b = b0 * didyre  # ∂²ψ/∂x∂y
    c = b0 * didyre  # ∂²ψ/∂y∂x
    d = b0 * didyim  # ∂²ψ/∂y²

    return a, b, c, d


def _pi05(x, y, eps, rcore, xp=np):
    """
    Returns the lensing potential psi / b0 of the single-core PIEMass (Kassiola &
    Kovner 1993 I0.5c model) at positions (x, y), ported from Lenstool's ``pi05``
    (``e_pcpx.c``). Note b0 is outside ``_pi05``, mirroring ``_ci05``.

    The dPIE potential is the two-component difference (Lenstool ``e_pot.c`` case 81):
    psi = b0 * rs / (rs - ra) * (_pi05(rcore=ra) - _pi05(rcore=rs)).

    Parameters
    ----------
    eps
        The ellipticity of the corresponding profiles.
    rcore
        The core radius of the corresponding profiles.
    """
    sqe = xp.sqrt(eps)
    ci = 0.5 * (1.0 - eps**2) / sqe
    cxro = (1.0 + eps) ** 2
    cyro = (1.0 - eps) ** 2
    rem2 = x**2 / cxro + y**2 / cyro
    e1 = 2.0 * sqe / (1.0 - eps)
    e2 = 2.0 * sqe / (1.0 + eps)
    z = xp.sqrt(x**2 + y**2)

    eta = -0.5 * xp.arcsinh(e1 * y / z) + 0.5j * xp.arcsin(e2 * x / z)
    zeta = 0.5 * xp.log((xp.sqrt(rem2) + xp.sqrt(rcore**2 + rem2)) / rcore) + 0.0j

    b1 = xp.cosh(eta + zeta)
    b2 = xp.cosh(eta - zeta)
    a1 = xp.log(xp.cosh(eta) ** 2 / (b1 * b2))
    a2 = xp.log(b1 / b2)
    c1 = xp.sinh(2.0 * eta) * a1
    c2 = xp.sinh(2.0 * zeta) * a2
    ckk = c1 + c2

    return ci * rcore / xp.sqrt(rem2) * (ckk.imag * x - ckk.real * y)


class PIEMass(MassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        b0: float = 0.1,
    ):
        """
        The Pseudo Isothermal Elliptical Mass Distribution(PIEMass) profiles, based on the formulaiton from
        Kassiola & Kovner(1993) https://articles.adsabs.harvard.edu/pdf/1993ApJ...417..450K.
        This profile is ported from Lenstool's C code, which has the same formulation.

        This proflie describes an elliptic isothermal mass distribution with a finite core:
        \\rho \\propto (ra^2 + R^2)^{-1}

        The convergence is given by:
        \\kappa(r_{em}) = \\kappa_0 * ra / \\sqrt{ ra^2 + r_{em}^2 }
        (see Kassiola & Kovner(1993), Eq. 4.1.1)
        where r_{em}^2 = x^2 / (1 + \\epsilon)^2 + y^2 / (1 - \\epsilon)^2, (see Kassiola & Kovner(1993), Eq. 2.3.6)
        and \\kappa_0 = b_0 / 2 / r_a.

        In this implementation:
        - `ra` is the core radius in unit of arcseconds.
        - `b0` is the lens strength in unit of arcseconds, when ra->0 & q->1, b0 is the Einstein radius.
          `b0` is related to the central velocity dispersion \\sigma_0: b_0 = 4\\pi * \\sigma_0^2 / c^2 * (D_{LS} / D_{S}).
          `b0` is not in the Intermediate-Axis-Convention for its r_{em}^2 = x^2 / (1 + \\epsilon)^2 + y^2 / (1 - \\epsilon)^2

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ra
            The inner core radius in arcseconds.
        b0
            The lens strength in arcseconds.
        """
        super().__init__(centre=centre, ell_comps=ell_comps)

        self.ra = ra
        self.b0 = b0

    def _ellip(self, xp=np):
        ellip = xp.sqrt(self.ell_comps[0] ** 2 + self.ell_comps[1] ** 2)
        # The ci05 deflection integral is degenerate (NaN) at exactly zero ellipticity;
        # Lenstool clamps to 1e-5 at setup for the same reason (set_lens.c).
        MIN_ELLIP = 0.00001
        MAX_ELLIP = 0.99999
        return xp.clip(ellip, MIN_ELLIP, MAX_ELLIP)

    @aa.decorators.to_vector_yx
    @aa.decorators.transform(rotate_back=True)
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        ellip = self._ellip(xp)
        factor = self.b0
        zis = _ci05(
            x=grid.array[:, 1], y=grid.array[:, 0], eps=ellip, rcore=self.ra, xp=xp
        )

        # This is in axes aligned to the major/minor axis
        deflection_x = zis.real
        deflection_y = zis.imag

        return xp.multiply(factor, xp.vstack((deflection_y, deflection_x)).T)

    def _convergence(self, radii, xp=np):

        radsq = radii * radii
        a = self.ra

        return self.b0 / 2 * (1 / xp.sqrt(a**2 + radsq))

    @aa.decorators.to_array
    @aa.decorators.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Returns the two-dimensional projected convergence on a grid of (y,x)
        arc-second coordinates.

        The `grid_2d_to_structure` decorator reshapes the ndarrays the convergence
        is outputted on. See *aa.grid_2d_to_structure* for details.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates on which the convergence is computed.
        """
        ellip = self._ellip(xp)
        grid_radii = xp.sqrt(
            grid.array[:, 1] ** 2 / (1 + ellip) ** 2
            + grid.array[:, 0] ** 2 / (1 - ellip) ** 2
        )
        # Compute the convergence and deflection of a *circular* profile
        kappa = self._convergence(grid_radii, xp)

        return kappa

    @aa.decorators.transform
    def analytical_hessian_2d_from(self, grid: "aa.type.Grid2DLike", xp=np, **kwargs):
        """
        Calculate the hessian matrix on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        if grid.ndim != 2 or grid.shape[1] != 2:
            raise ValueError("Grid must be a 2D array with shape (n, 2)")
        ellip = self._ellip()

        hessian_xx, hessian_xy, hessian_yx, hessian_yy = _mdci05(
            x=grid.array[:, 1],
            y=grid.array[:, 0],
            eps=ellip,
            rcore=self.ra,
            b0=self.b0,
            xp=xp,
        )

        return hessian_yy, hessian_xy, hessian_yx, hessian_xx

    def analytical_magnification_2d_from(
        self, grid: "aa.type.Grid2DLike", xp=np, **kwargs
    ):

        hessian_yy, hessian_xy, hessian_yx, hessian_xx = (
            self.analytical_hessian_2d_from(grid=grid, xp=np)
        )

        det_A = (1 - hessian_xx) * (1 - hessian_yy) - hessian_xy * hessian_yx

        return aa.Array2D(values=1.0 / det_A, mask=grid.mask)


class dPIEMassB0(MassProfile):
    r"""Dual pseudo-isothermal elliptical mass distribution in the internal
    (``ra``, ``rs``, ``b0``) parameterisation.

    **This is the non-standard parameterisation.** The default dPIE profile is
    :class:`dPIEMass`, whose free parameters are Lenstool's native ones
    (``ellipticity``, ``angle_pos``, ``sigma``, ``r_core``, ``r_cut``) as used
    by essentially every published cluster- and group-scale analysis. Use this
    class (or ``dPIEMass.from_b0``) only when composing a model directly on the
    lens strength ``b0``.

    A two-component PIE profile with both a core radius :math:`r_a` and a
    truncation radius :math:`r_s`.  The three-dimensional density scales as
    :math:`\rho \propto r^{-2}` in the transition region
    :math:`r_a \leq R \leq r_s` and as :math:`\rho \propto r^{-4}` in the outer
    parts, with the full form:

    .. math::

        \rho \propto \bigl[(r_a^2 + R^2)(r_s^2 + R^2)\bigr]^{-1}

    The projected convergence is the difference of two PIE profiles:

    .. math::

        \kappa(r_{\rm em}) = \frac{b_0}{2} \frac{r_s}{r_s - r_a}
        \left(
            \frac{1}{\sqrt{r_a^2 + r_{\rm em}^2}}
            - \frac{1}{\sqrt{r_s^2 + r_{\rm em}^2}}
        \right)

    where :math:`r_{\rm em}^2 = x^2/(1+\epsilon)^2 + y^2/(1-\epsilon)^2` is
    the pseudo-elliptical radius and :math:`b_0` is the lens strength (equal to
    the Einstein radius when :math:`r_a \to 0`, :math:`r_s \to \infty`, and
    :math:`q \to 1`).  This profile is ported directly from Lenstool's C code.

    Parameters
    ----------
    centre : (float, float)
        (y, x) arc-second coordinates of the profile centre.
    ell_comps : (float, float)
        Ellipticity components (e1, e2) of the elliptical coordinate system.
    ra : float
        Inner core radius in arcseconds.
    rs : float
        Outer truncation radius in arcseconds.
    b0 : float
        Lens strength in arcseconds (Einstein radius in the limit
        :math:`r_a \to 0`, :math:`r_s \to \infty`, :math:`q \to 1`).

    References
    ----------
    Kassiola & Kovner (1993), ApJ, 417, 450.
    Eliasdottir et al. (2007), arXiv:0710.5636.
    Limousin et al. (2005), A&A, 461, 881.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.0,
        rs: float = 2.0,
        b0: float = 0.1,
    ):
        """
        The dual Pseudo Isothermal Elliptical Mass Distribution(dPIEMass) profiles, which is a *two component PIEMass* with both a core radius and a truncation radius,
        see Eliasdottir (2007): https://arxiv.org/abs/0710.5636
        This profile is ported from Lenstool's C code, which has the same formulation.

        This proflie describes an elliptic isothermal mass distribution with a finite core, \\rho  r^{-2} while in
        the transition region (ra<=R<=rs),
        and \\rho r^{-4} in the outer parts:
        \\rho \\propto [(ra^2 + R^2) (rs^2 + R^2)]^{-1}

        The convergence is given by two PIEMass with core radius ra and rs:
        \\kappa(r_{em}) = rs / (rs - ra) * (\\kappa_{PIEMass,ra} - \\kappa_{PIEMass,rs})
                        = b_0 / 2 * rs / (rs - ra) * ( \\frac{1}{\\sqrt{ ra^2 + r_{em}^2}} - \\frac{1}{\\sqrt{ rs^2 + r_{em}^2}} )
        where r_{em}^2 = x^2 / (1 + \\epsilon)^2 + y^2 / (1 - \\epsilon)^2.
        Note in Eliasdottir (2007), E0 = 6\\pi * \\sigma_{dPIEPotential}^2 / c^2 * (D_{LS} / D_{S}). Eliasdottir's E0 is not the same as E0 in Kassiola & Kovner(1993) which is also b0.
        There is \\frac{\\sigma_{dPIEPotential}^2}{\\sigma_0^2} = \\frac{2}{3} \frac{rs^2}{rs^2-ra^2},
        thus E0(Kassiola & Kovner(1993)) = b0 = E0(Eliasdottir (2007)) * (rs^2 - ra^2) / rs^2. So when s->\\infty and a->0, they are equivalent.

        In this implementation:
        - `ra` is the core radius in unit of arcseconds.
        - `rs` is the truncation radius in unit of arcseconds.
        - `b0` is the lens strength in unit of arcseconds, when ra->0 & rs->\\infty & q->1, b0 is the Einstein radius.
          `b0` is related to the central velocity dispersion \\sigma_0: b_0 = 4\\pi * \\sigma_0^2 / c^2 * (D_{LS} / D_{S})
          `b0` is not in the Intermediate-Axis-Convention for its r_{em}^2 = x^2 / (1 + \\epsilon)^2 + y^2 / (1 - \\epsilon)^2

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ra
            The inner core radius in arcseconds.
        rs
            The outer truncation radius in arcseconds.
        b0
            The lens strength in arcseconds.
        """
        super().__init__(centre=centre, ell_comps=ell_comps)

        self.ra = ra
        self.rs = rs
        self.b0 = b0

    @classmethod
    def from_lenstool(
        cls,
        centre: Tuple[float, float] = (0.0, 0.0),
        ellipticity: float = 0.0,
        angle_pos: float = 0.0,
        sigma: float = 200.0,
        r_core: float = 0.1,
        r_cut: float = 20.0,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
        cosmology: LensingCosmology = None,
    ) -> "dPIEMassB0":
        """
        Construct a ``dPIEMassB0`` from Lenstool's native dPIE / PIEMD parameterization, as
        read directly out of a Lenstool ``.par`` file (``potentiel`` profil 81) or the
        parameter tables of Lenstool-based papers.

        For *model-fitting* in the Lenstool parameters use :class:`dPIEMass`, whose
        constructor takes the same inputs (with a flat H0/Om0 cosmology so priors can be
        composed); this classmethod is the general converter accepting an arbitrary
        ``cosmology`` object (e.g. ``Planck15``) and returning the internal
        parameterization.

        Three Lenstool conventions are converted (each verified against the Lenstool C
        source):

        - ``sigma`` is Lenstool's **fiducial** velocity dispersion ``v_disp`` (sigma_LT),
          *not* the central velocity dispersion sigma_0 of the dPIE profile. They differ
          by sigma_0 = sqrt(3/2) * sigma_LT (Eliasdottir et al. 2007, App. A; Bergamini
          et al. 2019, Eq. 5) — quoting a measured central/aperture dispersion here
          overestimates the mass by 50%. The lens strength is
          b0 = 6 * 648000 * (sigma_LT / c)^2 * (D_LS / D_S) arcsec, where Lenstool's
          stored ``b0 = 6 * pia_c2 * sigma^2`` (``set_potfile.c``) carries no distance
          ratio — Lenstool applies D_LS / D_S separately at deflection time
          (``e_grad.c``), whereas PyAutoGalaxy's ``b0`` is fully normalized.
        - ``ellipticity`` is Lenstool's ``ellipticite`` for mass-type profiles,
          emass = (a^2 - b^2) / (a^2 + b^2). Lenstool converts it internally
          (``set_lens.c``) to epot = (1 - q) / (1 + q) before evaluating deflections;
          that epot is exactly the magnitude of PyAutoGalaxy's ``ell_comps``, so the
          conversion here is emass -> q = sqrt((1 - e) / (1 + e)) -> ``ell_comps``.
        - ``r_core`` / ``r_cut`` (Lenstool ``core_radius`` / ``cut_radius``, arcsec) map
          one-to-one onto ``ra`` / ``rs``. For ``.par`` files using the kpc variants
          (``core_radius_kpc`` / ``cut_radius_kpc``), pre-convert with
          ``r_core = r_core_kpc / cosmology.kpc_per_arcsec_from(redshift=redshift_object)``.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ellipticity
            Lenstool mass ellipticity, (a^2 - b^2) / (a^2 + b^2).
        angle_pos
            Position angle in degrees, counter-clockwise from the positive x-axis
            (Lenstool ``angle_pos`` in its tangent plane; axis flips from WCS
            conventions must be handled when ingesting real-data catalogues).
        sigma
            Lenstool fiducial velocity dispersion ``v_disp`` (sigma_LT) in km/s.
        r_core
            Lenstool ``core_radius`` in arcseconds (becomes ``ra``).
        r_cut
            Lenstool ``cut_radius`` in arcseconds (becomes ``rs``).
        redshift_object
            The redshift of the lens, used for the D_LS / D_S normalization of ``b0``.
        redshift_source
            The redshift of the source used to normalize ``b0``. For multi-plane cluster
            models this is the reference source plane the Lenstool model was normalized
            to.
        cosmology
            The cosmology used to compute the distance ratio (default ``Planck15``; pass
            the cosmology of the Lenstool run for exact comparisons).
        """
        if cosmology is None:
            from autogalaxy.cosmology.model import Planck15

            cosmology = Planck15()

        axis_ratio = np.sqrt((1.0 - ellipticity) / (1.0 + ellipticity))
        ell_comps = convert.ell_comps_from(axis_ratio=axis_ratio, angle=angle_pos)

        b0 = _b0_from_lenstool_sigma(
            sigma=sigma,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
            cosmology=cosmology,
        )

        return cls(
            centre=centre,
            ell_comps=ell_comps,
            ra=r_core,
            rs=r_cut,
            b0=b0,
        )

    def _ellip(self, xp=np):
        ellip = xp.sqrt(self.ell_comps[0] ** 2 + self.ell_comps[1] ** 2)
        # The ci05 deflection integral is degenerate (NaN) at exactly zero ellipticity;
        # Lenstool clamps to 1e-5 at setup for the same reason (set_lens.c).
        MIN_ELLIP = 0.00001
        MAX_ELLIP = 0.99999
        return xp.clip(ellip, MIN_ELLIP, MAX_ELLIP)

    @aa.decorators.to_vector_yx
    @aa.decorators.transform(rotate_back=True)
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        ellip = self._ellip(xp)
        factor = self.b0 * self.rs / (self.rs - self.ra)
        zis = _ci05f(
            x=grid.array[:, 1],
            y=grid.array[:, 0],
            eps=ellip,
            rcore=self.ra,
            rcut=self.rs,
            xp=xp,
        )

        # This is in axes aligned to the major/minor axis
        deflection_x = zis.real
        deflection_y = zis.imag

        return xp.multiply(factor, xp.vstack((deflection_y, deflection_x)).T)

    def _convergence(self, radii, xp=np):

        radsq = radii * radii
        a, s = self.ra, self.rs

        return (
            self.b0
            / 2
            * s
            / (s - a)
            * (1 / xp.sqrt(a**2 + radsq) - 1 / xp.sqrt(s**2 + radsq))
        )

    def convergence_func(self, grid_radius, xp=np):
        return self._convergence(grid_radius, xp=xp)

    @aa.decorators.to_array
    @aa.decorators.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Returns the two dimensional projected convergence on a grid of (y,x) arc-second coordinates.

        The `grid_2d_to_structure` decorator reshapes the ndarrays the convergence is outputted on. See
        *aa.grid_2d_to_structure* for a description of the output.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.
        """
        ellip = self._ellip(xp)
        grid_radii = xp.sqrt(
            grid.array[:, 1] ** 2 / (1 + ellip) ** 2
            + grid.array[:, 0] ** 2 / (1 - ellip) ** 2
        )
        kappa = self._convergence(grid_radii, xp)
        return kappa

    @aa.decorators.transform
    def analytical_hessian_2d_from(self, grid: "aa.type.Grid2DLike", xp=np, **kwargs):
        """
        Calculate the hessian matrix on a grid of (y,x) arc-second coordinates.
        Hessian_dPIEMass = rs * (rs - ra) * ( Hessian_PIEMass(ra) - Hessian_PIEMass(rs))

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        if grid.ndim != 2 or grid.shape[1] != 2:
            raise ValueError("Grid must be a 2D array with shape (n, 2)")
        ellip = self._ellip()

        t05 = self.rs / (self.rs - self.ra)
        g05c_a, g05c_b, g05c_c, g05c_d = _mdci05(
            x=grid.array[:, 1],
            y=grid.array[:, 0],
            eps=ellip,
            rcore=self.ra,
            b0=self.b0,
            xp=xp,
        )
        g05cut_a, g05cut_b, g05cut_c, g05cut_d = _mdci05(
            x=grid.array[:, 1],
            y=grid.array[:, 0],
            eps=ellip,
            rcore=self.rs,
            b0=self.b0,
            xp=xp,
        )

        # Compute Hessian matrix components
        hessian_xx = t05 * (g05c_a - g05cut_a)
        hessian_xy = t05 * (g05c_b - g05cut_b)
        hessian_yx = t05 * (g05c_c - g05cut_c)
        hessian_yy = t05 * (g05c_d - g05cut_d)

        return hessian_yy, hessian_xy, hessian_yx, hessian_xx

    def analytical_magnification_2d_from(
        self, grid: "aa.type.Grid2DLike", xp=np, **kwargs
    ):

        hessian_yy, hessian_xy, hessian_yx, hessian_xx = (
            self.analytical_hessian_2d_from(grid=grid, xp=xp)
        )

        det_A = (1 - hessian_xx) * (1 - hessian_yy) - hessian_xy * hessian_yx

        return aa.Array2D(values=1.0 / det_A, mask=grid.mask)

    @aa.over_sample
    @aa.decorators.to_array
    @aa.decorators.transform
    @aa.decorators.to_array
    @aa.decorators.transform
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Returns the two dimensional projected lensing potential on a grid of (y,x) arc-second
        coordinates.

        The analytic Kassiola & Kovner (1993) I0.5 potential of the dPIE is the same
        two-component difference as the deflections, ported from Lenstool's C code
        (``e_pot.c`` case 81, ``pi05`` in ``e_pcpx.c``):

            psi = b0 * rs / (rs - ra) * (pi05(ra) - pi05(rs))

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the potential is computed on.
        """
        ellip = self._ellip(xp)
        factor = self.b0 * self.rs / (self.rs - self.ra)

        pot_core = _pi05(
            x=grid.array[:, 1],
            y=grid.array[:, 0],
            eps=ellip,
            rcore=self.ra,
            xp=xp,
        )
        pot_cut = _pi05(
            x=grid.array[:, 1],
            y=grid.array[:, 0],
            eps=ellip,
            rcore=self.rs,
            xp=xp,
        )

        return factor * (pot_core - pot_cut)


class dPIEMassB0Sph(dPIEMassB0):
    r"""Spherical dual pseudo-isothermal mass distribution in the internal
    (``ra``, ``rs``, ``b0``) parameterisation.

    **This is the non-standard parameterisation** — the default spherical dPIE is
    :class:`dPIEMassSph` (Lenstool-native parameters); see :class:`dPIEMassB0`.

    The spherical limit of :class:`dPIEMassB0`.  The projected convergence is:

    .. math::

        \kappa(r) = \frac{b_0}{2} \frac{r_s}{r_s - r_a}
        \left(
            \frac{1}{\sqrt{r_a^2 + r^2}}
            - \frac{1}{\sqrt{r_s^2 + r^2}}
        \right)

    where :math:`r` is the circular projected radius, :math:`r_a` is the core
    radius, :math:`r_s` is the truncation radius, and :math:`b_0` is the lens
    strength (Einstein radius in the limits :math:`r_a \to 0`,
    :math:`r_s \to \infty`).

    Parameters
    ----------
    centre : (float, float)
        (y, x) arc-second coordinates of the profile centre.
    ra : float
        Inner core radius in arcseconds.
    rs : float
        Outer truncation radius in arcseconds.
    b0 : float
        Lens strength in arcseconds (Einstein radius in the limit
        :math:`r_a \to 0`, :math:`r_s \to \infty`).

    References
    ----------
    Kassiola & Kovner (1993), ApJ, 417, 450.
    Eliasdottir et al. (2007), arXiv:0710.5636.
    Limousin et al. (2005), A&A, 461, 881.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        b0: float = 1.0,
    ):
        """
        The dual Pseudo Isothermal Elliptical Mass Distribution(dPIEMass) profiles without ellipticity, which is a *two component PIEMass* with both a core radius and a truncation radius,
        see Eliasdottir (2007): https://arxiv.org/abs/0710.5636
        This profile is ported from Lenstool's C code, which has the same formulation.

        This proflie describes an spherical isothermal mass distribution with a finite core, \\rho r^{-2} while in the transition region (ra<=R<=rs),
        and \\rho r^{-4} in the outer parts:
        \\rho \\propto [(ra^2 + R^2) (rs^2 + R^2)]^{-1}

        The convergence is given by two PIEMass with core radius ra and rs:
        \\kappa(r_{em}) = rs / (rs - ra) * (\\kappa_{PIEMass,ra} - \\kappa_{PIEMass,rs})
                        = b_0 / 2 * rs / (rs - ra) * ( \\frac{1}{\\sqrt{ ra^2 + r_{em}^2}} - \\frac{1}{\\sqrt{ rs^2 + r_{em}^2}} )
        where r_{em}^2 = x^2 / (1 + \\epsilon)^2 + y^2 / (1 - \\epsilon)^2.
        Note in Eliasdottir (2007), E0 = 6\\pi * \\sigma_{dPIEPotential}^2 / c^2 * (D_{LS} / D_{S}). Eliasdottir's E0 is not the same as E0 in Kassiola & Kovner(1993) which is also b0.
        There is \\frac{\\sigma_{dPIEPotential}^2}{\\sigma_0^2} = \\frac{2}{3} \frac{rs^2}{rs^2-ra^2},
        thus E0(Kassiola & Kovner(1993)) = b0 = E0(Eliasdottir (2007)) * (rs^2 - ra^2) / rs^2. So when s->\\infty and a->0, they are equivalent.

        In this implementation:
        - `ra` is the core radius in unit of arcseconds.
        - `rs` is the truncation radius in unit of arcseconds.
        - `b0` is the lens strength in unit of arcseconds, when ra->0 & rs->\\infty & q->1, b0 is the Einstein radius.
          `b0` is related to the central velocity dispersion \\sigma_0: b_0 = 4\\pi * \\sigma_0^2 / c^2 * (D_{LS} / D_{S})
          `b0` is not in the Intermediate-Axis-Convention for its r_{em}^2 = x^2 / (1 + \\epsilon)^2 + y^2 / (1 - \\epsilon)^2

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ra
            The inner core radius in arcseconds.
        rs
            The outer truncation radius in arcseconds.
        b0
            The lens strength in arcseconds.
        """
        super().__init__(centre=centre, ell_comps=(0.0, 0.0))

        self.ra = ra
        self.rs = rs
        self.b0 = b0

    @classmethod
    def from_lenstool(
        cls,
        centre: Tuple[float, float] = (0.0, 0.0),
        sigma: float = 200.0,
        r_core: float = 0.1,
        r_cut: float = 20.0,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
        cosmology: LensingCosmology = None,
    ) -> "dPIEMassB0Sph":
        """
        Construct a ``dPIEMassB0Sph`` from Lenstool's native dPIE / PIEMD parameterization
        (circular case). See ``dPIEMassB0.from_lenstool`` for the full conversion
        conventions; the ellipticity and angle inputs are absent here. For model-fitting
        in the Lenstool parameters use :class:`dPIEMassSph`.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        sigma
            Lenstool fiducial velocity dispersion ``v_disp`` (sigma_LT) in km/s — not the
            central velocity dispersion sigma_0 = sqrt(3/2) * sigma_LT.
        r_core
            Lenstool ``core_radius`` in arcseconds (becomes ``ra``).
        r_cut
            Lenstool ``cut_radius`` in arcseconds (becomes ``rs``).
        redshift_object
            The redshift of the lens, used for the D_LS / D_S normalization of ``b0``.
        redshift_source
            The redshift of the source used to normalize ``b0``.
        cosmology
            The cosmology used to compute the distance ratio (default ``Planck15``).
        """
        if cosmology is None:
            from autogalaxy.cosmology.model import Planck15

            cosmology = Planck15()

        b0 = _b0_from_lenstool_sigma(
            sigma=sigma,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
            cosmology=cosmology,
        )

        return cls(
            centre=centre,
            ra=r_core,
            rs=r_cut,
            b0=b0,
        )

    @aa.decorators.to_vector_yx
    @aa.decorators.transform(rotate_back=True)
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.
        Faster and equivalent to Eliasdottir (2007), see Eq. A19 and Eq. A20.

        f(R,a,s) = {R/a} / {1 + \\sqrt{1 + (R/a)^2}} - {R/s} / {1 + \\sqrt{1 + (R/s)^2}}
                 = R / {\\sqrt{a^2 + R^2} + a} - R / {\\sqrt{s^2 + R^2} + s}
                 = R * (\\sqrt{a^2 + R^2} - a) / {a^2 + R^2 - a^2} - R * (\\sqrt{s^2 + R^2} - s) / {s^2 + R^2 - s^2}
                 = (\\sqrt{R^2 + a^2} - a - \\sqrt{R^2 + s^2} + s) / R
        \\alpha = b0 * s / (s - a) * f(R,a,s)
        deflection_x = \\alpha * grid[:, 1] / R
                     = grid[:, 1] * b0 * s / (s - a) * (\\sqrt{R^2 + a^2} - a - \\sqrt{R^2 + s^2} + s) / R^2
        deflection_y = \\alpha * grid[:, 0] / R
                     = grid[:, 0] * b0 * s / (s - a) * (\\sqrt{R^2 + a^2} - a - \\sqrt{R^2 + s^2} + s) / R^2

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        a = self.ra
        s = self.rs
        # radii = self.radial_grid_from(grid=grid, xp=xp, **kwargs)
        # R2 = radii * radii
        R2 = grid.array[:, 1] * grid.array[:, 1] + grid.array[:, 0] * grid.array[:, 0]
        factor = xp.sqrt(R2 + a * a) - a - xp.sqrt(R2 + s * s) + s
        factor *= self.b0 * s / (s - a) / R2

        # This is in axes aligned to the major/minor axis
        deflection_x = grid.array[:, 1] * factor
        deflection_y = grid.array[:, 0] * factor

        return xp.vstack((deflection_y, deflection_x)).T

    @aa.decorators.to_array
    @aa.decorators.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Returns the two dimensional projected convergence on a grid of (y,x) arc-second coordinates.

        The `grid_2d_to_structure` decorator reshapes the ndarrays the convergence is outputted on. See
        *aa.grid_2d_to_structure* for a description of the output.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.
        """
        # already transformed to center on profile centre so this works
        radsq = grid.array[:, 0] ** 2 + grid.array[:, 1] ** 2

        return self._convergence(xp.sqrt(radsq), xp)

    @aa.decorators.transform
    def analytical_hessian_2d_from(self, grid: "aa.type.Grid2DLike", xp=np, **kwargs):
        """
        Calculate the hessian matrix on a grid of (y,x) arc-second coordinates.
        Chain rule of second derivatives:
        ∂²ψ/∂x² = ∂²ψ/∂R² * (∂R/∂x)² + ∂²R/∂x² * ∂ψ/∂R
        ∂²ψ/∂y² = ∂²ψ/∂R² * (∂R/∂y)² + ∂²R/∂y² * ∂ψ/∂R
        ∂²ψ/∂x∂y = ∂²ψ/∂R² * ∂R/∂x * ∂R/∂y + ∂²R/∂x∂y * ∂ψ/∂R


        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        if grid.ndim != 2 or grid.shape[1] != 2:
            raise ValueError("Grid must be a 2D array with shape (n, 2)")

        a = self.ra
        s = self.rs
        t05 = self.b0 * s / (s - a)

        # We have known the first derivatives as `deflections_yx`:
        # ∂ψ/∂R ∝ f(R,a,s) = (\\sqrt{R^2 + a^2} - a - \\sqrt{R^2 + s^2} + s) / R = z / R
        # ∂ψ/∂x ∝ x * (\\sqrt{R^2 + a^2} - a - \\sqrt{R^2 + s^2} + s) / R^2 = x * z / R^2
        # ∂ψ/∂y ∝ y * (\\sqrt{R^2 + a^2} - a - \\sqrt{R^2 + s^2} + s) / R^2 = y * z / R^2

        # where z = (\\sqrt{R^2 + a^2} - a - \\sqrt{R^2 + s^2} + s) / R^2

        # R = (x^2 + y^2)^(0.5)
        # ∂R/∂x = x / R
        # ∂R/∂y = y / R
        # ∂²R/∂²x = y^2 / R^3
        # ∂²R/∂²y = x^2 / R^3
        # ∂²R/∂x∂y = - x*y / R^3

        # ∂²ψ/∂²R = ∂(z/R)/∂R = (∂z/∂R * R - z * 1) / R^2
        #                     = {( R^2 / √(R^2 + a^2)) - ( R^2 / √(R^2 + s^2)) - z} / R^2
        #                     = p
        R2 = grid.array[:, 1] * grid.array[:, 1] + grid.array[:, 0] * grid.array[:, 0]
        z = xp.sqrt(R2 + a * a) - a - xp.sqrt(R2 + s * s) + s
        p = (1.0 - a / xp.sqrt(a * a + R2)) * a / R2 - (
            1.0 - s / xp.sqrt(s * s + R2)
        ) * s / R2
        X = grid.array[:, 1] * grid.array[:, 1] / R2  # x^2 / R^2
        Y = grid.array[:, 0] * grid.array[:, 0] / R2  # y^2 / R^2
        XY = grid.array[:, 1] * grid.array[:, 0] / R2  # x*y / R^2

        # ∂²ψ/∂x²  = ∂²ψ/∂R² * (∂R/∂x)² + ∂²R/∂x² * ∂ψ/∂R
        #          = p * (x / R)^2 + y^2 / R^3 * z / R
        #          = p * x^2 / R^2 + z * y^2 / R^2 / R^2
        #          = p * X + z * Y / R2
        # ∂²ψ/∂y²  = ∂²ψ/∂R² * (∂R/∂y)² + ∂²R/∂y² * ∂ψ/∂R
        #          = p * (y / R)^2 + x^2 / R^3 * z / R
        #          = p * y^2 / R^2 + z * x^2 / R^2 / R^2
        #          = p * Y + z * X / R2
        # ∂²ψ/∂x∂y = ∂²ψ/∂R² * ∂R/∂x * ∂R/∂y + ∂²R/∂x∂y * ∂ψ/∂R
        #          = p * (x / R) * (y / R) + (- x*y / R^3) * z / R
        #          = p * x*y / R^2 - z * x*y / R^2 / R^2
        #          = p * XY + z * XY / R2

        # Compute Hessian matrix components
        hessian_xx = t05 * (p * X + z * Y / R2)
        hessian_xy = t05 * (p * XY - z * XY / R2)
        hessian_yx = t05 * (p * XY - z * XY / R2)
        hessian_yy = t05 * (p * Y + z * X / R2)

        return hessian_yy, hessian_xy, hessian_yx, hessian_xx


class dPIEMass(dPIEMassB0):
    r"""Dual pseudo-isothermal elliptical mass distribution (dPIE) in Lenstool's
    native parameterization — **the default dPIE profile**.

    The dPIE (Elíasdóttir et al. 2007, App. A; also PIEMD, Kassiola & Kovner 1993)
    is the standard profile of published cluster- and group-scale strong-lensing
    analyses (Limousin et al. 2005; Bergamini et al. 2019; and essentially every
    Lenstool-based paper). Its free parameters here are exactly those of a Lenstool
    ``.par`` file (``potentiel`` profil 81) / paper results table, so a fitted
    posterior reads like a Lenstool results table:

    - ``ellipticity`` — Lenstool ``ellipticite``, emass = (a^2 - b^2) / (a^2 + b^2)
      (Elíasdóttir et al. 2007, Eq. A26). Converted internally to
      epot = (1 - q) / (1 + q) exactly as Lenstool's ``set_lens.c`` does.
    - ``angle_pos`` — position angle in degrees, counter-clockwise from the
      positive x-axis (Lenstool's tangent plane; axis flips from WCS conventions
      must be handled when ingesting real-data catalogues).
    - ``sigma`` — Lenstool's **fiducial** velocity dispersion ``v_disp``
      (sigma_LT, km/s), *not* the central dispersion:
      sigma_0 = sqrt(3/2) * sigma_LT (Elíasdóttir et al. 2007, App. A;
      Bergamini et al. 2019, Eq. 5). Quoting a measured stellar dispersion here
      overestimates the mass by 50%.
    - ``r_core`` / ``r_cut`` — Lenstool ``core_radius`` / ``cut_radius`` in
      arcseconds (the internal ``ra`` / ``rs``). For ``.par`` files using the kpc
      variants, pre-convert with
      ``r_core = r_core_kpc / cosmology.kpc_per_arcsec_from(redshift=redshift_object)``.

    The lens strength is fully normalized internally:
    b0 = 6 * 648000 * (sigma_LT / c)^2 * (D_LS / D_S) arcsec — equivalently
    E_0 = 6 pi (D_LS / D_S) (sigma_LT / c)^2 in radians (Elíasdóttir et al. 2007,
    Eq. A24) with the E_0-to-b0 prefactor folded in. Lenstool stores its ``b0``
    without the distance ratio and applies D_LS / D_S at deflection time
    (``e_grad.c``); the two conventions are verified equivalent against the
    Lenstool C source and reference deflections
    (``autolens_workspace_test/scripts/cluster/lenstool_parity.py``).

    The internal (``ell_comps``, ``ra``, ``rs``, ``b0``) parameterization — the
    non-standard variant — remains available via :class:`dPIEMassB0` (for models
    with priors on ``b0``) and :meth:`from_b0`.

    Parameters
    ----------
    centre : (float, float)
        (y, x) arc-second coordinates of the profile centre.
    ellipticity : float
        Lenstool mass ellipticity ``ellipticite`` = (a^2 - b^2) / (a^2 + b^2).
    angle_pos : float
        Position angle in degrees, counter-clockwise from the positive x-axis.
    sigma : float
        Lenstool fiducial velocity dispersion ``v_disp`` (sigma_LT) in km/s — not the
        central velocity dispersion sigma_0 = sqrt(3/2) * sigma_LT.
    r_core : float
        Lenstool ``core_radius`` in arcseconds (the dPIE ``ra``).
    r_cut : float
        Lenstool ``cut_radius`` in arcseconds (the dPIE ``rs``).
    redshift_object : float
        The redshift of the lens, used for the D_LS / D_S normalization of ``b0``.
    redshift_source : float
        The redshift of the source used to normalize ``b0``. For multi-plane cluster
        models this is the reference source plane the Lenstool model was normalized to.
    H0, Om0 : float
        Flat-input cosmology (defaults are Planck15 values) so the profile is fully
        constructable from flat inputs — priors configs, CSV rows — while a Lenstool
        run's own cosmology (typically H0=70, Om0=0.3) can be matched exactly. Model
        *constants* in practice. For an arbitrary cosmology object (e.g. ``Planck15``
        with massive neutrinos) use ``dPIEMassB0.from_lenstool(..., cosmology=...)``.

    References
    ----------
    Kassiola & Kovner (1993), ApJ, 417, 450.
    Elíasdóttir et al. (2007), arXiv:0710.5636 (App. A).
    Limousin et al. (2005), MNRAS, 356, 309.
    Bergamini et al. (2019), A&A, 631, A130.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ellipticity: float = 0.0,
        angle_pos: float = 0.0,
        sigma: float = 200.0,
        r_core: float = 0.1,
        r_cut: float = 20.0,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
        H0: float = 67.66,
        Om0: float = 0.30966,
    ):
        from autogalaxy.cosmology.model import FlatLambdaCDM

        # H0 / Om0 are plain floats (Planck15 values by default) so the profile is
        # fully constructable from flat inputs — CSV rows, prior configs — while a
        # Lenstool run's own cosmology (typically H0=70, Om0=0.3) can be matched
        # exactly. They are model *constants* in practice; the priors config carries
        # them only so af.Model composition works.
        cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)

        axis_ratio = np.sqrt((1.0 - ellipticity) / (1.0 + ellipticity))
        ell_comps = convert.ell_comps_from(axis_ratio=axis_ratio, angle=angle_pos)

        b0 = _b0_from_lenstool_sigma(
            sigma=sigma,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
            cosmology=cosmology,
        )

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            ra=r_core,
            rs=r_cut,
            b0=b0,
        )

        self.ellipticity = ellipticity
        self.angle_pos = angle_pos
        self.sigma = sigma
        self.r_core = r_core
        self.r_cut = r_cut
        self.redshift_object = redshift_object
        self.redshift_source = redshift_source
        self.H0 = H0
        self.Om0 = Om0

    @classmethod
    def from_b0(
        cls,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.0,
        rs: float = 2.0,
        b0: float = 0.1,
    ) -> "dPIEMassB0":
        """
        Construct a dPIE in the internal, non-standard (``ell_comps``, ``ra``, ``rs``,
        ``b0``) parameterization — returns a :class:`dPIEMassB0` instance.

        This is the pre-2026-07 default parameterization of the profile, kept for
        direct control of the lens strength ``b0`` (e.g. scaling relations composed
        on ``b0``). Published Lenstool-based analyses parameterize in
        (``ellipticity``, ``angle_pos``, ``sigma``, ``r_core``, ``r_cut``) — the
        default :class:`dPIEMass` constructor.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            Ellipticity components (e1, e2) of the elliptical coordinate system.
        ra
            The inner core radius in arcseconds.
        rs
            The outer truncation radius in arcseconds.
        b0
            The lens strength in arcseconds.
        """
        return dPIEMassB0(centre=centre, ell_comps=ell_comps, ra=ra, rs=rs, b0=b0)


class dPIEMassSph(dPIEMassB0Sph):
    """
    The spherical dPIE mass profile in Lenstool's native parameterization — **the
    default spherical dPIE profile** — supporting model-fitting with priors placed
    directly on the Lenstool parameters (``sigma``, ``r_core``, ``r_cut``). See
    :class:`dPIEMass` for the full conventions; the internal non-standard variant
    is :class:`dPIEMassB0Sph` / :meth:`from_b0`.

    Parameters
    ----------
    centre : (float, float)
        (y, x) arc-second coordinates of the profile centre.
    sigma : float
        Lenstool fiducial velocity dispersion ``v_disp`` (sigma_LT) in km/s.
    r_core : float
        Lenstool ``core_radius`` in arcseconds (the dPIE ``ra``).
    r_cut : float
        Lenstool ``cut_radius`` in arcseconds (the dPIE ``rs``).
    redshift_object : float
        The redshift of the lens, used for the D_LS / D_S normalization of ``b0``.
    redshift_source : float
        The redshift of the source used to normalize ``b0``.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        sigma: float = 200.0,
        r_core: float = 0.1,
        r_cut: float = 20.0,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
        H0: float = 67.66,
        Om0: float = 0.30966,
    ):
        from autogalaxy.cosmology.model import FlatLambdaCDM

        cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)

        b0 = _b0_from_lenstool_sigma(
            sigma=sigma,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
            cosmology=cosmology,
        )

        super().__init__(
            centre=centre,
            ra=r_core,
            rs=r_cut,
            b0=b0,
        )

        self.sigma = sigma
        self.r_core = r_core
        self.r_cut = r_cut
        self.redshift_object = redshift_object
        self.redshift_source = redshift_source
        self.H0 = H0
        self.Om0 = Om0

    @classmethod
    def from_b0(
        cls,
        centre: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        b0: float = 1.0,
    ) -> "dPIEMassB0Sph":
        """
        Construct a spherical dPIE in the internal, non-standard (``ra``, ``rs``,
        ``b0``) parameterization — returns a :class:`dPIEMassB0Sph` instance. See
        ``dPIEMass.from_b0``.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ra
            The inner core radius in arcseconds.
        rs
            The outer truncation radius in arcseconds.
        b0
            The lens strength in arcseconds.
        """
        return dPIEMassB0Sph(centre=centre, ra=ra, rs=rs, b0=b0)
