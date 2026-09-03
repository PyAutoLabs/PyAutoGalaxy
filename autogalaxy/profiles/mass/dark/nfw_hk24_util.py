"""
Created on Wed Apr  3 15:07:27 2024

@author: felixvecchi
"""

import numpy as np


def semi_major_axis_from(x1, x2, e, xp=np):
    """
    Returns the semi-major axis of the ellipse at a given point.

    Parameters
    ----------
    x1
        Horizontal coordinate, scaled by r_s, so unitless
    x2
        Vertical coordinate, scaled by r_s, so unitless
    e
        Eccentricity.
    """
    return xp.sqrt(x1**2 + x2**2 / (1 - e**2))


def capital_F_from(chi, xp=np):
    """
    Equation 16 from Heyrovský & Karamazov.

    Parameters
    ----------
    chi
        Dimenionless radial coordinate.

    Returns
    -------
    F(chi)
    """
    chi = xp.asarray(chi)
    eps = 1e-12

    less = chi < 1 - eps
    greater = chi > 1 + eps

    if xp is np:
        # Each branch is evaluated on its own subset of the grid only: the arctanh branch
        # inside the scale radius and the arctan branch outside it never see each other's
        # coordinates, so neither the special functions nor the square roots run twice over
        # the whole grid, and no masked-out input reaches a function it is invalid for.
        F = np.ones(chi.shape, dtype=np.result_type(chi, 1.0))

        chi_less = chi[less]
        root_min = np.sqrt(1.0 - chi_less**2)
        F[less] = np.arctanh(root_min) / root_min

        chi_greater = chi[greater]
        root_plus = np.sqrt(chi_greater**2 - 1.0)
        F[greater] = np.arctan(root_plus) / root_plus

        return F

    root_min_arg = xp.where(less, 1 - chi**2, 0.0)
    root_min = xp.sqrt(root_min_arg)
    root_min_safe = xp.where(less, root_min, 1.0)
    F_less = xp.where(less, xp.arctanh(root_min) / root_min_safe, 0.0)

    root_plus_arg = xp.where(greater, chi**2 - 1, 0.0)
    root_plus = xp.sqrt(root_plus_arg)
    root_plus_safe = xp.where(greater, root_plus, 1.0)
    F_greater = xp.where(greater, xp.arctan(root_plus) / root_plus_safe, 0.0)

    F_equal = xp.ones_like(chi)

    return xp.where(less, F_less, xp.where(greater, F_greater, F_equal))


def kappa_from(k_s, a, xp=np):
    """
    Equation 21 from Heyrovský & Karamazov.

    Parameters
    ----------
    k_s
        Halo convergence parameter.
    a
        Semi major axis scaled by a_scale.

    Returns
    -------
    Convergence as a function of a
    """
    F = capital_F_from(a, xp=xp)
    kappa = xp.where(xp.abs(a - 1) < 1e-12, 2 / 3 * k_s, 2 * k_s * (1 - F) / (a**2 - 1))
    return kappa


def small_f_1(x1, x2, e, xp=np):
    """
    Equation 32 HK+24

    Parameters
    ----------
    x1
        Horizontal coordinate, scaled by r_s, so unitless
    x2
        Vertical coordinate, scaled by r_s, so unitless
    e
        Eccentricity.

    Returns
    -------
    f_1
    """
    a = semi_major_axis_from(x1, x2, e, xp=xp)
    F = capital_F_from(a, xp=xp)
    f1 = (1 - e**2) ** (-1 / 2) * F
    return f1


def small_f_2(x1, x2, e, xp=np):
    """
    Equation 32 HK+24

    Parameters
    ----------
    x1
        Horizontal coordinate, scaled by r_s, so unitless
    x2
        Vertical coordinate, scaled by r_s, so unitless
    e
        Eccentricity.

    Returns
    -------
    f_3

    """
    norm = xp.sqrt(x1**2 + x2**2)
    f2 = xp.log(norm / (1 + xp.sqrt(1 - e**2)))
    return f2


def small_f_3(x1, x2, e, xp=np):
    """
    Equation 32 HK+24

    Parameters
    ----------
    x1
        Horizontal coordinate, scaled by r_s, so unitless
    x2
        Vertical coordinate, scaled by r_s, so unitless
    e
        Eccentricity.

    Returns
    -------
    f_3

    """
    root = xp.sqrt(1 - e**2)
    f3 = xp.arctan(x1 * x2 * (1 - root) / (x1**2 * root + x2**2))
    return f3


def small_f_0(x1, x2, e, xp=np):
    """
    Equation 37 HK+24

    Parameters
    ----------
    x1
        Horizontal coordinate, scaled by r_s, so unitless
    x2
        Vertical coordinate, scaled by r_s, so unitless
    e
        Eccentricity.

    Returns
    -------
    f_0

    """
    a = semi_major_axis_from(x1, x2, e, xp=xp)
    F = capital_F_from(a, xp=xp)
    pre_factor = 1 / (2 * xp.sqrt(1 - e**2))
    nominator = x1**2 + x2**2 + e**2 - 2 + (1 - e**2 * x1**2) * F
    denominator = 1 - x1**2 - x2**2 / (1 - e**2)

    f0 = 1 + pre_factor * (nominator / denominator)

    return f0


def g1_g2_from(x1, x2, e, k_s, xp=np):
    """
    Both components of the shear

    Equation 35 and 36 HK+24

    Parameters
    ----------
    x1 : numpy array
         horizontal coordinate, scaled by r_s, so unitless
    x2 : numpy array
        vertical coordinate, scaled by r_s, so unitless
    e : scalar
        eccentricity.
    k_s : scaler
        halo convergence parameter

    Returns
    -------
    tuple of 2 numpy arrays
    gamma1, gamma2 (g1 in '+' and g2 in 'x' shape)
    """

    # Factorized functions g1 and g2
    f0 = small_f_0(x1, x2, e, xp=xp)
    f1 = small_f_1(x1, x2, e, xp=xp)
    f2 = small_f_2(x1, x2, e, xp=xp)
    f3 = small_f_3(x1, x2, e, xp=xp)

    # Prefactor for both g1 and g2
    full_pre_factor = (
        4
        * k_s
        * xp.sqrt(1 - e**2)
        / (((x1 - e) ** 2 + x2**2) ** 2 * ((x1 + e) ** 2 + x2**2) ** 2)
    )

    # Prefactors for g1
    pre_f0_g1 = (
        ((x1 - e) ** 2 + x2**2) * ((x1 + e) ** 2 + x2**2) * (x1**2 - x2**2 - e**2)
    )

    pre_f1_g1 = (
        2 * e**2 * (x1**2 - 1) * ((x1**2 - x2**2 - e**2) ** 2 - 4 * x1**2 * x2**2)
        - (x1**2 - x2**2 - e**2) ** 3 * (3 + e**2) / 2
        + 6 * x1**2 * x2**2 * (e**2 - 1) * (x1**2 - x2**2 - e**2)
    )

    pre_f2_g1 = -(
        (x1**2 - x2**2 - e**2) * ((x1**2 + x2**2) ** 2 - e**4)
        - 8 * e**2 * x1**2 * x2**2
    )

    pre_f3_g1 = 2 * x1 * x2 * ((x1**2 + x2**2 + e**2) ** 2 - 4 * e**2 * (x2**2 + e**2))

    # First component shear
    g1 = full_pre_factor * (
        pre_f0_g1 * f0 + pre_f1_g1 * f1 + pre_f2_g1 * f2 + pre_f3_g1 * f3
    )

    # Prefactors g2
    pre_f0_g2 = 2 * x1 * x2 * ((x1 - e) ** 2 + x2**2) * ((x1 + e) ** 2 + x2**2)

    pre_f1_g2 = (
        x1
        * x2
        * (
            (x1**2 + x2**2 + e**2)
            * ((5 * e**2 - 3) * x1**2 - 3 * (1 + e**2) * x2**2 + (5 - 3 * e**2) * e**2)
            - 4 * e**2 * x1**2 * (1 + e**2)
        )
    )

    pre_f2_g2 = -2 * x1 * x2 * ((x1**2 + x2**2 + e**2) ** 2 - 4 * e**2 * (x2**2 + e**2))

    pre_f3_g2 = -(
        (x1**2 - x2**2 - e**2) * ((x1**2 + x2**2) ** 2 - e**4)
        - 8 * e**2 * x1**2 * x2**2
    )

    # Second component shear
    g2 = full_pre_factor * (
        pre_f0_g2 * f0 + pre_f1_g2 * f1 + pre_f2_g2 * f2 + pre_f3_g2 * f3
    )

    return g1, g2
