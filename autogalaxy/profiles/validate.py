"""
Profile constructor guards.

These are thin, named wrappers over the shared helpers in ``autoarray.validate``
(landed by PyAutoArray#440 for PyAutoArray#333). PyAutoArray is the floor both
PyAutoGalaxy and PyAutoLens build on, so the *rules* and the *message shape* are
defined once there and reused here — the failure mode being avoided is three repos
telling a user three different things about the same mistake.

What lives here is only the per-parameter explanation, written once per parameter
rather than once per profile class. Every message therefore reads the same way:
name the parameter, state the rule, show the received value, then explain why.

__Tracer safety__

Profile parameters are free model parameters, so under a JAX-traced fit a
constructor is handed a tracer rather than a number, and a plain Python
``if value <= 0`` would raise ``TracerBoolConversionError``. The shared helpers gate
every comparison on ``autoarray.validate.is_concrete_scalar`` and pass non-concrete
values straight through, so these guards catch hand-written mistakes and cost
nothing inside a trace.
"""

import numpy as np

from autoarray import validate


def validate_scale_radius(scale_radius, name: str = "scale_radius"):
    """
    Raise if a dark-matter halo scale radius is a concrete scalar which is not finite
    and positive.

    A ``scale_radius`` of zero divides the grid by zero in the very first step of the
    deflection-angle calculation, so every returned value is NaN rather than an error —
    the profile appears to work and quietly poisons the whole fit.

    Parameters
    ----------
    scale_radius
        The scale radius to validate.
    name
        The parameter's name, used in the error message.
    """
    validate.validate_positive_finite(
        value=scale_radius,
        name=name,
        extra=(
            "The scale radius is the angular radius at which the halo's log-slope "
            "changes, so it must be above zero. A value of 0.0 divides the grid by "
            "zero when computing deflection angles, which returns an all-NaN result "
            "instead of raising"
        ),
    )


def validate_sersic_index(sersic_index, name: str = "sersic_index"):
    """
    Raise if a Sersic index is a concrete scalar which is not finite and positive.

    A ``sersic_index`` of zero reaches a division by ``n`` deep inside the profile's
    ``image_2d_from``, surfacing as a bare ``ZeroDivisionError`` several calls away
    from the constructor that accepted it.

    Parameters
    ----------
    sersic_index
        The Sersic index to validate.
    name
        The parameter's name, used in the error message.
    """
    validate.validate_positive_finite(
        value=sersic_index,
        name=name,
        extra=(
            "The Sersic index controls the concentration of the profile and appears "
            "as a divisor in its normalisation, so it must be above zero. A value of "
            "0.0 raises ZeroDivisionError from inside image_2d_from rather than at "
            "construction"
        ),
    )


def validate_redshift(redshift, name: str = "redshift"):
    """
    Raise if a galaxy redshift is a concrete scalar which is negative or non-finite.

    A negative redshift is unphysical and produces meaningless angular diameter
    distances in every multi-plane calculation that consumes it.

    Zero is permitted: ``redshift=0.0`` is a legitimate way to place a galaxy at the
    observer, and is used in single-plane work where the redshift is a label rather
    than a cosmological quantity.

    **This does not touch the ``z_lens > z_source`` question.** Multi-plane lensing
    genuinely supports geometries that look wrong under two-plane naming, so that
    case must warn at most, never raise, and is held pending the reporter's answer on
    PyAutoLens#532.

    Parameters
    ----------
    redshift
        The redshift to validate.
    name
        The parameter's name, used in the error message.
    """
    validate.validate_non_negative_finite(
        value=redshift,
        name=name,
        extra=(
            "A redshift is a cosmological distance measure and cannot be negative — "
            "a negative value yields meaningless angular diameter distances in every "
            "multi-plane calculation that consumes it. Note that a lens redshift "
            "above a source redshift is NOT rejected: multi-plane lensing supports "
            "geometries that look inverted under two-plane naming"
        ),
    )


def validate_ell_comps(ell_comps, name: str = "ell_comps"):
    """
    Raise if the elliptical components are concrete scalars whose magnitude is not
    below one.

    The axis ratio is defined as ``q = (1 - f) / (1 + f)`` with
    ``f = sqrt(e_y**2 + e_x**2)``. That is a valid axis ratio in ``(0, 1]`` only while
    ``f < 1``; at ``f == 1`` the ellipse degenerates to ``q == 0``, and beyond it ``q``
    goes negative and the profile has no geometric meaning. Today such a profile is
    accepted and returns a finite but non-physical image.

    Applied at ``EllProfile``, the single base every elliptical light and mass profile
    inherits, so the rule is stated once rather than per subclass.

    Parameters
    ----------
    ell_comps
        The (e_y, e_x) elliptical components to validate.
    name
        The parameter's name, used in the error message.
    """
    if ell_comps is None:
        return

    try:
        ell_y, ell_x = ell_comps
    except (TypeError, ValueError):
        return

    if not validate.is_concrete_scalar(ell_y) or not validate.is_concrete_scalar(ell_x):
        return

    magnitude_squared = ell_y * ell_y + ell_x * ell_x

    if not np.isfinite(magnitude_squared) or magnitude_squared >= 1.0:
        raise ValueError(
            f"{name} must satisfy {name}[0]**2 + {name}[1]**2 < 1; got "
            f"{tuple(ell_comps)!r}, whose magnitude is "
            f"{np.sqrt(magnitude_squared) if np.isfinite(magnitude_squared) else magnitude_squared!r}. "
            f"The axis ratio is q = (1 - f) / (1 + f) with f the magnitude of "
            f"{name}, so f must be below 1 for q to be a valid axis ratio — at f = 1 "
            f"the ellipse degenerates to q = 0 and beyond it q is negative and the "
            f"profile has no geometric meaning"
        )
