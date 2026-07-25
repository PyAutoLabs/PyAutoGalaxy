"""
Weak-lensing shear field data structures.

This module extends **PyAutoArray**'s vector-field classes with properties specific to weak gravitational
lensing.  A *shear field* is a set of complex shear components :math:`(\\gamma_2, \\gamma_1)` stored on a 2D
grid; each vector encodes the shape distortion induced on a background source galaxy by the intervening
mass distribution.

Where shear fields come from
----------------------------
The shear field of a strong lens model is produced by ``LensCalc.shear_yx_2d_via_hessian_from`` (in
``autogalaxy/operate/lens_calc.py``) which derives the shear from the Hessian of the lensing potential.
For analytic profiles such as ``Isothermal`` the shear is also available directly via
``shear_yx_2d_from``; the two routines agree by construction (see the cross-check test in
``test_autogalaxy/profiles/mass/total/test_isothermal.py``).

For weak-lensing simulations the same Hessian-derived shear is the natural input to a simulator: evaluating
``shear_yx_2d_via_hessian_from`` on the (y, x) positions of a population of background source galaxies
produces the noise-free shear measurement at each source — a direct analogue of the deflection-angle field
used for strong-lens simulators.

Convention
----------
The shear components are stored as ``[\\gamma_2, \\gamma_1]`` (``[:, 0]`` is :math:`\\gamma_2`, ``[:, 1]`` is
:math:`\\gamma_1`):

- :math:`\\gamma_1` produces stretching along the x/y axes
- :math:`\\gamma_2` produces stretching along the diagonals

The shear *position angle* :math:`\\phi = \\tfrac{1}{2} \\mathrm{arctan2}(\\gamma_2, \\gamma_1)` is measured
counter-clockwise from the positive x-axis, matching the convention used by ``ExternalShear`` and the helpers
in ``autogalaxy/convert.py``.  Storing :math:`\\gamma_2` first matches the ``[y, x]`` ordering used elsewhere
in the project for vector fields, since :math:`\\gamma_2` plays the role of the "y" component and
:math:`\\gamma_1` plays the role of the "x" component when shear is treated as a 2D pseudo-vector.

Two concrete classes are provided:

- ``ShearYX2D`` — regular-grid variant (inherits ``aa.VectorYX2D``).
- ``ShearYX2DIrregular`` — irregular-grid variant (inherits ``aa.VectorYX2DIrregular``).

Both classes inherit the shared mixin ``AbstractShearField`` which provides derived quantities for
visualisation and analysis: ``ellipticities``, ``semi_major_axes`` / ``semi_minor_axes``, position angles
(``phis``), and ``matplotlib`` ellipse patches (``elliptical_patches``).
"""
import logging
import numpy as np
import typing

import autoarray as aa

logging.basicConfig()
logger = logging.getLogger(__name__)


class AbstractShearField:
    """
    Mixin providing weak-lensing-specific derived quantities for ``ShearYX2D`` / ``ShearYX2DIrregular``.

    Treats the underlying ``[\\gamma_2, \\gamma_1]`` vector field as a set of weak-lensing shear
    measurements and exposes the corresponding ellipticity, semi-axes, position angle, and
    visualisation-ready ``matplotlib`` ellipse patches for each grid point.
    """

    @property
    def ellipticities(self) -> aa.ArrayIrregular:
        r"""
        Returns the shear magnitude :math:`|\gamma| = \sqrt{\gamma_1^2 + \gamma_2^2}` at each grid point,
        which in the weak-lensing limit equals the induced ellipticity of a background source galaxy.
        """
        return aa.ArrayIrregular(
            values=np.sqrt(self.slim[:, 0] ** 2 + self.slim[:, 1] ** 2.0)
        )

    @property
    def semi_major_axes(self) -> aa.ArrayIrregular:
        """
        Returns the semi-major axis ``3 * (1 + |gamma|)`` of the ellipse representing each shear vector.

        The factor of 3 is a fixed visual scale used so the rendered ellipses are large enough to be visible
        on top of a typical lensing image.  This is a *visualisation* quantity, not a physical galaxy size.
        """
        return aa.ArrayIrregular(values=3 * (1 + self.ellipticities))

    @property
    def semi_minor_axes(self) -> aa.ArrayIrregular:
        """
        Returns the semi-minor axis ``3 * (1 - |gamma|)`` of the ellipse representing each shear vector.

        See ``semi_major_axes`` for the role of the ``3`` factor.
        """
        return aa.ArrayIrregular(values=3 * (1 - self.ellipticities))

    @property
    def phis(self) -> aa.ArrayIrregular:
        r"""
        Returns the shear position angle :math:`\phi = \tfrac{1}{2} \, \mathrm{arctan2}(\gamma_2, \gamma_1)`
        in degrees, measured counter-clockwise from the positive x-axis.  This is the angle along which a
        background source is *stretched* by the shear.
        """
        return aa.ArrayIrregular(
            values=np.arctan2(self.slim[:, 0], self.slim[:, 1]) * 180.0 / np.pi / 2.0
        )

    @property
    def elliptical_patches(self) -> "typing.List[Ellipse]":
        """
        Returns a list of ``matplotlib.patches.Ellipse`` objects, one per grid point, ready to be added to a
        plotted image to visualise the shear field.  The width / height / angle of each patch are taken from
        ``semi_major_axes``, ``semi_minor_axes`` and ``phis``.
        """
        from matplotlib.patches import Ellipse

        return [
            Ellipse(
                xy=(x, y), width=semi_major_axis, height=semi_minor_axis, angle=angle
            )
            for x, y, semi_major_axis, semi_minor_axis, angle in zip(
                self.grid.slim[:, 1],
                self.grid.slim[:, 0],
                self.semi_major_axes,
                self.semi_minor_axes,
                self.phis,
            )
        ]


class ShearYX2D(aa.VectorYX2D, AbstractShearField):
    r"""
    A weak-lensing shear field on a uniform (regular) grid of ``(y, x)`` coordinates.

    The underlying storage is inherited from ``autoarray.structures.vectors.uniform.VectorYX2D``.  This
    class adds the weak-lensing-specific properties defined on ``AbstractShearField`` (ellipticities,
    semi-major / semi-minor axes, position angles, ``matplotlib`` ellipse patches).

    Storage convention
    ------------------
    Shape: ``[total_vectors, 2]``.  Column 0 is :math:`\gamma_2` (cross / diagonal stretch), column 1 is
    :math:`\gamma_1` (plus / axis-aligned stretch).  This matches the output of
    ``LensCalc.shear_yx_2d_via_hessian_from`` and ``Isothermal.shear_yx_2d_from``.

    Parameters
    ----------
    vectors
        The 2D ``[\gamma_2, \gamma_1]`` shear components on a uniform grid.
    grid
        The uniform grid of ``(y, x)`` coordinates where each shear vector is located.
    """


class ShearYX2DIrregular(aa.VectorYX2DIrregular, AbstractShearField):
    r"""
    A weak-lensing shear field on an irregular grid of ``(y, x)`` coordinates.

    The irregular variant is the natural data structure for shear measurements at the discrete positions of
    a population of background source galaxies.  The underlying storage is inherited from
    ``autoarray.structures.vectors.irregular.VectorYX2DIrregular``.  This class adds the
    weak-lensing-specific properties defined on ``AbstractShearField``.

    ``LensCalc.shear_yx_2d_via_hessian_from`` returns ``ShearYX2DIrregular`` on the NumPy path; on the JAX
    path it returns a raw ``(N, 2)`` array (because ``ShearYX2DIrregular`` is not a registered JAX pytree).

    Storage convention
    ------------------
    Shape: ``[total_vectors, 2]``.  Column 0 is :math:`\gamma_2` (cross / diagonal stretch), column 1 is
    :math:`\gamma_1` (plus / axis-aligned stretch).

    Parameters
    ----------
    vectors
        The 2D ``[\gamma_2, \gamma_1]`` shear components on an irregular grid.
    grid
        The irregular grid of ``(y, x)`` coordinates where each shear vector is located.
    """
