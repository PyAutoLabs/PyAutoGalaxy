"""
Conversion between PyAutoGalaxy profiles and the COOLEST standard
(https://github.com/aymgal/COOLEST).

This package converts profile *parameterisations* only — every function is
dict-in / dict-out on plain floats, so ``autogalaxy`` gains no dependency on
the ``coolest`` package. Reading and writing COOLEST JSON template files is
performed one layer up, in ``autolens.interop.coolest``, which consumes the
converters defined here.

COOLEST conventions (see https://coolest.readthedocs.io -> Conventions):

- Cartesian coordinates (x to the right, y up), positions in arcseconds.
- Position angles ``phi`` measured counter-clockwise from the positive y axis
  ("East-of-North"), in degrees, in the interval (-90, +90].
- Ellipticity described by the axis ratio ``q = b / a`` and ``phi``.
- Characteristic radii (Einstein radius, effective radius, scale radius) are
  defined along the *intermediate axis* of the ellipse, r = sqrt(a * b).

PyAutoGalaxy conventions:

- (y, x) coordinates in arcseconds.
- Position angles measured counter-clockwise from the positive x axis.
- Ellipticity described by the elliptical components ``ell_comps`` (e1, e2).
- Radius conventions are profile specific (e.g. the ``PowerLaw`` Einstein
  radius follows an average-axis convention, the ``Sersic`` effective radius
  is already an intermediate-axis radius) — see the converters in
  ``mass.py`` / ``light.py`` for the exact factor each profile requires.
"""

from autogalaxy.interop.coolest import conventions
from autogalaxy.interop.coolest import light
from autogalaxy.interop.coolest import mass
from autogalaxy.interop.coolest.light import coolest_dict_from_light
from autogalaxy.interop.coolest.light import light_profile_from
from autogalaxy.interop.coolest.mass import coolest_dict_from_mass
from autogalaxy.interop.coolest.mass import mass_profile_from
