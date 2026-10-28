===========
Source Code
===========

This page provided API docs for functionality which is typically not used by users, but is used internally in the
**PyAutoGalaxy** source code.

These docs are intended for developers, or users doing non-standard computations using internal **PyAutoFit** objects.

Geometry Profiles
-----------------

.. currentmodule:: autogalaxy.profiles.geometry_profiles

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   EllProfile
   SphProfile

Operators
---------

.. currentmodule:: autogalaxy

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   OperateImage
   LensCalc

Projected mass inside a circle (convergence, angular units), grid-integrated so it is defined for any
object with ``convergence_2d_from`` (a mass profile, ``Basis``, ``Galaxy`` or ``Galaxies``):

.. autosummary::
   :toctree: _autosummary

   mass_angular_within_circle_via_grid_from

Total [ag.mp]
-------------

.. currentmodule:: autogalaxy.profiles.mass

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   PointMass
   PowerLawCore
   PowerLawCoreSph
   PowerLawBroken
   PowerLawBrokenSph
   IsothermalCore
   IsothermalCoreSph
   PowerLaw
   PowerLawSph
   Isothermal
   IsothermalSph

Mass Sheets [ag.mp]
-------------------

.. currentmodule:: autogalaxy.profiles.mass

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   ExternalShear
   MassSheet

Multipoles [ag.mp]
------------------

.. currentmodule:: autogalaxy.profiles.mass

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   PowerLawMultipole

Stellar [ag.mp]
---------------

.. currentmodule:: autogalaxy.profiles.mass

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   Gaussian
   Sersic
   SersicSph
   Exponential
   ExponentialSph
   DevVaucouleurs
   DevVaucouleursSph
   SersicGradient
   SersicGradientSph
   Chameleon
   ChameleonSph

Dark [ag.mp]
------------

.. currentmodule:: autogalaxy.profiles.mass

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   gNFW
   gNFWSph
   NFWTruncatedSph
   NFWTruncatedMCRDuffySph
   NFWTruncatedMCRLudlowSph
   NFWTruncatedMCRScatterLudlowSph
   NFW
   NFWSph
   NFWMCRDuffySph
   NFWMCRLudlowSph
   NFWMCRScatterLudlow
   NFWMCRScatterLudlowSph
   NFWMCRLudlow
   gNFWMCRLudlow