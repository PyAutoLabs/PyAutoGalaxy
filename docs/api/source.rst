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