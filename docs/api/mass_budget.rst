===========
Mass Budget
===========

Tools for re-normalising a galaxy's mass: a reference projected mass measured inside an aperture is
split between a stellar component, whose mass follows the light, and a dark-matter halo. They are
the mass-normalisation mathematics of the lens-twin workflow, where a fitted total-mass profile
(e.g. an isothermal) is replaced by a stellar-plus-dark decomposition that keeps the same projected
mass inside the Einstein radius.

All quantities are angular: masses are projected masses in convergence units (the integral of the
dimensionless convergence over the aperture, in arc-seconds squared) and radii, centres and scale
radii are in arc-seconds. They convert to physical units with the critical surface density
:math:`\Sigma_{\rm cr}` of the lens and source redshifts and the cosmology, which is done
downstream (in **PyAutoLens**).

Convention
----------

**Aperture (ruling R3).** Masses are measured inside a circle of radius
:math:`\theta_{\rm E,parent}`, the Einstein radius of the parent lens model, centred on the mass
centre of the galaxy being replaced. :math:`\theta_{\rm E,parent}` is computed at the tracer level
(from the critical-curve area of all mass in the parent, via ``LensCalc``), so it is defined for
multi-galaxy and non-isothermal parents; computing it is the job of **PyAutoLens** (lens-twin
phase 3), and **PyAutoGalaxy** takes the aperture as a ``CircularAperture`` of given radius and
centre.

**Reference mass (ruling R3).** :math:`M_{\rm ref}` is the projected mass of the *replaced mass
component(s)* inside the aperture, obtained by grid integration of their convergence
(``mass_angular_within_circle_via_grid_from``), not of the whole lens model. External shear and
all other galaxies are left untouched and do not enter :math:`M_{\rm ref}`.

**Stellar component.** A ``Basis`` of ``lmp.Gaussian`` profiles copied from the galaxy's light
basis (centre, ``ell_comps``, ``sigma`` and solved ``intensity`` of every Gaussian), with one
mass-to-light ratio :math:`\Upsilon` shared by all members:

.. math::

    \Upsilon = \frac{f_* \, M_{\rm ref}}{M_{\rm light}(\theta_{\rm E})}

where :math:`f_*` is the stellar fraction and :math:`M_{\rm light}` is the same grid integral of
the basis at :math:`\Upsilon = 1`.

**Dark component (ruling R4).** The first halo is an ``NFWSph`` centred on the light with a fixed
scale radius :math:`r_s` (a fixed physical value; a concentration-mass relation :math:`c(M)` is a
later option). Its :math:`\kappa_s` is solved so its aperture mass is
:math:`(1 - f_*) \, M_{\rm ref}`. A ``gNFWSph`` with a requested ``inner_slope`` uses the same
solver, with the slope passed as a fixed parameter. Because the convergence of the NFW family is
linear in :math:`\kappa_s`, the solution is exact, :math:`\kappa_s = M / M(\kappa_s = 1)`, and needs
no root find. In **PyAutoGalaxy** :math:`r_s` is given in arc-seconds; its conversion from kpc is
done in **PyAutoLens**.

Since every aperture mass uses the same grid integration, the stellar and dark aperture masses sum
to :math:`M_{\rm ref}`. A fraction of zero creates no component for that part (it is returned as
``None``, and its solved parameter recorded as ``0.0``).

**Derived record.** ``MassBudget.solve`` returns the stellar basis, the dark profile and a
``derived`` dictionary with the aperture radius and centre, :math:`M_{\rm ref}`,
:math:`M_{\rm light}` at :math:`\Upsilon = 1`, :math:`\Upsilon`, :math:`r_s`, :math:`\kappa_s`,
the dark fraction :math:`f_{\rm dark}` inside the aperture and the further fixed dark-profile
parameters (e.g. ``inner_slope``).

**Multi-galaxy budgets (ruling R7).** A ``MassBudget`` is keyed per galaxy
(``{"lens": {"stellar": 0.5, "dark": 0.5}}``). Only the single-galaxy case is implemented: a budget
naming more than one galaxy raises ``NotImplementedError``, because the multi-galaxy normalisation
convention is deferred until it is ruled with real systems.

Example
-------

.. code-block:: python

    import autogalaxy as ag

    grid = ag.Grid2D.uniform(shape_native=(88, 88), pixel_scales=0.025)

    reference_mass = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0)

    light_basis = ag.lp_basis.Basis(
        profile_list=[
            ag.lp.GaussianSph(centre=(0.0, 0.0), intensity=1.0, sigma=0.5),
            ag.lp.GaussianSph(centre=(0.0, 0.0), intensity=0.3, sigma=1.0),
        ]
    )

    budget = ag.MassBudget(
        aperture=ag.CircularAperture(radius=1.0, centre=(0.0, 0.0)),
        fractions={"lens": {"stellar": 0.5, "dark": 0.5}},
    )

    result = budget.solve(
        reference_mass=reference_mass,
        light_basis=light_basis,
        dark_cls=ag.mp.NFWSph,
        scale_radius=5.0,
        centre=(0.0, 0.0),
        grid=grid,
    )

    result.stellar  # Basis of lmp.Gaussian with the solved mass-to-light ratio
    result.dark  # NFWSph with the solved kappa_s
    result.derived  # the record of the solve

API
---

.. currentmodule:: autogalaxy

.. autosummary::
   :toctree: _autosummary
   :template: custom-class-template.rst
   :recursive:

   MassBudget
   CircularAperture

.. autosummary::
   :toctree: _autosummary

   mass_to_light_ratio_for_mass_from
   kappa_s_for_mass_from
   mass_angular_within_circle_via_grid_from
