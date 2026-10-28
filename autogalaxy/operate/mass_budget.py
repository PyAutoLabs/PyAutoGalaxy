"""
Mass-normalisation mathematics: split a reference projected mass inside an aperture into a stellar
component (a ``Basis`` of ``lmp.Gaussian`` with one shared mass-to-light ratio) and a dark-matter
component (e.g. ``NFWSph`` / ``gNFWSph`` with a solved ``kappa_s``).

All quantities are angular: masses are projected masses in convergence units (the integral of the
dimensionless convergence over the aperture, in arc-seconds squared), radii and centres are in
arc-seconds. Converting to physical units (multiplying by the critical surface density, or a scale
radius in kpc) is the caller's job.

Every aperture mass is computed by grid integration of the convergence
(``mass_angular_within_circle_via_grid_from``), so the same convention applies to the reference
mass, the stellar basis and the dark profile, and the stellar and dark aperture masses sum exactly
(to the integrator's arithmetic precision) to the reference mass on the same grid.

The convention these functions implement is documented in ``docs/api/mass_budget.rst``.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple, runtime_checkable

import autoarray as aa

from autogalaxy.operate.lens_calc import mass_angular_within_circle_via_grid_from

_FRACTION_KEYS = ("stellar", "dark")
_FRACTION_SUM_TOLERANCE = 1.0e-9


@runtime_checkable
class Aperture(Protocol):
    """
    The protocol of an aperture in which projected masses are measured.

    An aperture is any object with a ``mass_from(mass_obj, grid)`` method returning the projected
    mass of ``mass_obj`` inside the aperture, in convergence (angular) units. ``CircularAperture``
    is the only implementation in **PyAutoGalaxy**; **PyAutoLens** builds apertures whose radius is
    derived from a tracer (e.g. its Einstein radius) on top of it.
    """

    def mass_from(self, mass_obj, grid: aa.type.Grid2DLike) -> float:
        """
        Returns the projected mass of ``mass_obj`` inside the aperture, in convergence units.

        Parameters
        ----------
        mass_obj
            Any object with a ``convergence_2d_from(grid=...)`` method.
        grid
            The uniform grid on which the convergence is integrated.
        """
        ...


class CircularAperture:
    def __init__(
        self,
        radius: float,
        centre: Tuple[float, float] = (0.0, 0.0),
        sub_size_interior: int = 8,
        sub_size_edge: int = 16,
        sub_size_centre: int = 64,
        centre_refine_pixels: float = 2.0,
    ):
        """
        A circular aperture of a given radius and centre, inside which projected masses are measured
        by grid integration of the convergence.

        The integration is performed by ``mass_angular_within_circle_via_grid_from``; the
        ``sub_size_*`` and ``centre_refine_pixels`` attributes are passed straight through to it and
        set its accuracy (their defaults are the integrator's defaults). For a profile with a
        central cusp centred in the aperture (e.g. an isothermal profile), ``sub_size_centre`` is
        the parameter that controls the leading error term.

        Parameters
        ----------
        radius
            The radius of the circle, in arc-seconds.
        centre
            The (y,x) centre of the circle, in arc-seconds.
        sub_size_interior
            The sub-grid size used for pixels entirely inside the circle (away from ``centre``).
        sub_size_edge
            The sub-grid size used to refine pixels crossing the circle edge.
        sub_size_centre
            The sub-grid size used to refine pixels near ``centre``.
        centre_refine_pixels
            Pixels whose nearest point lies within this many (largest) pixel scales of ``centre``
            are refined with ``sub_size_centre``.
        """
        self.radius = radius
        self.centre = centre
        self.sub_size_interior = sub_size_interior
        self.sub_size_edge = sub_size_edge
        self.sub_size_centre = sub_size_centre
        self.centre_refine_pixels = centre_refine_pixels

    def mass_from(self, mass_obj, grid: aa.type.Grid2DLike) -> float:
        """
        Returns the projected mass of ``mass_obj`` inside the circle, in convergence (angular) units,
        by grid integration of its convergence.

        Parameters
        ----------
        mass_obj
            Any object with a ``convergence_2d_from(grid=...)`` method (e.g. a ``MassProfile``,
            ``Basis``, ``Galaxy`` or ``Galaxies``).
        grid
            A uniform ``Grid2D`` whose shape, pixel scales and origin define the integration
            pixels. The circle must lie within its extent.

        Returns
        -------
        The projected mass inside the circle, in convergence units.
        """
        return float(
            mass_angular_within_circle_via_grid_from(
                mass_obj=mass_obj,
                radius=self.radius,
                centre=self.centre,
                grid=grid,
                sub_size_interior=self.sub_size_interior,
                sub_size_edge=self.sub_size_edge,
                sub_size_centre=self.sub_size_centre,
                centre_refine_pixels=self.centre_refine_pixels,
            )
        )

    def __repr__(self):
        return f"CircularAperture(radius={self.radius}, centre={self.centre})"


def _check_target_mass(mass: float, function_name: str):
    if mass < 0.0:
        raise ValueError(
            f"{function_name}: the target mass must be non-negative, got {mass}."
        )


def mass_to_light_ratio_for_mass_from(
    basis, mass: float, aperture: Aperture, grid: aa.type.Grid2DLike
) -> float:
    """
    Returns the single mass-to-light ratio which, shared by every Gaussian of a light ``basis``,
    gives a projected mass ``mass`` inside ``aperture``.

    The basis is converted to a mass basis with a unit mass-to-light ratio
    (``basis.mass_basis_from(1.0)``) and its aperture mass ``M_light`` is measured. Because the
    convergence of the mass basis is linear in the mass-to-light ratio, the solution is exact:

    .. math::

        \\Upsilon = M / M_{\\rm light}(\\Upsilon = 1)

    Parameters
    ----------
    basis
        A ``Basis`` of standard ``lp.Gaussian`` (or ``lmp.Gaussian``) profiles with known
        intensities, e.g. the solved-intensity light basis of a fit. See ``Basis.mass_basis_from``
        for the members it accepts.
    mass
        The target projected mass inside the aperture, in convergence (angular) units.
    aperture
        The aperture in which the mass is measured (e.g. a ``CircularAperture``).
    grid
        The uniform grid on which the convergence is integrated.

    Returns
    -------
    The mass-to-light ratio, in convergence units per unit of the basis's intensity.

    Raises
    ------
    ValueError
        If ``mass`` is negative or the aperture mass of the basis at unit mass-to-light ratio is
        not positive.
    """
    _check_target_mass(mass=mass, function_name="mass_to_light_ratio_for_mass_from")

    mass_light = aperture.mass_from(
        mass_obj=basis.mass_basis_from(mass_to_light_ratio=1.0), grid=grid
    )

    if not mass_light > 0.0:
        raise ValueError(
            "mass_to_light_ratio_for_mass_from: the aperture mass of the basis at unit "
            f"mass-to-light ratio is {mass_light}, which is not positive, so no mass-to-light "
            "ratio can be solved for."
        )

    return mass / mass_light


def kappa_s_for_mass_from(
    dark_cls,
    mass: float,
    aperture: Aperture,
    scale_radius: float,
    centre: Tuple[float, float],
    grid: aa.type.Grid2DLike,
    **fixed,
) -> float:
    """
    Returns the ``kappa_s`` of a dark-matter profile of class ``dark_cls`` whose projected mass
    inside ``aperture`` is ``mass``, at a fixed ``scale_radius`` and ``centre``.

    The convergence of the NFW family (``NFWSph``, ``NFW``, ``gNFWSph``, ``gNFW``) is proportional
    to ``kappa_s`` at fixed shape parameters, so the aperture mass is linear in ``kappa_s`` and the
    solution is exact, with no root find:

    .. math::

        \\kappa_s = M / M(\\kappa_s = 1)

    Parameters
    ----------
    dark_cls
        The dark-matter profile class, e.g. ``ag.mp.NFWSph`` or ``ag.mp.gNFWSph``. It must take
        ``centre``, ``kappa_s`` and ``scale_radius`` constructor arguments and have a convergence
        linear in ``kappa_s``.
    mass
        The target projected mass inside the aperture, in convergence (angular) units.
    aperture
        The aperture in which the mass is measured (e.g. a ``CircularAperture``).
    scale_radius
        The fixed scale radius of the profile, in arc-seconds.
    centre
        The (y,x) centre of the profile, in arc-seconds.
    grid
        The uniform grid on which the convergence is integrated.
    fixed
        Any further fixed constructor arguments of ``dark_cls``, e.g. ``inner_slope`` for
        ``gNFWSph`` or ``ell_comps`` for ``NFW``.

    Returns
    -------
    The ``kappa_s`` of the profile.

    Raises
    ------
    ValueError
        If ``mass`` is negative or the aperture mass at ``kappa_s = 1`` is not positive.
    """
    _check_target_mass(mass=mass, function_name="kappa_s_for_mass_from")

    unit_profile = dark_cls(
        centre=centre, kappa_s=1.0, scale_radius=scale_radius, **fixed
    )

    mass_unit = aperture.mass_from(mass_obj=unit_profile, grid=grid)

    if not mass_unit > 0.0:
        raise ValueError(
            f"kappa_s_for_mass_from: the aperture mass of {dark_cls.__name__} at kappa_s = 1 is "
            f"{mass_unit}, which is not positive, so no kappa_s can be solved for."
        )

    return mass / mass_unit


@dataclass
class MassBudgetResult:
    """
    The output of ``MassBudget.solve``.

    Attributes
    ----------
    stellar
        The stellar mass component: a ``Basis`` of ``lmp.Gaussian`` profiles with one shared
        mass-to-light ratio, or ``None`` if the budget's stellar fraction is zero.
    dark
        The dark-matter profile instance (of the requested class, with the solved ``kappa_s``), or
        ``None`` if the budget's dark fraction is zero.
    derived
        The record of the solve (see ``MassBudget.solve`` for its keys).
    """

    stellar: Optional[object]
    dark: Optional[object]
    derived: Dict


class MassBudget:
    def __init__(self, aperture: Aperture, fractions: Dict[str, Dict[str, float]]):
        """
        A budget which splits a reference projected mass inside an aperture between a stellar and
        a dark-matter component, for one galaxy.

        ``fractions`` maps a galaxy name to its stellar and dark fractions of the reference
        aperture mass, e.g. ``{"lens": {"stellar": 0.5, "dark": 0.5}}``. The two fractions must be
        non-negative and sum to one (to 1e-9). A fraction of zero means that component is not
        created.

        Only single-galaxy budgets are implemented: a budget naming more than one galaxy raises
        ``NotImplementedError``, because the convention for normalising several galaxies inside a
        shared aperture has not yet been ruled (ruling R7 of the lens-twin epic).

        Parameters
        ----------
        aperture
            The aperture in which the reference, stellar and dark masses are measured (e.g. a
            ``CircularAperture`` of radius the parent's Einstein radius centred on the replaced
            galaxy's mass centre).
        fractions
            A dictionary ``{galaxy_name: {"stellar": f_stellar, "dark": f_dark}}`` with exactly one
            galaxy.

        Raises
        ------
        ValueError
            If ``fractions`` is empty, a galaxy's fractions do not have exactly the keys
            ``"stellar"`` and ``"dark"``, a fraction is negative, or the fractions do not sum to one.
        NotImplementedError
            If ``fractions`` names more than one galaxy.
        """
        if not fractions:
            raise ValueError("MassBudget: fractions must name one galaxy.")

        for galaxy_name, galaxy_fractions in fractions.items():
            if set(galaxy_fractions) != set(_FRACTION_KEYS):
                raise ValueError(
                    f"MassBudget: the fractions of galaxy '{galaxy_name}' must have exactly the "
                    f"keys {_FRACTION_KEYS}, got {tuple(galaxy_fractions)}."
                )

            for key in _FRACTION_KEYS:
                if galaxy_fractions[key] < 0.0:
                    raise ValueError(
                        f"MassBudget: the {key} fraction of galaxy '{galaxy_name}' is "
                        f"{galaxy_fractions[key]}, but fractions must be non-negative."
                    )

            total = galaxy_fractions["stellar"] + galaxy_fractions["dark"]

            if abs(total - 1.0) > _FRACTION_SUM_TOLERANCE:
                raise ValueError(
                    f"MassBudget: the fractions of galaxy '{galaxy_name}' sum to {total}, but "
                    "must sum to 1."
                )

        if len(fractions) > 1:
            raise NotImplementedError(
                "MassBudget: a budget naming more than one galaxy is not supported; the "
                "multi-galaxy normalisation convention is deferred (ruling R7 of the lens-twin "
                "epic). Use one budget per replaced galaxy only once that convention is ruled."
            )

        self.aperture = aperture
        self.fractions = fractions

    @property
    def galaxy_name(self) -> str:
        """
        The name of the (single) galaxy this budget normalises.
        """
        return next(iter(self.fractions))

    @property
    def f_stellar(self) -> float:
        """
        The stellar fraction of the reference aperture mass.
        """
        return self.fractions[self.galaxy_name]["stellar"]

    @property
    def f_dark(self) -> float:
        """
        The dark-matter fraction of the reference aperture mass.
        """
        return self.fractions[self.galaxy_name]["dark"]

    def solve(
        self,
        reference_mass,
        light_basis,
        dark_cls,
        scale_radius: float,
        centre: Tuple[float, float],
        grid: aa.type.Grid2DLike,
        **fixed,
    ) -> MassBudgetResult:
        """
        Returns the stellar and dark-matter components which reproduce the projected mass of
        ``reference_mass`` inside the aperture, split according to the budget's fractions.

        The steps are:

        1. ``M_ref`` = the aperture mass of ``reference_mass`` (the replaced mass component(s)
           only, not the whole system: external shear and other galaxies are not part of it).
        2. The stellar component is ``light_basis.mass_basis_from(Upsilon)``: a ``Basis`` of
           ``lmp.Gaussian`` copied from the light basis with one shared mass-to-light ratio
           ``Upsilon``, solved so its aperture mass is ``f_stellar * M_ref``
           (``mass_to_light_ratio_for_mass_from``).
        3. The dark component is ``dark_cls(centre=centre, kappa_s=kappa_s,
           scale_radius=scale_radius, **fixed)``, with ``kappa_s`` solved so its aperture mass is
           ``f_dark * M_ref`` (``kappa_s_for_mass_from``).

        On the same grid and aperture, the stellar and dark aperture masses therefore sum to
        ``M_ref``. A component whose fraction is zero is returned as ``None``; its solved parameter
        (``mass_to_light_ratio`` or ``kappa_s``) is recorded as ``0.0``.

        Parameters
        ----------
        reference_mass
            The mass being replaced: any object with a ``convergence_2d_from(grid=...)`` method
            (e.g. a ``MassProfile``, a ``Basis`` or a list of profiles wrapped in a ``Galaxy``).
        light_basis
            The galaxy's light ``Basis`` of standard Gaussians with known intensities (see
            ``Basis.mass_basis_from``).
        dark_cls
            The dark-matter profile class, e.g. ``ag.mp.NFWSph`` or ``ag.mp.gNFWSph``.
        scale_radius
            The fixed scale radius of the dark profile, in arc-seconds.
        centre
            The (y,x) centre of the dark profile, in arc-seconds (by convention the centre of the
            light).
        grid
            The uniform grid on which every aperture mass is integrated.
        fixed
            Further fixed constructor arguments of ``dark_cls``, e.g. ``inner_slope`` for
            ``gNFWSph``.

        Returns
        -------
        A ``MassBudgetResult`` whose ``derived`` dictionary has the keys:

        - ``"aperture_radius"``: the aperture radius in arc-seconds (``None`` if the aperture has
          no ``radius`` attribute);
        - ``"aperture_centre"``: the aperture centre in arc-seconds (``None`` if the aperture has
          no ``centre`` attribute);
        - ``"mass_reference"``: ``M_ref``, the reference aperture mass in convergence units;
        - ``"mass_light"``: ``M_light``, the aperture mass of the light basis at unit
          mass-to-light ratio, in convergence units;
        - ``"mass_to_light_ratio"``: the solved shared mass-to-light ratio ``Upsilon``;
        - ``"scale_radius"``: the dark profile's fixed scale radius in arc-seconds;
        - ``"kappa_s"``: the dark profile's solved ``kappa_s``;
        - ``"f_dark"``: the dark fraction of ``M_ref`` inside the aperture;
        - ``"fixed"``: a dictionary of the further fixed dark-profile parameters.

        Raises
        ------
        ValueError
            If ``M_ref`` is not positive.
        """
        mass_reference = self.aperture.mass_from(mass_obj=reference_mass, grid=grid)

        if not mass_reference > 0.0:
            raise ValueError(
                f"MassBudget.solve: the reference aperture mass is {mass_reference}, which is "
                "not positive."
            )

        mass_light = self.aperture.mass_from(
            mass_obj=light_basis.mass_basis_from(mass_to_light_ratio=1.0), grid=grid
        )

        if self.f_stellar > 0.0:
            mass_to_light_ratio = mass_to_light_ratio_for_mass_from(
                basis=light_basis,
                mass=self.f_stellar * mass_reference,
                aperture=self.aperture,
                grid=grid,
            )
            stellar = light_basis.mass_basis_from(
                mass_to_light_ratio=mass_to_light_ratio
            )
        else:
            mass_to_light_ratio = 0.0
            stellar = None

        if self.f_dark > 0.0:
            kappa_s = kappa_s_for_mass_from(
                dark_cls=dark_cls,
                mass=self.f_dark * mass_reference,
                aperture=self.aperture,
                scale_radius=scale_radius,
                centre=centre,
                grid=grid,
                **fixed,
            )
            dark = dark_cls(
                centre=centre, kappa_s=kappa_s, scale_radius=scale_radius, **fixed
            )
        else:
            kappa_s = 0.0
            dark = None

        derived = {
            "aperture_radius": getattr(self.aperture, "radius", None),
            "aperture_centre": getattr(self.aperture, "centre", None),
            "mass_reference": mass_reference,
            "mass_light": mass_light,
            "mass_to_light_ratio": mass_to_light_ratio,
            "scale_radius": scale_radius,
            "kappa_s": kappa_s,
            "f_dark": self.f_dark,
            "fixed": dict(fixed),
        }

        return MassBudgetResult(stellar=stellar, dark=dark, derived=derived)
