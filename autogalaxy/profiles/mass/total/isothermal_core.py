from typing import Tuple

from autogalaxy.profiles.mass.total.power_law_core import PowerLawCore
from autogalaxy.profiles.mass.total.power_law_core import PowerLawCoreSph


class IsothermalCore(PowerLawCore):
    r"""Cored elliptical isothermal (SIE with core) mass profile.

    The convergence of the cored isothermal ellipsoid is:

    .. math::

        \kappa(R) = \frac{\theta_{\rm E}}{2\sqrt{R^2 + s^2}}

    where :math:`\theta_{\rm E}` is the Einstein radius, :math:`R` is the
    elliptical radius, and :math:`s` is the core radius.  In the limit
    :math:`s \to 0` this reduces to the standard SIE.

    Parameters
    ----------
    centre : (float, float)
        (y, x) arc-second coordinates of the profile centre.
    ell_comps : (float, float)
        Ellipticity components (e1, e2) of the elliptical coordinate system.
    einstein_radius : float
        Einstein radius in arcseconds.
    core_radius : float
        Core radius :math:`s` in arcseconds.

    References
    ----------
    Kormann, Schneider & Bartelmann (1994), A&A, 284, 285.
    Keeton (2001), arXiv:astro-ph/0102341.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        core_radius: float = 0.01,
    ):
        """
        Represents a cored elliptical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        einstein_radius
            The arc-second Einstein radius.
        core_radius
            The arc-second radius of the inner core.
        """
        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            einstein_radius=einstein_radius,
            slope=2.0,
            core_radius=core_radius,
        )


class IsothermalCoreSph(PowerLawCoreSph):
    r"""Cored spherical isothermal (SIS with core) mass profile.

    The convergence of the cored spherical isothermal is:

    .. math::

        \kappa(r) = \frac{\theta_{\rm E}}{2\sqrt{r^2 + s^2}}

    where :math:`\theta_{\rm E}` is the Einstein radius, :math:`r` is the
    circular projected radius, and :math:`s` is the core radius.  This is
    the spherical special case of :class:`IsothermalCore`.

    Parameters
    ----------
    centre : (float, float)
        (y, x) arc-second coordinates of the profile centre.
    einstein_radius : float
        Einstein radius in arcseconds.
    core_radius : float
        Core radius :math:`s` in arcseconds.

    References
    ----------
    Kormann, Schneider & Bartelmann (1994), A&A, 284, 285.
    Keeton (2001), arXiv:astro-ph/0102341.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        core_radius: float = 0.01,
    ):
        """
        Represents a cored spherical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius
            The arc-second Einstein radius.
        core_radius
            The arc-second radius of the inner core.
        """
        super().__init__(
            centre=centre,
            einstein_radius=einstein_radius,
            slope=2.0,
            core_radius=core_radius,
        )
