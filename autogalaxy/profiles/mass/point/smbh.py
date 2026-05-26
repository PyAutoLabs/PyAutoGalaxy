import numpy as np
from typing import Tuple

from autogalaxy.profiles.mass.point.point import PointMass


class SMBH(PointMass):
    r"""
    Supermassive black hole (SMBH) modelled as a point mass lens.

    The SMBH is represented by a :class:`PointMass` profile whose Einstein radius
    :math:`\theta_E` is derived from the physical mass :math:`M` and the critical
    surface density :math:`\Sigma_{\rm crit}` between the SMBH and the source:

    .. math::

        \theta_E = \sqrt{\frac{M}{\pi \, \Sigma_{\rm crit}}}

    The lensing potential and deflections then follow the point-mass expressions:

    .. math::

        \psi(\boldsymbol{\theta}) = \theta_E^2 \ln r, \qquad
        \boldsymbol{\alpha}(\boldsymbol{\theta}) = \frac{\theta_E^2}{r}\,\hat{r}

    This profile is used to model the gravitational influence of a central SMBH on
    lensed images passing near the nucleus.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        mass: float = 1e10,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):
        r"""
        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        mass
            The mass of the SMBH in solar masses :math:`M_\odot`.
        redshift_object
            The redshift of the SMBH (lens plane), used to convert mass to an Einstein radius.
        redshift_source
            The redshift of the lensed source galaxy, used to compute
            :math:`\Sigma_{\rm crit}` and hence the Einstein radius.
        """
        from autogalaxy.cosmology.model import Planck15

        cosmology = Planck15()

        self.mass = mass

        critical_surface_density = (
            cosmology.critical_surface_density_between_redshifts_from(
                redshift_0=redshift_object,
                redshift_1=redshift_source,
            )
        )
        mass_angular = mass / critical_surface_density
        einstein_radius = np.sqrt(mass_angular / np.pi)

        super().__init__(centre=centre, einstein_radius=einstein_radius)

        self.redshift_object = redshift_object
        self.redshift_source = redshift_source
