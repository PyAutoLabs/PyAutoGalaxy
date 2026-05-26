import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.profiles.mass.point.smbh import SMBH


class SMBHBinary(MassProfile):
    r"""
    Binary supermassive black hole (SMBH) system modelled as two point masses.

    The binary is represented by two :class:`SMBH` mass profiles placed symmetrically
    about the system ``centre``.  The total mass :math:`M_{\rm tot}` and mass ratio
    :math:`q_m` determine the individual masses:

    .. math::

        M_0 = M_{\rm tot} \frac{q_m}{1 + q_m}, \qquad
        M_1 = M_{\rm tot} \frac{1}{1 + q_m}

    Each component follows the point-mass lensing law:

    .. math::

        \boldsymbol{\alpha}_i(\boldsymbol{\theta}) =
            \frac{\theta_{E,i}^2}{|\boldsymbol{\theta} - \boldsymbol{\theta}_i|}\,
            \hat{r}_i

    The total deflection, convergence, and potential are the sum of the two individual
    :class:`SMBH` contributions.

    This profile is used to model the lensing effect of SMBH binaries expected to
    form following galaxy mergers, whose gravitational influence can perturb the
    positions of images near the nucleus.
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        separation: float = 1.0,
        angle_binary: float = 0.0,
        mass: float = 1e10,
        mass_ratio: float = 1.0,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):
        r"""
        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the binary mid-point.
        separation
            The angular separation between the two SMBHs in arcseconds.
        angle_binary
            The orientation angle of the binary axis relative to the positive
            x-axis (degrees, anticlockwise).
        mass
            The total mass of the binary system in solar masses :math:`M_\odot`.
        mass_ratio
            The mass ratio :math:`q_m = M_0 / M_1` (:math:`q_m \geq 1` by convention).
        redshift_object
            The redshift of the SMBH binary (lens plane).
        redshift_source
            The redshift of the lensed source galaxy.
        """

        self.separation = separation
        self.angle_binary = angle_binary
        self.mass = mass
        self.mass_ratio = mass_ratio
        self.redshift_object = redshift_object
        self.redshift_source = redshift_source

        x_0 = centre[1] + (self.separation / 2.0) * np.cos(self.angle_binary_radians)
        y_0 = centre[0] + (self.separation / 2.0) * np.sin(self.angle_binary_radians)

        if mass_ratio >= 1.0:
            mass_0 = mass * (mass_ratio / mass)
        else:
            mass_0 = mass - mass * ((1.0 / mass_ratio) / mass)

        self.smbh_0 = SMBH(
            centre=(y_0, x_0),
            mass=mass_0,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )

        x_1 = centre[1] + (self.separation / 2.0) * np.cos(
            self.angle_binary_radians - np.pi
        )
        y_1 = centre[0] + (self.separation / 2.0) * np.sin(
            self.angle_binary_radians - np.pi
        )

        mass_1 = mass - mass_0

        self.smbh_1 = SMBH(
            centre=(y_1, x_1),
            mass=mass_1,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )

        super().__init__(centre=centre, ell_comps=(0.0, 0.0))

    @property
    def angle_binary_radians(self) -> float:
        """
        The angle between the two SMBHs in radians.
        """
        return self.angle_binary * np.pi / 180.0

    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Returns the two dimensional projected convergence on a grid of (y,x) arc-second coordinates.

        The convergence is computed as the sum of the convergence of the two individual `SMBH` profiles in the binary.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.
        """
        return self.smbh_0.convergence_2d_from(
            grid=grid, xp=xp, **kwargs
        ) + self.smbh_1.convergence_2d_from(grid=grid, xp=xp, **kwargs)

    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Returns the two dimensional projected potential on a grid of (y,x) arc-second coordinates.

        The potential is computed as the sum of the potential of the two individual `SMBH` profiles in the binary.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the potential is computed on.
        """
        return self.smbh_0.potential_2d_from(
            grid=grid, **kwargs
        ) + self.smbh_1.potential_2d_from(grid=grid)

    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Returns the two dimensional deflection angles on a grid of (y,x) arc-second coordinates.

        The deflection angles are computed as the sum of the convergence of the two individual `SMBH` profiles in the
        binary.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        return self.smbh_0.deflections_yx_2d_from(
            grid=grid
        ) + self.smbh_1.deflections_yx_2d_from(grid=grid, xp=xp, **kwargs)
