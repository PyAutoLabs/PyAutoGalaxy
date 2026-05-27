from typing import Tuple

import numpy as np

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile


class ExternalPotential(MassProfile):
    r"""
    Higher-order external potential extending the constant external shear.

    The ``ExternalPotential`` captures up to spin-3 contributions from line-of-sight
    mass in strong-lens models, following Powell et al. (2022) Eq. 4.  The lensing
    potential is:

    .. math::

        \psi(\boldsymbol{r}) =
            \tfrac{1}{2} r^2 (\gamma_1 \cos 2\theta + \gamma_2 \sin 2\theta)
            + \tfrac{1}{4} r^3 (\tau_1 \cos\theta + \tau_2 \sin\theta)
            + \tfrac{1}{6} r^3 (\delta_1 \cos 3\theta + \delta_2 \sin 3\theta)

    where :math:`(r, \theta)` are polar coordinates centred on ``centre``.

    - The :math:`\gamma` term is a constant external shear (spin-2); with
      :math:`\tau_i = \delta_i = 0` this reduces to :class:`ExternalShear`.
    - The :math:`\tau` term (spin-1) introduces a linear convergence gradient:
      :math:`\kappa(x, y) = \tau_1 x + \tau_2 y`.
    - The :math:`\delta` term (spin-3) is a higher-order generalised shear with zero
      convergence contribution.

    References
    ----------
    - Powell, Vegetti, McKean et al. 2022, MNRAS, 516, 1808
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        gamma_1: float = 0.0,
        gamma_2: float = 0.0,
        tau_1: float = 0.0,
        tau_2: float = 0.0,
        delta_1: float = 0.0,
        delta_2: float = 0.0,
    ):
        r"""
        A line-of-sight external potential that generalises the constant external shear used in
        strong-lens models by adding the two next-order terms from Powell et al. 2022 (Eq. 4):

        .. math::

            \psi(\mathbf{r}) =
                \tfrac{1}{2}\, r^2 \big(\gamma_1 \cos 2\theta + \gamma_2 \sin 2\theta\big)
                + \tfrac{1}{4}\, r^3 \big(\tau_1 \cos\theta + \tau_2 \sin\theta\big)
                + \tfrac{1}{6}\, r^3 \big(\delta_1 \cos 3\theta + \delta_2 \sin 3\theta\big)

        where :math:`(r, \theta)` are polar coordinates centred on the profile's ``centre``.

        Term-by-term:

        - :math:`\gamma_1, \gamma_2` — the constant external shear contribution. With
          :math:`\tau_i = \delta_i = 0` and ``centre = (0, 0)`` this reduces exactly to
          :class:`ExternalShear`.
        - :math:`\tau_1, \tau_2` — a linear gradient in the surface mass density (spin-1), giving a
          non-zero convergence :math:`\kappa(x, y) = \tau_1 x + \tau_2 y`.
        - :math:`\delta_1, \delta_2` — a higher-order spin-3 generalised-shear term.

        Unlike :class:`ExternalShear`, where the deflection field is a constant in the lens plane
        and the source position is degenerate with ``centre``, the :math:`\tau` and :math:`\delta`
        deflections have explicit radial dependence — so ``centre`` is a free parameter (typically
        tied to the primary lens centre when modelling).

        Parameters
        ----------
        centre
            The (y, x) arc-second coordinates of the profile centre.
        gamma_1, gamma_2
            The two components of the constant external shear (spin-2).
        tau_1, tau_2
            The two components of the linear surface-mass-density gradient (spin-1).
        delta_1, delta_2
            The two components of the higher-order spin-3 generalised shear.
        """

        super().__init__(centre=centre, ell_comps=(0.0, 0.0))
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.delta_1 = delta_1
        self.delta_2 = delta_2

    @staticmethod
    def _magnitude_from(c1, c2, xp=np):
        return xp.sqrt(c1 * c1 + c2 * c2)

    @staticmethod
    def _angle_from(c1, c2, harmonic: int, xp=np):
        r"""
        Return the principal angle in degrees for a harmonic of order ``harmonic``.

        - ``harmonic = 1`` -> angle in [0, 360)
        - ``harmonic = 2`` -> angle in [0, 180) (shear convention)
        - ``harmonic = 3`` -> angle in [0, 120)
        """
        angle = xp.rad2deg(xp.arctan2(c2, c1)) / harmonic
        period = 360.0 / harmonic
        return angle % period

    @classmethod
    def from_magnitudes_and_angles(
        cls,
        centre: Tuple[float, float] = (0.0, 0.0),
        gamma: float = 0.0,
        theta_gamma: float = 0.0,
        tau: float = 0.0,
        theta_tau: float = 0.0,
        delta: float = 0.0,
        theta_delta: float = 0.0,
    ):
        r"""
        Build an :class:`ExternalPotential` from per-term magnitudes and position angles, matching
        the paper-style parameterisation. Angles are in degrees, anticlockwise from the +x axis.
        """
        tg = np.deg2rad(theta_gamma)
        tt = np.deg2rad(theta_tau)
        td = np.deg2rad(theta_delta)

        gamma_1 = gamma * np.cos(2.0 * tg)
        gamma_2 = gamma * np.sin(2.0 * tg)
        tau_1 = tau * np.cos(tt)
        tau_2 = tau * np.sin(tt)
        delta_1 = delta * np.cos(3.0 * td)
        delta_2 = delta * np.sin(3.0 * td)

        return cls(
            centre=centre,
            gamma_1=gamma_1,
            gamma_2=gamma_2,
            tau_1=tau_1,
            tau_2=tau_2,
            delta_1=delta_1,
            delta_2=delta_2,
        )

    def gamma_magnitude(self, xp=np):
        return self._magnitude_from(self.gamma_1, self.gamma_2, xp=xp)

    def gamma_angle(self, xp=np):
        return self._angle_from(self.gamma_1, self.gamma_2, harmonic=2, xp=xp)

    def tau_magnitude(self, xp=np):
        return self._magnitude_from(self.tau_1, self.tau_2, xp=xp)

    def tau_angle(self, xp=np):
        return self._angle_from(self.tau_1, self.tau_2, harmonic=1, xp=xp)

    def delta_magnitude(self, xp=np):
        return self._magnitude_from(self.delta_1, self.delta_2, xp=xp)

    def delta_angle(self, xp=np):
        return self._angle_from(self.delta_1, self.delta_2, harmonic=3, xp=xp)

    def convergence_func(self, grid_radius: float, xp=np) -> float:
        return 0.0

    def average_convergence_of_1_radius(self):
        return 0.0

    @aa.decorators.to_array
    @aa.decorators.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        r"""
        Returns the convergence :math:`\kappa = \tfrac{1}{2}\nabla^2 \psi` at each grid point.

        Only the :math:`\tau` term contributes; the :math:`\gamma` (spin-2 shear) and
        :math:`\delta` (spin-3) terms are harmonic and yield zero convergence:

        .. math::

            \kappa(x, y) = \tau_1 \, x + \tau_2 \, y

        where :math:`(x, y)` are coordinates relative to ``centre``.
        """
        y = grid.array[:, 0]
        x = grid.array[:, 1]
        return self.tau_1 * x + self.tau_2 * y

    @aa.decorators.to_array
    @aa.decorators.transform
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        r"""
        Returns the lensing potential of the external potential at each grid point, following
        Powell et al. 2022 Eq. 4.
        """
        y = grid.array[:, 0]
        x = grid.array[:, 1]
        r2 = x * x + y * y
        r = xp.sqrt(r2)
        theta = xp.arctan2(y, x)

        gamma_term = 0.5 * r2 * (
            self.gamma_1 * xp.cos(2.0 * theta) + self.gamma_2 * xp.sin(2.0 * theta)
        )
        tau_term = 0.25 * r2 * r * (
            self.tau_1 * xp.cos(theta) + self.tau_2 * xp.sin(theta)
        )
        delta_term = (1.0 / 6.0) * r2 * r * (
            self.delta_1 * xp.cos(3.0 * theta) + self.delta_2 * xp.sin(3.0 * theta)
        )

        return gamma_term + tau_term + delta_term

    @aa.decorators.to_vector_yx
    @aa.decorators.transform
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        r"""
        Returns the deflection vector :math:`\boldsymbol{\alpha} = \nabla \psi` at each grid point.

        Computed in polar form (`alpha_r = d\psi/dr`, `alpha_theta = (1/r)\,d\psi/d\theta`) and
        then projected back to ``(y, x)`` Cartesian components. ``centre`` enters through the
        ``@transform`` decorator, which shifts the grid before the function body runs.
        """
        y = grid.array[:, 0]
        x = grid.array[:, 1]
        r = xp.sqrt(x * x + y * y)
        theta = xp.arctan2(y, x)
        cos_t = xp.cos(theta)
        sin_t = xp.sin(theta)
        cos_2t = xp.cos(2.0 * theta)
        sin_2t = xp.sin(2.0 * theta)
        cos_3t = xp.cos(3.0 * theta)
        sin_3t = xp.sin(3.0 * theta)

        alpha_r = (
            r * (self.gamma_1 * cos_2t + self.gamma_2 * sin_2t)
            + 0.75 * r * r * (self.tau_1 * cos_t + self.tau_2 * sin_t)
            + 0.5 * r * r * (self.delta_1 * cos_3t + self.delta_2 * sin_3t)
        )
        alpha_theta = (
            r * (-self.gamma_1 * sin_2t + self.gamma_2 * cos_2t)
            + 0.25 * r * r * (-self.tau_1 * sin_t + self.tau_2 * cos_t)
            + 0.5 * r * r * (-self.delta_1 * sin_3t + self.delta_2 * cos_3t)
        )

        alpha_y = sin_t * alpha_r + cos_t * alpha_theta
        alpha_x = cos_t * alpha_r - sin_t * alpha_theta

        return xp.vstack((alpha_y, alpha_x)).T
