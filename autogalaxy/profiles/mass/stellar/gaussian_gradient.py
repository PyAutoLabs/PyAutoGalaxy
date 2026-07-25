from typing import Tuple

from autogalaxy.profiles.mass.stellar.gaussian import Gaussian


class GaussianGradient(Gaussian):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        sigma: float = 1.0,
        mass_to_light_ratio_base: float = 1.0,
        mass_to_light_gradient: float = 0.0,
        mass_to_light_radius: float = 1.0,
    ):
        r"""
        The elliptical Gaussian light profile with a gradient in its mass to light conversion.

        $\Psi (r) = \Psi_{o} \frac{(\sigma + 0.01)}{R_{ref}}^{\Tau}$

        Where:

        $\Psi (r)$ is the 1D convergence profile of the Gaussian [dimensionless].
        $\Psi_{o}$ is the base mass-to-light ratio of the profile [dimensionless].
        $\sigma$ is the sigma value of the Gaussian [arc-seconds].
        $r$ is the radius from the centre of the profile [arc-seconds].
        $R_{ref}$ is the reference radius where the mass-to-light ratio is equal to $\Psi_{o} [arc-seconds].
        $\Tau$ is the mass-to-light gradient of the profile [dimensionless].

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        sigma
            The sigma value of the Gaussian.
        mass_to_light_ratio_base
            The base mass-to-light ratio of the profile, which is the mass-to-light ratio of the Gaussian before it
            is scaled by values that adjust its mass-to-light ratio based on the reference radius and gradient.
        mass_to_light_gradient
            The mass-to-light radial gradient of the profile, whereby positive values means there is more mass
            per unit light within the reference radius.
        mass_to_light_radius
            The radius where the mass-to-light ratio is equal to the base mass-to-light ratio, such that there will be
            more of less mass per unit light within this radius depending on the mass-to-light gradient.
        """

        self.mass_to_light_ratio_base = mass_to_light_ratio_base
        self.mass_to_light_gradient = mass_to_light_gradient
        self.mass_to_light_radius = mass_to_light_radius
        mass_to_light_ratio = (
            self.mass_to_light_ratio_base
            * ((sigma + 0.01) / self.mass_to_light_radius)
            ** self.mass_to_light_gradient
        )

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            sigma=sigma,
            mass_to_light_ratio=mass_to_light_ratio,
        )
