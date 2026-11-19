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
        The elliptical Gaussian mass profile with a gradient in its mass-to-light conversion.

        Each Gaussian takes a single mass-to-light ratio set by its `sigma`, following a power law in radius:

        $\Upsilon(\sigma) = \Upsilon_{0} \left( \frac{\sigma}{R_{\rm ref}} \right)^{-\Gamma}$

        Where:

        $\Upsilon(\sigma)$ is the mass-to-light ratio of the Gaussian [dimensionless].
        $\Upsilon_{0}$ is `mass_to_light_ratio_base`, the mass-to-light ratio at the reference radius [dimensionless].
        $\sigma$ is the sigma value of the Gaussian [arc-seconds].
        $R_{\rm ref}$ is `mass_to_light_radius`, the reference radius [arc-seconds].
        $\Gamma$ is `mass_to_light_gradient` [dimensionless].

        A positive `mass_to_light_gradient` means the mass-to-light ratio falls outward: Gaussians with
        `sigma` below the reference radius carry more mass per unit light than those above it. This is the
        same convention as `SersicGradient`, so the same positive gradient makes both profiles' mass-to-light
        ratio fall with radius.

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
            The mass-to-light ratio of a Gaussian whose `sigma` equals `mass_to_light_radius`.
        mass_to_light_gradient
            The power-law slope of the mass-to-light ratio with `sigma`, M/L proportional to
            `(sigma / mass_to_light_radius) ** -mass_to_light_gradient`: positive values mean the mass-to-light
            ratio falls outward.
        mass_to_light_radius
            The reference radius at which the mass-to-light ratio equals `mass_to_light_ratio_base`.
        """

        self.mass_to_light_ratio_base = mass_to_light_ratio_base
        self.mass_to_light_gradient = mass_to_light_gradient
        self.mass_to_light_radius = mass_to_light_radius
        mass_to_light_ratio = (
            self.mass_to_light_ratio_base
            * (sigma / self.mass_to_light_radius) ** -self.mass_to_light_gradient
        )

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            sigma=sigma,
            mass_to_light_ratio=mass_to_light_ratio,
        )
