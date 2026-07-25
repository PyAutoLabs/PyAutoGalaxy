from typing import Tuple

from autogalaxy.profiles.mass.dark.nfw_truncated import NFWTruncatedSph

from autogalaxy.profiles.mass.dark import mcr_util


class NFWTruncatedMCRScatterLudlowSph(NFWTruncatedSph):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        mass_at_200: float = 1e9,
        scatter_sigma: float = 0.0,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):
        self.mass_at_200 = mass_at_200
        self.scatter_sigma = scatter_sigma
        self.redshift_object = redshift_object
        self.redshift_source = redshift_source

        (
            kappa_s,
            scale_radius,
            radius_at_200,
        ) = mcr_util.kappa_s_and_scale_radius_for_ludlow(
            mass_at_200=mass_at_200,
            scatter_sigma=scatter_sigma,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )

        super().__init__(
            centre=centre,
            kappa_s=kappa_s,
            scale_radius=scale_radius,
            truncation_radius=2.0 * radius_at_200,
        )
