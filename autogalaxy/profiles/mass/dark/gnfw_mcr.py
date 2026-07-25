from typing import Tuple

from autogalaxy.profiles.mass.dark.gnfw import gNFW

from autogalaxy.profiles.mass.dark import mcr_util


class gNFWMCRLudlow(gNFW):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        mass_at_200: float = 1e9,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
        inner_slope: float = 1.0,
    ):
        self.mass_at_200 = mass_at_200
        self.redshift_object = redshift_object
        self.redshift_source = redshift_source

        (
            kappa_s,
            scale_radius,
            radius_at_200,
        ) = mcr_util.kappa_s_and_scale_radius_for_ludlow(
            mass_at_200=mass_at_200,
            scatter_sigma=0.0,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            kappa_s=kappa_s,
            inner_slope=inner_slope,
            scale_radius=scale_radius,
        )
