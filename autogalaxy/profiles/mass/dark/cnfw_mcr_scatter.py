from typing import Tuple

from autogalaxy.profiles.mass.dark.cnfw import (cNFW, cNFWSph)

from autogalaxy.profiles.mass.dark import mcr_util


class cNFWMCRScatterLudlow(cNFW):
    def __init__(
            self,
            centre: Tuple[float, float] = (0.0, 0.0),
            ell_comps: Tuple[float, float] = (0.0, 0.0),
            mass_at_200: float = 1e9,
            scatter_sigma: float = 0.0,
            f_c: float = 0.01,
            redshift_object: float = 0.5,
            redshift_source: float = 1.0,
    ):
        self.mass_at_200 = mass_at_200
        self.scatter_sigma = scatter_sigma
        self.f_c = f_c
        self.redshift_object = redshift_object
        self.redshift_source = redshift_source

        (
            kappa_s,
            scale_radius,
            core_radius,
            radius_at_200,
        ) = mcr_util.kappa_s_scale_radius_and_core_radius_for_ludlow(
            mass_at_200=mass_at_200,
            scatter_sigma=scatter_sigma,
            f_c=f_c,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            kappa_s=kappa_s,
            scale_radius=scale_radius,
            core_radius=core_radius,
        )

class cNFWMCRScatterLudlowSph(cNFWSph):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        mass_at_200: float = 1e9,
        scatter_sigma: float = 0.0,
        f_c: float = 0.01,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):
        self.mass_at_200 = mass_at_200
        self.scatter_sigma = scatter_sigma
        self.f_c = f_c
        self.redshift_object = redshift_object
        self.redshift_source = redshift_source

        (
            kappa_s,
            scale_radius,
            core_radius,
            radius_at_200,
        ) = mcr_util.kappa_s_scale_radius_and_core_radius_for_ludlow(
            mass_at_200=mass_at_200,
            scatter_sigma=scatter_sigma,
            f_c=f_c,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )

        super().__init__(
            centre=centre,
            kappa_s=kappa_s,
            scale_radius=scale_radius,
            core_radius=core_radius,
        )
