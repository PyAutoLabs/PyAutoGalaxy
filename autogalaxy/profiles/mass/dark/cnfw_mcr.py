from typing import Tuple

from autogalaxy.profiles.mass.dark.cnfw_mcr_scatter import (
    cNFWMCRScatterLudlow,
    cNFWMCRScatterLudlowSph,
)

class cNFWMCRLudlow(cNFWMCRScatterLudlow):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        mass_at_200: float = 1e9,
        f_c: float = 0.01,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):
        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            mass_at_200=mass_at_200,
            scatter_sigma=0.0,
            f_c=f_c,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )

class cNFWMCRLudlowSph(cNFWMCRScatterLudlowSph):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        mass_at_200: float = 1e9,
        f_c: float = 0.01,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):
        super().__init__(
            centre=centre,
            mass_at_200=mass_at_200,
            scatter_sigma=0.0,
            f_c=f_c,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )
