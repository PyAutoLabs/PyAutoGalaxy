from typing import Tuple

from autogalaxy.profiles.mass.dark.nfw_truncated import NFWTruncatedSph
from autogalaxy.profiles.mass.dark.nfw_truncated_mcr_scatter import (
    NFWTruncatedMCRScatterLudlowSph,
)

from autogalaxy.profiles.mass.dark import mcr_util


class NFWTruncatedMCRDuffySph(NFWTruncatedSph):
    """
    This function only applies for the lens configuration as follows:
    Cosmology: FlatLamdaCDM
    H0 = 70.0 km/sec/Mpc
    Omega_Lambda = 0.7
    Omega_m = 0.3
    Redshfit of Main Lens: 0.6
    Redshift of Source: 2.5
    A truncated NFW halo at z = 0.6 with tau = 2.0
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        mass_at_200: float = 1e9,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):
        """
        Input m200: The m200 of the NFW part of the corresponding tNFW part. Unit: M_sun.
        """

        self.mass_at_200 = mass_at_200

        (
            kappa_s,
            scale_radius,
            radius_at_200,
        ) = mcr_util.kappa_s_and_scale_radius_for_duffy(
            mass_at_200=mass_at_200,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )

        super().__init__(
            centre=centre,
            kappa_s=kappa_s,
            scale_radius=scale_radius,
            truncation_radius=2.0 * radius_at_200,
        )


class NFWTruncatedMCRLudlowSph(NFWTruncatedMCRScatterLudlowSph):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        mass_at_200: float = 1e9,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):
        """
        Input m200: The m200 of the NFW part of the corresponding tNFW part. Unit: M_sun.
        """

        super().__init__(
            centre=centre,
            mass_at_200=mass_at_200,
            scatter_sigma=0.0,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )
