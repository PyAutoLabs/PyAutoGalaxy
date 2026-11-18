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
        """
        Elliptical gNFW profile whose `kappa_s` and `scale_radius` follow from `mass_at_200`
        through the Ludlow et al. 2016 mass-concentration relation.

        The 3D mass inside r200 equals `mass_at_200` for every `inner_slope`, so a prior on
        `mass_at_200` is a prior on M200. The Ludlow concentration is taken as c = r200 / r_s, so
        `scale_radius` does not depend on `inner_slope` and `inner_slope=1` gives the
        `NFWMCRLudlow` halo (see `mcr_util.kappa_s_and_scale_radius_for_ludlow_gnfw`).

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        mass_at_200
            The mass of the halo inside r200 (M_200c), in solar masses.
        redshift_object
            Lens redshift.
        redshift_source
            Source redshift.
        inner_slope
            The inner logarithmic density slope gamma of the gNFW profile (gamma < 3).
        """
        self.mass_at_200 = mass_at_200
        self.redshift_object = redshift_object
        self.redshift_source = redshift_source

        (
            kappa_s,
            scale_radius,
            radius_at_200,
        ) = mcr_util.kappa_s_and_scale_radius_for_ludlow_gnfw(
            mass_at_200=mass_at_200,
            scatter_sigma=0.0,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
            inner_slope=inner_slope,
        )

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            kappa_s=kappa_s,
            inner_slope=inner_slope,
            scale_radius=scale_radius,
        )
