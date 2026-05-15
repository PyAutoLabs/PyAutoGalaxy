from typing import Tuple
import numpy as np
from autogalaxy.profiles.mass.dark.nfw import NFWSph
from autogalaxy.profiles.mass.dark.nfw import NFW

from autogalaxy.profiles.mass.dark import mcr_util
from autogalaxy import convert as al_convert
import jax.numpy as jnp

class NFWMCRScatterLudlowSph(NFWSph):
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

        super().__init__(centre=centre, kappa_s=kappa_s, scale_radius=scale_radius)

class NFWMCRScatterLudlowSphMultiBand(NFWSph):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        mass_at_200: float = 1e9,
        scatter_sigma: float = 0.0,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
        shift_centre_0: float = 0.0,
        shift_centre_1: float = 0.0,
        shift_angle: float = 0.0,
        rot_ref_centre_0: float = 0.0,
        rot_ref_centre_1: float = 0.0
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

        centre_0_shifted = centre[0] + shift_centre_0
        centre_1_shifted = centre[1] + shift_centre_1

        rot_ref_centre_0_shifted = rot_ref_centre_0 + shift_centre_0
        rot_ref_centre_1_shifted = rot_ref_centre_1 + shift_centre_1

        centre_0_rotated, centre_1_rotated = al_convert.rotate_centre(
            centre_0=centre_0_shifted,
            centre_1=centre_1_shifted,
            ref_centre_0=rot_ref_centre_0_shifted,
            ref_centre_1=rot_ref_centre_1_shifted,
            angle=shift_angle,
            xp=jnp
        )

        centre = (centre_0_rotated, centre_1_rotated)

        super().__init__(centre=centre, kappa_s=kappa_s, scale_radius=scale_radius)

        self.shift_centre_0 = shift_centre_0
        self.shift_centre_1 = shift_centre_1
        self.shift_angle = shift_angle

class NFWMCRScatterLudlow(NFW):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
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

        """
        #Make correction that Andrew proposed
        fac = np.sqrt(ell_comps[1] ** 2 + ell_comps[0] ** 2)
        if fac > 0.999:
            fac = 0.999  # avoid unphysical solution
        # if fac > 1: print('unphysical e1,e2')
        axis_ratio = (1 - fac) / (1 + fac)
        scale_radius = scale_radius / np.sqrt(axis_ratio)
        
        print('With Correction')
        """
        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            kappa_s=kappa_s,
            scale_radius=scale_radius,
        )
