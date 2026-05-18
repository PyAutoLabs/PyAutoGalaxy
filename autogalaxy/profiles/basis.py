"""
The `Basis` class groups multiple light or mass profiles into a single object that acts like a single profile.

A basis is typically used for multi-component decompositions of a galaxy's light or mass distribution — for
example a multi-Gaussian expansion (MGE), which represents a galaxy as a sum of many Gaussian profiles, or
a shapelet decomposition. Each component of the basis captures a distinct spatial scale of the galaxy.

When linear light profiles are used in a basis, their individual intensities are solved simultaneously via a
linear inversion (a single matrix solve), making the inference highly efficient regardless of how many basis
components are included.
"""
import numpy as np
from typing import Dict, List, Optional, Union

import autoarray as aa

from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.mass.abstract.abstract import MassProfile

from autogalaxy.profiles.light import linear as lp_linear


class Basis(LightProfile, MassProfile):
    def __init__(
        self,
        profile_list: List[Union[LightProfile, MassProfile]],
        regularization: Optional[aa.AbstractRegularization] = None,
    ):
        """
        A basis is a collection of multiple light or mass profiles that are used to represent the light or mass
        distribution of a galaxy.

        A basis is typically used to decompose a galaxy's light or mass distribution into many profiles, each of
        which fit a small part of the overall light or mass distribution.

        For example, a common basis uses of order 10-100 Gaussian light or mass profiles, where each Gaussian
        represents a small part of the overall light or mass distribution of a galaxy. By decomposing the light or mass
        distribution into many profiles, more detailed structures can be captured and fitted that ordinary profiles
        would struggle to capture.

        This contrasts most standard light profiles (e.g. a Sersic) or mass profiles (e.g. an Isothermal) which
        represent the entire light or mass distribution of a galaxy.

        Parameters
        ----------
        profile_list
            The light or mass profiles that make up the basis.
        regularization
            The regularization scheme applied to the basis, which is used to regularize the solution to the linear
            inversion that fits the basis to the data.
        """
        super().__init__(
            centre=profile_list[0].centre,
            ell_comps=profile_list[0].ell_comps,
        )

        self.profile_list = profile_list
        self.regularization = regularization

    @property
    def light_profile_list(self) -> List[LightProfile]:
        """
        Returns a list of all light profiles in the `Basis` object.

        This is used for computing light profile quantities of each individual light profile in the `Basis` object and
        then summing them to get the overall quantity (e.g. the image, surface brightness, etc.).

        Returns
        -------
            The list of light profiles in the `Basis` object.
        """
        return aa.util.misc.cls_list_from(values=self.profile_list, cls=LightProfile)

    @property
    def mass_profile_list(self) -> List[MassProfile]:
        """
        Returns a list of all mass profiles in the `Basis` object.

        This is used for computing mass profile quantities of each individual mass profile in the `Basis` object and
        then summing them to get the overall quantity (e.g. the convergence, potential, etc.).

        Returns
        -------
            The list of mass profiles in the `Basis` object.
        """
        return aa.util.misc.cls_list_from(values=self.profile_list, cls=MassProfile)

    def image_2d_from(
        self,
        grid: aa.type.Grid2DLike,
        xp=np,
        operated_only: Optional[bool] = None,
        **kwargs,
    ) -> aa.Array2D:
        """
        Returns the summed image of all light profiles in the basis from a 2D grid of Cartesian (y,x) coordinates.

        Normal steps in the calculation of an image, like shifting the input grid to the profile's centre, rotating
        it to its position angle, and checking if its already operated on are all handled internally by
        each profiles `image_2d_from` method when it is called.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.
        operated_only
            By default, the returned list contains all light profile images (irrespective of whether they have been
            operated on or not). If this input is included as a bool, only images which are or are not already
            operated are included in the list, with the images of other light profiles created as a numpy array of
            zeros.

        Returns
        -------
        The image of the light profiles in the basis summed together.
        """
        return sum(
            self.image_2d_list_from(grid=grid, xp=xp, operated_only=operated_only)
        )

    def image_2d_list_from(
        self, grid: aa.type.Grid2DLike, xp=np, operated_only: Optional[bool] = None
    ) -> List[aa.Array2D]:
        """
        Returns each image of each light profiles in the basis as a list, from a 2D grid of Cartesian (y,x) coordinates.

        Normal steps in the calculation of an image, like shifting the input grid to the profile's centre, rotating
        it to its position angle, and checking if its already operated on are all handled internally by
        each profiles `image_2d_from` method when it is called.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.
        operated_only
            By default, the returned list contains all light profile images (irrespective of whether they have been
            operated on or not). If this input is included as a bool, only images which are or are not already
            operated are included in the list, with the images of other light profiles created as a numpy array of
            zeros.

        Returns
        -------
        The image of the light profiles in the basis summed together.
        """
        return [
            (
                light_profile.image_2d_from(
                    grid=grid, xp=xp, operated_only=operated_only
                )
                if not isinstance(light_profile, lp_linear.LightProfileLinear)
                else aa.Array2D(
                    values=xp.zeros((grid.shape[0],)), mask=grid.mask
                )
            )
            for light_profile in self.light_profile_list
        ]

    def convergence_2d_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ) -> aa.Array2D:
        """
        Returns the summed convergence of all mass profiles in the basis from a 2D grid of Cartesian (y,x) coordinates.

        Normal steps in the calculation of a convergence, like shifting the input grid to the profile's centre and
        rotating it to its position angle are all handled internally by each profile's `convergence_2d_from` method
        when it is called.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        The convergence of the mass profiles in the basis summed together.
        """
        if len(self.mass_profile_list) > 0:
            return sum(
                [
                    mass.convergence_2d_from(grid=grid, xp=xp)
                    for mass in self.profile_list
                ]
            )
        return xp.zeros((grid.shape[0],))

    def potential_2d_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ) -> aa.Array2D:
        """
        Returns the summed potential of all mass profiles in the basis from a 2D grid of Cartesian (y,x) coordinates.

        Normal steps in the calculation of a potential, like shifting the input grid to the profile's centre and
        rotating it to its position angle are all handled internally by each profile's `potential_2d_from` method
        when it is called.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        The potential of the mass profiles in the basis summed together.
        """
        if len(self.mass_profile_list) > 0:
            return sum(
                [mass.potential_2d_from(grid=grid, xp=xp) for mass in self.profile_list]
            )
        return xp.zeros((grid.shape[0],))

    def deflections_yx_2d_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ) -> aa.Array2D:
        """
        Returns the summed deflections of all mass profiles in the basis from a 2D grid of Cartesian (y,x) coordinates.

        Normal steps in the calculation of a deflections, like shifting the input grid to the profile's centre and
        rotating it to its position angle are all handled internally by each profile's `deflections_2d_from` method
        when it is called.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        The deflections of the mass profiles in the basis summed together.
        """
        if len(self.mass_profile_list) > 0:
            return sum(
                [
                    mass.deflections_yx_2d_from(grid=grid, xp=xp)
                    for mass in self.profile_list
                ]
            )
        return xp.zeros((grid.shape[0], 2))

    def lp_instance_from(self, linear_light_profile_intensity_dict: Dict):
        light_profile_list = []

        for light_profile in self.profile_list:
            if isinstance(light_profile, lp_linear.LightProfileLinear):
                light_profile = light_profile.lp_instance_from(
                    linear_light_profile_intensity_dict=linear_light_profile_intensity_dict
                )

            light_profile_list.append(light_profile)

        return Basis(
            profile_list=light_profile_list, regularization=self.regularization
        )
