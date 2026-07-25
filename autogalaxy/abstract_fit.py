from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from autofit import ModelInstance

if TYPE_CHECKING:
    from autogalaxy.galaxy.galaxy import Galaxy

import autoarray as aa

from autogalaxy.profiles.light.linear import LightProfileLinear
from autogalaxy.profiles.basis import Basis


class AbstractFitInversion:
    def __init__(self, model_obj, settings: aa.Settings, xp=np):
        """
        An abstract fit object which fits to datasets (e.g. imaging, interferometer) inherit from.

        This object primarily inspects the `model_obj` (e.g. a galaxies object PyAutoGalaxy or tracer in PyAutoLens)
        and determines the properties used for the fit by inspecting the galaxies / light profiles in this object.

        Parameters
        ----------
        model_obj
            The object which contains the model components (e.g. light profiles, galaxies, etc) which are used to
            create the model-data that fits the data. In PyAutoGalaxy this is a list of galaxies and PyAutoLens
            it is a `Tracer`.
        settings
            Settings controlling how an inversion is fitted for example which linear algebra formalism is used.
        """
        self.model_obj = model_obj
        self.settings = settings or aa.Settings()
        self.use_jax = xp is not np

    @property
    def _xp(self):
        if self.use_jax:
            import jax.numpy as jnp

            return jnp
        return np

    @property
    def total_mappers(self) -> int:
        """
        The total number of `Mapper` objects used by the inversion in this fit.

        A mapper is created for each galaxy with a pixelization object.
        """
        # TODO : When we add regularization to basis need to change this to reflect mappers.

        return len(
            list(filter(None, self.model_obj.cls_list_from(cls=aa.Pixelization)))
        )

    @property
    def perform_inversion(self) -> bool:
        """
        Returns a bool specifying whether this fit object performs an inversion.

        This is based on whether any of the galaxies in the `model_obj` have a `Pixelization` or `LightProfileLinear`
        object, in which case an inversion is performed.

        Returns
        -------
            A bool which is True if an inversion is performed.
        """

        return self.model_obj.perform_inversion

    @property
    def sparse_operator(self) -> Optional[aa.ImagingSparseOperator]:
        """
        Only call the `sparse_operator` property of a dataset used to perform efficient linear algebra calculations if
        the Settings()` object has `use_sparse_operator=True`, to avoid unnecessary computation.

        Returns
        -------
        The sparse operator used for efficient linear algebra calculations for this inversion, if enabled.
        """
        if self.dataset.sparse_operator is not None:
            if self.total_mappers > 0:
                return self.dataset.sparse_operator

    @property
    def inversion(self) -> Optional[aa.Inversion]:
        raise NotImplementedError

    @property
    def linear_light_profile_intensity_dict(
        self,
    ) -> Optional[Dict[LightProfileLinear, float]]:
        """
        When linear light profiles are used in an inversion, their `intensity` parameter values are solved for via
        linear algebra.

        These values are contained in the `reconstruction` ndarray of the inversion, however their location in this
        ndarray depends how the inversion was performed.

        This function returns a dictionary which maps every linear light profile instance to its solved for
        `intensity` value in the inversion, so that the intensity value of every light profile can be accessed.

        Type casting is complicated by JAX. When this function is used in a JAX.jit (e.g. computed latent varialbes)
        it requires the reconstruction values to be JAX arrays, but when it is used outside of JAX certain taks
        requires the reconstruction values to be floats.

        An example of the latter is using a tracer inferred in one search to pass the solved for intensity values of
        linear light profiles to a subsequent search, for example setting up the intensities of the mass components
        of a light dark model.
        """

        if self.inversion is None:
            return None

        linear_obj_func_list = self.inversion.cls_list_from(
            cls=aa.AbstractLinearObjFuncList
        )

        linear_light_profile_intensity_dict = {}

        for linear_obj_func in linear_obj_func_list:
            reconstruction = self.inversion.reconstruction_dict[linear_obj_func]

            for i, light_profile in enumerate(linear_obj_func.light_profile_list):
                if self.use_jax:
                    linear_light_profile_intensity_dict[light_profile] = reconstruction[
                        i
                    ]
                else:
                    linear_light_profile_intensity_dict[light_profile] = float(
                        reconstruction[i]
                    )

        return linear_light_profile_intensity_dict

    def galaxy_linear_obj_data_dict_from(
        self,
        use_operated: bool = True,
    ) -> Dict[Galaxy, aa.Array2D]:
        """
        Returns a dictionary mapping every galaxy containing a linear
        object (e.g. a linear light profile / pixelization) in the `model_obj` to the `model_data` of its linear
        objects.

        The `model_data` is the `reconstructed_data` solved for in the inversion`.

        This is used to create the overall `galaxy_model_image_dict`, which maps every galaxy to its
        overall `model_data` (e.g. including the `model_data` of orindary light profiles too).

        If `use_operated=False`, the `reconstructed_data` of the inversion (e.g. an image for dataset data,
        visibilities for  interferometer data) is input in the dictionary.

        if `use_operated=True`, the `reconstructed_operated_data` of the inversion (e.g. the image for dataset data, the
        real-space image for interferometer data) is input in the dictionary.

        Parameters
        ----------
        use_operated
            Whether to use the operated (e.g PSF convolved) images of the linear objects in the dictionary, or
            the unoperated images.

        Returns
        -------
        The dictionary mapping all galaxies with linear objects to the model data / images of those linear objects
        reconstructed by the inversion.
        """
        if self.inversion is None:
            return {}

        galaxy_linear_obj_image_dict = {}

        for linear_obj in self.inversion.linear_obj_list:
            try:
                galaxy = self.inversion.linear_obj_galaxy_dict[linear_obj]
            except KeyError:
                continue

            if use_operated:
                mapped_reconstructed = (
                    self.inversion.mapped_reconstructed_operated_data_dict[linear_obj]
                )
            else:
                mapped_reconstructed = self.inversion.mapped_reconstructed_data_dict[
                    linear_obj
                ]

            if galaxy in galaxy_linear_obj_image_dict:
                galaxy_linear_obj_image_dict[galaxy] += mapped_reconstructed

            else:
                galaxy_linear_obj_image_dict.update({galaxy: mapped_reconstructed})

        return galaxy_linear_obj_image_dict

    @property
    def model_obj_linear_light_profiles_to_light_profiles(self):
        """
        The model object may contain linear light profiles, which solve for the `intensity` during the `Inversion`.

        This means they are difficult to visualize, because they do not have a valid `intensity` parameter.

        To address this, this property creates a new `model_obj` where all linear light profiles are converted to
        ordinary light profiles whose `intensity` parameters are set to the results of the Inversion.

        Returns
        -------
        A `model_obj` (E.g. galaxies or `Tracer`) where the light profile intensity values are set to the results
        of those inferred via the `Inversion`.
        """

        if self.linear_light_profile_intensity_dict is None:
            return self.model_obj

        model_instance = self.append_linear_light_profiles_to_model(
            model_instance=ModelInstance(dict(model_obj=self.model_obj))
        )

        return model_instance.model_obj

    def append_linear_light_profiles_to_model(self, model_instance):
        """
        For a model instance, this function replaces all linear light profiles with instances of their standard
        light profile counterparts.

        The `intensity` parameter of each light profile is set to the value inferred via the `Inversion`.

        Parameters
        ----------
        model_instance
            An instance of the model object (e.g. galaxies or `Tracer`) whose linear light profiles are to be
            replaced with instances of their standard light profile counterparts.

        Returns
        -------
        A model instance with all linear light profiles replaced with instances of their standard light profile
        """

        for path, instance in model_instance.path_instance_tuples_for_class(
            (LightProfileLinear, Basis)
        ):
            model_instance = model_instance.replacing_for_path(
                path,
                instance.lp_instance_from(self.linear_light_profile_intensity_dict),
            )

        return model_instance
