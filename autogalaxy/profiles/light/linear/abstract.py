"""
Abstract base classes for linear light profiles in **PyAutoGalaxy**.

A linear light profile behaves identically to a standard light profile (e.g. `Sersic`, `Exponential`,
`Gaussian`) except that its `intensity` is not a free parameter — instead, it is solved for analytically via
a linear inversion during every likelihood evaluation.

This reduces the number of non-linear parameters in a model fit by one per linear profile, which can
substantially speed up non-linear sampling without any loss of accuracy.

The `LightProfileLinear` class is the abstract base from which all linear light profile variants inherit.
The `LightProfileLinearObjFuncList` subclass additionally supports regularization, allowing the solved
intensities to be penalized by a smoothness prior.
"""
import inspect
import itertools
import numpy as np
from typing import Dict, List, Optional

from autoconf import cached_property
import autoarray as aa

from autogalaxy.profiles.light.operated.abstract import (
    LightProfileOperated,
)

from autogalaxy.profiles.light.abstract import LightProfile

from autogalaxy import exc


class LightProfileLinear(LightProfile):
    """
    A linear light profile, which is a light profile whose `intensity` value is solved for via linear algebra
    using an inversion.

    Every standard light profile (e.g. `Serisic`, `Exponential`, etc.) has a linear light profile equivalent that
    behaves identically, except that the `intensity` is not explicitly set by the user but instead is inferred via
    a linear inversion.

    This means that when a linear light profile is used to perform a model-fit, it reduces the number of free parameters
    in the model-fit by 1, as the `intensity` parameter is inferred via the inversion.

    The `LightProfileLinear` class is an abstract class that should be used to make specific linear light profiles.
    This inheritance is used throughout the `galaxy.py`, `tracer.py` and other modules to extract linear light
    profiles when performing a fit to data.
    """

    _pytree_token_counter = itertools.count()
    __exclude_identifier_fields__ = ("pytree_token",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pytree_token = next(LightProfileLinear._pytree_token_counter)

    def __hash__(self):
        return self.pytree_token

    def __eq__(self, other):
        return (
            isinstance(other, LightProfileLinear)
            and self.pytree_token == other.pytree_token
        )

    def __getstate__(self):
        # pytree_token is a process-local counter increment used only as a
        # stable hash/eq identity for jax.jit flatten/unflatten round-trips.
        # Persisting it would copy a number from one process's counter into
        # another's — the JAX path uses register_model's attr_const which
        # reads vars(self) directly and is unaffected by __getstate__.
        return {k: v for k, v in self.__dict__.items() if k != "pytree_token"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "pytree_token" not in state:
            self.pytree_token = next(LightProfileLinear._pytree_token_counter)

    @property
    def regularization(self):
        return None

    @property
    def _intensity(self):
        return 1.0

    @property
    def standard_lp_parent(self):
        """
        Returns the first parent class of the linear light profile which is not a linear light profile itself.

        The functions below (e.g. `lp_instance_from`) are used to convert a linear light profile to its parent
        standard light profile, where the input `intensity` is the value solved for in the linear inversion.

        This function is used to determine the parent class of the linear light profile that is used to create the
        standard light profile.

        This function also maps linear light and mass profiles to their standard light and mass profile counterparts.
        This specific mapping is required because their inheritence structure is different to the other light profiles.
        In the future, the ugly dictionary used to do this mapping should be removed for better code.

        Returns
        -------
        The parent class of the linear light profile that is not a linear light profile itself.
        """

        from autogalaxy.profiles import light_linear_and_mass_profiles as lmp_linear
        from autogalaxy.profiles import light_and_mass_profiles as lmp

        if isinstance(self, lmp.LightMassProfile):
            lmp_mapping_dict = {
                lmp_linear.Gaussian: lmp.Gaussian,
                lmp_linear.GaussianGradient: lmp.GaussianGradient,
                lmp_linear.SersicSph: lmp.SersicSph,
                lmp_linear.Sersic: lmp.Sersic,
                lmp_linear.SersicGradient: lmp.SersicGradient,
                lmp_linear.SersicGradientSph: lmp.SersicGradientSph,
                lmp_linear.ExponentialSph: lmp.ExponentialSph,
                lmp_linear.Exponential: lmp.Exponential,
                lmp_linear.ExponentialGradient: lmp.ExponentialGradient,
                lmp_linear.ExponentialGradientSph: lmp.ExponentialGradientSph,
                lmp_linear.DevVaucouleursSph: lmp.DevVaucouleursSph,
                lmp_linear.DevVaucouleurs: lmp.DevVaucouleurs,
                lmp_linear.SersicCoreSph: lmp.SersicCoreSph,
                lmp_linear.SersicCore: lmp.SersicCore,
            }

            return lmp_mapping_dict[self.__class__]

        bases = self.__class__.__bases__

        if self.__class__.__name__.endswith("Sph") or isinstance(
            self, LightProfileOperated
        ):
            bases = bases[0].__bases__

        for cls in bases:
            if not issubclass(cls, LightProfileLinear):
                return cls

    def parameters_dict_from(self, intensity: float) -> Dict[str, float]:
        """
        Returns a dictionary of the parameters of the linear light profile with the `intensity` added.

        This `intenisty` will likely have come from the value inferred via the linear inversion.

        Parameters
        ----------
        intensity
            Overall intensity normalisation of the not linear light profile that is created (units are dimensionless
            and derived from the data the light profile's image is compared too, which is expected to be electrons
            per second).
        """
        parameters_dict = vars(self)
        args = inspect.getfullargspec(self.standard_lp_parent).args
        args.remove("self")

        parameters_dict = {key: parameters_dict[key] for key in args}
        parameters_dict["intensity"] = intensity

        return parameters_dict

    def lp_instance_from(
        self, linear_light_profile_intensity_dict: Dict
    ) -> LightProfile:
        """
        Creates an instance of a linear light profile using its parent normal light profile (e.g. the non linear
        variant which has an `intensity` parameter).

        The `intensity` value of the profile created is passed into this function and used.

        Parameters
        ----------
        intensity
            Overall intensity normalisation of the not linear light profile that is created (units are dimensionless
            and derived from the data the light profile's image is compared too, which is expected to be electrons
            per second).
        """
        intensity = linear_light_profile_intensity_dict[self]
        parameters_dict = self.parameters_dict_from(intensity=intensity)

        return self.standard_lp_parent(**parameters_dict)


class LightProfileLinearObjFuncList(aa.AbstractLinearObjFuncList):
    def __init__(
        self,
        grid: aa.type.Grid1D2DLike,
        blurring_grid: aa.type.Grid1D2DLike,
        psf: Optional[aa.Convolver],
        light_profile_list: List[LightProfileLinear],
        regularization=Optional[aa.reg.Regularization],
        settings=aa.Settings(),
        xp=np,
    ):
        """
        A list of linear light profiles which fits a dataset via linear algebra using the images of each linear light
        profile and the dataset.

        By inheriting from `AbstractLinearObjFuncList` this tells the parent project PyAutoArray that this class is a
        linear object that can be used in a linear inversion via its `Inversion` object. This means the linear light
        profiles are used to perform a linear inversion to fit the data, which solves for the `intensity` of each light profile.

        By overwriting the `mapping_matrix` function with a method that fills in each column of the matrix with the
        image of each linear light profile, this is then passed through the `inversion` package to perform the
        linear inversion. The API is identical to `Mapper` objects such that linear functions can easily be combined
        with mappers.

        The autoarray inversion module treats separely the `mapping_matrix` and the `operated_mapping_matrix`,
        where the latter is the mapping matrix after all additional operations (e.g. convolution, FFT) have been applied.
        The `operated_mapping_matrix` is used to perform the linear inversion. This class defines a function
        `operated_mapping_matrix_override` which overrides the `operated_mapping_matrix` calculation in autoarray
        to account for PSF convolution of the light profile images.

        This is a slightly weird design, is is required because although the autoarray inversion module has
        functionality to convolve ta `mapping_matrix` with a PSF in order to compute the `operated_mapping_matrix`,
        it does not have functionality to account for the fact that the light profile images have flux outside the
        masked region which is blurred into the masked region.

        For example, in `PyAutoGalaxy` and `PyAutoLens` the light of galaxies is represented using `LightProfile`
        objects, which describe the surface brightness of a galaxy as a function. By grouping lists of these linear
        light profiles into this class, autoarray solves for their intensity's linearly.

        Parameters
        ----------
        grid
            The (y,x) grid aligned with the data that is fitted, which is used to evaluate the image of each light
            profile.
        blurring_grid
            The blurring grid is all points whose light is outside the data's mask but close enough to the mask that
            it may be blurred into the mask. This is also used when evaluating the image of each light profile.
        psf
            The psf used to blur the light profile images of each light profile, the output of which
            makes up the columns of the `operated_mapping matrix`.
        light_profile_list
            A list of the linear light profiles that are used to fit the data via linear algebra.
        regularization
            The regularization scheme which may be applied to this linear object in order to smooth its solution.
        """
        for light_profile in light_profile_list:
            if not isinstance(light_profile, LightProfileLinear):
                raise exc.ProfileException(
                    """
                    A light profile that is not a LightProfileLinear object has been input into the
                    LightProfileLinearObjFuncList object.

                    Only children of the LightProfileLinear class can be used in a linear inversion.
                    """
                )

        super().__init__(
            grid=grid, regularization=regularization, settings=settings, xp=xp
        )

        self.blurring_grid = blurring_grid
        self.psf = psf
        self.light_profile_list = light_profile_list

    @property
    def params(self) -> int:
        """
        The `params` property is used by the autoarray inversion module to track how many parameters are solved
        for in the linear inversion.

        This is used to define the dimensions of certain matrices used in the linear algebra, for example the
        `curvature_matrix` and `regularization_matrix`.

        For this class, the number of parameters is equal to the number of light profiles in the list,
        as each light profile contributes a single parameter to the inversion, its `intensity`.

        Returns
        -------
        The number of parameters that are solved for in the linear inversion.
        """
        return len(self.light_profile_list)

    @property
    def pixels_in_mask(self) -> int:
        """
        The number of pixels in the mask of the grid, which is used to define the dimensions of the `mapping_matrix`
        used in the linear inversion.

        This function has two ways to return the number of pixels in the mask, depending on whether the inversion
        is being performed on an imaging or interferometer dataset.

        Returns
        -------
        The number of pixels in the mask of the grid.
        """
        return self.grid.mask.pixels_in_mask

    @property
    def mapping_matrix(self) -> np.ndarray:
        """
        Returns the `mapping_matrix` of the linear light profiles, where each column is the image of each light profile
        evaluated on the grid before operations are applied (e.g. convolution, FFT).

        This function iterates over each light profile in the list and evaluates its image on the grid, storing this
        image in the `mapping_matrix`.

        The standard behaviour is that this is only used for linear light profiles computed when fitting interferometer
        data, where the `mapping_matrix` is used to compute the `operated_mapping_matrix` via a fast Fourier transform.
        This is because the FFT does not require light outside the masked region to computed, unlike for imaging
        data which omits this function and uses `operated_mapping_matrix_override` instead.

        Returns
        -------
        The `mapping_matrix` of the linear light profiles.
        """

        image_2d_list = []

        for pixel, light_profile in enumerate(self.light_profile_list):

            image_2d = light_profile.image_2d_from(grid=self.grid, xp=self._xp).slim

            image_2d_list.append(image_2d.array)

        return self._xp.stack(image_2d_list, axis=1)

    @cached_property
    def operated_mapping_matrix_override(self) -> Optional[np.ndarray]:
        """
        The inversion object takes the `mapping_matrix` of each linear object and combines it with the PSF
        operator to perform a 2D convolution and compute the `operated_mapping_matrix`.

        If this property is overwritten this operation is not performed, with the `operated_mapping_matrix` output this
        property automatically used instead.

        This is used for a linear light profile because the in-built mapping matrix convolution does not account for
        how light profile images have flux outside the masked region which is blurred into the masked region. This
        flux is outside the region that defines the `mapping_matrix` and thus this override is required to properly
        incorporate it.

        Returns
        -------
        A blurred mapping matrix of dimensions (total_mask_pixels, 1) which overrides the mapping matrix calculations
        performed in the linear equation solvers.
        """

        if isinstance(self.light_profile_list[0], LightProfileOperated):
            return self.mapping_matrix

        if self.psf.convolve_over_sample_size > 1:
            # Evaluate each profile on the over-sampled coordinates (per-pixel
            # sub-block order, unbinned — the oversampled Convolver's input format)
            # so convolution runs at the fine resolution, mirroring
            # OperateImage.blurred_image_2d_from.
            evaluation_grid = self.grid.over_sampled
            evaluation_blurring_grid = self.blurring_grid.over_sampled
            convolution_mask = self.grid.mask
        else:
            evaluation_grid = self.grid
            evaluation_blurring_grid = self.blurring_grid
            convolution_mask = None

        blurred_image_2d_list = []

        for pixel, light_profile in enumerate(self.light_profile_list):
            image_2d = light_profile.image_2d_from(grid=evaluation_grid, xp=self._xp)

            blurring_image_2d = light_profile.image_2d_from(
                grid=evaluation_blurring_grid, xp=self._xp
            )

            blurred_image_2d = self.psf.convolved_image_from(
                image=image_2d,
                blurring_image=blurring_image_2d,
                use_mixed_precision=self.settings.use_mixed_precision,
                mask=convolution_mask,
                xp=self._xp,
            )

            blurred_image_2d_list.append(blurred_image_2d.array)

        return self._xp.stack(blurred_image_2d_list, axis=1)
