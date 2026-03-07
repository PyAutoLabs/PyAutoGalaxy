from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Type, Union

from autoconf import cached_property

import autoarray as aa

from autoarray.inversion.inversion.factory import inversion_from
from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages
from autogalaxy.profiles.light.linear import (
    LightProfileLinearObjFuncList,
)
from autogalaxy.profiles.basis import Basis
from autogalaxy.profiles.light.linear import LightProfileLinear
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.galaxy.galaxies import Galaxies


class AbstractToInversion:
    def __init__(
        self,
        dataset: Optional[Union[aa.Imaging, aa.Interferometer, aa.DatasetInterface]],
        adapt_images: Optional[AdaptImages] = None,
        settings: aa.Settings = None,
        xp=np,
    ):
        """
        Abstract class which interfaces a dataset and input modeling object (e.g. galaxies, a tracer) with the
        inversion module. to setup a linear algebra calculation.

        The galaxies may contain linear light profiles whose `intensity` values are solved for via linear algebra in
        order to best-fit the data. In this case, this class extracts the linear light profiles, of all galaxies,
        computes their images and passes them to the `inversion` module such that they become the  `mapping_matrix`
        used in the linear algebra calculation.

        The galaxies may also contain pixelizations, which use a mesh (e.g. a Voronoi mesh) and regularization scheme
        to reconstruct the galaxy's light. This class extracts all pixelizations and uses them to set up `Mapper`
        objects which pair the dataset and pixelization to again set up the appropriate `mapping_matrix` and
        other linear algebra matrices (e.g. the `regularization_matrix`).

        This class does not perform the inversion or compute any of the linear algebra matrices itself. Instead,
        it acts as an interface between the dataset and galaxies and the inversion module, extracting the
        necessary information from galaxies and passing it to the inversion module.

        The modeling object may also contain standard light profiles which have an input `intensity` which is not
        solved for via linear algebra. These profiles should have already been evaluated and subtracted from the
        dataset before the inversion is performed. This is how an inversion is set up in the fit
        modules (e.g. `FitImaging`).

        Parameters
        ----------
        dataset
            The dataset containing the data which the inversion is performed on.
        adapt_images
            Images which certain pixelizations use to adapt their properties to the dataset, for example congregating
            the pixelization's pixels to the brightest regions of the image.
        settings
            The settings of the inversion, which controls how the linear algebra calculation is performed.
        """
        if dataset is not None:
            if dataset.noise_covariance_matrix is not None:
                raise aa.exc.InversionException(
                    """
                    You cannot perform an inversion (e.g. use a linear light profile or pixelization) 
                    if the dataset has a `noise_covariance_matrix`.
                    
                    This is because the linear algebra implementation is only valid under the assumption 
                    of independent gaussian noise.
                    """
                )

        self.dataset = dataset

        self.adapt_images = adapt_images

        self.settings = settings or aa.Settings()

        self.use_jax = xp is not np

    @property
    def _xp(self):
        if self.use_jax:
            import jax.numpy as jnp

            return jnp
        return np

    @property
    def psf(self) -> Optional[aa.Convolver]:
        """
        Returns the PSF of the imaging dataset, if the inversion is performed on an imaging dataset.

        The `GalaxiesToInversion` class acts as an interface between the dataset and inversion module for
        both imaging and interferometer datasets. Only imaging datasets have a PSF, thus this property
        ensures that for an interferometer dataset code which references a PSF does not raise an error.

        Returns
        -------
        The psf of the imaging dataset, if it is an imaging dataset.
        """
        try:
            return self.dataset.psf
        except AttributeError:
            return None

    @property
    def transformer(self) -> Optional[Union[aa.TransformerNUFFT, aa.TransformerDFT]]:
        """
        Returns the transformer of the interferometer dataset, if the inversion is performed on an interferometer
        dataset.

        The `GalaxiesToInversion` class acts as an interface between the dataset and inversion module for
        both imaging and interferometer datasets. Only interferometer datasets have a transformer, thus this property
        ensures that for an imaging dataset code which references a transformer does not raise an error.

        Returns
        -------
        The transformer of the interferometer dataset, if it is an interferometer dataset.
        """
        try:
            return self.dataset.transformer
        except AttributeError:
            return None

    @property
    def border_relocator(self) -> Optional[aa.BorderRelocator]:
        """
        Returns the border relocator, which relocates pixels from the border of the inversion to the edge of the
        inversion, which is used to prevent edge effects in the reconstruction.

        A full description of the border relocator is given in the `BorderRelocator` class in PyAutoArray.

        Border relocation is only used if the `use_border_relocator` attribute is True in the `Settings` object.
        """
        if self.settings.use_border_relocator:
            return self.dataset.grids.border_relocator

    def cls_light_profile_func_list_galaxy_dict_from(
        self, cls: Type
    ) -> Dict[LightProfileLinearObjFuncList, Galaxy]:
        raise NotImplementedError

    @cached_property
    def lp_linear_func_list_galaxy_dict(
        self,
    ) -> Dict[LightProfileLinearObjFuncList, Galaxy]:
        raise NotImplementedError

    @cached_property
    def mapper_galaxy_dict(self):
        raise NotImplementedError

    @cached_property
    def linear_obj_galaxy_dict(
        self,
    ) -> Dict[Union[LightProfileLinearObjFuncList, aa.Mapper], Galaxy]:
        """
        Returns a dictionary associating every linear object (e.g. a linear light profile or mapper) with the galaxy
        it belongs to.

        This object is primarily used by fit objects (e.g. `FitImaging`, `FitInterferometer`) after the
        inversion has been performed, to extract the individual reconstruct images of each galaxy.

        For example, if three galaxies are fitted via an inversion, we need to be able to extract the individual
        linear profiles from the inversion and associate them with the correct galaxy. This dictionary provides
        this association.

        Returns
        -------
        The dictionary associating linear objects with galaxies.
        """
        lp_linear_func_galaxy_dict = self.lp_linear_func_list_galaxy_dict

        mapper_galaxy_dict = self.mapper_galaxy_dict

        return {**lp_linear_func_galaxy_dict, **mapper_galaxy_dict}

    @cached_property
    def linear_obj_list(self) -> List[aa.LinearObj]:
        """
        Returns a list of linear objects (e.g. linear light profiles or mappers) which are used to perform the
        inversion.

        This list is passed directly to the inversion module, and thus acts as a the final interface between the
        galaxies and the inversion module.

        Returns
        -------
        The list of linear objects used to perform the inversion.
        """
        return list(self.linear_obj_galaxy_dict.keys())


class GalaxiesToInversion(AbstractToInversion):
    def __init__(
        self,
        dataset: Optional[Union[aa.Imaging, aa.Interferometer, aa.DatasetInterface]],
        galaxies: List[Galaxy],
        adapt_images: Optional[AdaptImages] = None,
        settings: aa.Settings = None,
        xp=np,
    ):
        """
        Interfaces a dataset and input list of galaxies with the inversion module. to setup a
        linear algebra calculation.

        The galaxies may contain linear light profiles whose `intensity` values are solved for via linear algebra in
        order to best-fit the data. In this case, this class extracts the linear light profiles, of all galaxies,
        computes their images and passes them to the `inversion` module such that they become the  `mapping_matrix`
        used in the linear algebra calculation.

        The galaxies may also contain pixelizations, which use a mesh (e.g. a Voronoi mesh) and regularization scheme
        to reconstruct the galaxy's light. This class extracts all pixelizations and uses them to set up `Mapper`
        objects which pair the dataset and pixelization to again set up the appropriate `mapping_matrix` and
        other linear algebra matrices (e.g. the `regularization_matrix`).

        This class does not perform the inversion or compute any of the linear algebra matrices itself. Instead,
        it acts as an interface between the dataset and galaxies and the inversion module, extracting the
        necessary information from galaxies and passing it to the inversion module.

        The galaxies may also contain standard light profiles which have an input `intensity` which is not solved for
        via linear algebra. These profiles should have already been evaluated and subtracted from the dataset before
        the inversion is performed. This is how an inversion is set up in the fit modules (e.g. `FitImaging`).

        Parameters
        ----------
        dataset
            The dataset containing the data which the inversion is performed on.
        galaxies
            The list of galaxies which are fitted to the dataset via the inversion.
        adapt_images
            Images which certain pixelizations use to adapt their properties to the dataset, for example congregating
            the pixelization's pixels to the brightest regions of the image.
        settings
            The settings of the inversion, which controls how the linear algebra calculation is performed.
        """
        self.galaxies = Galaxies(galaxies)

        super().__init__(
            dataset=dataset,
            adapt_images=adapt_images,
            settings=settings,
            xp=xp,
        )

    @property
    def has_mapper(self):
        if self.galaxies.has(cls=aa.Pixelization):
            return True

    def cls_light_profile_func_list_galaxy_dict_from(
        self, cls: Type
    ) -> Dict[LightProfileLinearObjFuncList, Galaxy]:
        """
        Returns a dictionary associating each list of linear light profiles with the galaxy they belong to.

        This function iterates over all galaxies and their light profiles, extracting their linear light profiles and
        for each galaxy grouping them into a `LightProfileLinearObjFuncList` object, which is associated with the
        galaxy via the dictionary.

        This `LightProfileLinearObjFuncList` object contains the attributes (e.g. the data `grid`, `light_profiles`)
        and functionality (e.g. a `mapping_matrix` method) that are required to perform the inversion. It is
        in this method that the `image_2d_from` method of each light profile is used to compute the linear algebra
        matrices.

        Special behaviour is implemented for the case where a galaxy has a `Basis` object, which contains a list
        of light profiles grouped into one object (e.g. when performing a multi Gaussian expansion where 30+ linear
        light profile Gaussians are often used). In this case, the function extracts all linear light profiles from
        the `Basis` object and associates them with the galaxy.

        It is expected that only two inputs are passed into the `cls` parameter, either `LightProfileLinear` or `Basis`.
        However, a specific linear light profile class (e.g. `Sersic`) could be input if extracting a specific type of
        light profile is desired.

        There is a noteable reason why a different `LightProfileLinearObjFuncList` object is created for each
        galaxy. In the project PyAutoLens, an inversion is performed on a strong lens, consisting of a lens galaxy
        and source galaxy whose grids are different (the source-plane grid is deflected by the lens galaxy).

        In this case, because each `LightProfileLinearObjFuncList` object is associated with a different galaxy
        it is also associated with a different grid. Users of PyAutoLens should therefore be aware that in
        the `lens.to_inversion` module multiple `GalaxiesToInversion` objects are used to setup the inversion in this
        way.

        Parameters
        ----------
        cls
            The class of the light profiles which are extracted from the galaxies, which is
            typically `LightProfileLinear` or `Basis`.

        Returns
        -------
        A dictionary associating each list of linear light profiles with the galaxy they belong to.
        """
        if not self.galaxies.has(cls=cls):
            return {}

        lp_linear_func_galaxy_dict = {}

        for galaxy in self.galaxies:
            if galaxy.has(cls=cls):
                for light_profile in galaxy.cls_list_from(cls=cls):
                    if isinstance(light_profile, LightProfileLinear):
                        light_profile_list = [light_profile]
                    else:
                        light_profile_list = light_profile.light_profile_list
                        light_profile_list = [
                            light_profile
                            for light_profile in light_profile_list
                            if isinstance(light_profile, LightProfileLinear)
                        ]

                    if len(light_profile_list) > 0:
                        lp_linear_func = LightProfileLinearObjFuncList(
                            grid=self.dataset.grids.lp,
                            blurring_grid=self.dataset.grids.blurring,
                            psf=self.dataset.psf,
                            light_profile_list=light_profile_list,
                            regularization=light_profile.regularization,
                            settings=self.settings,
                            xp=self._xp,
                        )

                        lp_linear_func_galaxy_dict[lp_linear_func] = galaxy

        return lp_linear_func_galaxy_dict

    @cached_property
    def lp_linear_func_list_galaxy_dict(
        self,
    ) -> Dict[LightProfileLinearObjFuncList, Galaxy]:
        """
        Returns a dictionary associating each list of linear light profiles with the galaxy they belong to.

        You should first refer to the docstring of the `cls_light_profile_func_list_galaxy_dict_from` method for a
        description of this method.

        In brief, this method iterates over all galaxies and their light profiles, extracting their linear light profiles
        and for each galaxy grouping them into a `LightProfileLinearObjFuncList` object, which is associated with the
        galaxy via the dictionary. It also extracts linear light profiles from `Basis` objects and makes this
        associated.

        The `LightProfileLinearObjFuncList` object contains the attributes (e.g. the data `grid`, `light_profiles`)
        and functionality (e.g. a `mapping_matrix` method) that are required to perform the inversion.

        This function first creates a dictionary of linear light profiles associated with each galaxy, and then
        does the same for all `Basis` objects. The two dictionaries are then combined and returned.

        In the project PyAutoLens, this function is overwritten in order to fully account how ray tracing
        changes the images of each galaxy light profile. Users of PyAutoLens should therefore be aware that in
        the `lens.to_inversion` module this function behaves differently.

        Returns
        -------
        A dictionary associating each list of linear light profiles and basis objects with the galaxy they belong to.
        """
        lp_linear_light_profile_func_list_galaxy_dict = (
            self.cls_light_profile_func_list_galaxy_dict_from(cls=LightProfileLinear)
        )

        lp_basis_func_list_galaxy_dict = (
            self.cls_light_profile_func_list_galaxy_dict_from(cls=Basis)
        )

        return {
            **lp_linear_light_profile_func_list_galaxy_dict,
            **lp_basis_func_list_galaxy_dict,
        }

    @cached_property
    def image_plane_mesh_grid_list(
        self,
    ) -> Optional[List[aa.Grid2DIrregular]]:
        """
        Returns a list of image-plane mesh-grids, which are image-plane grids defining the centres of the pixels of
        the pixelization's mesh (e.g. the centres of Voronoi pixels).

        The `image_mesh` attribute of the pixelization object defines whether the centre of each mesh pixel are
        determined in the image-plane. When this is the case, the pixelization therefore has an image-plane mesh-grid,
        which needs to be computed before the inversion is performed.

        This function iterates over all galaxies with pixelizations, determines which pixelizations have an
        `image_mesh` and for these pixelizations computes the image-plane mesh-grid.

        It returns a list of all image-plane mesh-grids, which in the functions `mapper_from` and `mapper_galaxy_dict`
        are grouped into a `Mapper` object with other information required to perform the inversion using the
        pixelization.

        The order of this list is not important, because the `linear_obj_galaxy_dict` function associates each
        mapper object (and therefore image-plane mesh-grid) with the galaxy it belongs to and is therefore used
        elsewhere in the code (e.g. the fit module) to match inversion results to galaxies.

        Certain image meshes adapt their pixels to the dataset, for example congregating the pixels to the brightest
        regions of the image. This requires that `adapt_images` are used when setting up the image-plane mesh-grid.
        This function uses the `adapt_images` attribute of the `GalaxiesToInversion` object pass these images and
        raise an error if they are not present.

        Returns
        -------
        A list of image-plane mesh-grids, one for each pixelization with an image mesh.
        """
        if not self.galaxies.galaxy_has_cls(cls=aa.Pixelization):
            return None

        image_plane_mesh_grid_list = []

        for galaxy in self.galaxies.galaxies_with_cls_list_from(cls=aa.Pixelization):
            try:
                image_plane_mesh_grid = (
                    self.adapt_images.galaxy_image_plane_mesh_grid_dict[galaxy]
                )
            except (AttributeError, KeyError, TypeError):
                image_plane_mesh_grid = None

            image_plane_mesh_grid_list.append(image_plane_mesh_grid)

        return image_plane_mesh_grid_list

    def mapper_from(
        self,
        mesh: aa.AbstractMesh,
        regularization: aa.AbstractRegularization,
        source_plane_mesh_grid: aa.Grid2DIrregular,
        source_plane_data_grid: aa.Grid2D,
        adapt_galaxy_image: aa.Array2D,
        image_plane_mesh_grid: Optional[aa.Grid2DIrregular] = None,
    ) -> aa.Mapper:
        """
        Returns a `Mapper` object from the attributes required to create one, which are extracted and computed
        from the dataset and galaxies.

        The `Mapper` object is used by a pixelization to perform an inversion. It maps pixels from the dataset's
        `data` to the pixels of the `mesh` which reconstruct the data via the inversion. These mappings are used to
        construct the `mapping_matrix` and other linear algebra matrices used in the inversion.

        This function is quite complex due to requirements from the child project PyAutoLens. In PyAutoLens, the
        `Mapper` object has grids corresponding to both the image-plane (e.g. the pixels of the dataset) and the
        source-plane (e.g. after gravitational lensing). There are also separate grids for the data and mesh pixels.
        In total, this means there are four grids: `image_plane_data_grid`, `image_plane_mesh_grid`,
        `source_plane_data_grid` and `source_plane_mesh_grid`. Lensing calculations are performed before these
        grids are passed  to this function.

        In PyAutoGalaxy, there is no lensing and therefore the `image_plane_data_grid` and `source_plane_data_grid`
        are identical, as are the `image_plane_mesh_grid` and `source_plane_mesh_grid`. This function therefore has
        an overly complex API, as it is designed to support PyAutoLens's use-cases.

        Parameters
        ----------
        mesh
            The mesh of the pixelization, which defines the pixels used to reconstruct the data (e.g. `Voronoi`).
        regularization
            The regularization scheme used to regularize the mesh pixel's reconstructed fluxes.
        source_plane_mesh_grid
            The mesh-grid of the source-plane which reconstructs the data (e.g. the centre of the `Voronoi` cells
            after lensing). In PyAutoGalaxy, this is identical to the `image_plane_mesh_grid`.
        source_plane_data_grid
            The data-grid of the source-plane, which are the ray-traced coordinates of the image-plane pixels
            that align with the image data. In PyAutoGalaxy, this is identical to the `image_plane_data_grid`.
        adapt_galaxy_image
            Images which certain pixelizations use to adapt their properties to the dataset, for example congregating
            the pixelization's pixels to the brightest regions of the image.
        image_plane_mesh_grid
            The mesh-grid of the image-plane, which are the centres of the pixels of the dataset. This is only required
            if the pixelization has an `image_mesh` attribute.

        Returns
        -------
        A `Mapper` object which maps the dataset's data to the pixelization's mesh.
        """
        interpolator = mesh.interpolator_from(
            border_relocator=self.border_relocator,
            source_plane_data_grid=source_plane_data_grid,
            source_plane_mesh_grid=source_plane_mesh_grid,
            adapt_data=adapt_galaxy_image,
            xp=self._xp,
        )

        return aa.Mapper(
            interpolator=interpolator,
            image_plane_mesh_grid=image_plane_mesh_grid,
            regularization=regularization,
            settings=self.settings,
            xp=self._xp,
        )

    @cached_property
    def mapper_galaxy_dict(self) -> Dict[aa.Mapper, Galaxy]:
        """
        Returns a dictionary associating each `Mapper` object with the galaxy it belongs to.

        The docstring of the function `mapper_from` describes the `Mapper` object in detail, and is used
        in this function to create the `Mapper` objects which are associated with the galaxies.

        In brief, the `Mappers` are used by pixelizations to perform an inversion. They map pixels from the dataset's
        `data` to the pixels of the `mesh` which reconstruct the data via the inversion. These mappings are used to
        construct the `mapping_matrix` and other linear algebra matrices used in the inversion.

        This function essentially finds all galaxies with pixelizations, performs all necessary calculations to
        set up the `Mapper` objects (e.g. compute the `image_plane_mesh_grid`), and then associates each `Mapper`
        with the galaxy it belongs to.

        Returns
        -------
        A dictionary associating each `Mapper` object with the galaxy it belongs to.
        """
        if not self.galaxies.galaxy_has_cls(cls=aa.Pixelization):
            return {}

        mesh_grid_list = self.image_plane_mesh_grid_list

        mapper_galaxy_dict = {}

        pixelization_list = []

        galaxies_with_pixelization_list = self.galaxies.galaxies_with_cls_list_from(
            cls=aa.Pixelization
        )

        for pix in self.galaxies.cls_list_from(cls=aa.Pixelization):
            pixelization_list.append(pix)

        for mapper_index in range(len(mesh_grid_list)):
            galaxy = galaxies_with_pixelization_list[mapper_index]

            try:
                adapt_galaxy_image = self.adapt_images.galaxy_image_dict[galaxy]
            except (AttributeError, KeyError):
                adapt_galaxy_image = None

            mapper = self.mapper_from(
                mesh=pixelization_list[mapper_index].mesh,
                regularization=pixelization_list[mapper_index].regularization,
                source_plane_data_grid=self.dataset.grids.pixelization,
                source_plane_mesh_grid=mesh_grid_list[mapper_index],
                adapt_galaxy_image=adapt_galaxy_image,
                image_plane_mesh_grid=mesh_grid_list[mapper_index],
            )

            mapper_galaxy_dict[mapper] = galaxy

        return mapper_galaxy_dict

    @property
    def inversion(self) -> aa.AbstractInversion:
        """
        Returns an inversion object from the dataset, galaxies and inversion settings.

        The inversion uses all linear light profiles and pixelizations in the galaxies to fit the data.

        It solves for the linear light profile intensities and pixelization mesh pixel values via linear algebra,
        finding the solution which best fits the data after regularization is applied.

        The `GalaxiesToInversion` object acts as an interface between the dataset and galaxies and the inversion module,
        with many of its functions required to set up the inputs to the inversion object, primarily
        the `linear_obj_list` and `linear_obj_galaxy_dict` properties.

        Returns
        -------
        The inversion object which fits the dataset using the galaxies.
        """
        inversion = inversion_from(
            dataset=self.dataset,
            linear_obj_list=self.linear_obj_list,
            settings=self.settings,
            xp=self._xp,
        )

        inversion.linear_obj_galaxy_dict = self.linear_obj_galaxy_dict

        return inversion
