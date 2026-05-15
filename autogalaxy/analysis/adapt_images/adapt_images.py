from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from autoconf import conf
from autoconf import cached_property

import autoarray as aa

if TYPE_CHECKING:
    from autogalaxy.galaxy.galaxy import Galaxy


def galaxy_name_image_dict_via_result_from(
    result, use_model_images: bool = False
) -> "AdaptImages":
    """
    Returns the adapt-images from a non-linear search result.

    For model-fitting, the adapt-images are typically setup using the maximum log likelihood model of the
    previous model-fit. This means the model-fitting is used to cleanly deblend the light of the different
    galaxies in the image (e.g. separate the lens light from the source light).

    This method uses attributes of a result (e.g. dictionary mapping galaxy instances to their model-images)
    to create the adapt-images.

    This can use either:

    - The model image of each galaxy in the best-fit model.
    - The subtracted image of each galaxy in the best-fit model, where the subtracted image is the dataset
      minus the model images of all other galaxies.

    Certain models produce galaxy-images with negative flux values (e.g. a pixelization), which can cause
    numerical issues with the adaptive schemes. To prevent this, we set a minimum flux value for each
    galaxy-image, which is a fraction of the maximum flux value of that image defined via a config file.

    Parameters
    ----------
    result
        The result of a previous model-fit, which contains the model-image of each galaxy.
    use_model_images
        If True, the model images of the galaxies are used to create the adapt images. If False, the subtracted
        images of the galaxies are used.

    Returns
    -------
    The adapt-images, which are the model-image of each galaxy inferred via the previous model-fit.
    """
    adapt_minimum_percent = conf.instance["general"]["adapt"]["adapt_minimum_percent"]

    galaxy_name_image_dict = {}

    for path, galaxy in result.path_galaxy_tuples:
        if use_model_images:
            galaxy_image = result.model_image_galaxy_dict[path]
        else:
            galaxy_image = result.subtracted_signal_to_noise_map_galaxy_dict[path]

        minimum_galaxy_value = adapt_minimum_percent * np.max(galaxy_image.array)
        galaxy_image[galaxy_image < minimum_galaxy_value] = minimum_galaxy_value

        galaxy_name_image_dict[path] = galaxy_image

    return galaxy_name_image_dict


class AdaptImages:
    def __init__(
        self,
        galaxy_image_dict: Optional[Dict[Galaxy, aa.Array2D]] = None,
        galaxy_name_image_dict: Optional[Dict[Tuple[str, ...], aa.Array2D]] = None,
        galaxy_image_plane_mesh_grid_dict: Optional[Dict[Galaxy, aa.Array2D]] = None,
        galaxy_name_image_plane_mesh_grid_dict: Optional[
            Dict[Tuple[str, ...], aa.Grid2DIrregular]
        ] = None,
        galaxy_path_list: Optional[List[str]] = None,
    ):
        """
        Contains the adapt-images which are used to make a pixelization's mesh and regularization adapt to the
        reconstructed galaxy's morphology.

        Pixelization image-mesh objects (e.g. `KMeans`, `Hilbert`) adapt the distribution of pixels to the observed
        image's brightness and therefore to the reconstructed source's morphology.

        Certain regularization schemes (e.g. `Adapt`) adapt their regularization coefficients to the
        reconstructed source's morphology.

        These adaptive schemes use "adapt-images", which are images of each galaxy (e.g. the lens and source of a
        strong lens) estimated via an earlier model-fit.

        The adapt-images are stored as the model-image of each galaxy in a model (e.g. the lens and source for a
        strong lens). They are stored as a dictionary mapping each instance of the galaxy to its model-image.

        For model-fitting, the galaxy instances are updated for every iteration of the non-linear search. This means
        an `AdaptImages` instance cannot be passed directly to an `Analysis` class, as the galaxy instances need to be
        updated for every iteration of the non-linear search.

        A dictionary mapping the path name of each galaxy (e.g. "galaxies.lens") to its model-image is therefore used
        which is called inside the `log_likelihood_function` o map the model-image of each galaxy to the galaxy
        instance of that iteration's specific model.

        Parameters
        ----------
        galaxy_image_dict
            A dictionary associating each galaxy instance to an image of only that galaxy (e.g. for a strong lens
            one entry will map an instance of the source galaxy entry to an image of the lensed source.
        galaxy_name_image_dict
            A dictionary associating each galaxy path name (e.g. "galaxies.source") to an image of only that
            galaxy (e.g. for a strong lens the `source` entry is an image of the lensed source, without the lens light).
        """

        self.galaxy_image_dict = galaxy_image_dict
        self.galaxy_name_image_dict = galaxy_name_image_dict

        self.galaxy_image_plane_mesh_grid_dict = galaxy_image_plane_mesh_grid_dict
        self.galaxy_name_image_plane_mesh_grid_dict = (
            galaxy_name_image_plane_mesh_grid_dict
        )

        # Parallel to the analysis-time galaxies list (as built by
        # ``Analysis.galaxies_via_instance_from``). Populated by
        # ``updated_via_instance_from`` and used by ``image_for_galaxy`` to
        # recover the galaxy's path-tuple key after a JAX unflatten has produced
        # fresh ``Galaxy`` objects whose hashes no longer match
        # ``galaxy_image_dict`` keys.
        self.galaxy_path_list = galaxy_path_list

    @property
    def mask(self) -> aa.Mask2D:
        """
        The mask of the adapt images.
        """
        try:
            return list(self.galaxy_image_dict.values())[0].mask
        except AttributeError:
            return list(self.galaxy_name_image_dict.values())[0].mask

    @cached_property
    def model_image(self) -> aa.Array2D:
        """
        The model-image is the sum of all individual galaxy images in the image dictionary.

        This is computed by summing the model-image of each individual adapt galaxy contained in the dictionary.
        """
        adapt_model_image = aa.Array2D(
            values=np.zeros(self.mask.pixels_in_mask),
            mask=self.mask,
        )

        try:
            for path in self.galaxy_image_dict.keys():
                adapt_model_image += self.galaxy_image_dict[path]
        except AttributeError:
            for path in self.galaxy_name_image_dict.keys():
                adapt_model_image += self.galaxy_name_image_dict[path]

        return adapt_model_image

    def updated_via_instance_from(
        self,
        instance,
        dataset_model: Optional["aa.DatasetModel"] = None,
        mask=None,
        galaxies: Optional[List["Galaxy"]] = None,
        xp=np,
    ) -> "AdaptImages":
        """
        Returns adapt-images which have been updated to map galaxy instances instead of galaxy names.

        For model-fitting, the galaxy instances are updated for every iteration of the non-linear search. This means
        an `AdaptImages` instance cannot be passed directly to an `Analysis` class, as the galaxy instances need to be
        updated for every iteration of the non-linear search.

        A dictionary mapping the path name of each galaxy (e.g. "galaxies.lens") to its model-image is therefore used
        which is called inside the `log_likelihood_function` o map the model-image of each galaxy to the galaxy
        instance of that iteration's specific model.

        This function is also called when loading an `AdaptImages` instance from a PyAutoFit database, as the
        galaxy instances are also created on-fly from the database. Database images do not have a mask, so it is
        also applied to the adapt images on-the-fly during database loading.

        When a ``dataset_model`` is supplied with a non-trivial ``grid_offset`` or ``grid_rotation_angle``, the cached
        ``galaxy_name_image_plane_mesh_grid_dict`` entries are transformed into the same frame as the dataset's
        image-plane grid (which ``FitDataset.grids`` rotates by the same amount). Without this transform the cached
        mesh and the data grid would sit in different frames, producing a misaligned source reconstruction.

        Parameters
        ----------
        instance
            The instance of the model-fit (e.g. in a non-linear search) which is used to update the adapt images.
        dataset_model
            The dataset model whose ``grid_offset`` and ``grid_rotation_angle`` are applied to cached mesh grids so
            they remain consistent with the rotated/shifted data grid produced by ``FitDataset.grids``. If ``None``,
            the cached mesh grids are passed through unchanged.
        mask
            A mask which can be applied to the adapt images, which is used when setting up the adaptive images
            via the aggregator and autofit database tools.
        galaxies
            Optional list of galaxies in the order used by the calling ``Analysis`` (i.e. the list passed to
            ``FitImaging`` / ``Tracer``). When provided, a parallel ``galaxy_path_list`` is populated so that
            ``image_for_galaxy`` can recover the path-tuple key for each galaxy after JAX has unflattened the
            galaxy instances into fresh objects. When ``None`` the path list is populated in ``path_instance_tuples_for_class``
            order, which matches ``Analysis.galaxies_via_instance_from`` for the common case (no
            ``extra_galaxies`` / ``scaling_galaxies``).
        xp
            Array backend (``numpy`` or ``jax.numpy``) used when transforming cached mesh grids.

        Returns
        -------

        """
        from autogalaxy.galaxy.galaxy import Galaxy

        path_by_id = {
            id(galaxy): str(galaxy_name)
            for galaxy_name, galaxy in instance.path_instance_tuples_for_class(Galaxy)
        }

        galaxy_image_dict = None

        if self.galaxy_name_image_dict is not None:

            galaxy_image_dict = {}

            for galaxy_name, galaxy in instance.path_instance_tuples_for_class(Galaxy):
                galaxy_name = str(galaxy_name)

                if galaxy_name in self.galaxy_name_image_dict:
                    galaxy_image_dict[galaxy] = self.galaxy_name_image_dict[galaxy_name]

            if mask is not None:
                for key, image in galaxy_image_dict.items():
                    galaxy_image_dict[key] = aa.Array2D(values=image, mask=mask)

        galaxy_image_plane_mesh_grid_dict = None

        if self.galaxy_name_image_plane_mesh_grid_dict is not None:

            galaxy_image_plane_mesh_grid_dict = {}

            for galaxy_name, galaxy in instance.path_instance_tuples_for_class(Galaxy):
                galaxy_name = str(galaxy_name)

                if galaxy_name in self.galaxy_name_image_plane_mesh_grid_dict:
                    cached_mesh = self.galaxy_name_image_plane_mesh_grid_dict[galaxy_name]
                    if dataset_model is not None:
                        cached_mesh = cached_mesh.subtracted_and_rotated_from(
                            offset=dataset_model.grid_offset,
                            angle=dataset_model.grid_rotation_angle,
                            xp=xp,
                        )
                    galaxy_image_plane_mesh_grid_dict[galaxy] = cached_mesh

        if galaxies is not None:
            galaxy_path_list = [path_by_id.get(id(g)) for g in galaxies]
        else:
            galaxy_path_list = [
                str(galaxy_name)
                for galaxy_name, _ in instance.path_instance_tuples_for_class(Galaxy)
            ]

        return AdaptImages(
            galaxy_image_dict=galaxy_image_dict,
            galaxy_image_plane_mesh_grid_dict=galaxy_image_plane_mesh_grid_dict,
            galaxy_name_image_dict=self.galaxy_name_image_dict,
            galaxy_name_image_plane_mesh_grid_dict=self.galaxy_name_image_plane_mesh_grid_dict,
            galaxy_path_list=galaxy_path_list,
        )

    def image_for_galaxy(
        self, galaxy: "Galaxy", galaxies: Optional[List["Galaxy"]] = None
    ) -> Optional[aa.Array2D]:
        """
        Return the adapt image for ``galaxy``, robust to JAX ``jit`` boundaries.

        ``galaxy_image_dict`` is keyed by the trace-time ``Galaxy`` instances. After ``jax.jit`` has flattened
        and unflattened a ``FitImaging``, the galaxies inside it are fresh Python objects whose ``__hash__``
        differs from the trace-time keys, so a direct lookup misses. This helper falls back to the path-tuple
        keyed ``galaxy_name_image_dict`` using ``galaxy_path_list`` to map the post-unflatten galaxy back to
        its trace-time path.

        Returns ``None`` when no adapt image is associated with the galaxy.
        """
        try:
            return self.galaxy_image_dict[galaxy]
        except (AttributeError, KeyError, TypeError):
            pass

        path = self._path_for_galaxy(galaxy, galaxies)
        if path is None or self.galaxy_name_image_dict is None:
            return None
        return self.galaxy_name_image_dict.get(path)

    def image_plane_mesh_grid_for_galaxy(
        self, galaxy: "Galaxy", galaxies: Optional[List["Galaxy"]] = None
    ) -> Optional[aa.Grid2DIrregular]:
        """
        Return the image-plane mesh grid for ``galaxy``, robust to JAX ``jit`` boundaries.

        Companion to :meth:`image_for_galaxy` for ``galaxy_image_plane_mesh_grid_dict`` /
        ``galaxy_name_image_plane_mesh_grid_dict``.
        """
        try:
            return self.galaxy_image_plane_mesh_grid_dict[galaxy]
        except (AttributeError, KeyError, TypeError):
            pass

        path = self._path_for_galaxy(galaxy, galaxies)
        if path is None or self.galaxy_name_image_plane_mesh_grid_dict is None:
            return None
        return self.galaxy_name_image_plane_mesh_grid_dict.get(path)

    def _path_for_galaxy(
        self, galaxy: "Galaxy", galaxies: Optional[List["Galaxy"]]
    ) -> Optional[str]:
        if not self.galaxy_path_list or galaxies is None:
            return None
        for index, candidate in enumerate(galaxies):
            if candidate is galaxy:
                if index < len(self.galaxy_path_list):
                    return self.galaxy_path_list[index]
                return None
        return None
