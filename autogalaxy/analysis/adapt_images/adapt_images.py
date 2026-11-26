from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from autonerves import conf
from autonerves import cached_property

import autoarray as aa

if TYPE_CHECKING:
    from autogalaxy.galaxy.galaxy import Galaxy


_GALAXY_IMAGES_CACHE_VERSION_KEY = "CACHEVER"
_GALAXY_IMAGES_CACHE_VERSION = 1


def _galaxy_images_cache_path(result, use_model_images: bool):
    """
    The on-disk cache file for the raw per-galaxy images of a result, inside the
    result's own ``files/`` folder, or ``None`` when the result has no on-disk
    output (e.g. ``NullPaths``) and therefore cannot cache.
    """
    from pathlib import Path

    paths = getattr(result, "paths", None)
    files_path = getattr(paths, "_files_path", None)
    if files_path is None or not Path(files_path).is_dir():
        return None
    name = "galaxy_images_model" if use_model_images else "galaxy_images_snr"
    return Path(files_path) / f"{name}.fits"


def _masks_are_equal(mask_0, mask_1) -> bool:
    """
    Returns whether two masks describe the same set of unmasked pixels on the same
    geometry, comparing shape, pixel scales, origin and the mask array itself.

    Used to decide whether an on-disk adapt-image cache was written under the mask
    the caller is now fitting with.
    """
    if mask_0.shape_native != mask_1.shape_native:
        return False
    if not np.allclose(mask_0.pixel_scales, mask_1.pixel_scales):
        return False
    if not np.allclose(mask_0.origin, mask_1.origin):
        return False
    return bool(np.array_equal(np.asarray(mask_0), np.asarray(mask_1)))


def _result_has_mixed_light(result) -> bool:
    """Check whether earlier image dictionaries could omit a galaxy's ordinary light."""
    from autogalaxy.galaxy.galaxies import Galaxies
    from autogalaxy.interferometer.fit_interferometer import _has_light_profile_non_linear

    return any(
        galaxy is not None
        and Galaxies(galaxies=[galaxy]).perform_inversion
        and _has_light_profile_non_linear([galaxy])
        for _, galaxy in result.path_galaxy_tuples
    )


def _galaxy_image_dict_from_cache(cache_path, mask=None, result=None) -> Optional[Dict]:
    """
    Load the raw (pre minimum-percent clip) per-galaxy image dictionary from a
    result's cache file, or ``None`` when the file does not exist (the first
    arrival at this result computes and writes it).

    The FITS layout mirrors the ``adapt_images.fits`` artifact the aggregator
    reads (``agg_util.adapt_images_from``): HDU 0 is the mask (header carries
    pixel scales and origin), HDU 1+ are one image per galaxy with the galaxy
    path as ``EXTNAME``.

    ``None`` is also returned when ``mask`` is supplied and does not match the
    mask the cache was written under, which makes the mismatch a cache miss so
    the caller recomputes and overwrites the stale file. The cache lives in a
    directory keyed by the search identifier, which encodes the model and the
    search but **not** the dataset — so a run whose dataset shaping changed while
    the model stayed identical lands on the previous run's cache. Without this
    check the stale images load silently and surface much later as an
    ``IndexError`` deep in the inversion, when adapt data sized for the old mask
    is indexed by the new mask's slim indexes (PyAutoGalaxy#516).

    Parameters
    ----------
    cache_path
        The cache file to load, or ``None`` when the result cannot cache.
    mask
        The mask the caller expects the cached images to be defined on. When
        ``None`` no mask validation is performed.
    result
        The result used to invalidate older caches when a galaxy has both
        ordinary light and a linear reconstruction.
    """
    from astropy.io import fits as astropy_fits

    from autoarray.mask.mask_2d import Mask2DKeys
    from autonerves.fitsable import ndarray_via_hdu_from

    if cache_path is None or not cache_path.exists():
        return None

    with astropy_fits.open(cache_path) as hdu_list:
        header = hdu_list[0].header
        if (
            header.get(_GALAXY_IMAGES_CACHE_VERSION_KEY, 0) < _GALAXY_IMAGES_CACHE_VERSION
            and result is not None
            and _result_has_mixed_light(result)
        ):
            return None

        pixel_scales = (
            header[Mask2DKeys.PIXSCAY.value],
            header[Mask2DKeys.PIXSCAX.value],
        )
        origin = (
            header[Mask2DKeys.ORIGINY.value],
            header[Mask2DKeys.ORIGINX.value],
        )
        cached_mask = aa.Mask2D(
            mask=ndarray_via_hdu_from(hdu_list[0]),
            pixel_scales=pixel_scales,
            origin=origin,
        )

        if mask is not None and not _masks_are_equal(cached_mask, mask):
            return None

        galaxy_name_image_dict = {}
        for hdu in hdu_list[1:]:
            image = aa.Array2D.no_mask(
                values=ndarray_via_hdu_from(hdu),
                pixel_scales=cached_mask.pixel_scales,
                origin=cached_mask.origin,
            )
            galaxy_name_image_dict[hdu.header["EXTNAME"].lower()] = image.apply_mask(
                mask=cached_mask
            )

    return galaxy_name_image_dict


def _galaxy_image_dict_to_cache(cache_path, galaxy_name_image_dict: Dict, paths):
    """
    Persist the raw per-galaxy image dictionary to the result's cache file, in
    the same FITS layout ``_galaxy_image_dict_from_cache`` reads, and preserve
    it in the search's zip archive so later resumes keep it.
    """
    from autonerves.fitsable import hdu_list_for_output_from

    image_list = [
        galaxy_name_image_dict[name].native_for_fits
        for name in galaxy_name_image_dict
    ]
    hdu_list = hdu_list_for_output_from(
        values_list=[image_list[0].mask.astype("float")] + image_list,
        ext_name_list=["mask"] + list(galaxy_name_image_dict.keys()),
        header_dict=next(iter(galaxy_name_image_dict.values())).mask.header_dict,
    )
    hdu_list[0].header[_GALAXY_IMAGES_CACHE_VERSION_KEY] = _GALAXY_IMAGES_CACHE_VERSION
    hdu_list.writeto(cache_path, overwrite=True)
    paths.preserve_in_zip(cache_path)


def galaxy_name_image_dict_via_result_from(
    result, use_model_images: bool = False, minimum_percent: Optional[float] = None
) -> "AdaptImages":
    """
    Returns per-galaxy subtracted signal-to-noise maps by default, or model images.

    For model-fitting, the adapt-images are typically setup using the maximum log likelihood model of the
    previous model-fit. This means the model-fitting is used to cleanly deblend the light of the different
    galaxies in the image (e.g. separate the lens light from the source light).

    This method uses attributes of a result (e.g. dictionary mapping galaxy instances to their model-images)
    to create the adapt-images.

    This can use either:

    - The model image of each galaxy in the best-fit model.
    - The subtracted signal-to-noise map of each galaxy in the best-fit model, where the dataset minus
      the model images of all other galaxies is divided by the dataset noise map (the default).

    Certain models produce galaxy-images with negative values (e.g. a pixelization), which can cause
    numerical issues with the adaptive schemes. To prevent this, we set a minimum value for each
    galaxy-image, which is a fraction of its maximum value defined via a config file. This flooring applies
    to both signal-to-noise maps and model images.

    The `HilbertSNR` image mesh and the soft maps of `soft_snr_maps_from` need the
    signal-to-noise maps *before* that floor: they work in absolute S/N, and above a peak S/N of 200 the default
    1% floor lifts empty sky over their noise threshold of S/N 2 (sky reads 2.5 at a peak of 250), which is then
    weighted as source. Pass ``minimum_percent=0.0`` for those: negative values are set to 0 and nothing else
    changes.

    The raw per-galaxy images are cached to the result's own ``files/`` folder on first computation
    (``galaxy_images_model.fits`` / ``galaxy_images_snr.fits``) and loaded from there on every later call —
    computing them rebuilds the result's maximum log likelihood fit, which on a resumed pipeline pays a fresh
    JIT compile plus (for pixelized fits) an inversion, and dominates SLaM resume overhead
    (autolens_profiling#70). Results with no on-disk output (e.g. ``NullPaths``) always compute.

    Two things can make a cache file stale, and they are guarded differently. Changing the upstream model or
    search is guarded structurally: it produces a new search identifier and therefore a fresh output directory
    with no cache file. Changing the *dataset* is not, because the identifier does not encode it — a rerun
    whose mask changed while the model stayed identical lands on the previous run's directory and its cache.
    That case is caught by validating the cached mask against ``result.mask`` on load and treating a mismatch
    as a cache miss, so the images are recomputed on the current mask. Reading ``result.mask`` resolves to
    ``analysis.dataset.mask`` and does not rebuild the maximum log likelihood fit, so the check costs nothing
    the cache was saving (PyAutoGalaxy#516).

    A stale cache is also *repaired* on disk: the recomputed images are written back to the loose ``files/``
    folder and ``Paths.preserve_in_zip`` replaces the search's now-outdated zip member, so the next
    ``restore()`` re-extracts the recomputed copy. A dataset change therefore costs one recompute rather than
    one per run (PyAutoFit#1414).

    Parameters
    ----------
    result
        The result of a previous model-fit, providing per-galaxy subtracted signal-to-noise maps and model images.
    use_model_images
        If True, use the model images of the galaxies. If False (the default), use their subtracted
        signal-to-noise maps, which have already been divided by the noise map.
    minimum_percent
        The fraction of each image's maximum it is floored at. None (the default) uses the config value
        ``general.adapt.adapt_minimum_percent`` (0.01). ``0.0`` gives the images before the floor, with negative
        values set to 0: the input for `HilbertSNR` and `soft_snr_maps_from` (see also `galaxy_snr_maps_from`).

    Returns
    -------
    A dictionary mapping galaxy paths to subtracted signal-to-noise maps by default, or model images when
    use_model_images is True, floored at ``minimum_percent`` of each image's maximum in either mode.
    """
    if minimum_percent is None:
        adapt_minimum_percent = conf.instance["general"]["adapt"]["adapt_minimum_percent"]
    else:
        adapt_minimum_percent = minimum_percent

    cache_path = _galaxy_images_cache_path(result, use_model_images=use_model_images)
    raw_image_dict = _galaxy_image_dict_from_cache(
        cache_path, mask=getattr(result, "mask", None), result=result
    )

    if raw_image_dict is None:
        raw_image_dict = {}

        for path, galaxy in result.path_galaxy_tuples:
            if use_model_images:
                raw_image_dict[path] = result.model_image_galaxy_dict[path]
            else:
                raw_image_dict[path] = result.subtracted_signal_to_noise_map_galaxy_dict[
                    path
                ]

        if cache_path is not None:
            _galaxy_image_dict_to_cache(cache_path, raw_image_dict, paths=result.paths)

    galaxy_name_image_dict = {}

    for path, galaxy_image in raw_image_dict.items():
        # Floor a copy: computed without a cache, these arrays are the result's own, and flooring them in place
        # would change what every later call on the same result returns (e.g. a later ``minimum_percent=0.0``).
        galaxy_image = galaxy_image.copy()

        minimum_galaxy_value = adapt_minimum_percent * np.max(galaxy_image.array)
        galaxy_image[galaxy_image < minimum_galaxy_value] = minimum_galaxy_value

        galaxy_name_image_dict[path] = galaxy_image

    return galaxy_name_image_dict


def galaxy_snr_maps_from(result) -> Dict[str, aa.Array2D]:
    """
    Returns each galaxy's signal-to-noise (S/N) map from a previous fit: the data minus the model images of every
    other galaxy, divided by the noise map, with negative values set to 0 and no other floor.

    This is the input the `HilbertSNR` image mesh and `soft_snr_maps_from` expect. It is
    ``galaxy_name_image_dict_via_result_from(result=result, minimum_percent=0.0)``: the same maps the default adapt
    images are built from (and cached as ``files/galaxy_images_snr.fits``), before the 1% minimum-percent floor,
    which above a peak S/N of 200 lifts empty sky over S/N 2.

    Parameters
    ----------
    result
        The result of a previous model-fit.

    Returns
    -------
    A dictionary mapping each galaxy's path name (e.g. ``"('galaxies', 'source')"``) to its S/N map.
    """
    return galaxy_name_image_dict_via_result_from(result=result, minimum_percent=0.0)


def soft_snr_maps_from(
    snr_maps: Dict[str, aa.Array2D],
    galaxy_names: Optional[List[str]] = None,
) -> Dict[str, aa.Array2D]:
    """
    Returns the maps to give adaptive regularizations (e.g. `AdaptSplit`) so they adapt to the same signal as the
    `HilbertSNR` image mesh: each S/N map passed through `aa.image_mesh.soft_snr_map_from`
    (``(x / (x + 3)) ** 2`` with ``x = max(S/N - 2, 0)``, floored at 1% of its maximum).

    The regularization schemes are unchanged; only the map they are given changes. `AdaptImages.from_result` does
    this (and builds the meshes) in one call:

        image_mesh = ag.image_mesh.HilbertSNR(pixels=1000, edge_pixels=30)
        adapt_images = ag.AdaptImages.from_result(result=result, image_mesh={"source": image_mesh})

    Parameters
    ----------
    snr_maps
        Galaxy path name -> S/N map, e.g. from `galaxy_snr_maps_from`.
    galaxy_names
        The galaxies to transform, as full path names or short names (e.g. ``"source"``); the others are returned
        unchanged. None (the default) transforms every galaxy.

    Returns
    -------
    A new dictionary with the same keys.
    """
    selected = None
    if galaxy_names is not None:
        selected = {galaxy_name_key_from(name, list(snr_maps)) for name in galaxy_names}

    soft_maps = {}

    for name, snr_map in snr_maps.items():
        if selected is not None and name not in selected:
            soft_maps[name] = snr_map
            continue

        soft_maps[name] = aa.Array2D(
            values=aa.image_mesh.soft_snr_map_from(np.asarray(snr_map.array)),
            mask=snr_map.mask,
        )

    return soft_maps


def _galaxy_path_forms(key: str) -> List[str]:
    """
    The names a galaxy path key can be written as: the full key (``"('galaxies', 'source')"``), the dotted path
    (``"galaxies.source"``) and the galaxy's own name (``"source"``).
    """
    import ast

    forms = [key]
    try:
        path = ast.literal_eval(key)
    except (ValueError, SyntaxError):
        return forms
    if isinstance(path, tuple) and path and all(isinstance(part, str) for part in path):
        forms += [".".join(path), path[-1]]
    return forms


def galaxy_name_key_from(name: str, keys: List[str]) -> str:
    """
    Returns the full galaxy path key in ``keys`` that ``name`` refers to.

    ``name`` may be the full key (``"('galaxies', 'source')"``), the dotted path (``"galaxies.source"``) or the
    galaxy's own name (``"source"``). A short name that matches more than one galaxy (e.g. two galaxies called
    ``"source"`` in different collections) raises an error; use the full or dotted form then.

    Parameters
    ----------
    name
        The name as written by the user.
    keys
        The full path keys to resolve against (e.g. ``str(path)`` of every galaxy in the model).
    """
    keys = [str(key) for key in keys]
    if name in keys:
        return name

    matches = [key for key in keys if name in _galaxy_path_forms(key)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"No galaxy called {name!r}; the galaxies are {keys}.")
    raise KeyError(
        f"{name!r} matches more than one galaxy ({matches}); use the full or dotted path name."
    )


def _resolved_name_dict(name_dict: Optional[Dict], keys: List[str]) -> Optional[Dict]:
    """
    ``name_dict`` with every short key (``"source"``, ``"galaxies.source"``) replaced by the full galaxy path key it
    refers to. A key that names no galaxy in ``keys`` is kept as it is (and, as before short keys existed, simply
    matches nothing); a short key naming more than one galaxy raises.
    """
    if name_dict is None:
        return None

    resolved = {}
    for name, value in name_dict.items():
        name = str(name)
        try:
            name = galaxy_name_key_from(name, keys)
        except KeyError as exc:
            if "more than one galaxy" in str(exc):
                raise
        resolved[name] = value
    return resolved


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

        A dictionary mapping the path name of each galaxy (e.g. "('galaxies', 'lens')", or in short "lens") to its model-image is therefore used
        which is called inside the `log_likelihood_function` o map the model-image of each galaxy to the galaxy
        instance of that iteration's specific model.

        Parameters
        ----------
        galaxy_image_dict
            A dictionary associating each galaxy instance to an image of only that galaxy (e.g. for a strong lens
            one entry will map an instance of the source galaxy entry to an image of the lensed source.
        galaxy_name_image_dict
            A dictionary associating each galaxy path name (e.g. "('galaxies', 'source')", or in short "source"
            or "galaxies.source") to an image of only that
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

    @classmethod
    def from_result(
        cls,
        result,
        image_mesh: Optional[Dict[str, "aa.AbstractImageMesh"]] = None,
        mask: Optional[aa.Mask2D] = None,
    ) -> "AdaptImages":
        """
        Returns the adapt images for the next model-fit of a pipeline, in one call, from the result of the previous
        one: each galaxy's S/N map (`galaxy_snr_maps_from`), the image-plane mesh grid of every galaxy listed in
        ``image_mesh`` drawn from its S/N map, and the soft S/N maps (`soft_snr_maps_from`) the adaptive
        regularizations (e.g. `AdaptSplit`) adapt to.

            image_mesh = ag.image_mesh.HilbertSNR(pixels=1000, edge_pixels=30)
            adapt_images = ag.AdaptImages.from_result(result=result, image_mesh={"source": image_mesh})

            mesh = ag.mesh.Delaunay(pixels=image_mesh.pixels, zeroed_pixels=image_mesh.edge_pixels)

        This is the recommended set-up with `HilbertSNR`. For the older peak-normalised set-up (`Hilbert` with
        ``weight_power`` / ``weight_floor`` on the floored S/N map), build `AdaptImages` directly from
        `galaxy_name_image_dict_via_result_from`.

        Parameters
        ----------
        result
            The result of the previous model-fit.
        image_mesh
            Galaxy name -> the image mesh to draw that galaxy's mesh grid with. Names may be short (``"source"``),
            dotted (``"galaxies.source"``) or full (``"('galaxies', 'source')"``). Edge points are appended when the
            image mesh has ``edge_pixels``. None draws no mesh grids.
        mask
            The mask the mesh grids are drawn on; defaults to the result's mask.
        """
        snr_maps = galaxy_snr_maps_from(result=result)

        mesh_grid_dict = None
        if image_mesh is not None:
            mask = mask if mask is not None else result.mask
            keys = list(snr_maps)
            mesh_grid_dict = {}
            for name, mesh in image_mesh.items():
                key = galaxy_name_key_from(name, keys)
                mesh_grid_dict[key] = mesh.image_plane_mesh_grid_from(
                    mask=mask, adapt_data=snr_maps[key]
                )

        return cls(
            galaxy_name_image_dict=soft_snr_maps_from(snr_maps),
            galaxy_name_image_plane_mesh_grid_dict=mesh_grid_dict,
        )

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

        A dictionary mapping the path name of each galaxy (e.g. "('galaxies', 'lens')", or in short "lens") to its model-image is therefore used
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

        # Keys may be written short ("source", "galaxies.source"); resolve them to the full path keys once.
        galaxy_keys = list(path_by_id.values())
        resolved = AdaptImages(
            galaxy_name_image_dict=_resolved_name_dict(
                self.galaxy_name_image_dict, galaxy_keys
            ),
            galaxy_name_image_plane_mesh_grid_dict=_resolved_name_dict(
                self.galaxy_name_image_plane_mesh_grid_dict, galaxy_keys
            ),
        )

        galaxy_image_dict = None

        if resolved.galaxy_name_image_dict is not None:

            galaxy_image_dict = {}

            for galaxy_name, galaxy in instance.path_instance_tuples_for_class(Galaxy):
                galaxy_name = str(galaxy_name)

                if galaxy_name in resolved.galaxy_name_image_dict:
                    galaxy_image_dict[galaxy] = resolved.galaxy_name_image_dict[galaxy_name]

            if mask is not None:
                for key, image in galaxy_image_dict.items():
                    galaxy_image_dict[key] = aa.Array2D(values=image, mask=mask)

        galaxy_image_plane_mesh_grid_dict = None

        if resolved.galaxy_name_image_plane_mesh_grid_dict is not None:

            galaxy_image_plane_mesh_grid_dict = {}

            for galaxy_name, galaxy in instance.path_instance_tuples_for_class(Galaxy):
                galaxy_name = str(galaxy_name)

                if galaxy_name in resolved.galaxy_name_image_plane_mesh_grid_dict:
                    cached_mesh = resolved.galaxy_name_image_plane_mesh_grid_dict[galaxy_name]
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
            galaxy_name_image_dict=resolved.galaxy_name_image_dict,
            galaxy_name_image_plane_mesh_grid_dict=resolved.galaxy_name_image_plane_mesh_grid_dict,
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
