import pytest
import numpy as np

import autofit as af
import autogalaxy as ag


def test__instance_with_associated_adapt_images_from(masked_imaging_7x7):
    g0 = ag.Galaxy(redshift=0.5)
    g1 = ag.Galaxy(redshift=1.0)

    galaxy_image_dict = {
        g0: ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
        g1: ag.Array2D.full(fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0),
    }

    adapt_images = ag.AdaptImages(
        galaxy_image_dict=galaxy_image_dict,
    )

    assert adapt_images.model_image.native == pytest.approx(
        3.0 * np.ones((3, 3)), 1.0e-4
    )


def test__image_for_galaxy__resolves_after_galaxy_identity_changes():
    """
    Simulates the post-``jax.jit`` unflatten boundary: ``adapt_images.galaxy_image_dict`` is keyed by the
    trace-time ``Galaxy`` instances, but the lookup at ``GalaxiesToInversion.mapper_galaxy_dict`` is performed
    against fresh ``Galaxy`` objects whose ``__hash__`` differs. The path-tuple lookup via
    ``galaxy_name_image_dict`` must still resolve to the right adapt image.
    """
    galaxies = af.ModelInstance()
    galaxies.lens = ag.Galaxy(redshift=0.5)
    galaxies.source = ag.Galaxy(redshift=1.0)

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    galaxy_name_image_dict = {
        str(("galaxies", "lens")): ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
        str(("galaxies", "source")): ag.Array2D.full(
            fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0
        ),
    }

    trace_galaxies = [galaxies.lens, galaxies.source]

    adapt_images = ag.AdaptImages(
        galaxy_name_image_dict=galaxy_name_image_dict,
    ).updated_via_instance_from(instance=instance, galaxies=trace_galaxies)

    assert adapt_images.galaxy_path_list == [
        str(("galaxies", "lens")),
        str(("galaxies", "source")),
    ]

    # Fast path: by-instance lookup still works for the trace-time galaxies.
    assert adapt_images.image_for_galaxy(
        trace_galaxies[0], trace_galaxies
    ).native == pytest.approx(np.ones((3, 3)), 1.0e-4)

    # Simulate post-unflatten: fresh ``Galaxy`` objects with new ``.id`` values
    # placed at the same positions as the trace-time list. ``galaxy_image_dict``
    # cannot resolve them (hash mismatch) so the helper must fall back to
    # ``galaxy_name_image_dict`` via ``galaxy_path_list``.
    fresh_galaxies = [ag.Galaxy(redshift=0.5), ag.Galaxy(redshift=1.0)]

    assert adapt_images.galaxy_image_dict.get(fresh_galaxies[0]) is None
    assert adapt_images.image_for_galaxy(
        fresh_galaxies[0], fresh_galaxies
    ).native == pytest.approx(np.ones((3, 3)), 1.0e-4)
    assert adapt_images.image_for_galaxy(
        fresh_galaxies[1], fresh_galaxies
    ).native == pytest.approx(2.0 * np.ones((3, 3)), 1.0e-4)


def test__image_plane_mesh_grid_for_galaxy__resolves_after_galaxy_identity_changes():
    """
    Companion to :func:`test__image_for_galaxy__resolves_after_galaxy_identity_changes` for the mesh-grid
    lookup path used by ``GalaxiesToInversion.image_plane_mesh_grid_list``.
    """
    galaxies = af.ModelInstance()
    galaxies.lens = ag.Galaxy(redshift=0.5)
    galaxies.source = ag.Galaxy(redshift=1.0)

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    galaxy_name_image_plane_mesh_grid_dict = {
        str(("galaxies", "lens")): ag.Grid2DIrregular(values=[(3.0, 3.0), (3.0, 3.0)]),
        str(("galaxies", "source")): ag.Grid2DIrregular(values=[(4.0, 4.0), (4.0, 4.0)]),
    }

    trace_galaxies = [galaxies.lens, galaxies.source]

    adapt_images = ag.AdaptImages(
        galaxy_name_image_plane_mesh_grid_dict=galaxy_name_image_plane_mesh_grid_dict,
    ).updated_via_instance_from(instance=instance, galaxies=trace_galaxies)

    fresh_galaxies = [ag.Galaxy(redshift=0.5), ag.Galaxy(redshift=1.0)]

    assert adapt_images.image_plane_mesh_grid_for_galaxy(
        fresh_galaxies[0], fresh_galaxies
    ) == pytest.approx(3.0 * np.ones((2, 2)), 1.0e-4)
    assert adapt_images.image_plane_mesh_grid_for_galaxy(
        fresh_galaxies[1], fresh_galaxies
    ) == pytest.approx(4.0 * np.ones((2, 2)), 1.0e-4)


class _StubCachePaths:
    """Duck-typed paths: only what the galaxy-image cache helpers touch."""

    def __init__(self, files_path):
        self._files_path = files_path


class _StubCacheResult:
    """Duck-typed result for `galaxy_name_image_dict_via_result_from`."""

    def __init__(self, files_path, image_dict, poisoned=False):
        self.paths = _StubCachePaths(files_path)
        self._image_dict = image_dict
        self._poisoned = poisoned

    @property
    def path_galaxy_tuples(self):
        return [(name, None) for name in self._image_dict]

    @property
    def subtracted_signal_to_noise_map_galaxy_dict(self):
        if self._poisoned:
            raise AssertionError(
                "recompute path taken — the cache should have been loaded"
            )
        return self._image_dict


def _cache_test_image_dict():
    mask = ag.Mask2D.circular(shape_native=(7, 7), pixel_scales=0.1, radius=0.3)
    image_0 = ag.Array2D.ones(shape_native=(7, 7), pixel_scales=0.1).apply_mask(
        mask=mask
    )
    image_1 = ag.Array2D.full(
        fill_value=2.0, shape_native=(7, 7), pixel_scales=0.1
    ).apply_mask(mask=mask)
    return {
        "('galaxies', 'lens')": image_0,
        "('galaxies', 'source')": image_1,
    }


def test__galaxy_image_dict_cache__round_trip(tmp_path):
    from autogalaxy.analysis.adapt_images.adapt_images import (
        _galaxy_image_dict_from_cache,
        _galaxy_image_dict_to_cache,
    )

    image_dict = _cache_test_image_dict()
    cache_path = tmp_path / "galaxy_images_snr.fits"

    _galaxy_image_dict_to_cache(cache_path, image_dict, paths=_StubCachePaths(tmp_path))
    loaded = _galaxy_image_dict_from_cache(cache_path)

    assert set(loaded.keys()) == set(image_dict.keys())
    for name in image_dict:
        assert loaded[name].array == pytest.approx(image_dict[name].array, 1.0e-8)
        assert (loaded[name].mask == image_dict[name].mask).all()


def test__galaxy_name_image_dict_via_result_from__loads_cache_on_second_call(tmp_path):
    from autogalaxy.analysis.adapt_images.adapt_images import (
        galaxy_name_image_dict_via_result_from,
    )

    image_dict = _cache_test_image_dict()

    result = _StubCacheResult(tmp_path, image_dict)
    first = galaxy_name_image_dict_via_result_from(result=result)

    assert (tmp_path / "galaxy_images_snr.fits").exists()

    # A poisoned result raises if the compute path is taken — the second call
    # must come entirely from the cache written by the first.
    poisoned = _StubCacheResult(tmp_path, image_dict, poisoned=True)
    second = galaxy_name_image_dict_via_result_from(result=poisoned)

    assert set(second.keys()) == set(first.keys())
    for name in first:
        assert second[name].array == pytest.approx(first[name].array, 1.0e-8)


def test__galaxy_name_image_dict_via_result_from__no_paths_always_computes():
    from autogalaxy.analysis.adapt_images.adapt_images import (
        galaxy_name_image_dict_via_result_from,
    )

    image_dict = _cache_test_image_dict()

    result = _StubCacheResult(files_path=None, image_dict=image_dict)
    result.paths = None

    galaxy_name_image_dict = galaxy_name_image_dict_via_result_from(result=result)

    assert set(galaxy_name_image_dict.keys()) == set(image_dict.keys())
