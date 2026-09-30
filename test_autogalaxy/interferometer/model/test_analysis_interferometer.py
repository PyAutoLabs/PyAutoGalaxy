
import autofit as af
import autogalaxy as ag

from autogalaxy.interferometer.model.result import ResultInterferometer
from pathlib import Path


directory = Path(__file__).resolve().parent


def test__make_result__result_interferometer_is_returned(interferometer_7):
    model = af.Collection(galaxies=af.Collection(galaxy_0=ag.Galaxy(redshift=0.5)))

    analysis = ag.AnalysisInterferometer(dataset=interferometer_7, use_jax=False)

    search = ag.m.MockSearch(name="test_search")

    result = search.fit(model=model, analysis=analysis)

    assert isinstance(result, ResultInterferometer)


def test__fit_figure_of_merit__matches_correct_fit_given_galaxy_profiles(
    interferometer_7,
):
    galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=0.1))

    model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

    analysis = ag.AnalysisInterferometer(dataset=interferometer_7, use_jax=False)

    instance = model.instance_from_unit_vector([])
    fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

    galaxies = analysis.galaxies_via_instance_from(instance=instance)

    fit = ag.FitInterferometer(dataset=interferometer_7, galaxies=galaxies)

    assert fit.log_likelihood == fit_figure_of_merit


def _array_free_dataset_from(dataset):
    import autoarray as aa

    return aa.Interferometer.from_stream(
        [(dataset.uv_wavelengths, dataset.data, dataset.noise_map)],
        real_space_mask=dataset.real_space_mask,
        transformer_class=type(dataset.transformer),
    )


def _pixelization_only_galaxies():
    pixelization = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    return [ag.Galaxy(redshift=0.5, pixelization=pixelization)]


class _FitStub:
    """
    The part of a `PyAutoFit` aggregator `Fit` the interferometer loader reads: `value(name)` returning the
    `dataset.fits` HDU list written by `save_attributes` and the saved `transformer_class` json (if any).
    """

    def __init__(self, paths):
        self.paths = paths
        self.children = []

    def value(self, name):
        from astropy.io import fits
        from autonerves.dictable import from_dict

        if name == "dataset":
            return fits.open(self.paths.image_path / "dataset.fits")

        if name == "transformer_class":
            path = self.paths._files_path / "transformer_class.json"

            if not path.exists():
                return None

            return from_dict(self.paths.load_json("transformer_class"))

        return None


def test__interferometer_hdu_list_from__array_free_dataset__writes_sparse_terms(
    interferometer_7,
):
    import numpy as np

    from autogalaxy.interferometer.model.analysis import (
        SPARSE_TERMS_HEADER_KEYS,
        interferometer_hdu_list_from,
    )

    dataset = _array_free_dataset_from(interferometer_7)
    terms = dataset.sparse_terms

    hdu_list = interferometer_hdu_list_from(dataset=dataset)

    assert [hdu.name for hdu in hdu_list] == [
        "MASK",
        "NUFFT_PRECISION_OPERATOR",
        "DIRTY_IMAGE",
        "DIRTY_BEAM",
        "SPARSE_TERMS_SCALARS",
    ]

    np.testing.assert_array_equal(
        hdu_list["NUFFT_PRECISION_OPERATOR"].data, terms.nufft_precision_operator
    )
    np.testing.assert_array_equal(
        hdu_list["SPARSE_TERMS_SCALARS"].data,
        [
            terms.sum_weights,
            terms.data_term,
            terms.noise_normalization,
            terms.n_vis,
            terms.eps,
        ],
    )
    np.testing.assert_array_equal(
        hdu_list["DIRTY_IMAGE"].data, terms.dirty_image_native
    )
    np.testing.assert_array_equal(hdu_list["DIRTY_BEAM"].data, terms.dirty_beam_native)

    header = hdu_list[0].header

    for key in SPARSE_TERMS_HEADER_KEYS.values():
        assert len(key) <= 8

    assert header["SUMW"] == terms.sum_weights
    assert header["DATATERM"] == terms.data_term
    assert header["NOISENRM"] == terms.noise_normalization
    assert header["NVIS"] == terms.n_vis
    assert header["EPS"] == terms.eps
    assert header["TRNSFRMR"] == terms.transformer_class_name
    assert header["PIXSCAY"] == dataset.real_space_mask.pixel_scales[0]


def test__interferometer_hdu_list_from__in_memory_dataset__unchanged_layout(
    interferometer_7,
):
    import numpy as np

    from autogalaxy.interferometer.model.analysis import interferometer_hdu_list_from

    hdu_list = interferometer_hdu_list_from(dataset=interferometer_7)

    assert [hdu.name.lower() for hdu in hdu_list] == [
        "mask",
        "data",
        "noise_map",
        "uv_wavelengths",
    ]

    np.testing.assert_array_equal(
        hdu_list[0].data, interferometer_7.real_space_mask.astype("float")
    )
    np.testing.assert_array_equal(hdu_list[1].data, interferometer_7.data.in_array)
    np.testing.assert_array_equal(
        hdu_list[2].data, interferometer_7.noise_map.in_array
    )
    np.testing.assert_array_equal(hdu_list[3].data, interferometer_7.uv_wavelengths)


def test__save_attributes__array_free_dataset__aggregator_round_trip(
    interferometer_7, tmp_path
):
    """
    `save_attributes` writes an array-free dataset's `SparseTerms` to `dataset.fits` and the aggregator loader
    rebuilds it via `Interferometer.from_sparse_terms`, so a fit of the reloaded dataset reproduces the
    original `log_evidence`.
    """
    import pytest

    from autogalaxy.aggregator.interferometer.interferometer import (
        _interferometer_from,
    )

    dataset = _array_free_dataset_from(interferometer_7)

    paths = af.DirectoryPaths(
        name="array_free_round_trip", path_prefix=str(tmp_path)
    )

    analysis = ag.AnalysisInterferometer(dataset=dataset, use_jax=False)
    analysis.save_attributes(paths=paths)

    assert (paths.image_path / "dataset.fits").exists()
    assert not (paths._files_path / "transformer_class.json").exists()

    dataset_list = _interferometer_from(fit=_FitStub(paths=paths))

    assert len(dataset_list) == 1

    dataset_reloaded = dataset_list[0]

    assert dataset_reloaded.is_array_free
    assert dataset_reloaded.sparse_operator is not None

    assert (
        dataset_reloaded.sparse_operator.data_term
        == dataset.sparse_operator.data_term
    )
    assert (
        dataset_reloaded.sparse_operator.noise_normalization
        == dataset.sparse_operator.noise_normalization
    )

    terms = dataset.sparse_terms
    terms_reloaded = dataset_reloaded.sparse_terms

    for name in (
        "sum_weights",
        "data_term",
        "noise_normalization",
        "n_vis",
        "shape_native",
        "pixel_scales",
        "origin",
        "eps",
        "transformer_class_name",
    ):
        assert getattr(terms_reloaded, name) == getattr(terms, name), name

    galaxies = _pixelization_only_galaxies()

    log_evidence = ag.FitInterferometer(dataset=dataset, galaxies=galaxies).log_evidence
    log_evidence_reloaded = ag.FitInterferometer(
        dataset=dataset_reloaded, galaxies=galaxies
    ).log_evidence

    assert log_evidence_reloaded == pytest.approx(log_evidence, rel=1.0e-8)


def test__save_attributes__in_memory_dataset__aggregator_round_trip(
    interferometer_7, tmp_path
):
    import numpy as np

    from autogalaxy.aggregator.interferometer.interferometer import (
        _interferometer_from,
    )

    paths = af.DirectoryPaths(name="in_memory_round_trip", path_prefix=str(tmp_path))

    analysis = ag.AnalysisInterferometer(dataset=interferometer_7, use_jax=False)
    analysis.save_attributes(paths=paths)

    dataset_reloaded = _interferometer_from(fit=_FitStub(paths=paths))[0]

    assert not dataset_reloaded.is_array_free
    assert type(dataset_reloaded.transformer) is type(interferometer_7.transformer)

    np.testing.assert_array_equal(
        dataset_reloaded.data.array, interferometer_7.data.array
    )
    np.testing.assert_array_equal(
        dataset_reloaded.noise_map.array, interferometer_7.noise_map.array
    )
    np.testing.assert_array_equal(
        dataset_reloaded.uv_wavelengths, interferometer_7.uv_wavelengths
    )


def test__interferometer_hdu_list_from__array_free_dataset__scalars_round_trip_losslessly_through_disk(
    interferometer_7, tmp_path
):
    """
    A FITS header card holds at most 20 characters of value, so astropy truncates exponent-form float64
    values (e.g. `1.2345678901234567e+20` is written as `1.23456789012345E+20`). The `SparseTerms` scalars
    are therefore stored losslessly in the `SPARSE_TERMS_SCALARS` HDU, which this test round-trips through
    a real `writeto` / `fits.open` for 17-significant-digit values.
    """
    import dataclasses

    import autoarray as aa
    from astropy.io import fits

    from autogalaxy.aggregator.interferometer.interferometer import _sparse_terms_from
    from autogalaxy.interferometer.model.analysis import interferometer_hdu_list_from

    dataset = _array_free_dataset_from(interferometer_7)

    terms = dataclasses.replace(
        dataset.sparse_terms,
        sum_weights=1.2345678901234567e20,
        data_term=1.0e-9 * (1.0 + 1.0e-15),
        noise_normalization=-1.2345678901234567e-20,
        eps=9.876543210987654e-13,
    )

    dataset = aa.Interferometer.from_sparse_terms(
        terms, real_space_mask=dataset.real_space_mask
    )

    file_path = tmp_path / "dataset.fits"
    interferometer_hdu_list_from(dataset=dataset).writeto(file_path)

    with fits.open(file_path) as hdu_list:
        terms_reloaded = _sparse_terms_from(
            hdu_list, real_space_mask=dataset.real_space_mask
        )

    for name in ("sum_weights", "data_term", "noise_normalization", "n_vis", "eps"):
        assert getattr(terms_reloaded, name) == getattr(terms, name), (
            name,
            repr(getattr(terms_reloaded, name)),
            repr(getattr(terms, name)),
        )


def test__save_attributes__array_free_dataset__non_square_pixel_scales_round_trip(
    interferometer_7, tmp_path
):
    """
    `agg_util.mask_header_from` reloads both axes of the mask pixel scales (`PIXSCAY`, `PIXSCAX`), so a
    non-square mask, and the `SparseTerms` provenance copied from it, round-trip through `save_attributes`.
    """
    import dataclasses

    import autoarray as aa

    from autogalaxy.aggregator.interferometer.interferometer import (
        _interferometer_from,
    )

    real_space_mask = aa.Mask2D(
        mask=interferometer_7.real_space_mask,
        pixel_scales=(0.1, 0.2),
        origin=(0.3, -0.4),
    )

    terms = dataclasses.replace(
        _array_free_dataset_from(interferometer_7).sparse_terms,
        pixel_scales=(0.1, 0.2),
        origin=(0.3, -0.4),
    )

    dataset = aa.Interferometer.from_sparse_terms(
        terms, real_space_mask=real_space_mask
    )

    paths = af.DirectoryPaths(name="non_square_round_trip", path_prefix=str(tmp_path))

    ag.AnalysisInterferometer(dataset=dataset, use_jax=False).save_attributes(
        paths=paths
    )

    dataset_reloaded = _interferometer_from(fit=_FitStub(paths=paths))[0]

    assert tuple(dataset_reloaded.real_space_mask.pixel_scales) == (0.1, 0.2)
    assert tuple(dataset_reloaded.real_space_mask.origin) == (0.3, -0.4)
    assert dataset_reloaded.sparse_terms.pixel_scales == terms.pixel_scales
    assert dataset_reloaded.sparse_terms.origin == terms.origin


def test__sparse_terms_from__no_scalars_hdu__falls_back_to_header_cards(
    interferometer_7,
):
    from astropy.io import fits

    from autogalaxy.aggregator.interferometer.interferometer import _sparse_terms_from
    from autogalaxy.interferometer.model.analysis import interferometer_hdu_list_from

    dataset = _array_free_dataset_from(interferometer_7)
    terms = dataset.sparse_terms

    hdu_list = interferometer_hdu_list_from(dataset=dataset)
    hdu_list = fits.HDUList([hdu for hdu in hdu_list if hdu.name != "SPARSE_TERMS_SCALARS"])

    terms_reloaded = _sparse_terms_from(hdu_list, real_space_mask=dataset.real_space_mask)

    for name in ("sum_weights", "data_term", "noise_normalization", "n_vis", "eps"):
        assert getattr(terms_reloaded, name) == getattr(terms, name), name
