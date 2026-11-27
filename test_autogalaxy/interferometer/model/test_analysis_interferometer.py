import pytest
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
            *terms.phase_centre,
            np.nan,
            np.nan,
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
    assert (header["PHCENTY"], header["PHCENTX"]) == terms.phase_centre
    assert header["MASKFP"] == terms.mask_fingerprint
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
    np.testing.assert_array_equal(hdu_list[2].data, interferometer_7.noise_map.in_array)
    np.testing.assert_array_equal(hdu_list[3].data, interferometer_7.uv_wavelengths)

    # A dataset without a primary beam writes no `PRIMARY_BEAM` HDU, so the layout is pinned at 4 HDUs.
    assert len(hdu_list) == 4


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

    paths = af.DirectoryPaths(name="array_free_round_trip", path_prefix=str(tmp_path))

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
        dataset_reloaded.sparse_operator.data_term == dataset.sparse_operator.data_term
    )
    assert (
        dataset_reloaded.sparse_operator.noise_normalization
        == dataset.sparse_operator.noise_normalization
    )

    terms = dataset.sparse_terms
    terms_reloaded = dataset_reloaded.sparse_terms

    assert terms.mask_fingerprint is not None

    for name in (
        "sum_weights",
        "data_term",
        "noise_normalization",
        "n_vis",
        "shape_native",
        "pixel_scales",
        "origin",
        "mask_fingerprint",
        "eps",
        "transformer_class_name",
        "phase_centre",
    ):
        assert getattr(terms_reloaded, name) == getattr(terms, name), name

    galaxies = _pixelization_only_galaxies()

    log_evidence = ag.FitInterferometer(dataset=dataset, galaxies=galaxies).log_evidence
    log_evidence_reloaded = ag.FitInterferometer(
        dataset=dataset_reloaded, galaxies=galaxies
    ).log_evidence

    assert log_evidence_reloaded == pytest.approx(log_evidence, rel=1.0e-8)

    # Ordinary light profiles (fitted via the data-term identity) also reproduce on the reloaded dataset,
    # with and without an inversion.
    light = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=0.1))

    for galaxies in ([light, *_pixelization_only_galaxies()], [light]):
        fit = ag.FitInterferometer(dataset=dataset, galaxies=galaxies)
        fit_reloaded = ag.FitInterferometer(dataset=dataset_reloaded, galaxies=galaxies)

        assert fit_reloaded.figure_of_merit == pytest.approx(
            fit.figure_of_merit, rel=1.0e-8
        )
        assert fit_reloaded.figure_of_merit == pytest.approx(
            ag.FitInterferometer(
                dataset=interferometer_7, galaxies=galaxies
            ).figure_of_merit,
            rel=1.0e-8,
        )


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
    hdu_list = fits.HDUList(
        [hdu for hdu in hdu_list if hdu.name != "SPARSE_TERMS_SCALARS"]
    )

    terms_reloaded = _sparse_terms_from(
        hdu_list, real_space_mask=dataset.real_space_mask
    )

    for name in (
        "sum_weights",
        "data_term",
        "noise_normalization",
        "n_vis",
        "eps",
        "phase_centre",
    ):
        assert getattr(terms_reloaded, name) == getattr(terms, name), name


def test__save_attributes__array_free_dataset__mask_with_different_masked_pixels_refused_on_reload(
    interferometer_7, tmp_path
):
    """
    The mask fingerprint survives the `dataset.fits` round trip (PyAutoArray#707): a saved mask edited to mask
    one more pixel (same shape, pixel scales and origin) no longer matches the terms, so the reload raises
    instead of rebuilding the dataset on the wrong pixels.
    """
    import autoarray as aa
    import numpy as np
    import pytest
    from astropy.io import fits

    from autogalaxy.aggregator.interferometer.interferometer import (
        _interferometer_from,
    )

    dataset = _array_free_dataset_from(interferometer_7)

    paths = af.DirectoryPaths(name="mask_fingerprint", path_prefix=str(tmp_path))
    ag.AnalysisInterferometer(dataset=dataset, use_jax=False).save_attributes(
        paths=paths
    )

    with fits.open(paths.image_path / "dataset.fits", mode="update") as hdu_list:
        mask_values = np.asarray(hdu_list[0].data).copy()
        row, column = np.argwhere(mask_values == 0.0)[0]
        mask_values[row, column] = 1.0
        hdu_list[0].data = mask_values

    with pytest.raises(aa.exc.DatasetException, match="fingerprint"):
        _interferometer_from(fit=_FitStub(paths=paths))


def test__sparse_terms_from__file_predating_mask_fingerprint__reloads_none(
    interferometer_7,
):
    from autogalaxy.aggregator.interferometer.interferometer import _sparse_terms_from
    from autogalaxy.interferometer.model.analysis import interferometer_hdu_list_from

    dataset = _array_free_dataset_from(interferometer_7)

    hdu_list = interferometer_hdu_list_from(dataset=dataset)
    del hdu_list[0].header["MASKFP"]

    terms_reloaded = _sparse_terms_from(
        hdu_list, real_space_mask=dataset.real_space_mask
    )

    assert terms_reloaded.mask_fingerprint is None


def test__interferometer_hdu_list_from__array_free_dataset__phase_centre_round_trips_through_disk(
    interferometer_7, tmp_path
):
    """
    The `(y, x)` phase centre the streamed visibilities were re-centred on is persisted (losslessly in the
    `SPARSE_TERMS_SCALARS` HDU, readably in the `PHCENTY` / `PHCENTX` cards) and reloaded, so reloaded
    shifted terms still refuse to be summed with unshifted ones.
    """
    import pytest

    import autoarray as aa
    from astropy.io import fits

    from autogalaxy.aggregator.interferometer.interferometer import _sparse_terms_from
    from autogalaxy.interferometer.model.analysis import interferometer_hdu_list_from

    chunks = [
        (
            interferometer_7.uv_wavelengths,
            interferometer_7.data,
            interferometer_7.noise_map,
        )
    ]
    transformer_class = type(interferometer_7.transformer)

    dataset = aa.Interferometer.from_stream(
        chunks,
        real_space_mask=interferometer_7.real_space_mask,
        transformer_class=transformer_class,
        phase_centre=(0.7, -1.3),
    )
    terms_unshifted = _array_free_dataset_from(interferometer_7).sparse_terms

    file_path = tmp_path / "dataset.fits"
    interferometer_hdu_list_from(dataset=dataset).writeto(file_path)

    with fits.open(file_path) as hdu_list:
        assert (hdu_list[0].header["PHCENTY"], hdu_list[0].header["PHCENTX"]) == (
            0.7,
            -1.3,
        )

        terms_reloaded = _sparse_terms_from(
            hdu_list, real_space_mask=dataset.real_space_mask
        )

    assert terms_reloaded.phase_centre == (0.7, -1.3)

    with pytest.raises(aa.exc.InversionException, match="phase_centre"):
        terms_reloaded + terms_unshifted


def test__sparse_terms_from__phase_centre_unrecorded_or_predating_file__reloads_none(
    interferometer_7,
):
    """
    Unrecorded phase centre is written as `NaN` and reloads as `None`; a file written before the phase centre
    was persisted (a 5-entry `SPARSE_TERMS_SCALARS` array and no `PHCENTY` / `PHCENTX` cards) also reloads
    it as `None`, with the other scalars intact.
    """
    import dataclasses

    import numpy as np
    import autoarray as aa
    from astropy.io import fits

    from autogalaxy.aggregator.interferometer.interferometer import _sparse_terms_from
    from autogalaxy.interferometer.model.analysis import interferometer_hdu_list_from

    dataset = _array_free_dataset_from(interferometer_7)
    terms = dataset.sparse_terms

    dataset_unrecorded = aa.Interferometer.from_sparse_terms(
        dataclasses.replace(terms, phase_centre=None),
        real_space_mask=dataset.real_space_mask,
    )
    hdu_list = interferometer_hdu_list_from(dataset=dataset_unrecorded)

    assert np.isnan(hdu_list["SPARSE_TERMS_SCALARS"].data[5:7]).all()
    assert "PHCENTY" not in hdu_list[0].header

    assert (
        _sparse_terms_from(
            hdu_list, real_space_mask=dataset.real_space_mask
        ).phase_centre
        is None
    )

    # A file predating the phase centre: drop the two appended entries and the header cards.
    hdu_list = interferometer_hdu_list_from(dataset=dataset)
    header = hdu_list[0].header.copy()
    del header["PHCENTY"]
    del header["PHCENTX"]

    hdu_list = fits.HDUList(
        [fits.PrimaryHDU(data=hdu_list[0].data, header=header)]
        + [
            (
                fits.ImageHDU(
                    data=hdu.data[:5], header=hdu.header, name="SPARSE_TERMS_SCALARS"
                )
                if hdu.name == "SPARSE_TERMS_SCALARS"
                else hdu
            )
            for hdu in hdu_list[1:]
        ]
    )

    terms_reloaded = _sparse_terms_from(
        hdu_list, real_space_mask=dataset.real_space_mask
    )

    assert terms_reloaded.phase_centre is None
    for name in ("sum_weights", "data_term", "noise_normalization", "n_vis", "eps"):
        assert getattr(terms_reloaded, name) == getattr(terms, name), name


def _fine_array_free_dataset_from(dataset, oversample=2):
    import autoarray as aa

    return aa.Interferometer.from_stream(
        [(dataset.uv_wavelengths, dataset.data, dataset.noise_map)],
        real_space_mask=dataset.real_space_mask,
        transformer_class=aa.TransformerNUFFT,
        oversample=oversample,
    )


def test__interferometer_hdu_list_from__fine_grids__written_only_when_opted_in(
    interferometer_7,
):
    """
    The `oversample` fine grids are hundreds of MB at realistic sizes (~840 MB at 400 pixels, q = 8) and
    `dataset.fits` is written into every search's output, so they are written only with
    `include_fine_grids=True`; by default the layout and scalars are those of terms without fine grids.
    """
    import numpy as np
    import pytest

    pytest.importorskip("nufftax")

    from autogalaxy.interferometer.model.analysis import interferometer_hdu_list_from

    dataset = _fine_array_free_dataset_from(interferometer_7)
    terms = dataset.sparse_terms

    assert terms.oversample == 2

    hdu_list = interferometer_hdu_list_from(dataset=dataset)

    assert [hdu.name for hdu in hdu_list] == [
        "MASK",
        "NUFFT_PRECISION_OPERATOR",
        "DIRTY_IMAGE",
        "DIRTY_BEAM",
        "SPARSE_TERMS_SCALARS",
    ]
    assert np.isnan(hdu_list["SPARSE_TERMS_SCALARS"].data[7:9]).all()
    assert "OVERSAMP" not in hdu_list[0].header

    hdu_list = interferometer_hdu_list_from(dataset=dataset, include_fine_grids=True)

    assert [hdu.name for hdu in hdu_list] == [
        "MASK",
        "NUFFT_PRECISION_OPERATOR",
        "DIRTY_IMAGE",
        "DIRTY_BEAM",
        "SPARSE_TERMS_SCALARS",
        "PRECISION_OPERATOR_FINE",
        "DIRTY_IMAGE_FINE",
    ]
    np.testing.assert_array_equal(
        hdu_list["PRECISION_OPERATOR_FINE"].data, terms.precision_operator_fine
    )
    np.testing.assert_array_equal(
        hdu_list["DIRTY_IMAGE_FINE"].data, terms.dirty_image_fine
    )
    np.testing.assert_array_equal(hdu_list["SPARSE_TERMS_SCALARS"].data[7:9], [2, 0.25])
    assert hdu_list[0].header["OVERSAMP"] == 2

    # Terms without fine grids: opting in writes nothing extra.
    hdu_list = interferometer_hdu_list_from(
        dataset=_array_free_dataset_from(interferometer_7), include_fine_grids=True
    )

    assert "PRECISION_OPERATOR_FINE" not in [hdu.name for hdu in hdu_list]
    assert "OVERSAMP" not in hdu_list[0].header


def test__sparse_terms_from__fine_grids_round_trip_through_disk(
    interferometer_7, tmp_path
):
    import numpy as np
    import pytest

    pytest.importorskip("nufftax")

    import autoarray as aa
    from astropy.io import fits

    from autogalaxy.aggregator.interferometer.interferometer import _sparse_terms_from
    from autogalaxy.interferometer.model.analysis import interferometer_hdu_list_from

    dataset = _fine_array_free_dataset_from(interferometer_7)
    terms = dataset.sparse_terms

    file_path = tmp_path / "dataset.fits"
    interferometer_hdu_list_from(dataset=dataset, include_fine_grids=True).writeto(
        file_path
    )

    with fits.open(file_path) as hdu_list:
        terms_reloaded = _sparse_terms_from(
            hdu_list, real_space_mask=dataset.real_space_mask
        )

    np.testing.assert_array_equal(
        terms_reloaded.precision_operator_fine, terms.precision_operator_fine
    )
    np.testing.assert_array_equal(
        terms_reloaded.dirty_image_fine, terms.dirty_image_fine
    )
    assert terms_reloaded.oversample == 2
    assert isinstance(terms_reloaded.oversample, int)
    assert terms_reloaded.oversample_pad == 0.25

    # Reloaded fine terms still refuse to be summed with terms of another q.
    other = _fine_array_free_dataset_from(interferometer_7, oversample=4).sparse_terms

    with pytest.raises(aa.exc.InversionException, match="oversample"):
        terms_reloaded + other

    # Written without opting in: reloads with no fine grids and no oversample.
    file_path = tmp_path / "dataset_default.fits"
    interferometer_hdu_list_from(dataset=dataset).writeto(file_path)

    with fits.open(file_path) as hdu_list:
        terms_reloaded = _sparse_terms_from(
            hdu_list, real_space_mask=dataset.real_space_mask
        )

    assert terms_reloaded.precision_operator_fine is None
    assert terms_reloaded.dirty_image_fine is None
    assert terms_reloaded.oversample is None
    assert terms_reloaded.oversample_pad is None


def test__sparse_terms_from__file_predating_fine_grids__reloads_none(interferometer_7):
    """
    A file written before the fine grids existed -- a 7-entry `SPARSE_TERMS_SCALARS` array, no `OVERSAMP`
    card and no fine-grid HDUs -- reloads with `None` fine grids and oversampling, other terms intact.
    """
    from astropy.io import fits

    from autogalaxy.aggregator.interferometer.interferometer import _sparse_terms_from
    from autogalaxy.interferometer.model.analysis import interferometer_hdu_list_from

    dataset = _array_free_dataset_from(interferometer_7)
    terms = dataset.sparse_terms

    hdu_list = interferometer_hdu_list_from(dataset=dataset)

    hdu_list = fits.HDUList(
        [hdu_list[0]]
        + [
            (
                fits.ImageHDU(
                    data=hdu.data[:7], header=hdu.header, name="SPARSE_TERMS_SCALARS"
                )
                if hdu.name == "SPARSE_TERMS_SCALARS"
                else hdu
            )
            for hdu in hdu_list[1:]
        ]
    )

    terms_reloaded = _sparse_terms_from(
        hdu_list, real_space_mask=dataset.real_space_mask
    )

    assert terms_reloaded.precision_operator_fine is None
    assert terms_reloaded.dirty_image_fine is None
    assert terms_reloaded.oversample is None
    assert terms_reloaded.oversample_pad is None
    assert terms_reloaded.phase_centre == terms.phase_centre
    for name in ("sum_weights", "data_term", "noise_normalization", "n_vis", "eps"):
        assert getattr(terms_reloaded, name) == getattr(terms, name), name


def _primary_beam_from(dataset):
    """
    A Gaussian primary beam on the real-space mask of `dataset` (peak 1 at the centre) whose FWHM is 2.2
    arcsec, spanning a few of the mask's 1 arcsec pixels so the beam runs from 1 down to ~0.3.
    """
    import autoarray as aa
    from autoarray.dataset.interferometer.primary_beam import ARCSEC_PER_RADIAN

    dish_diameter_m = 12.0
    fwhm_arcsec = 2.2
    frequency_hz = 1.13 * 299792458.0 / dish_diameter_m / (fwhm_arcsec / ARCSEC_PER_RADIAN)

    return aa.primary_beam_gaussian_from(
        mask=dataset.real_space_mask,
        frequency_hz=frequency_hz,
        dish_diameter_m=dish_diameter_m,
    )


def _beamed_dataset_from(dataset, primary_beam):
    import autoarray as aa

    return aa.Interferometer(
        data=dataset.data,
        noise_map=dataset.noise_map,
        uv_wavelengths=dataset.uv_wavelengths,
        real_space_mask=dataset.real_space_mask,
        transformer_class=type(dataset.transformer),
        primary_beam=primary_beam,
    )


def _array_free_beamed_dataset_from(dataset, primary_beam):
    import autoarray as aa

    return aa.Interferometer.from_stream(
        [(dataset.uv_wavelengths, dataset.data, dataset.noise_map)],
        real_space_mask=dataset.real_space_mask,
        transformer_class=type(dataset.transformer),
        primary_beam=primary_beam,
    )


def _light_galaxies():
    return [ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=0.1))]


def test__primary_beam__fixture_beam_spans_expected_range(interferometer_7):
    import numpy as np

    beam = np.asarray(_primary_beam_from(interferometer_7))

    assert beam.max() == 1.0
    assert 0.25 < beam.min() < 0.4


def test__save_attributes__in_memory_beamed_dataset__aggregator_round_trip(
    interferometer_7, tmp_path
):
    """
    An in-memory dataset with a primary beam writes it as a trailing `PRIMARY_BEAM` HDU, and the aggregator
    passes it back as `primary_beam=`, so the reloaded dataset carries an identical beam (on the dataset and
    its transformer) and a fit of it reproduces the beamed log likelihood, which differs from the beamless one.
    """
    import numpy as np
    import pytest
    from astropy.io import fits

    from autogalaxy.aggregator.interferometer.interferometer import (
        _interferometer_from,
    )

    dataset = _beamed_dataset_from(
        interferometer_7, primary_beam=_primary_beam_from(interferometer_7)
    )

    paths = af.DirectoryPaths(name="in_memory_beamed", path_prefix=str(tmp_path))

    ag.AnalysisInterferometer(dataset=dataset, use_jax=False).save_attributes(
        paths=paths
    )

    with fits.open(paths.image_path / "dataset.fits") as hdu_list:
        assert [hdu.name for hdu in hdu_list] == [
            "MASK",
            "DATA",
            "NOISE_MAP",
            "UV_WAVELENGTHS",
            "PRIMARY_BEAM",
        ]
        assert hdu_list["PRIMARY_BEAM"].data.shape == tuple(
            dataset.real_space_mask.shape_native
        )

    dataset_reloaded = _interferometer_from(fit=_FitStub(paths=paths))[0]

    assert not dataset_reloaded.is_array_free

    np.testing.assert_array_equal(
        dataset_reloaded.primary_beam_slim, dataset.primary_beam_slim
    )
    np.testing.assert_array_equal(
        dataset_reloaded.transformer.primary_beam_slim,
        dataset.transformer.primary_beam_slim,
    )

    for galaxies in (_light_galaxies(), _pixelization_only_galaxies()):
        log_likelihood = ag.FitInterferometer(
            dataset=dataset, galaxies=galaxies
        ).log_likelihood
        log_likelihood_reloaded = ag.FitInterferometer(
            dataset=dataset_reloaded, galaxies=galaxies
        ).log_likelihood
        log_likelihood_beamless = ag.FitInterferometer(
            dataset=interferometer_7, galaxies=galaxies
        ).log_likelihood

        assert log_likelihood_reloaded == pytest.approx(log_likelihood, rel=1.0e-10)
        assert log_likelihood_reloaded != pytest.approx(
            log_likelihood_beamless, rel=1.0e-6
        )


def test__save_attributes__array_free_beamed_dataset__aggregator_round_trip(
    interferometer_7, tmp_path
):
    """
    An array-free dataset with a primary beam writes it as a trailing `PRIMARY_BEAM` HDU after its
    `SparseTerms`, and the aggregator passes it back to `from_sparse_terms`, so the reloaded sparse operator
    carries an identical beam and fits reproduce the beamed log evidence / figure of merit.
    """
    import numpy as np
    import pytest
    from astropy.io import fits

    from autogalaxy.aggregator.interferometer.interferometer import (
        _interferometer_from,
    )

    dataset = _array_free_beamed_dataset_from(
        interferometer_7, primary_beam=_primary_beam_from(interferometer_7)
    )

    paths = af.DirectoryPaths(name="array_free_beamed", path_prefix=str(tmp_path))

    ag.AnalysisInterferometer(dataset=dataset, use_jax=False).save_attributes(
        paths=paths
    )

    with fits.open(paths.image_path / "dataset.fits") as hdu_list:
        assert [hdu.name for hdu in hdu_list][-1] == "PRIMARY_BEAM"

    dataset_reloaded = _interferometer_from(fit=_FitStub(paths=paths))[0]

    assert dataset_reloaded.is_array_free

    np.testing.assert_array_equal(
        dataset_reloaded.primary_beam_slim, dataset.primary_beam_slim
    )
    np.testing.assert_array_equal(
        dataset_reloaded.sparse_operator.primary_beam_slim,
        dataset.sparse_operator.primary_beam_slim,
    )

    galaxies = _pixelization_only_galaxies()

    log_evidence = ag.FitInterferometer(dataset=dataset, galaxies=galaxies).log_evidence
    log_evidence_reloaded = ag.FitInterferometer(
        dataset=dataset_reloaded, galaxies=galaxies
    ).log_evidence

    assert log_evidence_reloaded == pytest.approx(log_evidence, rel=1.0e-10)
    assert log_evidence_reloaded != pytest.approx(
        ag.FitInterferometer(
            dataset=_array_free_dataset_from(interferometer_7), galaxies=galaxies
        ).log_evidence,
        rel=1.0e-6,
    )

    galaxies = [*_light_galaxies(), *_pixelization_only_galaxies()]

    figure_of_merit = ag.FitInterferometer(
        dataset=dataset, galaxies=galaxies
    ).figure_of_merit
    figure_of_merit_reloaded = ag.FitInterferometer(
        dataset=dataset_reloaded, galaxies=galaxies
    ).figure_of_merit

    assert figure_of_merit_reloaded == pytest.approx(figure_of_merit, rel=1.0e-10)


def test__save_attributes__beamless_datasets__no_primary_beam_hdu_and_reload_without_beam(
    interferometer_7, tmp_path
):
    from astropy.io import fits

    from autogalaxy.aggregator.interferometer.interferometer import (
        _interferometer_from,
    )

    for name, dataset in (
        ("in_memory", interferometer_7),
        ("array_free", _array_free_dataset_from(interferometer_7)),
    ):
        paths = af.DirectoryPaths(name=f"beamless_{name}", path_prefix=str(tmp_path))

        ag.AnalysisInterferometer(dataset=dataset, use_jax=False).save_attributes(
            paths=paths
        )

        with fits.open(paths.image_path / "dataset.fits") as hdu_list:
            assert "PRIMARY_BEAM" not in [hdu.name for hdu in hdu_list]

        dataset_reloaded = _interferometer_from(fit=_FitStub(paths=paths))[0]

        assert dataset_reloaded.primary_beam is None
        assert dataset_reloaded.primary_beam_slim is None


def test__interferometer_from__outputs_written_before_primary_beam_persisted__reload_without_beam(
    interferometer_7, tmp_path
):
    """
    `dataset.fits` files written before the primary beam was persisted have no `PRIMARY_BEAM` HDU, even for a
    beamed dataset. Both layouts reload with `primary_beam=None`, including a legacy in-memory file with no
    EXTNAMEs at all (read by position), whose beam must not be read from any positional HDU.
    """
    from astropy.io import fits

    from autogalaxy.aggregator.interferometer.interferometer import (
        _interferometer_from,
    )

    beam = _primary_beam_from(interferometer_7)

    for name, dataset, strip_ext_names in (
        ("in_memory", _beamed_dataset_from(interferometer_7, beam), False),
        ("in_memory_legacy", _beamed_dataset_from(interferometer_7, beam), True),
        ("array_free", _array_free_beamed_dataset_from(interferometer_7, beam), False),
    ):
        paths = af.DirectoryPaths(name=f"old_{name}", path_prefix=str(tmp_path))

        ag.AnalysisInterferometer(dataset=dataset, use_jax=False).save_attributes(
            paths=paths
        )

        dataset_path = paths.image_path / "dataset.fits"

        with fits.open(dataset_path) as hdu_list:
            hdus = [hdu.copy() for hdu in hdu_list if hdu.name != "PRIMARY_BEAM"]

        if strip_ext_names:
            for hdu in hdus:
                hdu.header.remove("EXTNAME", ignore_missing=True)

        fits.HDUList(hdus).writeto(dataset_path, overwrite=True)

        dataset_reloaded = _interferometer_from(fit=_FitStub(paths=paths))[0]

        assert dataset_reloaded.primary_beam is None
        assert dataset_reloaded.primary_beam_slim is None
        assert dataset_reloaded.is_array_free == dataset.is_array_free


@pytest.mark.parametrize("array_free", [False, True])
def test__save_attributes__registers_dataset_fits_in_database(
    interferometer_7, tmp_path, array_free
):
    import numpy as np

    from autofit.aggregator.search_output import SearchOutput
    from autofit.database import Fit, open_database
    from autogalaxy.aggregator import agg_util
    from autogalaxy.aggregator.interferometer.interferometer import _interferometer_from

    if array_free:
        dataset = _array_free_dataset_from(interferometer_7)
    else:
        dataset = interferometer_7
    analysis = ag.AnalysisInterferometer(dataset=dataset, use_jax=False)
    directory_paths = af.DirectoryPaths(
        name="directory", path_prefix=str(tmp_path), is_identifier_in_paths=False
    )
    analysis.save_attributes(paths=directory_paths)
    directory_fit = SearchOutput(directory_paths.output_path)
    (directory_dataset,) = _interferometer_from(fit=directory_fit)

    with open_database(str(tmp_path / "fits.sqlite")) as session:
        database_paths = af.DatabasePaths(
            session=session,
            name="database",
            path_prefix=str(tmp_path),
            is_identifier_in_paths=False,
        )
        analysis.save_attributes(paths=database_paths)
        fit_id = database_paths.fit.id
        session.commit()
        session.expunge_all()
        database_fit = session.get(Fit, fit_id)

        with database_fit.value("dataset") as database_hdus, directory_fit.value(
            "dataset"
        ) as directory_hdus:
            assert [hdu.name for hdu in database_hdus] == [
                hdu.name for hdu in directory_hdus
            ]
            for database_hdu, directory_hdu in zip(database_hdus, directory_hdus):
                np.testing.assert_array_equal(database_hdu.data, directory_hdu.data)

        mask, _ = agg_util.mask_header_from(fit=database_fit)
        directory_mask, _ = agg_util.mask_header_from(fit=directory_fit)
        np.testing.assert_array_equal(mask.array, directory_mask.array)
        assert mask.pixel_scales == directory_mask.pixel_scales
        assert mask.origin == directory_mask.origin

        (database_dataset,) = _interferometer_from(fit=database_fit)

    assert database_dataset.is_array_free == directory_dataset.is_array_free == array_free
    if array_free:
        np.testing.assert_array_equal(
            database_dataset.sparse_terms.nufft_precision_operator,
            directory_dataset.sparse_terms.nufft_precision_operator,
        )
        np.testing.assert_array_equal(
            database_dataset.sparse_terms.dirty_image_native,
            directory_dataset.sparse_terms.dirty_image_native,
        )
    else:
        np.testing.assert_array_equal(
            database_dataset.data.array, directory_dataset.data.array
        )
        np.testing.assert_array_equal(
            database_dataset.noise_map.array, directory_dataset.noise_map.array
        )
        np.testing.assert_array_equal(
            database_dataset.uv_wavelengths, directory_dataset.uv_wavelengths
        )
        assert type(database_dataset.transformer) is type(directory_dataset.transformer)


# --- setup="auto" (likelihood lifecycle 5C) ---------------------------------------------

import pytest

from autonerves.environment import Environment


def _environment(gpu=False, gpu_name=None, cores=64):
    return Environment(
        gpu=gpu,
        gpu_name=gpu_name,
        gpu_memory_gb=None,
        cores=cores,
        ram_gb=None,
        numba=True,
        nufftax=True,
        site="local",
    )


@pytest.fixture(autouse=True)
def unset_setup_env(monkeypatch):
    monkeypatch.delenv("PYAUTO_SETUP", raising=False)
    monkeypatch.delenv("PYAUTO_DISABLE_JAX", raising=False)


def _rectangular_model():
    pixelization = af.Model(
        ag.Pixelization,
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant,
    )
    return af.Collection(
        galaxies=af.Collection(
            source=af.Model(ag.Galaxy, redshift=1.0, pixelization=pixelization)
        )
    )


def test__setup_auto__cpu_many_core_rectangular__numba_wtilde_sparse_operator(
    interferometer_7, monkeypatch
):
    pytest.importorskip("finufft")

    monkeypatch.setattr(
        Environment, "detect", classmethod(lambda cls: _environment(cores=64))
    )

    analysis = ag.AnalysisInterferometer(
        dataset=interferometer_7, setup="auto", use_jax=True
    )
    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=_rectangular_model())

    assert analysis.setup.setup_id == "numba_cpu_sparse_fp64_wtilde"
    assert analysis._use_jax is False
    assert analysis.dataset.sparse_operator is not None
    assert analysis.dataset.setup == analysis.setup


def test__setup_auto__gpu_mge__jax_dense_nufft(interferometer_7, monkeypatch):
    pytest.importorskip("jax")

    monkeypatch.setattr(
        Environment,
        "detect",
        classmethod(
            lambda cls: _environment(gpu=True, gpu_name="NVIDIA A100-SXM4-80GB")
        ),
    )

    model = af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(ag.Galaxy, redshift=0.5, light=ag.lp.Sersic)
        )
    )

    analysis = ag.AnalysisInterferometer(
        dataset=interferometer_7, setup="auto", use_jax=False
    )
    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=model)

    assert analysis.setup.setup_id == "jax_gpu_dense_fp64_nufft"
    assert analysis._use_jax is True
    assert analysis.dataset.sparse_operator is None
    assert isinstance(analysis.dataset.transformer, ag.TransformerNUFFT)


def test__no_setup__dataset_unchanged(interferometer_7, monkeypatch):
    monkeypatch.setattr(
        Environment, "detect", classmethod(lambda cls: _environment(cores=64))
    )

    analysis = ag.AnalysisInterferometer(dataset=interferometer_7, use_jax=False)
    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=_rectangular_model())

    assert analysis.dataset is interferometer_7
    assert analysis._use_jax is False
    assert analysis.setup is None
