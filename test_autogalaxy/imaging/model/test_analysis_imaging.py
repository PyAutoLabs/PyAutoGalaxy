from pathlib import Path

import autofit as af
import autogalaxy as ag

from autogalaxy.imaging.model.result import ResultImaging

directory = Path(__file__).resolve().parent


def test__make_result__result_imaging_is_returned(masked_imaging_7x7):
    model = af.Collection(galaxies=af.Collection(galaxy_0=ag.Galaxy(redshift=0.5)))

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)

    search = ag.m.MockSearch(name="test_search")

    result = search.fit(model=model, analysis=analysis)

    assert isinstance(result, ResultImaging)


def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
    masked_imaging_7x7,
):
    galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=0.1))

    model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)

    instance = model.instance_from_unit_vector([])
    fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

    galaxies = analysis.galaxies_via_instance_from(instance=instance)

    fit = ag.FitImaging(dataset=masked_imaging_7x7, galaxies=galaxies)

    assert fit.log_likelihood == fit_figure_of_merit


def test__register_fit_imaging_pytrees__registers_fit_galaxies_and_dataset_model():
    from autoarray.abstract_ndarray import _pytree_registered_classes
    from autoarray.dataset.dataset_model import DatasetModel
    from autogalaxy.galaxy.galaxies import Galaxies
    from autogalaxy.imaging.fit_imaging import FitImaging

    ag.AnalysisImaging._register_fit_imaging_pytrees()

    assert FitImaging in _pytree_registered_classes
    assert DatasetModel in _pytree_registered_classes
    assert Galaxies in _pytree_registered_classes


def test__save_attributes__registers_dataset_fits_in_database(masked_imaging_7x7, tmp_path):
    import numpy as np

    from autofit.aggregator.search_output import SearchOutput
    from autofit.database import Fit, Fits, open_database
    from autogalaxy.aggregator import agg_util
    from autogalaxy.aggregator.imaging.imaging import _imaging_from

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)
    directory_paths = af.DirectoryPaths(
        name="directory", path_prefix=str(tmp_path), is_identifier_in_paths=False
    )
    analysis.save_attributes(paths=directory_paths)
    directory_fit = SearchOutput(directory_paths.output_path)
    (directory_dataset,) = _imaging_from(fit=directory_fit)

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
        resumed_paths = af.DatabasePaths(
            session=session,
            name="database",
            path_prefix=str(tmp_path),
            is_identifier_in_paths=False,
        )
        analysis.save_attributes(paths=resumed_paths)
        session.commit()
        assert session.query(Fits).count() == 1
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

        (database_dataset,) = _imaging_from(fit=database_fit)

    np.testing.assert_array_equal(
        database_dataset.data.array, directory_dataset.data.array
    )
    np.testing.assert_array_equal(
        database_dataset.noise_map.array, directory_dataset.noise_map.array
    )
    np.testing.assert_array_equal(
        database_dataset.psf.kernel.array, directory_dataset.psf.kernel.array
    )
