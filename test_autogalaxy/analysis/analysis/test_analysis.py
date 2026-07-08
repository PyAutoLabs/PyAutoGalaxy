import os
from pathlib import Path

from autoconf.dictable import from_json

import autofit as af
import autogalaxy as ag

directory = Path(__file__).resolve().parent


def test__galaxies_via_instance(masked_imaging_7x7):
    galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=0.1))
    extra_galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=0.2))

    model = af.Collection(
        galaxies=af.Collection(galaxy=galaxy),
        extra_galaxies=af.Collection(extra_galaxy_0=extra_galaxy),
    )

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)

    instance = model.instance_from_unit_vector([])

    galaxies = analysis.galaxies_via_instance_from(instance=instance)

    assert galaxies[0].light.intensity == 0.1
    assert galaxies[1].light.intensity == 0.2


def test__save_results__galaxies_output_to_json(analysis_imaging_7x7):
    galaxy = ag.Galaxy(redshift=0.5)

    model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

    paths = af.DirectoryPaths()

    analysis_imaging_7x7.save_results(
        paths=paths,
        result=ag.m.MockResult(max_log_likelihood_galaxies=[galaxy], model=model),
    )

    galaxies = from_json(file_path=paths._files_path / "galaxies.json")

    assert galaxies[0].redshift == 0.5

    os.remove(paths._files_path / "galaxies.json")


def test__save_attributes__dataset_fits_output_for_aggregator(analysis_imaging_7x7):
    # Regression guard: `save_attributes` must always write `dataset.fits` to the
    # `files` folder so the aggregator loaders (`ImagingAgg`,
    # `agg_util.mask_header_from`) can reload the dataset via
    # `fit.value(name="dataset")`, independently of whether visualization ran.
    from astropy.io import fits

    paths = af.DirectoryPaths()

    analysis_imaging_7x7.save_attributes(paths=paths)

    dataset_fits_path = paths._files_path / "dataset.fits"

    assert dataset_fits_path.exists()

    with fits.open(dataset_fits_path) as hdu_list:
        ext_names = [hdu.name for hdu in hdu_list]

    assert ext_names[:4] == ["MASK", "DATA", "NOISE_MAP", "PSF"]

    os.remove(dataset_fits_path)
