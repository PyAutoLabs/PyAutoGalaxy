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
