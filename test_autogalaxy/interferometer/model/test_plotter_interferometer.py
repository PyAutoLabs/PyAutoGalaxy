import pytest
from pathlib import Path

import autogalaxy as ag

from autogalaxy.interferometer.model.plotter import (
    PlotterInterferometer,
)

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_plotter_plotter_setup():
    return directory / "files"


def test__interferometer(interferometer_7, plot_path, plot_patch):
    plotter = PlotterInterferometer(image_path=plot_path)

    plotter.interferometer(dataset=interferometer_7)

    assert str(Path(plot_path) / "dataset.png") in plot_patch.paths


def test__fit_interferometer(
    interferometer_7,
    fit_interferometer_x2_galaxy_inversion_7x7,
    plot_path,
    plot_patch,
):
    plotter = PlotterInterferometer(image_path=plot_path)

    plotter.fit_interferometer(
        fit=fit_interferometer_x2_galaxy_inversion_7x7,
    )

    assert str(Path(plot_path) / "fit.png") in plot_patch.paths

    image = ag.ndarray_via_fits_from(
        file_path=Path(plot_path) / "galaxy_images.fits", hdu=1
    )

    assert image.shape == (5, 5)

    image = ag.ndarray_via_fits_from(
        file_path=Path(plot_path) / "fit_dirty_images.fits", hdu=1
    )

    assert image.shape == (5, 5)


def _array_free_dataset_from(dataset):
    import autoarray as aa

    return aa.Interferometer.from_stream(
        [(dataset.uv_wavelengths, dataset.data, dataset.noise_map)],
        real_space_mask=dataset.real_space_mask,
        transformer_class=type(dataset.transformer),
    )


def _pixelization_only_model():
    import autofit as af

    pixelization = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    return af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(ag.Galaxy, redshift=0.5, pixelization=pixelization)
        )
    )


def _light_profile_and_pixelization_model():
    import autofit as af

    model = _pixelization_only_model()
    model.galaxies.light = af.Model(
        ag.Galaxy, redshift=0.5, bulge=ag.lp.Sersic(intensity=0.1)
    )

    return model


def _visualize(dataset, image_path, model=None):
    """
    Run the interferometer visualizer's `visualize_before_fit` and `visualize` for a
    pixelization-only model on `dataset`, as a non-linear search would.
    """
    from types import SimpleNamespace

    from autogalaxy.interferometer.model.visualizer import VisualizerInterferometer

    model = model or _pixelization_only_model()
    instance = model.instance_from_prior_medians()

    analysis = ag.AnalysisInterferometer(dataset=dataset, use_jax=False)
    paths = SimpleNamespace(image_path=image_path)

    VisualizerInterferometer.visualize_before_fit(
        analysis=analysis, paths=paths, model=model
    )
    VisualizerInterferometer.visualize(
        analysis=analysis, paths=paths, instance=instance, during_analysis=False
    )

    return analysis.fit_from(instance=instance)


def _ext_names_from(file_path):
    from astropy.io import fits

    with fits.open(file_path) as hdu_list:
        return [hdu.name for hdu in hdu_list]


def test__visualizer__array_free_dataset(interferometer_7, tmp_path, plot_patch):
    pytest.importorskip("nufftax")

    import numpy as np

    dataset = _array_free_dataset_from(interferometer_7)

    fit = _visualize(dataset=dataset, image_path=tmp_path)

    for filename in ("dataset", "fit", "galaxies", "inversion_0_0"):
        assert str(tmp_path / f"{filename}.png") in plot_patch.paths, filename

    assert (tmp_path / "galaxy_images.fits").exists()
    assert _ext_names_from(tmp_path / "fit_dirty_images.fits") == [
        "MASK",
        "DIRTY_IMAGE_NATURAL",
        "DIRTY_BEAM",
        "DIRTY_MODEL_IMAGE_NATURAL",
        "DIRTY_RESIDUAL_MAP_NATURAL",
    ]

    for hdu, array in (
        (1, dataset.dirty_image_natural),
        (2, dataset.dirty_beam),
        (3, fit.dirty_model_image_natural),
        (4, fit.dirty_residual_map_natural),
    ):
        np.testing.assert_allclose(
            ag.ndarray_via_fits_from(
                file_path=tmp_path / "fit_dirty_images.fits", hdu=hdu
            ),
            array.native_for_fits,
            rtol=1.0e-6,
            atol=1.0e-12,
        )


def test__visualizer__array_free_dataset__ordinary_light_profile(
    interferometer_7, tmp_path, plot_patch
):
    """
    A model with an ordinary light profile alongside a pixelization visualizes on an array-free dataset: the
    visualizer reads only `model_image_natural` / `dirty_model_image_natural`, never the model visibilities.
    """
    pytest.importorskip("nufftax")

    import numpy as np

    dataset = _array_free_dataset_from(interferometer_7)

    fit = _visualize(
        dataset=dataset,
        image_path=tmp_path,
        model=_light_profile_and_pixelization_model(),
    )

    assert np.abs(fit.profile_image.array).max() > 0.0

    for filename in ("dataset", "fit", "galaxies", "inversion_0_0"):
        assert str(tmp_path / f"{filename}.png") in plot_patch.paths, filename

    np.testing.assert_allclose(
        ag.ndarray_via_fits_from(file_path=tmp_path / "fit_dirty_images.fits", hdu=3),
        fit.dirty_model_image_natural.native_for_fits,
        rtol=1.0e-6,
        atol=1.0e-12,
    )


def test__visualizer__in_memory_dataset__fit_dirty_images_unchanged(
    interferometer_7, tmp_path, plot_patch
):
    import numpy as np

    fit = _visualize(dataset=interferometer_7, image_path=tmp_path)

    for filename in ("dataset", "fit", "galaxies", "inversion_0_0"):
        assert str(tmp_path / f"{filename}.png") in plot_patch.paths, filename

    assert _ext_names_from(tmp_path / "fit_dirty_images.fits") == [
        "MASK",
        "DIRTY_IMAGE",
        "DIRTY_NOISE_MAP",
        "DIRTY_MODEL_IMAGE",
        "DIRTY_RESIDUAL_MAP",
        "DIRTY_NORMALIZED_RESIDUAL_MAP",
        "DIRTY_CHI_SQUARED_MAP",
    ]

    for hdu, array in enumerate(
        (
            fit.dirty_image,
            fit.dirty_noise_map,
            fit.dirty_model_image,
            fit.dirty_residual_map,
            fit.dirty_normalized_residual_map,
            fit.dirty_chi_squared_map,
        ),
        start=1,
    ):
        np.testing.assert_array_equal(
            ag.ndarray_via_fits_from(
                file_path=tmp_path / "fit_dirty_images.fits", hdu=hdu
            ),
            np.asarray(array.native_for_fits),
        )
