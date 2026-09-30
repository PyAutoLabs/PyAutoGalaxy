from pathlib import Path

import autogalaxy.plot as aplt
import pytest


@pytest.fixture(name="plot_path")
def make_fit_dataset_plotter_setup():
    return Path(__file__).resolve().parent / "files" / "plots" / "fit"


def test__fit_sub_plot_real_space(
    fit_interferometer_7x7,
    fit_interferometer_x2_galaxy_inversion_7x7,
    plot_path,
    plot_patch,
):
    aplt.subplot_fit_real_space(
        fit=fit_interferometer_7x7,
        output_path=plot_path,
        output_format="png",
    )

    assert str(plot_path / "fit_real_space.png") in plot_patch.paths

    plot_patch.paths = []

    aplt.subplot_fit_real_space(
        fit=fit_interferometer_x2_galaxy_inversion_7x7,
        output_path=plot_path,
        output_format="png",
    )

    assert str(plot_path / "fit_real_space.png") in plot_patch.paths


def test__fit_sub_plots__array_free_dataset(
    interferometer_7, plot_path, plot_patch, monkeypatch
):
    pytest.importorskip("nufftax")

    import autoarray as aa
    import autogalaxy as ag
    from autogalaxy.interferometer.plot import fit_interferometer_plots

    dataset = aa.Interferometer.from_stream(
        [(interferometer_7.uv_wavelengths, interferometer_7.data, interferometer_7.noise_map)],
        real_space_mask=interferometer_7.real_space_mask,
        transformer_class=type(interferometer_7.transformer),
    )

    pixelization = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    fit = ag.FitInterferometer(
        dataset=dataset,
        galaxies=[ag.Galaxy(redshift=0.5, pixelization=pixelization)],
    )

    titles = []
    plot_array = fit_interferometer_plots.plot_array

    def _plot_array(*args, title=None, **kwargs):
        titles.append(title)
        return plot_array(*args, title=title, **kwargs)

    monkeypatch.setattr(fit_interferometer_plots, "plot_array", _plot_array)

    natural_titles = [
        "Dirty Image (Natural)",
        "Dirty Model Image (Natural)",
        "Dirty Residual Map (Natural)",
    ]

    for function, filename in (
        (aplt.subplot_fit_interferometer, "fit"),
        (aplt.subplot_fit_dirty_images, "fit_dirty_images"),
        (aplt.subplot_fit_real_space, "fit_real_space"),
    ):
        titles.clear()
        plot_patch.paths = []

        function(fit=fit, output_path=plot_path, output_format="png")

        assert str(plot_path / f"{filename}.png") in plot_patch.paths
        assert titles == natural_titles
