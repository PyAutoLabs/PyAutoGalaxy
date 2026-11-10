import shutil
import pytest

import autogalaxy as ag
from autogalaxy.ellipse.model import plotter as plotter_module
from autogalaxy.ellipse.model.plotter import PlotterEllipse
from pathlib import Path

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_plotter_plotter_setup():
    return directory / "files"


@pytest.fixture(name="fit_list")
def make_fit_list(masked_imaging_7x7):
    return [
        ag.FitEllipse(
            dataset=masked_imaging_7x7,
            ellipse=ag.Ellipse(
                centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=major_axis
            ),
        )
        for major_axis in (0.2, 0.4)
    ]


def _ellipse_fit_filenames(plot_patch):
    return sorted(
        Path(path).name
        for path in plot_patch.paths
        if Path(path).name.startswith("ellipse_fit")
    )


def test__fit_ellipse__data_writes_linear_and_log10_to_own_filenames(
    fit_list, plot_path, plot_patch
):
    if Path(plot_path).exists():
        shutil.rmtree(plot_path)

    plotter = PlotterEllipse(image_path=plot_path)

    plotter.fit_ellipse(fit_list=fit_list)

    assert _ellipse_fit_filenames(plot_patch) == [
        "ellipse_fit.png",
        "ellipse_fit_log10.png",
    ]


def test__fit_ellipse__data_off__no_ellipse_fit_images(
    fit_list, plot_path, plot_patch, monkeypatch
):
    if Path(plot_path).exists():
        shutil.rmtree(plot_path)

    monkeypatch.setattr(
        plotter_module, "plot_setting", lambda section, name: name != "data"
    )

    plotter = PlotterEllipse(image_path=plot_path)

    plotter.fit_ellipse(fit_list=fit_list)

    assert _ellipse_fit_filenames(plot_patch) == []
    assert str(Path(plot_path) / "ellipse_residuals.png") in plot_patch.paths
    assert str(Path(plot_path) / "fit_ellipse.png") in plot_patch.paths
