import shutil
import pytest

import autogalaxy as ag
from autogalaxy.ellipse.model.plotter import PlotterEllipse
from pathlib import Path

directory = Path(__file__).resolve().parent


@pytest.fixture(name="plot_path")
def make_plotter_plotter_setup():
    return directory / "files"


def test__fit_ellipse__each_data_variant_has_own_filename(
    masked_imaging_7x7, plot_path, plot_patch
):
    if Path(plot_path).exists():
        shutil.rmtree(plot_path)

    fit_list = [
        ag.FitEllipse(
            dataset=masked_imaging_7x7,
            ellipse=ag.Ellipse(
                centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=major_axis
            ),
        )
        for major_axis in (0.2, 0.4)
    ]

    plotter = PlotterEllipse(image_path=plot_path)

    plotter.fit_ellipse(fit_list=fit_list)

    for filename in (
        "ellipse_fit.png",
        "ellipse_fit_no_ellipse.png",
        "ellipse_fit_log10.png",
        "ellipse_fit_no_ellipse_log10.png",
    ):
        assert str(Path(plot_path) / filename) in plot_patch.paths
