
import os
from os import path
import pytest
from matplotlib import pyplot
from unittest.mock import MagicMock

from autoconf import conf
from autogalaxy import fixtures

import logging

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class PlotPatch:
    def __init__(self):
        self.paths = []

    def __call__(self, path, *args, **kwargs):
        self.paths.append(path)


@pytest.fixture(name="plot_patch")
def make_plot_patch(monkeypatch):
    import matplotlib.figure

    plot_patch = PlotPatch()
    monkeypatch.setattr(pyplot, "savefig", plot_patch)
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", plot_patch)
    monkeypatch.setattr(pyplot, "tight_layout", lambda *a, **kw: None)
    monkeypatch.setattr(pyplot, "colorbar", lambda *a, **kw: MagicMock())
    return plot_patch


directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path(request):
    conf.instance.push(
        new_path=path.join(directory, "config"),
        output_path=path.join(directory, "output"),
    )


@pytest.fixture(autouse=True, scope="session")
def remove_logs():
    yield
    for d, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".log"):
                os.remove(path.join(d, file))


### Datasets ###


@pytest.fixture(name="psf_3x3")
def make_psf_3x3():
    return fixtures.make_psf_3x3()


@pytest.fixture(name="psf_3x3_no_blur")
def make_psf_3x3_no_blur():
    return fixtures.make_psf_3x3_no_blur()


@pytest.fixture(name="rectangular_inversion_7x7_3x3")
def make_rectangular_inversion_7x7_3x3():
    return fixtures.make_rectangular_inversion_7x7_3x3()


@pytest.fixture(name="delaunay_inversion_9_3x3")
def make_delaunay_inversion_9_3x3():
    return fixtures.make_delaunay_inversion_9_3x3()


@pytest.fixture(name="voronoi_inversion_9_3x3")
def make_voronoi_inversion_9_3x3():
    return fixtures.make_voronoi_inversion_9_3x3()


@pytest.fixture(name="rectangular_mapper_7x7_3x3")
def make_rectangular_mapper_7x7_3x3():
    return fixtures.make_rectangular_mapper_7x7_3x3()


@pytest.fixture(name="interferometer_7_lop")
def make_interferometer_7_lop():
    return fixtures.make_interferometer_7_lop()


@pytest.fixture(name="fit_interferometer_7")
def make_fit_interferometer_7(interferometer_7):
    return fixtures.make_fit_interferometer_7()


@pytest.fixture(name="noise_map_7x7")
def make_noise_map_7x7():
    return fixtures.make_noise_map_7x7()


@pytest.fixture(name="imaging_7x7")
def make_imaging_7x7():
    return fixtures.make_imaging_7x7()


@pytest.fixture(name="image_7x7")
def make_image_7x7():
    return fixtures.make_image_7x7()


@pytest.fixture(name="masked_imaging_7x7")
def make_masked_imaging_7x7():
    return fixtures.make_masked_imaging_7x7()


@pytest.fixture(name="masked_imaging_7x7_sub_2")
def make_masked_imaging_7x7_sub_2():
    return fixtures.make_masked_imaging_7x7_sub_2()


@pytest.fixture(name="masked_imaging_covariance_7x7")
def make_masked_imaging_covariance_7x7():
    return fixtures.make_masked_imaging_covariance_7x7()


@pytest.fixture(name="masked_imaging_7x7_no_blur")
def make_masked_imaging_7x7_no_blur():
    return fixtures.make_masked_imaging_7x7_no_blur()


@pytest.fixture(name="interferometer_7")
def make_interferometer_7():
    return fixtures.make_interferometer_7()


@pytest.fixture(name="mask_2d_7x7")
def make_mask_2d_7x7():
    return fixtures.make_mask_2d_7x7()


@pytest.fixture(name="mask_2d_7x7_1_pix")
def make_mask_2d_7x7_1_pix():
    return fixtures.make_mask_2d_7x7_1_pix()


@pytest.fixture(name="grid_1d_7")
def make_grid_1d_7():
    return fixtures.make_grid_1d_7()


@pytest.fixture(name="grid_2d_iterate_7x7")
def make_grid_2d_iterate_7x7():
    return fixtures.make_grid_2d_iterate_7x7()


@pytest.fixture(name="grid_2d_irregular_7x7")
def make_grid_2d_irregular_7x7():
    return fixtures.make_grid_2d_irregular_7x7()


@pytest.fixture(name="grid_2d_irregular_7x7_list")
def make_grid_2d_irregular_7x7_list():
    return fixtures.make_grid_2d_irregular_7x7_list()


@pytest.fixture(name="transformer_7x7_7")
def make_transformer_7x7_7():
    return fixtures.make_transformer_7x7_7()


@pytest.fixture(name="blurring_grid_2d_7x7")
def make_blurring_grid_2d_7x7():
    return fixtures.make_blurring_grid_2d_7x7()


@pytest.fixture(name="visibilities_7")
def make_visibilities_7():
    return fixtures.make_visibilities_7()


@pytest.fixture(name="visibilities_noise_map_7")
def make_noise_map_7():
    return fixtures.make_visibilities_noise_map_7()


@pytest.fixture(name="uv_wavelengths_7x2")
def make_uv_wavelengths_7x2():
    return fixtures.make_uv_wavelengths_7x2()


@pytest.fixture(name="grid_2d_7x7")
def make_grid_2d_7x7():
    return fixtures.make_grid_2d_7x7()


@pytest.fixture(name="grid_2d_7x7_simple")
def make_grid_2d_7x7_simple():
    return fixtures.make_grid_2d_7x7_simple()


@pytest.fixture(name="ps_0")
def make_ps_0():
    return fixtures.make_ps_0()


@pytest.fixture(name="ps_1")
def make_ps_1():
    return fixtures.make_ps_1()


@pytest.fixture(name="lp_0")
def make_lp_0():
    return fixtures.make_lp_0()


@pytest.fixture(name="lp_1")
def make_lp_1():
    return fixtures.make_lp_1()


@pytest.fixture(name="lp_linear_0")
def make_lp_linear_0():
    return fixtures.make_lp_linear_0()


@pytest.fixture(name="lp_operated_0")
def make_lp_operated_0():
    return fixtures.make_lp_operated_0()


@pytest.fixture(name="mp_0")
def make_mp_0():
    return fixtures.make_mp_0()


@pytest.fixture(name="mp_1")
def make_mp_1():
    return fixtures.make_mp_1()


@pytest.fixture(name="lmp_0")
def make_lmp_0():
    return fixtures.make_lmp_0()


@pytest.fixture(name="dmp_0")
def make_dmp_0():
    return fixtures.make_dmp_0()


@pytest.fixture(name="dmp_1")
def make_dmp_1():
    return fixtures.make_dmp_1()


@pytest.fixture(name="smp_0")
def make_smp_0():
    return fixtures.make_smp_0()


@pytest.fixture(name="smp_1")
def make_smp_1():
    return fixtures.make_smp_1()


@pytest.fixture(name="gal_x1_lp")
def make_gal_x1_lp():
    return fixtures.make_gal_x1_lp()


@pytest.fixture(name="gal_x2_lp")
def make_gal_x2_lp():
    return fixtures.make_gal_x2_lp()


@pytest.fixture(name="gal_x1_mp")
def make_gal_x1_mp():
    return fixtures.make_gal_x1_mp()


@pytest.fixture(name="gal_x2_mp")
def make_gal_x2_mp():
    return fixtures.make_gal_x2_mp()


@pytest.fixture(name="gal_x1_lp_x1_mp")
def make_gal_x1_lp_x1_mp():
    return fixtures.make_gal_x1_lp_x1_mp()


@pytest.fixture(name="galaxies_7x7")
def make_galaxies_7x7():
    return fixtures.make_galaxies_7x7()


@pytest.fixture(name="galaxies_x2_7x7")
def make_galaxies_x2_7x7():
    return fixtures.make_galaxies_x2_7x7()


@pytest.fixture(name="adapt_galaxy_name_image_dict_7x7")
def make_adapt_galaxy_name_image_dict_7x7():
    return fixtures.make_adapt_galaxy_name_image_dict_7x7()


@pytest.fixture(name="adapt_images_7x7")
def make_adapt_images_7x7():
    return fixtures.make_adapt_images_7x7()


### ELLIPSE ###


@pytest.fixture(name="dataset_interp_7x7")
def make_dataset_interp_7x7():
    return fixtures.make_dataset_interp_7x7()


### QUANTITY ###


@pytest.fixture(name="dataset_quantity_7x7_array_2d")
def make_dataset_quantity_7x7_array_2d():
    return fixtures.make_dataset_quantity_7x7_array_2d()


@pytest.fixture(name="dataset_quantity_7x7_vector_yx_2d")
def make_dataset_quantity_7x7_vector_yx_2d():
    return fixtures.make_dataset_quantity_7x7_vector_yx_2d()


@pytest.fixture(name="fit_quantity_7x7_array_2d")
def make_fit_quantity_7x7_array_2d():
    return fixtures.make_fit_quantity_7x7_array_2d()


@pytest.fixture(name="fit_quantity_7x7_vector_yx_2d")
def make_fit_quantity_7x7_vector_yx_2d():
    return fixtures.make_fit_quantity_7x7_vector_yx_2d()


### FITS ###


@pytest.fixture(name="fit_imaging_7x7")
def make_fit_imaging_7x7():
    return fixtures.make_fit_imaging_7x7()


@pytest.fixture(name="fit_imaging_x2_galaxy_7x7")
def make_fit_imaging_x2_galaxy_7x7():
    return fixtures.make_fit_imaging_x2_galaxy_7x7()


@pytest.fixture(name="fit_imaging_x2_galaxy_inversion_7x7")
def make_fit_imaging_x2_galaxy_inversion_7x7():
    return fixtures.make_fit_imaging_x2_galaxy_inversion_7x7()


@pytest.fixture(name="fit_interferometer_7x7")
def make_fit_interferometer_7x7():
    return fixtures.make_fit_interferometer_7x7()


@pytest.fixture(name="fit_interferometer_x2_galaxy_inversion_7x7")
def make_fit_interferometer_x2_galaxy_inversion_7x7():
    return fixtures.make_fit_interferometer_x2_galaxy_inversion_7x7()


@pytest.fixture(name="voronoi_mapper_9_3x3")
def make_voronoi_mapper_9_3x3():
    return fixtures.make_voronoi_mapper_9_3x3()


@pytest.fixture(name="samples_summary_with_result")
def make_samples_summary_with_result():
    return fixtures.make_samples_summary_with_result()


@pytest.fixture(name="analysis_imaging_7x7")
def make_analysis_imaging_7x7():
    return fixtures.make_analysis_imaging_7x7()


@pytest.fixture(name="analysis_interferometer_7")
def make_analysis_interferometer_7():
    return fixtures.make_analysis_interferometer_7()


@pytest.fixture(name="analysis_ellipse_7x7")
def make_analysis_ellipse_7x7():
    return fixtures.make_analysis_ellipse_7x7()
