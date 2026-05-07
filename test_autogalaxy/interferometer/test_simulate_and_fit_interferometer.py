import os
import shutil

import numpy as np
import pytest

import autogalaxy as ag
from pathlib import Path


def test__perfect_fit__chi_squared_0():
    grid = ag.Grid2D.uniform(
        shape_native=(31, 31),
        pixel_scales=0.1,
        over_sample_size=1,
    )

    galaxy_0 = ag.Galaxy(
        redshift=0.5, light=ag.lp.Sersic(centre=(0.1, 0.1), intensity=0.1)
    )

    galaxy_1 = ag.Galaxy(
        redshift=0.5, light=ag.lp.Exponential(centre=(0.1, 0.1), intensity=0.5)
    )

    simulator = ag.SimulatorInterferometer(
        uv_wavelengths=np.ones(shape=(3, 2)),
        transformer_class=ag.TransformerDFT,
        exposure_time=300.0,
        noise_if_add_noise_false=1.0,
        noise_sigma=None,
    )

    dataset = simulator.via_galaxies_from(galaxies=[galaxy_0, galaxy_1], grid=grid)

    file_path = Path(Path(__file__).resolve().parent) / "data_temp" / "simulate_and_fit"

    try:
        shutil.rmtree(file_path)
    except FileNotFoundError:
        pass

    if Path(file_path).exists() is False:
        os.makedirs(file_path)

    from autoarray.dataset.plot.interferometer_plots import fits_interferometer

    fits_interferometer(
        dataset=dataset,
        data_path=Path(file_path) / "data.fits",
        noise_map_path=Path(file_path) / "noise_map.fits",
        uv_wavelengths_path=Path(file_path) / "uv_wavelengths.fits",
        overwrite=True,
    )

    real_space_mask = ag.Mask2D.all_false(
        shape_native=(31, 31),
        pixel_scales=0.1,
    )

    dataset = ag.Interferometer.from_fits(
        data_path=Path(file_path) / "data.fits",
        noise_map_path=Path(file_path) / "noise_map.fits",
        uv_wavelengths_path=Path(file_path) / "uv_wavelengths.fits",
        real_space_mask=real_space_mask,
        transformer_class=ag.TransformerDFT,
    )

    fit = ag.FitInterferometer(
        dataset=dataset,
        galaxies=[galaxy_0, galaxy_1],
    )

    assert fit.chi_squared == pytest.approx(0.0)

    pixelization = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(5, 5)),
        regularization=ag.reg.Constant(coefficient=0.0001),
    )

    galaxy_0 = ag.Galaxy(
        redshift=0.5, light=ag.lp.Sersic(centre=(0.1, 0.1), intensity=0.1)
    )

    galaxy_1 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=dataset,
        galaxies=[galaxy_0, galaxy_1],
    )
    assert abs(fit.chi_squared) < 1.0e-4

    file_path = Path(Path(__file__).resolve().parent) / "data_temp"

    shutil.rmtree(file_path, ignore_errors=True)


def test__simulate_interferometer_data_and_fit__known_likelihood():
    mask = ag.Mask2D.circular(
        radius=3.0,
        shape_native=(31, 31),
        pixel_scales=0.2,
    )

    grid = ag.Grid2D.from_mask(mask=mask)

    galaxy_0 = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.Sersic(centre=(0.1, 0.1), intensity=0.1),
        mass=ag.mp.Isothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )

    pixelization = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(16, 16)),
        regularization=ag.reg.Constant(coefficient=(1.0)),
    )

    galaxy_1 = ag.Galaxy(redshift=1.0, pixelization=pixelization)

    simulator = ag.SimulatorInterferometer(
        uv_wavelengths=np.ones(shape=(7, 2)),
        transformer_class=ag.TransformerDFT,
        exposure_time=300.0,
        noise_seed=1,
    )

    dataset = simulator.via_galaxies_from(galaxies=[galaxy_0, galaxy_1], grid=grid)

    fit = ag.FitInterferometer(
        dataset=dataset,
        galaxies=[galaxy_0, galaxy_1],
    )

    assert fit.figure_of_merit == pytest.approx(-5.05513095, 1.0e-2)


def test__linear_light_profiles_agree_with_standard_light_profiles():
    grid = ag.Grid2D.uniform(
        shape_native=(31, 31),
        pixel_scales=0.1,
        over_sample_size=1,
    )

    galaxy = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.Sersic(intensity=0.1, sersic_index=1.0),
        disk=ag.lp.Sersic(intensity=0.2, sersic_index=4.0),
    )

    simulator = ag.SimulatorInterferometer(
        uv_wavelengths=np.array(
            [
                [0.04, 200.0, 0.3, 400000.0, 60000000.0],
                [0.00003, 500.0, 600000.0, 0.1, 75555555],
            ]
        ),
        transformer_class=ag.TransformerDFT,
        exposure_time=300.0,
        noise_if_add_noise_false=1.0,
        noise_sigma=None,
    )

    dataset = simulator.via_galaxies_from(galaxies=[galaxy], grid=grid)

    dataset = ag.Interferometer(
        data=dataset.data,
        noise_map=dataset.noise_map,
        uv_wavelengths=dataset.uv_wavelengths,
        real_space_mask=dataset.real_space_mask,
        transformer_class=ag.TransformerDFT,
    )

    fit = ag.FitInterferometer(
        dataset=dataset,
        galaxies=[galaxy],
    )

    galaxy_linear = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp_linear.Sersic(sersic_index=1.0),
        disk=ag.lp_linear.Sersic(sersic_index=4.0),
    )

    fit_linear = ag.FitInterferometer(
        dataset=dataset,
        galaxies=[galaxy_linear],
        settings=ag.Settings(no_regularization_add_to_curvature_diag_value=False),
    )

    assert fit_linear.inversion.reconstruction == pytest.approx(
        np.array([0.1, 0.2]), 1.0e-2
    )
    assert fit_linear.linear_light_profile_intensity_dict[
        galaxy_linear.bulge
    ] == pytest.approx(0.1, 1.0e-2)
    assert fit_linear.linear_light_profile_intensity_dict[
        galaxy_linear.disk
    ] == pytest.approx(0.2, 1.0e-2)
    assert fit.log_likelihood == pytest.approx(fit_linear.log_likelihood, 1.0e-4)

    galaxy_image = galaxy.image_2d_from(grid=dataset.grids.lp)

    assert fit_linear.galaxy_image_dict[galaxy_linear] == pytest.approx(
        galaxy_image.array, 1.0e-4
    )

    galaxy_visibilities = galaxy.visibilities_from(
        grid=dataset.grids.lp, transformer=dataset.transformer
    )

    assert fit_linear.galaxy_model_visibilities_dict[
        galaxy_linear
    ].array == pytest.approx(galaxy_visibilities.array, 1.0e-4)
