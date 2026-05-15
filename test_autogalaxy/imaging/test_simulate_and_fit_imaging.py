import os
import shutil

import numpy as np
import pytest

import autogalaxy as ag
from pathlib import Path


def test__perfect_fit__simulate_and_reload__chi_squared_zero():
    grid = ag.Grid2D.uniform(
        shape_native=(11, 11),
        pixel_scales=0.2,
        over_sample_size=1,
    )

    psf = ag.Convolver.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    galaxy_0 = ag.Galaxy(
        redshift=0.5, light=ag.lp.Sersic(centre=(0.1, 0.1), intensity=0.1)
    )
    galaxy_1 = ag.Galaxy(
        redshift=0.5, light=ag.lp.Exponential(centre=(0.1, 0.1), intensity=0.5)
    )

    simulator = ag.SimulatorImaging(
        exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False
    )

    dataset = simulator.via_galaxies_from(galaxies=[galaxy_0, galaxy_1], grid=grid)
    dataset.noise_map = ag.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )

    file_path = Path(Path(__file__).resolve().parent) / "data_temp" / "simulate_and_fit"

    try:
        shutil.rmtree(file_path)
    except FileNotFoundError:
        pass

    if Path(file_path).exists() is False:
        os.makedirs(file_path)

    from autoarray.dataset.plot.imaging_plots import fits_imaging

    fits_imaging(
        dataset=dataset,
        data_path=Path(file_path) / "data.fits",
        noise_map_path=Path(file_path) / "noise_map.fits",
        psf_path=Path(file_path) / "psf.fits",
        overwrite=True,
    )

    dataset = ag.Imaging.from_fits(
        data_path=Path(file_path) / "data.fits",
        noise_map_path=Path(file_path) / "noise_map.fits",
        psf_path=Path(file_path) / "psf.fits",
        pixel_scales=0.2,
        over_sample_size_lp=1,
    )

    mask = ag.Mask2D.circular(
        shape_native=dataset.data.shape_native,
        pixel_scales=0.2,
        radius=0.8,
    )

    masked_dataset = dataset.apply_mask(mask=mask)

    fit = ag.FitImaging(dataset=masked_dataset, galaxies=[galaxy_0, galaxy_1])

    assert fit.chi_squared == pytest.approx(0.0, 1e-4)

    file_path = Path(Path(__file__).resolve().parent) / "data_temp"

    if Path(file_path).exists() is True:
        shutil.rmtree(file_path)


def _perfect_fit_dataset(galaxies, grid):
    """Helper: simulate noiseless imaging and zero out the noise map for chi^2 tests."""
    psf = ag.Convolver.from_gaussian(
        shape_native=(3, 3), pixel_scales=grid.pixel_scales[0], sigma=0.05, normalize=True
    )
    simulator = ag.SimulatorImaging(
        exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False
    )
    dataset = simulator.via_galaxies_from(galaxies=galaxies, grid=grid)
    dataset.noise_map = ag.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=grid.pixel_scales
    )
    return dataset


def test__perfect_fit__sim_offset_centre__fit_with_dataset_model_grid_offset__chi_squared_zero():
    """Sim a profile with offset centre; fit with origin profile + DatasetModel.grid_offset."""
    grid = ag.Grid2D.uniform(shape_native=(31, 31), pixel_scales=0.2, over_sample_size=1)
    centre = (0.3, 0.2)

    sim_galaxy = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.Sersic(centre=centre, intensity=0.5, effective_radius=0.5),
    )
    dataset = _perfect_fit_dataset([sim_galaxy], grid)
    mask = ag.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=2.5
    )
    masked = dataset.apply_mask(mask=mask)

    fit_galaxy = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.Sersic(centre=(0.0, 0.0), intensity=0.5, effective_radius=0.5),
    )
    dataset_model = ag.DatasetModel(grid_offset=centre)
    fit = ag.FitImaging(
        dataset=masked, galaxies=[fit_galaxy], dataset_model=dataset_model
    )

    assert fit.chi_squared == pytest.approx(0.0, abs=1e-4)


def test__perfect_fit__sim_rotated_ellipse__fit_with_dataset_model_grid_rotation__chi_squared_zero():
    """Sim a rotated ellipse; fit with axis-aligned profile + DatasetModel.grid_rotation_angle.

    Convention: profile with ell-angle theta is equivalent to grid rotated by -theta,
    so fit with grid_rotation_angle=-theta to recover chi^2 = 0.
    """
    grid = ag.Grid2D.uniform(shape_native=(31, 31), pixel_scales=0.2, over_sample_size=1)
    theta = 15.0

    sim_galaxy = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.Sersic(
            centre=(0.0, 0.0),
            ell_comps=ag.convert.ell_comps_from(axis_ratio=0.6, angle=theta),
            intensity=0.5,
            effective_radius=0.5,
        ),
    )
    dataset = _perfect_fit_dataset([sim_galaxy], grid)
    mask = ag.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=2.5
    )
    masked = dataset.apply_mask(mask=mask)

    fit_galaxy = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.Sersic(
            centre=(0.0, 0.0),
            ell_comps=ag.convert.ell_comps_from(axis_ratio=0.6, angle=0.0),
            intensity=0.5,
            effective_radius=0.5,
        ),
    )
    dataset_model = ag.DatasetModel(grid_rotation_angle=-theta)
    fit = ag.FitImaging(
        dataset=masked, galaxies=[fit_galaxy], dataset_model=dataset_model
    )

    assert fit.chi_squared == pytest.approx(0.0, abs=1e-4)


def test__perfect_fit__sim_offset_and_rotated__fit_with_dataset_model_offset_and_rotation__chi_squared_zero():
    """Combined: sim with offset centre AND rotated ellipse, fit with identity profile +
    DatasetModel(grid_offset, grid_rotation_angle)."""
    grid = ag.Grid2D.uniform(shape_native=(31, 31), pixel_scales=0.2, over_sample_size=1)
    centre = (0.3, 0.2)
    theta = 12.0

    sim_galaxy = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.Sersic(
            centre=centre,
            ell_comps=ag.convert.ell_comps_from(axis_ratio=0.6, angle=theta),
            intensity=0.5,
            effective_radius=0.5,
        ),
    )
    dataset = _perfect_fit_dataset([sim_galaxy], grid)
    mask = ag.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=2.5
    )
    masked = dataset.apply_mask(mask=mask)

    fit_galaxy = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.Sersic(
            centre=(0.0, 0.0),
            ell_comps=ag.convert.ell_comps_from(axis_ratio=0.6, angle=0.0),
            intensity=0.5,
            effective_radius=0.5,
        ),
    )
    dataset_model = ag.DatasetModel(
        grid_offset=centre, grid_rotation_angle=-theta
    )
    fit = ag.FitImaging(
        dataset=masked, galaxies=[fit_galaxy], dataset_model=dataset_model
    )

    assert fit.chi_squared == pytest.approx(0.0, abs=1e-4)


def test__simulate_imaging_data_and_fit__standard_galaxies__known_figure_of_merit():
    grid = ag.Grid2D.uniform(shape_native=(31, 31), pixel_scales=0.2)

    psf = ag.Convolver.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    galaxy_0 = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.Sersic(centre=(0.1, 0.1), intensity=0.1),
        disk=ag.lp.Sersic(centre=(0.2, 0.2), intensity=0.2),
    )

    pixelization = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(16, 16)),
        regularization=ag.reg.Constant(coefficient=(1.0)),
    )

    galaxy_1 = ag.Galaxy(redshift=1.0, pixelization=pixelization)

    simulator = ag.SimulatorImaging(exposure_time=300.0, psf=psf, noise_seed=1)

    dataset = simulator.via_galaxies_from(galaxies=[galaxy_0, galaxy_1], grid=grid)

    mask = ag.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=2.005
    )

    masked_dataset = dataset.apply_mask(mask=mask)

    fit = ag.FitImaging(dataset=masked_dataset, galaxies=[galaxy_0, galaxy_1])

    assert fit.figure_of_merit == pytest.approx(579.015739085647, 1.0e-2)


def test__simulate_imaging_data_and_fit__basis_galaxies__same_figure_of_merit_as_standard():
    grid = ag.Grid2D.uniform(shape_native=(31, 31), pixel_scales=0.2)

    psf = ag.Convolver.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    pixelization = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(16, 16)),
        regularization=ag.reg.Constant(coefficient=(1.0)),
    )

    galaxy_1 = ag.Galaxy(redshift=1.0, pixelization=pixelization)

    simulator = ag.SimulatorImaging(exposure_time=300.0, psf=psf, noise_seed=1)

    basis = ag.lp_basis.Basis(
        profile_list=[
            ag.lp.Sersic(centre=(0.1, 0.1), intensity=0.1),
            ag.lp.Sersic(centre=(0.2, 0.2), intensity=0.2),
        ]
    )

    galaxy_0 = ag.Galaxy(redshift=0.5, bulge=basis)

    dataset = simulator.via_galaxies_from(galaxies=[galaxy_0, galaxy_1], grid=grid)

    mask = ag.Mask2D.circular(
        shape_native=dataset.data.shape_native, pixel_scales=0.2, radius=2.005
    )

    masked_dataset = dataset.apply_mask(mask=mask)

    fit = ag.FitImaging(dataset=masked_dataset, galaxies=[galaxy_0, galaxy_1])

    assert fit.figure_of_merit == pytest.approx(579.015739085647, 1.0e-2)


def test__linear_light_profiles_agree_with_standard__reconstruction_recovers_intensities():
    grid = ag.Grid2D.uniform(
        shape_native=(11, 11),
        pixel_scales=0.2,
        over_sample_size=1,
    )

    psf = ag.Convolver.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    galaxy = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.Sersic(centre=(0.05, 0.05), intensity=0.1, sersic_index=1.0),
        disk=ag.lp.Sersic(centre=(0.05, 0.05), intensity=0.2, sersic_index=4.0),
    )

    simulator = ag.SimulatorImaging(
        exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False
    )

    dataset = simulator.via_galaxies_from(galaxies=[galaxy], grid=grid)
    dataset.noise_map = ag.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )

    mask = ag.Mask2D.circular(
        shape_native=dataset.data.shape_native,
        pixel_scales=0.2,
        radius=0.81,
    )

    masked_dataset = dataset.apply_mask(mask=mask)
    masked_dataset = masked_dataset.apply_over_sampling(over_sample_size_lp=1)

    galaxy_linear = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=1.0),
        disk=ag.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=4.0),
    )

    fit_linear = ag.FitImaging(
        dataset=masked_dataset,
        galaxies=[galaxy_linear],
    )

    assert fit_linear.inversion.reconstruction == pytest.approx(
        np.array([0.1, 0.2]), 1.0e-2
    )


def test__linear_light_profiles_agree_with_standard__intensity_dict_recovers_correct_values():
    grid = ag.Grid2D.uniform(
        shape_native=(11, 11),
        pixel_scales=0.2,
        over_sample_size=1,
    )

    psf = ag.Convolver.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    galaxy = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.Sersic(centre=(0.05, 0.05), intensity=0.1, sersic_index=1.0),
        disk=ag.lp.Sersic(centre=(0.05, 0.05), intensity=0.2, sersic_index=4.0),
    )

    simulator = ag.SimulatorImaging(
        exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False
    )

    dataset = simulator.via_galaxies_from(galaxies=[galaxy], grid=grid)
    dataset.noise_map = ag.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )

    mask = ag.Mask2D.circular(
        shape_native=dataset.data.shape_native,
        pixel_scales=0.2,
        radius=0.81,
    )

    masked_dataset = dataset.apply_mask(mask=mask)
    masked_dataset = masked_dataset.apply_over_sampling(over_sample_size_lp=1)

    galaxy_linear = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=1.0),
        disk=ag.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=4.0),
    )

    fit_linear = ag.FitImaging(
        dataset=masked_dataset,
        galaxies=[galaxy_linear],
    )

    assert fit_linear.linear_light_profile_intensity_dict[
        galaxy_linear.bulge
    ] == pytest.approx(0.1, 1.0e-2)
    assert fit_linear.linear_light_profile_intensity_dict[
        galaxy_linear.disk
    ] == pytest.approx(0.2, 1.0e-2)


def test__linear_light_profiles_agree_with_standard__figure_of_merit_matches_standard_fit():
    grid = ag.Grid2D.uniform(
        shape_native=(11, 11),
        pixel_scales=0.2,
        over_sample_size=1,
    )

    psf = ag.Convolver.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    galaxy = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.Sersic(centre=(0.05, 0.05), intensity=0.1, sersic_index=1.0),
        disk=ag.lp.Sersic(centre=(0.05, 0.05), intensity=0.2, sersic_index=4.0),
    )

    simulator = ag.SimulatorImaging(
        exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False
    )

    dataset = simulator.via_galaxies_from(galaxies=[galaxy], grid=grid)
    dataset.noise_map = ag.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )

    mask = ag.Mask2D.circular(
        shape_native=dataset.data.shape_native,
        pixel_scales=0.2,
        radius=0.81,
    )

    masked_dataset = dataset.apply_mask(mask=mask)
    masked_dataset = masked_dataset.apply_over_sampling(over_sample_size_lp=1)

    fit = ag.FitImaging(dataset=masked_dataset, galaxies=[galaxy])

    galaxy_linear = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=1.0),
        disk=ag.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=4.0),
    )

    fit_linear = ag.FitImaging(
        dataset=masked_dataset,
        galaxies=[galaxy_linear],
    )

    assert fit.log_likelihood == fit_linear.figure_of_merit
    assert fit_linear.figure_of_merit == pytest.approx(-45.02798, 1.0e-4)


def test__linear_light_profiles_agree_with_standard__galaxy_model_image_matches_blurred_image():
    grid = ag.Grid2D.uniform(
        shape_native=(11, 11),
        pixel_scales=0.2,
        over_sample_size=1,
    )

    psf = ag.Convolver.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    galaxy = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.Sersic(centre=(0.05, 0.05), intensity=0.1, sersic_index=1.0),
        disk=ag.lp.Sersic(centre=(0.05, 0.05), intensity=0.2, sersic_index=4.0),
    )

    simulator = ag.SimulatorImaging(
        exposure_time=300.0, psf=psf, add_poisson_noise_to_data=False
    )

    dataset = simulator.via_galaxies_from(galaxies=[galaxy], grid=grid)
    dataset.noise_map = ag.Array2D.ones(
        shape_native=dataset.data.shape_native, pixel_scales=0.2
    )

    mask = ag.Mask2D.circular(
        shape_native=dataset.data.shape_native,
        pixel_scales=0.2,
        radius=0.81,
    )

    masked_dataset = dataset.apply_mask(mask=mask)
    masked_dataset = masked_dataset.apply_over_sampling(over_sample_size_lp=1)

    galaxy_linear = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=1.0),
        disk=ag.lp_linear.Sersic(centre=(0.05, 0.05), sersic_index=4.0),
    )

    fit_linear = ag.FitImaging(
        dataset=masked_dataset,
        galaxies=[galaxy_linear],
    )

    galaxy_image = galaxy.blurred_image_2d_from(
        grid=masked_dataset.grids.lp,
        psf=masked_dataset.psf,
        blurring_grid=masked_dataset.grids.blurring,
    )

    assert fit_linear.galaxy_model_image_dict[galaxy_linear] == pytest.approx(
        galaxy_image.array, 1.0e-4
    )
