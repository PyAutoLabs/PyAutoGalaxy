from typing import List

from autonerves.fitsable import hdu_list_for_output_from

import autoarray as aa

from autoarray.dataset.plot.imaging_plots import subplot_imaging_dataset, subplot_imaging_dataset_list

from autogalaxy.imaging.fit_imaging import FitImaging
from autogalaxy.imaging.plot import fit_imaging_plots
from autogalaxy.imaging.plot.fit_imaging_plots import (
    subplot_fit_imaging_list,
    fits_fit,
    fits_galaxy_images,
    fits_model_galaxy_images,
)
from autogalaxy.analysis.plotter import Plotter, plot_setting


class PlotterImaging(Plotter):
    def imaging(self, dataset: aa.Imaging):
        """
        Output visualization of an ``Imaging`` dataset.

        Controlled by the ``[dataset]`` / ``[imaging]`` sections of
        ``config/visualize/plots.yaml``.  Outputs a subplot of the imaging data
        and, when enabled, a FITS file containing the mask, data, noise map, PSF,
        and over-sample-size arrays.

        Parameters
        ----------
        dataset
            The imaging dataset to visualize.
        """
        def should_plot(name):
            return plot_setting(section=["dataset", "imaging"], name=name)

        if should_plot("subplot_dataset"):
            subplot_imaging_dataset(
                dataset,
                output_path=self.image_path,
                output_format=self.fmt,
                title_prefix=self.title_prefix,
            )

        if should_plot("fits_dataset"):
            image_list = [
                dataset.data.native_for_fits,
                dataset.noise_map.native_for_fits,
                dataset.psf.kernel.native_for_fits,
                dataset.grids.lp.over_sample_size.native_for_fits.astype("float"),
                dataset.grids.pixelization.over_sample_size.native_for_fits.astype("float"),
            ]

            hdu_list = hdu_list_for_output_from(
                values_list=[image_list[0].mask.astype("float")] + image_list,
                ext_name_list=["mask", "data", "noise_map", "psf",
                               "over_sample_size_lp", "over_sample_size_pixelization"],
                header_dict=dataset.mask.header_dict,
            )
            hdu_list.writeto(self.image_path / "dataset.fits", overwrite=True)

    def fit_imaging(self, fit: FitImaging, quick_update: bool = False):
        """
        Output visualization of a ``FitImaging`` object.

        Controlled by the ``[fit]`` / ``[fit_imaging]`` sections of
        ``config/visualize/plots.yaml``.  Outputs the main fit subplot and,
        optionally, per-galaxy subplots and FITS residual files.

        Parameters
        ----------
        fit
            The imaging fit to visualize.
        quick_update
            When ``True`` only the essential ``subplot_fit`` is written; all
            other outputs are skipped.
        """
        def should_plot(name):
            return plot_setting(section=["fit", "fit_imaging"], name=name)

        if should_plot("subplot_fit") or quick_update:
            fit_imaging_plots.subplot_fit(
                fit=fit,
                output_path=self.image_path,
                output_format=self.fmt,
                title_prefix=self.title_prefix,
            )

        if quick_update:
            return

        if should_plot("subplot_of_galaxies"):
            galaxy_indices = list(range(len(fit.galaxies)))
            for galaxy_index in galaxy_indices:
                fit_imaging_plots.subplot_of_galaxy(
                    fit=fit,
                    galaxy_index=galaxy_index,
                    output_path=self.image_path,
                    output_format=self.fmt,
                    title_prefix=self.title_prefix,
                )

        if should_plot("fits_fit"):
            fits_fit(fit=fit, output_path=self.image_path)

        if should_plot("fits_galaxy_images"):
            fits_galaxy_images(fit=fit, output_path=self.image_path)

        if should_plot("fits_model_galaxy_images"):
            fits_model_galaxy_images(fit=fit, output_path=self.image_path)

    def imaging_combined(self, dataset_list: List[aa.Imaging]):
        """
        Output visualization of a list of ``Imaging`` datasets from a combined analysis.

        Controlled by the ``[dataset]`` / ``[imaging]`` sections of
        ``config/visualize/plots.yaml``.  Outputs a combined subplot with one
        row per dataset.

        Parameters
        ----------
        dataset_list
            The list of imaging datasets to visualize.
        """
        def should_plot(name):
            return plot_setting(section=["dataset", "imaging"], name=name)

        if should_plot("subplot_dataset"):
            subplot_imaging_dataset_list(
                dataset_list,
                output_path=self.image_path,
                output_format=self.fmt,
                title_prefix=self.title_prefix,
            )

    def fit_imaging_combined(self, fit_list: List[FitImaging]):
        """
        Output visualization of a list of ``FitImaging`` objects from a combined analysis.

        Controlled by the ``[fit]`` / ``[fit_imaging]`` sections of
        ``config/visualize/plots.yaml``.  Outputs a combined subplot with one
        row per fit.

        Parameters
        ----------
        fit_list
            The list of imaging fits to visualize.
        """
        def should_plot(name):
            return plot_setting(section=["fit", "fit_imaging"], name=name)

        if should_plot("subplot_fit"):
            subplot_fit_imaging_list(
                fit_list,
                output_path=self.image_path,
                output_format=self.fmt,
                title_prefix=self.title_prefix,
            )
