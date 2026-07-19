from typing import List

from autonerves.fitsable import hdu_list_for_output_from

import autoarray as aa

from autoarray.dataset.plot.imaging_plots import subplot_imaging_dataset

from autogalaxy.ellipse.fit_ellipse import FitEllipse
from autogalaxy.ellipse.plot import fit_ellipse_plots
from autogalaxy.analysis.plotter import Plotter, plot_setting


class PlotterEllipse(Plotter):
    def imaging(self, dataset: aa.Imaging):
        """
        Output visualization of an ``Imaging`` dataset for ellipse fitting.

        Controlled by the ``[dataset]`` / ``[imaging]`` sections of
        ``config/visualize/plots.yaml``.  Outputs a subplot of the imaging data
        and a FITS file containing the mask, data, and noise-map arrays.

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

        image_list = [
            dataset.data.native,
            dataset.noise_map.native,
        ]

        hdu_list = hdu_list_for_output_from(
            values_list=[image_list[0].mask.astype("float")] + image_list,
            ext_name_list=[
                "mask",
                "data",
                "noise_map",
            ],
            header_dict=dataset.mask.header_dict,
        )

        hdu_list.writeto(self.image_path / "dataset.fits", overwrite=True)

    def fit_ellipse(
        self,
        fit_list: List[FitEllipse],
    ):
        """
        Output visualization of a list of ``FitEllipse`` objects.

        Controlled by the ``[fit]`` / ``[fit_ellipse]`` sections of
        ``config/visualize/plots.yaml``.  Outputs data images with ellipse
        overlays, ellipse residual plots, and a combined fit subplot.

        Parameters
        ----------
        fit_list
            The list of ellipse fits to visualize (one per ellipse).
        """
        def should_plot(name):
            return plot_setting(section=["fit", "fit_ellipse"], name=name)

        if should_plot("data"):
            fit_ellipse_plots._plot_data(
                fit_list=fit_list,
                output_path=self.image_path,
                output_format=self.fmt,
                title_prefix=self.title_prefix,
            )

        if should_plot("ellipse_residuals"):
            fit_ellipse_plots._plot_ellipse_residuals(
                fit_list=fit_list,
                output_path=self.image_path,
                output_format=self.fmt,
            )

        if should_plot("data_no_ellipse"):
            fit_ellipse_plots._plot_data(
                fit_list=fit_list,
                output_path=self.image_path,
                output_format=self.fmt,
                disable_data_contours=True,
                title_prefix=self.title_prefix,
            )

        if should_plot("subplot_fit_ellipse"):
            fit_ellipse_plots.subplot_fit_ellipse(
                fit_list=fit_list,
                output_path=self.image_path,
                output_format=self.fmt,
                title_prefix=self.title_prefix,
            )

        fit_ellipse_plots._plot_data(
            fit_list=fit_list,
            output_path=self.image_path,
            output_format=self.fmt,
            use_log10=True,
            title_prefix=self.title_prefix,
        )

        if should_plot("data_no_ellipse"):
            fit_ellipse_plots._plot_data(
                fit_list=fit_list,
                output_path=self.image_path,
                output_format=self.fmt,
                use_log10=True,
                disable_data_contours=True,
                title_prefix=self.title_prefix,
            )
