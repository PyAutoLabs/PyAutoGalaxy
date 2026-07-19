import autoarray as aa

from autoarray.dataset.plot.interferometer_plots import subplot_interferometer_dataset

from autonerves.fitsable import hdu_list_for_output_from

from autogalaxy.interferometer.fit_interferometer import FitInterferometer
from autogalaxy.interferometer.plot import fit_interferometer_plots
from autogalaxy.interferometer.plot.fit_interferometer_plots import (
    fits_galaxy_images,
    fits_dirty_images,
)
from autogalaxy.analysis.plotter import Plotter, plot_setting


class PlotterInterferometer(Plotter):
    def interferometer(self, dataset: aa.Interferometer):
        """
        Output visualization of an ``Interferometer`` dataset.

        Controlled by the ``[dataset]`` / ``[interferometer]`` sections of
        ``config/visualize/plots.yaml``.  Outputs a dirty-image subplot and,
        when enabled, a FITS file containing the mask, visibilities, noise map,
        and UV-wavelengths arrays.

        Parameters
        ----------
        dataset
            The interferometer dataset to visualize.
        """
        def should_plot(name):
            return plot_setting(section=["dataset", "interferometer"], name=name)

        if should_plot("subplot_dataset"):
            subplot_interferometer_dataset(
                dataset,
                output_path=self.image_path,
                output_filename="dataset",
                output_format=self.fmt[0] if isinstance(self.fmt, (list, tuple)) else self.fmt,
                title_prefix=self.title_prefix,
            )

        if should_plot("fits_dataset"):

            hdu_list = hdu_list_for_output_from(
                values_list=[
                    dataset.real_space_mask.astype("float"),
                    dataset.data.in_array,
                    dataset.noise_map.in_array,
                    dataset.uv_wavelengths,
                ],
                ext_name_list=["mask", "data", "noise_map", "uv_wavelengths"],
                header_dict=dataset.real_space_mask.header_dict,
            )

            hdu_list.writeto(self.image_path / "dataset.fits", overwrite=True)

    def fit_interferometer(
        self,
        fit: FitInterferometer,
        quick_update: bool = False,
    ):
        """
        Output visualization of a ``FitInterferometer`` object.

        Controlled by the ``[fit]`` / ``[fit_interferometer]`` sections of
        ``config/visualize/plots.yaml``.  Outputs the main fit subplot, a
        dirty-images subplot, and when enabled a real-space subplot and FITS
        residual files.

        Parameters
        ----------
        fit
            The interferometer fit to visualize.
        quick_update
            When ``True`` only the essential subplots are written; the
            real-space subplot and FITS outputs are skipped.
        """
        def should_plot(name):
            return plot_setting(section=["fit", "fit_interferometer"], name=name)

        if should_plot("subplot_fit") or quick_update:
            fit_interferometer_plots.subplot_fit(
                fit=fit,
                output_path=self.image_path,
                output_format=self.fmt,
                title_prefix=self.title_prefix,
            )

        if should_plot("subplot_fit_dirty_images") or quick_update:
            fit_interferometer_plots.subplot_fit_dirty_images(
                fit=fit,
                output_path=self.image_path,
                output_format=self.fmt,
                title_prefix=self.title_prefix,
            )

        if quick_update:
            return

        if should_plot("subplot_fit_real_space"):
            fit_interferometer_plots.subplot_fit_real_space(
                fit=fit,
                output_path=self.image_path,
                output_format=self.fmt,
                title_prefix=self.title_prefix,
            )

        if should_plot("fits_galaxy_images"):
            fits_galaxy_images(fit=fit, output_path=self.image_path)

        if should_plot("fits_dirty_images"):
            fits_dirty_images(fit=fit, output_path=self.image_path)
