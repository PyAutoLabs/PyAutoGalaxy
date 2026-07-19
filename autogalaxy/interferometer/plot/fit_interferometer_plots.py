import numpy as np
from pathlib import Path

import autoarray as aa
from autonerves.fitsable import hdu_list_for_output_from
from autoarray.plot import plot_visibilities_1d
from autoarray.plot.utils import subplots, conf_subplot_figsize, tight_layout

from autogalaxy.interferometer.fit_interferometer import FitInterferometer
from autogalaxy.galaxy.plot import galaxies_plots
from autogalaxy.util.plot_utils import plot_array, _save_subplot


def subplot_fit(
    fit: FitInterferometer,
    output_path=None,
    output_format=None,
    colormap="default",
    use_log10=False,
    residuals_symmetric_cmap: bool = True,
    title_prefix: str = None,
):
    """Create a three-panel subplot summarising a :class:`~autogalaxy.interferometer.fit_interferometer.FitInterferometer`.

    The panels show the visibility-space residual map, normalised residual
    map, and chi-squared map, each rendered as 1-D visibility plots via
    ``autoarray.plot.plot_visibilities_1d``.

    Parameters
    ----------
    fit : FitInterferometer
        The completed interferometer fit to visualise.
    output_path : str or None
        Directory in which to save the figure.  ``None`` → ``plt.show()``.
    output_format : str
        File format, e.g. ``"png"``.
    colormap : str
        Matplotlib colormap name, or ``"default"`` (passed through but not
        used by the 1-D visibility renderer).
    use_log10 : bool
        Reserved for future log-stretch support (currently unused).
    residuals_symmetric_cmap : bool
        Reserved for future symmetric-colormap support (currently unused).
    """
    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)
    panels = [
        (fit.residual_map, _pf("Residual Map")),
        (fit.normalized_residual_map, _pf("Normalized Residual Map")),
        (fit.chi_squared_map, _pf("Chi-Squared Map")),
    ]
    n = len(panels)
    fig, axes = subplots(1, n, figsize=conf_subplot_figsize(1, n))
    axes_flat = list(axes.flatten())

    for i, (vis, title) in enumerate(panels):
        plot_visibilities_1d(vis, axes_flat[i], title)

    tight_layout()
    _save_subplot(fig, output_path, "fit", output_format)


def subplot_fit_dirty_images(
    fit: FitInterferometer,
    output_path=None,
    output_format=None,
    colormap="default",
    use_log10=False,
    residuals_symmetric_cmap: bool = True,
    title_prefix: str = None,
):
    """Create a six-panel subplot of dirty-image diagnostics for an interferometer fit.

    Dirty images are the real-space counterparts of the visibility-space data,
    obtained via an inverse Fourier transform.  The panels show: dirty image,
    dirty signal-to-noise map, dirty model image, dirty residual map, dirty
    normalised residual map, and dirty chi-squared map.

    Parameters
    ----------
    fit : FitInterferometer
        The completed interferometer fit to visualise.
    output_path : str or None
        Directory in which to save the figure.  ``None`` → ``plt.show()``.
    output_format : str
        File format, e.g. ``"png"``.
    colormap : str
        Matplotlib colormap name, or ``"default"``.
    use_log10 : bool
        Apply a log₁₀ stretch to the plotted values.
    residuals_symmetric_cmap : bool
        Reserved for future symmetric-colormap support (currently unused).
    """
    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)
    panels = [
        (fit.dirty_image, _pf("Dirty Image"), None),
        (fit.dirty_signal_to_noise_map, _pf("Dirty Signal-To-Noise Map"), None),
        (fit.dirty_model_image, _pf("Dirty Model Image"), None),
        (fit.dirty_residual_map, _pf("Dirty Residual Map"), None),
        (fit.dirty_normalized_residual_map, _pf("Dirty Normalized Residual Map"), r"$\sigma$"),
        (fit.dirty_chi_squared_map, _pf("Dirty Chi-Squared Map"), r"$\chi^2$"),
    ]
    n = len(panels)
    fig, axes = subplots(1, n, figsize=conf_subplot_figsize(1, n))
    axes_flat = list(axes.flatten())

    for i, (array, title, cb_unit) in enumerate(panels):
        plot_array(
            array=array,
            title=title,
            colormap=colormap,
            use_log10=use_log10,
            cb_unit=cb_unit,
            ax=axes_flat[i],
        )

    tight_layout()
    _save_subplot(fig, output_path, "fit_dirty_images", output_format)


def subplot_fit_real_space(
    fit: FitInterferometer,
    output_path=None,
    output_format=None,
    colormap="default",
    use_log10=False,
    title_prefix: str = None,
):
    """Create a real-space summary subplot for an interferometer fit.

    The exact panels depend on whether the fit includes a pixelization:

    - **No pixelization**: delegates to
      :func:`~autogalaxy.galaxy.plot.galaxies_plots.subplot_galaxies` which
      shows image, convergence, potential, and deflections on the light-profile
      grid.
    - **With pixelization**: shows three dirty-image panels (dirty image, dirty
      model image, dirty residual map), which are the best real-space
      representation available when a pixelized source is used.

    Parameters
    ----------
    fit : FitInterferometer
        The completed interferometer fit to visualise.
    output_path : str or None
        Directory in which to save the figure.  ``None`` → ``plt.show()``.
    output_format : str
        File format, e.g. ``"png"``.
    colormap : str
        Matplotlib colormap name, or ``"default"``.
    use_log10 : bool
        Apply a log₁₀ stretch to the plotted values.
    """
    galaxy_list = fit.galaxies_linear_light_profiles_to_light_profiles

    if not galaxy_list.has(cls=aa.Pixelization):
        galaxies_plots.subplot_galaxies(
            galaxies=galaxy_list,
            grid=fit.grids.lp,
            output_path=output_path,
            output_format=output_format,
            colormap=colormap,
            use_log10=use_log10,
            auto_filename="fit_real_space",
            title_prefix=title_prefix,
        )
    else:
        _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)
        panels = [
            (fit.dirty_image, _pf("Dirty Image")),
            (fit.dirty_model_image, _pf("Dirty Model Image")),
            (fit.dirty_residual_map, _pf("Dirty Residual Map")),
        ]
        n = len(panels)
        fig, axes = subplots(1, n, figsize=conf_subplot_figsize(1, n))
        axes_flat = list(axes.flatten())
        for i, (array, title) in enumerate(panels):
            plot_array(
                array=array,
                title=title,
                colormap=colormap,
                use_log10=use_log10,
                ax=axes_flat[i],
            )
        tight_layout()
        _save_subplot(fig, output_path, "fit_real_space", output_format)


def fits_galaxy_images(fit: FitInterferometer, output_path) -> None:
    """Write per-galaxy images from a ``FitInterferometer`` to ``galaxy_images.fits``.

    Extensions: ``mask``, ``galaxy_0``, ``galaxy_1``, …

    Parameters
    ----------
    fit
        The interferometer fit whose per-galaxy images are saved.
    output_path
        Directory in which to write ``galaxy_images.fits``.
    """
    image_list = [image.native_for_fits for image in fit.galaxy_image_dict.values()]
    hdu_list = hdu_list_for_output_from(
        values_list=[image_list[0].mask.astype("float")] + image_list,
        ext_name_list=["mask"] + [f"galaxy_{i}" for i in range(len(image_list))],
        header_dict=fit.dataset.real_space_mask.header_dict,
    )
    hdu_list.writeto(Path(output_path) / "galaxy_images.fits", overwrite=True)


def fits_dirty_images(fit: FitInterferometer, output_path) -> None:
    """Write dirty-image residual maps from a ``FitInterferometer`` to ``fit_dirty_images.fits``.

    Extensions: ``mask``, ``dirty_image``, ``dirty_noise_map``, ``dirty_model_image``,
    ``dirty_residual_map``, ``dirty_normalized_residual_map``, ``dirty_chi_squared_map``.

    Parameters
    ----------
    fit
        The interferometer fit whose dirty-image arrays are saved.
    output_path
        Directory in which to write ``fit_dirty_images.fits``.
    """
    image_list = [
        fit.dirty_image.native_for_fits,
        fit.dirty_noise_map.native_for_fits,
        fit.dirty_model_image.native_for_fits,
        fit.dirty_residual_map.native_for_fits,
        fit.dirty_normalized_residual_map.native_for_fits,
        fit.dirty_chi_squared_map.native_for_fits,
    ]
    hdu_list = hdu_list_for_output_from(
        values_list=[image_list[0].mask.astype("float")] + image_list,
        ext_name_list=[
            "mask",
            "dirty_image",
            "dirty_noise_map",
            "dirty_model_image",
            "dirty_residual_map",
            "dirty_normalized_residual_map",
            "dirty_chi_squared_map",
        ],
        header_dict=fit.dataset.real_space_mask.header_dict,
    )
    hdu_list.writeto(Path(output_path) / "fit_dirty_images.fits", overwrite=True)
