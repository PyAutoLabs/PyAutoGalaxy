from pathlib import Path

import autoarray as aa
from autonerves.fitsable import hdu_list_for_output_from
from autoarray.plot.utils import subplots, conf_subplot_figsize, tight_layout

from autogalaxy.imaging.fit_imaging import FitImaging
from autogalaxy.util.plot_utils import plot_array, _save_subplot


def subplot_fit(
    fit: FitImaging,
    output_path=None,
    output_format=None,
    colormap="default",
    use_log10=False,
    positions=None,
    residuals_symmetric_cmap: bool = True,
    title_prefix: str = None,
):
    """Create a six-panel subplot summarising a :class:`~autogalaxy.imaging.fit_imaging.FitImaging`.

    The panels show, in order: data, signal-to-noise map, model image,
    residual map, normalised residual map, and chi-squared map.

    Parameters
    ----------
    fit : FitImaging
        The completed imaging fit to visualise.
    output_path : str or None
        Directory in which to save the figure.  ``None`` → ``plt.show()``.
    output_format : str
        File format, e.g. ``"png"``.
    colormap : str
        Matplotlib colormap name, or ``"default"``.
    use_log10 : bool
        Apply a log₁₀ stretch to the plotted values.
    positions : array-like or None
        Point positions to scatter-plot over each panel.
    residuals_symmetric_cmap : bool
        Reserved for future symmetric-colormap support on residual panels
        (currently unused).
    """
    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)
    panels = [
        (fit.data, _pf("Data"), None),
        (fit.signal_to_noise_map, _pf("Signal-To-Noise Map"), None),
        (fit.model_data, _pf("Model Image"), None),
        (fit.residual_map, _pf("Residual Map"), None),
        (fit.normalized_residual_map, _pf("Normalized Residual Map"), None),
        (fit.chi_squared_map, _pf("Chi-Squared Map"), r"$\chi^2$"),
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
            positions=positions,
            cb_unit=cb_unit,
            ax=axes_flat[i],
        )

    tight_layout()
    _save_subplot(fig, output_path, "fit", output_format)


def subplot_of_galaxy(
    fit: FitImaging,
    galaxy_index: int,
    output_path=None,
    output_format=None,
    colormap="default",
    use_log10=False,
    positions=None,
    residuals_symmetric_cmap: bool = True,
    title_prefix: str = None,
):
    """Create a three-panel subplot focused on a single galaxy contribution.

    Shows the observed data alongside the subtracted image and model image
    for the galaxy at *galaxy_index* in the fitted galaxy list.  This is
    useful for inspecting the contribution of individual galaxies when
    multiple galaxies are being fitted simultaneously.

    Parameters
    ----------
    fit : FitImaging
        The completed imaging fit to visualise.
    galaxy_index : int
        Index into ``fit.galaxies`` selecting which galaxy to highlight.
    output_path : str or None
        Directory in which to save the figure.  ``None`` → ``plt.show()``.
    output_format : str
        File format, e.g. ``"png"``.
    colormap : str
        Matplotlib colormap name, or ``"default"``.
    use_log10 : bool
        Apply a log₁₀ stretch to the plotted values.
    positions : array-like or None
        Point positions to scatter-plot over each panel.
    residuals_symmetric_cmap : bool
        Reserved for future symmetric-colormap support (currently unused).
    """
    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)
    panels = [
        (fit.data, _pf("Data")),
        (
            fit.subtracted_images_of_galaxies_list[galaxy_index],
            _pf(f"Subtracted Image of Galaxy {galaxy_index}"),
        ),
        (
            fit.model_images_of_galaxies_list[galaxy_index],
            _pf(f"Model Image of Galaxy {galaxy_index}"),
        ),
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
    _save_subplot(fig, output_path, f"of_galaxy_{galaxy_index}", output_format)


def subplot_fit_imaging_list(
    fit_list,
    output_path=None,
    output_filename: str = "fit_combined",
    output_format=None,
    title_prefix: str = None,
):
    """
    n×5 subplot summarising a list of ``FitImaging`` objects.

    Each row shows: Data | Signal-To-Noise Map | Model Image |
    Normalized Residual Map | Chi-Squared Map

    Parameters
    ----------
    fit_list
        List of ``FitImaging`` instances.
    output_path
        Directory to save the figure.  ``None`` calls ``plt.show()``.
    output_filename
        Base filename without extension.
    output_format
        File format string or list, e.g. ``"png"`` or ``["png"]``.
    """
    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)
    n = len(fit_list)
    fig, axes = subplots(n, 5, figsize=conf_subplot_figsize(n, 5))
    if n == 1:
        axes = [axes]
    for i, fit in enumerate(fit_list):
        plot_array(array=fit.data, title=_pf("Data"), ax=axes[i][0])
        plot_array(array=fit.signal_to_noise_map, title=_pf("Signal-To-Noise Map"), ax=axes[i][1])
        plot_array(array=fit.model_data, title=_pf("Model Image"), ax=axes[i][2])
        plot_array(array=fit.normalized_residual_map, title=_pf("Normalized Residual Map"), cb_unit=r"$\sigma$", ax=axes[i][3])
        plot_array(array=fit.chi_squared_map, title=_pf("Chi-Squared Map"), cb_unit=r"$\chi^2$", ax=axes[i][4])
    tight_layout()
    _save_subplot(fig, output_path, output_filename, output_format)


def fits_fit(fit: FitImaging, output_path) -> None:
    """Write fit residual maps from a ``FitImaging`` to ``fit.fits``.

    Extensions: ``mask``, ``model_data``, ``residual_map``,
    ``normalized_residual_map``, ``chi_squared_map``.

    Parameters
    ----------
    fit
        The imaging fit whose residual arrays are saved.
    output_path
        Directory in which to write ``fit.fits``.
    """
    image_list = [
        fit.model_data.native_for_fits,
        fit.residual_map.native_for_fits,
        fit.normalized_residual_map.native_for_fits,
        fit.chi_squared_map.native_for_fits,
    ]
    hdu_list = hdu_list_for_output_from(
        values_list=[image_list[0].mask.astype("float")] + image_list,
        ext_name_list=[
            "mask",
            "model_data",
            "residual_map",
            "normalized_residual_map",
            "chi_squared_map",
        ],
        header_dict=fit.mask.header_dict,
    )
    hdu_list.writeto(Path(output_path) / "fit.fits", overwrite=True)


def fits_galaxy_images(fit: FitImaging, output_path) -> None:
    """Write per-galaxy images from a ``FitImaging`` to ``galaxy_images.fits``.

    Extensions: ``mask``, ``galaxy_0``, ``galaxy_1``, …

    Parameters
    ----------
    fit
        The imaging fit whose per-galaxy images are saved.
    output_path
        Directory in which to write ``galaxy_images.fits``.
    """
    image_list = [image.native_for_fits for image in fit.galaxy_image_dict.values()]
    hdu_list = hdu_list_for_output_from(
        values_list=[image_list[0].mask.astype("float")] + image_list,
        ext_name_list=["mask"] + [f"galaxy_{i}" for i in range(len(image_list))],
        header_dict=fit.mask.header_dict,
    )
    hdu_list.writeto(Path(output_path) / "galaxy_images.fits", overwrite=True)


def fits_model_galaxy_images(fit: FitImaging, output_path) -> None:
    """Write per-galaxy model images from a ``FitImaging`` to ``model_galaxy_images.fits``.

    Extensions: ``mask``, ``galaxy_0``, ``galaxy_1``, …

    Parameters
    ----------
    fit
        The imaging fit whose per-galaxy model images are saved.
    output_path
        Directory in which to write ``model_galaxy_images.fits``.
    """
    image_list = [image.native_for_fits for image in fit.galaxy_model_image_dict.values()]
    hdu_list = hdu_list_for_output_from(
        values_list=[image_list[0].mask.astype("float")] + image_list,
        ext_name_list=["mask"] + [f"galaxy_{i}" for i in range(len(image_list))],
        header_dict=fit.mask.header_dict,
    )
    hdu_list.writeto(Path(output_path) / "model_galaxy_images.fits", overwrite=True)
