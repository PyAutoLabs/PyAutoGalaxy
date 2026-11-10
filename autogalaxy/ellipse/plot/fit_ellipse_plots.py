import numpy as np
import math
from typing import List, Optional

import autoarray as aa
from autoarray.plot.utils import subplots, conf_subplot_figsize, tight_layout

from autogalaxy.ellipse.plot import fit_ellipse_plot_util
from autogalaxy.ellipse.fit_ellipse import FitEllipse
from autogalaxy.util.plot_utils import plot_array, _save_subplot
from autogalaxy.util import error_util


def _plot_data(
    fit_list: List[FitEllipse],
    output_path=None,
    output_filename="ellipse_fit",
    output_format=None,
    colormap="default",
    use_log10=False,
    suffix: str = "",
    ax=None,
    title_prefix: str = None,
):
    """Plot the 2-D image data with fitted ellipse contours overlaid.

    For each :class:`~autogalaxy.ellipse.fit_ellipse.FitEllipse` in
    *fit_list* the major-axis sampling points are extracted and converted to
    ``Grid2DIrregular`` objects, which are then passed as *lines* and
    *positions* overlays to the underlying array-plot routine.

    Parameters
    ----------
    fit_list : list of FitEllipse
        The ellipse fits whose contours are to be overlaid.
    output_path : str or None
        Directory in which to save the figure.  ``None`` → ``plt.show()``.
    output_filename : str
        Stem of the output file name.
    output_format : str
        File format, e.g. ``"png"``.
    colormap : str
        Matplotlib colormap name, or ``"default"``.
    use_log10 : bool
        Apply a log₁₀ stretch to the image values.
    suffix : str
        Optional suffix appended to *output_filename* before the extension.
    ax : matplotlib.axes.Axes or None
        Existing ``Axes`` to draw into; the caller is responsible for saving
        when this is provided.
    """
    ellipse_list = []
    for fit in fit_list:
        points = fit.points_from_major_axis_from()
        x = points[:, 1]
        y = points[:, 0] * -1.0
        ellipse_list.append(aa.Grid2DIrregular.from_yx_1d(y=y, x=x))

    lines = [np.array(e.array) for e in ellipse_list if e is not None]
    positions = lines

    _title = f"{title_prefix.rstrip()} Ellipse Fit" if title_prefix else "Ellipse Fit"
    plot_array(
        array=fit_list[0].data,
        title=_title,
        output_path=output_path,
        output_filename=f"{output_filename}{suffix}",
        output_format=output_format,
        colormap=colormap,
        use_log10=use_log10,
        lines=lines or None,
        positions=positions or None,
        ax=ax,
    )


def _plot_ellipse_residuals(
    fit_list: List[FitEllipse],
    output_path=None,
    output_format=None,
    for_subplot: bool = False,
    suffix: str = "",
    ax=None,
):
    """Plot the 1-D ellipse residuals via the low-level utility function.

    Constructs the ``autoarray`` ``Output`` object required by
    :func:`~autogalaxy.ellipse.plot.fit_ellipse_plot_util.plot_ellipse_residuals`
    and then delegates to it.

    Parameters
    ----------
    fit_list : list of FitEllipse
        The ellipse fits to summarise.
    output_path : str or None
        Directory in which to save the figure.  ``None`` → an ``Output``
        with no path is created, falling back to ``plt.show()``.
    output_format : str
        File format, e.g. ``"png"``.
    for_subplot : bool
        If ``True``, draw into subplot position ``(1, 2, 2)`` of the current
        figure.
    suffix : str
        Reserved for future filename-suffix support (currently unused).
    ax : matplotlib.axes.Axes or None
        Reserved for future direct-axes support (currently unused).
    """
    from autoarray.plot.output import Output

    output = Output(path=output_path, format=output_format) if output_path else Output()

    fit_ellipse_plot_util.plot_ellipse_residuals(
        array=fit_list[0].dataset.data.native,
        fit_list=fit_list,
        colors="k",
        output=output,
        for_subplot=for_subplot,
    )


def subplot_fit_ellipse(
    fit_list: List[FitEllipse],
    output_path=None,
    output_format=None,
    colormap="default",
    use_log10=False,
    title_prefix: str = None,
):
    """Create a two-panel subplot summarising a list of ellipse fits.

    The left panel shows the 2-D image with fitted ellipse contours overlaid
    (via :func:`_plot_data`); the right panel shows the 1-D residuals as a
    function of position angle (via :func:`_plot_ellipse_residuals`).

    Parameters
    ----------
    fit_list : list of FitEllipse
        The ellipse fits to visualise.
    output_path : str or None
        Directory in which to save the figure.  ``None`` → ``plt.show()``.
    output_format : str
        File format, e.g. ``"png"``.
    colormap : str
        Matplotlib colormap name, or ``"default"``.
    use_log10 : bool
        Apply a log₁₀ stretch to the image values in the left panel.
    """
    fig, axes = subplots(1, 2, figsize=conf_subplot_figsize(1, 2))

    _plot_data(
        fit_list=fit_list,
        colormap=colormap,
        use_log10=use_log10,
        ax=axes[0],
        title_prefix=title_prefix,
    )
    _plot_ellipse_residuals(fit_list=fit_list, for_subplot=True, ax=axes[1])

    tight_layout()
    _save_subplot(fig, output_path, "fit_ellipse", output_format)


def subplot_ellipse_errors(
    fit_pdf_list: List[List[FitEllipse]],
    output_path=None,
    output_format=None,
    colormap="default",
    use_log10=False,
    sigma: Optional[float] = 3.0,
):
    """Create a subplot showing the median ellipse and its uncertainty region from a PDF sample.

    *fit_pdf_list* is a list of fit-lists — each inner list represents one
    posterior sample and contains one :class:`~autogalaxy.ellipse.fit_ellipse.FitEllipse`
    per ellipse.  For each ellipse position the median contour and the
    ``sigma``-level confidence interval are computed in polar coordinates
    (via :func:`~autogalaxy.util.error_util.ellipse_median_and_error_region_in_polar`)
    and overlaid on the 2-D image.

    One panel is produced per ellipse.

    Parameters
    ----------
    fit_pdf_list : list of list of FitEllipse
        Outer list: posterior samples.  Inner list: per-ellipse fits for that
        sample.
    output_path : str or None
        Directory in which to save the figure.  ``None`` → ``plt.show()``.
    output_format : str
        File format, e.g. ``"png"``.
    colormap : str
        Matplotlib colormap name, or ``"default"``.
    use_log10 : bool
        Apply a log₁₀ stretch to the image values.
    sigma : float or None
        Number of standard deviations defining the confidence interval
        (default ``3.0``).
    """
    low_limit = (1 - math.erf(sigma / math.sqrt(2))) / 2

    ellipse_centre_list = []
    fit_ellipse_list = [[] for _ in range(len(fit_pdf_list[0]))]

    for fit_list in fit_pdf_list:
        ellipse_centre_list.append(fit_list[0].ellipse.centre)
        for i, fit in enumerate(fit_list):
            points = fit.points_from_major_axis_from()
            x = points[:, 1]
            y = points[:, 0] * -1.0
            fit_ellipse_list[i].append(aa.Grid2DIrregular.from_yx_1d(y=y, x=x))

    n = len(fit_ellipse_list)
    fig, axes = subplots(1, n, figsize=conf_subplot_figsize(1, n))
    axes_flat = [axes] if n == 1 else list(axes.flatten())

    for i in range(n):
        median_ellipse, [lower_ellipse, upper_ellipse] = (
            error_util.ellipse_median_and_error_region_in_polar(
                fit_ellipse_list[i],
                low_limit=low_limit,
                center=ellipse_centre_list[i],
            )
        )

        try:
            median_arr = np.array(
                median_ellipse.array
                if hasattr(median_ellipse, "array")
                else median_ellipse
            )
            lines = [median_arr] if median_arr.ndim == 2 else None
        except Exception:
            lines = None

        plot_array(
            array=fit_pdf_list[0][0].data,
            title="Ellipse Fit",
            colormap=colormap,
            use_log10=use_log10,
            lines=lines,
            ax=axes_flat[i],
        )

    tight_layout()
    _save_subplot(fig, output_path, "ellipse_errors", output_format)
