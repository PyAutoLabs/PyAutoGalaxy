import numpy as np

import autoarray as aa

from autogalaxy.galaxy.galaxies import Galaxies
from autogalaxy.util.plot_utils import _to_lines, _to_positions, plot_array, plot_grid, _save_subplot, _critical_curves_from
from autoarray.plot.utils import subplots, hide_unused_axes, conf_subplot_figsize, tight_layout
from autogalaxy import exc


def _check_no_linear(galaxies):
    """Raise if any galaxy in *galaxies* contains a linear light profile.

    Plotting functions cannot render
    :class:`~autogalaxy.profiles.light.linear.LightProfileLinear` profiles
    directly because their intensity is not a fixed parameter — it is solved
    via a linear inversion.  This guard ensures a clear error message is
    produced rather than a silent wrong result.

    Parameters
    ----------
    galaxies : sequence of Galaxy
        The galaxies to inspect.

    Raises
    ------
    exc.LinearLightProfileInPlotError
        If at least one linear light profile is found.
    """
    from autogalaxy.profiles.light.linear import LightProfileLinear

    if Galaxies(galaxies=galaxies).has(cls=LightProfileLinear):
        raise exc.raise_linear_light_profile_in_plot(plotter_type="galaxies plot")


def _galaxies_critical_curves(galaxies, grid, tc=None, rc=None):
    """Return the tangential and radial critical curves for a set of galaxies.

    Thin wrapper around :func:`~autogalaxy.util.plot_utils._critical_curves_from`
    that keeps the galaxies-plot API decoupled from the utility layer.

    Parameters
    ----------
    galaxies : Galaxies
        The galaxies acting as a combined lens.
    grid : aa.type.Grid2DLike
        The grid on which to evaluate the critical curves.
    tc : list or None
        Pre-computed tangential critical curves; ``None`` to trigger
        computation.
    rc : list or None
        Pre-computed radial critical curves; ``None`` to trigger computation.

    Returns
    -------
    tuple[list, list or None]
        ``(tangential_critical_curves, radial_critical_curves)``.
    """
    return _critical_curves_from(galaxies, grid, tc=tc, rc=rc)


def subplot_galaxies(
    galaxies,
    grid: aa.type.Grid1D2DLike,
    output_path=None,
    output_format=None,
    colormap="default",
    use_log10=False,
    positions=None,
    light_profile_centres=None,
    mass_profile_centres=None,
    multiple_images=None,
    tangential_critical_curves=None,
    radial_critical_curves=None,
    auto_filename="galaxies",
    title_prefix: str = None,
):
    """Create a standard five-panel summary subplot for a collection of galaxies.

    The subplot shows: image, convergence, potential, deflections-y, and
    deflections-x.  Critical curves and various point overlays can be added
    to all lensing panels automatically.

    Parameters
    ----------
    galaxies : sequence of Galaxy
        The galaxies to plot.  Must not contain any
        :class:`~autogalaxy.profiles.light.linear.LightProfileLinear` profiles.
    grid : aa.type.Grid1D2DLike
        The grid on which all quantities are evaluated.
    output_path : str or None
        Directory in which to save the figure.  ``None`` → ``plt.show()``.
    output_format : str
        File format, e.g. ``"png"``.
    colormap : str
        Matplotlib colormap name, or ``"default"``.
    use_log10 : bool
        Apply a log₁₀ stretch to the plotted values.
    positions : array-like or None
        Arbitrary point positions to overlay on all panels.
    light_profile_centres : array-like or None
        Light-profile centre coordinates to overlay.
    mass_profile_centres : array-like or None
        Mass-profile centre coordinates to overlay.
    multiple_images : array-like or None
        Multiple-image positions to overlay on all panels.
    tangential_critical_curves : list or None
        Pre-computed tangential critical curves.  ``None`` triggers
        automatic computation.
    radial_critical_curves : list or None
        Pre-computed radial critical curves.  ``None`` triggers automatic
        computation.
    auto_filename : str
        Output filename stem (default ``"subplot_galaxies"``).
    """
    _check_no_linear(galaxies)
    gs = Galaxies(galaxies=galaxies)
    tc, rc = _galaxies_critical_curves(
        gs, grid, tc=tangential_critical_curves, rc=radial_critical_curves
    )
    lines = _to_lines(tc, rc)
    pos = _to_positions(positions, light_profile_centres, mass_profile_centres, multiple_images)
    pos_no_cc = _to_positions(positions, light_profile_centres, mass_profile_centres)

    def _defl_y():
        d = gs.deflections_yx_2d_from(grid=grid)
        return aa.Array2D(values=d.slim[:, 0], mask=grid.mask)

    def _defl_x():
        d = gs.deflections_yx_2d_from(grid=grid)
        return aa.Array2D(values=d.slim[:, 1], mask=grid.mask)

    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)
    panels = [
        ("image", gs.image_2d_from(grid=grid), _pf("Image"), pos_no_cc, None),
        ("convergence", gs.convergence_2d_from(grid=grid), _pf("Convergence"), pos, lines),
        ("potential", gs.potential_2d_from(grid=grid), _pf("Potential"), pos, lines),
        ("deflections_y", _defl_y(), _pf("Deflections Y"), pos, lines),
        ("deflections_x", _defl_x(), _pf("Deflections X"), pos, lines),
    ]

    n = len(panels)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = subplots(rows, cols, figsize=conf_subplot_figsize(rows, cols))
    axes_flat = [axes] if n == 1 else list(np.array(axes).flatten())

    for i, (_, array, title, p, l) in enumerate(panels):
        plot_array(
            array=array,
            title=title,
            colormap=colormap,
            use_log10=use_log10,
            positions=p,
            lines=l,
            ax=axes_flat[i],
        )

    hide_unused_axes(axes_flat)
    tight_layout()
    _save_subplot(fig, output_path, auto_filename, output_format)


def subplot_galaxy_images(
    galaxies,
    grid: aa.type.Grid1D2DLike,
    output_path=None,
    output_format=None,
    colormap="default",
    use_log10=False,
    tangential_critical_curves=None,
    radial_critical_curves=None,
    title_prefix: str = None,
):
    """Create a subplot showing the individual image of each galaxy.

    One panel is drawn per galaxy.  This is useful for inspecting which
    galaxy contributes how much light when multiple galaxies are present in
    a scene.

    Parameters
    ----------
    galaxies : sequence of Galaxy
        The galaxies whose images are to be plotted.  Must not contain any
        :class:`~autogalaxy.profiles.light.linear.LightProfileLinear`
        profiles.
    grid : aa.type.Grid1D2DLike
        The grid on which each galaxy image is evaluated.
    output_path : str or None
        Directory in which to save the figure.  ``None`` → ``plt.show()``.
    output_format : str
        File format, e.g. ``"png"``.
    colormap : str
        Matplotlib colormap name, or ``"default"``.
    use_log10 : bool
        Apply a log₁₀ stretch to the image values.
    tangential_critical_curves : list or None
        Reserved for future overlay support (currently unused).
    radial_critical_curves : list or None
        Reserved for future overlay support (currently unused).
    """
    _check_no_linear(galaxies)
    gs = Galaxies(galaxies=galaxies)
    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)

    n = len(gs)
    fig, axes = subplots(1, n, figsize=conf_subplot_figsize(1, n))
    axes_flat = [axes] if n == 1 else list(axes.flatten())

    for i in range(n):
        plot_array(
            array=gs[i].image_2d_from(grid=grid),
            title=_pf(f"Image Of Galaxies {i}"),
            colormap=colormap,
            use_log10=use_log10,
            ax=axes_flat[i],
        )

    tight_layout()
    _save_subplot(fig, output_path, "galaxy_images", output_format)


def fits_galaxy_images(
    galaxies,
    grid: aa.type.Grid1D2DLike,
    output_path,
) -> None:
    """Write a FITS file containing the 2D image of every galaxy.

    Produces ``galaxy_images.fits`` in *output_path*.  The file contains one
    HDU per galaxy, named ``galaxy_0``, ``galaxy_1``, …, plus a ``mask``
    extension as the first extension.

    Parameters
    ----------
    galaxies : Galaxies
        The galaxies whose images are evaluated.  Must not contain
        :class:`~autogalaxy.profiles.light.linear.LightProfileLinear` profiles.
    grid : aa.type.Grid1D2DLike
        The grid on which each galaxy image is evaluated.
    output_path : str or Path
        Directory in which to write ``galaxy_images.fits``.
    """
    from pathlib import Path
    from autonerves.fitsable import hdu_list_for_output_from

    image_list = [galaxy.image_2d_from(grid=grid).native_for_fits for galaxy in galaxies]
    hdu_list = hdu_list_for_output_from(
        values_list=[image_list[0].mask.astype("float")] + image_list,
        ext_name_list=["mask"] + [f"galaxy_{i}" for i in range(len(galaxies))],
        header_dict=grid.mask.header_dict,
    )
    hdu_list.writeto(Path(output_path) / "galaxy_images.fits", overwrite=True)


def plot_magnification_2d(
    mass_obj,
    grid: aa.type.Grid2DLike,
    contour_levels=(2.0, 5.0, 10.0, 20.0, 50.0, 100.0),
    include_critical_curves: bool = True,
    tangential_critical_curves=None,
    radial_critical_curves=None,
    vmin=None,
    vmax=None,
    title: str = r"Magnification $\log_{10}|\mu|$",
    colormap="default",
    output_path=None,
    output_filename: str = "magnification_2d",
    output_format=None,
    ax=None,
):
    """Plot the image-plane magnification map of a mass object, with constant-magnification contours.

    Every pixel shows log10 |mu| at its centre, where mu = 1 / det(A) is the signed magnification computed from the
    Hessian of the lensing potential. The magnification diverges on the critical curves, so a linear colour scale is
    dominated by the few pixels next to them; the log scale and a colour range set by the 1st and 99th percentiles
    of log10 |mu| keep the rest of the map readable.

    Contour lines are drawn at constant |mu| (by default 2, 5, 10, 20, 50 and 100), labelled with their |mu| value,
    as in the magnification maps of cluster lens models (e.g. LEGGOS II, Abedi et al. 2026, figure 4). Levels outside
    the range of the map are skipped. The critical curves can be overlaid, computed with the method configured in
    ``visualize/general.yaml``.

    The mass object can be anything ``LensCalc.from_mass_obj`` accepts (a ``Galaxy``, ``Galaxies`` or a ``Tracer``),
    so the same map is made for a single galaxy, a group or a cluster.

    Parameters
    ----------
    mass_obj
        The mass object whose magnification is plotted.
    grid
        The image-plane grid on which the magnification is evaluated, for example ``dataset.grid`` or a
        ``Grid2D.uniform``.
    contour_levels
        The |mu| values at which contour lines are drawn. ``None`` or an empty sequence draws no contours.
    include_critical_curves
        If ``True``, overlay the tangential and radial critical curves.
    tangential_critical_curves, radial_critical_curves
        Pre-computed critical curves; ``None`` computes them when ``include_critical_curves`` is ``True``.
    vmin, vmax
        Explicit colour-bar limits in log10 |mu|. ``None`` uses the 1st and 99th percentiles of the map.
    title
        Title displayed above the panel.
    colormap
        Matplotlib colormap name, or ``"default"``.
    output_path
        Directory in which to save the figure. ``None`` shows it instead.
    output_filename
        Stem of the output file name.
    output_format
        File format, e.g. ``"png"``.
    ax
        An existing ``Axes`` to draw into. When given the figure is not saved; the caller saves it.
    """
    from autogalaxy.operate.lens_calc import LensCalc

    magnification = LensCalc.from_mass_obj(mass_obj).magnification_2d_from(grid=grid)

    with np.errstate(divide="ignore"):
        log_abs_mu = np.log10(np.abs(np.asarray(magnification.slim)))

    finite = np.isfinite(log_abs_mu)
    log_abs_mu[~finite] = np.nanmax(log_abs_mu[finite]) if np.any(finite) else 0.0

    if vmin is None:
        vmin = float(np.percentile(log_abs_mu, 1.0))
    if vmax is None:
        vmax = float(np.percentile(log_abs_mu, 99.0))

    log_abs_mu_array = aa.Array2D(values=log_abs_mu, mask=grid.mask)

    lines = None
    if include_critical_curves:
        tc, rc = _critical_curves_from(
            mass_obj, grid, tc=tangential_critical_curves, rc=radial_critical_curves
        )
        lines = _to_lines(tc, rc)

    owns_figure = ax is None
    if owns_figure:
        fig, ax = subplots(1, 1, figsize=conf_subplot_figsize(1, 1))

    plot_array(
        array=log_abs_mu_array,
        title=title,
        colormap=colormap,
        vmin=vmin,
        vmax=vmax,
        lines=lines,
        ax=ax,
    )

    _magnification_contours(
        ax=ax, log_abs_mu_array=log_abs_mu_array, grid=grid, contour_levels=contour_levels
    )

    if owns_figure:
        tight_layout()
        _save_subplot(fig, output_path, output_filename, output_format)


def _magnification_contours(ax, log_abs_mu_array, grid, contour_levels):
    """Draw labelled contour lines of constant |mu| over a log10 |mu| map on ``ax``.

    The contours are drawn in arcsecond coordinates from the grid's native (y, x) values, so they line up with the
    image however ``plot_array`` zoomed it. The axes limits are restored afterwards so the contours never widen the
    panel.

    Parameters
    ----------
    ax
        The ``Axes`` the map was drawn on.
    log_abs_mu_array
        The ``Array2D`` of log10 |mu| that was plotted.
    grid
        The grid the magnification was evaluated on.
    contour_levels
        The |mu| values at which to draw contours; values outside the range of the map are skipped.
    """
    if not contour_levels:
        return

    values = np.asarray(log_abs_mu_array.native.array, dtype=float)
    values[np.asarray(grid.mask)] = np.nan

    levels = np.log10(np.asarray(sorted(contour_levels), dtype=float))
    levels = levels[(levels > np.nanmin(values)) & (levels < np.nanmax(values))]

    if len(levels) == 0:
        return

    native = np.asarray(grid.native.array)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    contours = ax.contour(
        native[:, :, 1], native[:, :, 0], values, levels=levels, colors="k", linewidths=0.8
    )
    ax.clabel(contours, fmt=lambda level: f"{10.0 ** level:g}", fontsize=8)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
