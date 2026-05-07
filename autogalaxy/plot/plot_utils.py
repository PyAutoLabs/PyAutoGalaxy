import logging
import os
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


def _to_lines(*items):
    """Convert multiple line sources into a flat list of (N,2) numpy arrays.

    Each item may be ``None`` (skipped), a list of line-like objects, or a
    single line-like object.  A line-like object is anything that either has
    an ``.array`` attribute or can be coerced to a 2-D numpy array with shape
    ``(N, 2)``.  Items that cannot be converted, or that are empty, are
    silently dropped.

    Parameters
    ----------
    *items
        Any number of line sources to merge.

    Returns
    -------
    list of np.ndarray or None
        A flat list of ``(N, 2)`` arrays, or ``None`` if nothing valid was
        found.
    """
    result = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, list):
            for sub in item:
                try:
                    arr = np.array(sub.array if hasattr(sub, "array") else sub)
                    if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
                        result.append(arr)
                except Exception:
                    pass
        else:
            try:
                arr = np.array(item.array if hasattr(item, "array") else item)
                if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
                    result.append(arr)
            except Exception:
                pass
    return result or None


def _to_positions(*items):
    """Convert multiple position sources into a flat list of (N,2) numpy arrays.

    Thin wrapper around :func:`_to_lines` — positions and lines share the same
    underlying representation (lists of ``(N, 2)`` coordinate arrays).

    Parameters
    ----------
    *items
        Any number of position sources to merge.

    Returns
    -------
    list of np.ndarray or None
        A flat list of ``(N, 2)`` arrays, or ``None`` if nothing valid was
        found.
    """
    return _to_lines(*items)


def _save_subplot(fig, output_path, output_filename, output_format=None,
                  dpi=300):
    """Save a subplot figure to disk (or show it if output_format/output_path say so).

    For FITS output use the dedicated ``fits_*`` functions instead.
    """
    import matplotlib.pyplot as plt
    from autoarray.plot.utils import _output_mode_save, _conf_output_format, _FAST_PLOTS

    if _output_mode_save(fig, output_filename):
        return

    if _FAST_PLOTS:
        plt.close(fig)
        return

    fmt = output_format[0] if isinstance(output_format, (list, tuple)) else (output_format or _conf_output_format())
    if fmt == "show" or not output_path:
        plt.show()
    else:
        os.makedirs(str(output_path), exist_ok=True)
        fpath = Path(output_path) / f"{output_filename}.{fmt}"
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def _resolve_colormap(colormap):
    """Resolve 'default' or None to the autoarray default colormap."""
    if colormap in ("default", None):
        from autoarray.plot.utils import _default_colormap
        return _default_colormap()
    return colormap


def _resolve_format(output_format):
    """Normalise output_format: accept a list/tuple or a plain string."""
    from autoarray.plot.utils import _conf_output_format

    if isinstance(output_format, (list, tuple)):
        return output_format[0]
    return output_format or _conf_output_format()


def _numpy_grid(grid):
    """Convert a grid-like object to a numpy array, or return None."""
    if grid is None:
        return None
    try:
        return np.array(grid.array if hasattr(grid, "array") else grid)
    except Exception:
        return None


def plot_array(
    array,
    title="",
    output_path=None,
    output_filename="array",
    output_format=None,
    colormap="default",
    use_log10=False,
    vmin=None,
    vmax=None,
    symmetric=False,
    positions=None,
    lines=None,
    line_colors=None,
    grid=None,
    cb_unit=None,
    ax=None,
):
    """Plot an autoarray ``Array2D`` to file or onto an existing ``Axes``.

    All array preprocessing (zoom, mask-edge extraction, native/extent
    unpacking) is handled internally so callers never need to duplicate it.
    The actual rendering is delegated to ``autoarray.plot.plot_array``.

    Parameters
    ----------
    array
        The ``Array2D`` (or array-like) to plot.
    title : str
        Title displayed above the panel.
    output_path : str or None
        Directory in which to save the figure.  ``None`` → call
        ``plt.show()`` instead.
    output_filename : str
        Stem of the output file name (extension is added from
        *output_format*).
    output_format : str
        File format, e.g. ``"png"`` or ``"pdf"``.
    colormap : str
        Matplotlib colormap name, or ``"default"`` to use the autoarray
        default (``"jet"``).
    use_log10 : bool
        If ``True`` apply a log₁₀ stretch to the array values.
    vmin, vmax : float or None
        Explicit colour-bar limits.  Ignored when *symmetric* is ``True``.
    symmetric : bool
        If ``True`` set ``vmin = -vmax`` so that zero maps to the middle of
        the colormap.
    positions : list or array-like or None
        Point positions to scatter-plot over the image.
    lines : list or array-like or None
        Line coordinates to overlay on the image.
    line_colors : list or None
        Colours for each entry in *lines*.
    grid : array-like or None
        An additional grid of points to overlay.
    ax : matplotlib.axes.Axes or None
        Existing ``Axes`` to draw into.  When provided the figure is *not*
        saved — the caller is responsible for saving.
    """
    from autoarray.plot import plot_array as _aa_plot_array

    colormap = _resolve_colormap(colormap)
    output_format = _resolve_format(output_format)

    if symmetric:
        try:
            arr = array.native.array
        except AttributeError:
            arr = np.asarray(array)
        finite = arr[np.isfinite(arr)]
        abs_max = float(np.max(np.abs(finite))) if len(finite) > 0 else 1.0
        vmin, vmax = -abs_max, abs_max

    _positions_list = positions if isinstance(positions, list) else _to_positions(positions)
    _lines_list = lines if isinstance(lines, list) else _to_lines(lines)

    if ax is not None:
        _output_path = None
    else:
        _output_path = output_path if output_path is not None else "."

    _aa_plot_array(
        array=array,
        ax=ax,
        grid=_numpy_grid(grid),
        positions=_positions_list,
        lines=_lines_list,
        line_colors=line_colors,
        title=title or "",
        colormap=colormap,
        use_log10=use_log10,
        vmin=vmin,
        vmax=vmax,
        cb_unit=cb_unit,
        output_path=_output_path,
        output_filename=output_filename,
        output_format=output_format,
    )


def _fits_values_and_header(array):
    """Extract raw numpy values and header dict from an autoarray object.

    Returns ``(values, header_dict, ext_name)`` where *header_dict* and
    *ext_name* may be ``None`` for plain arrays.
    """
    from autoarray.structures.visibilities import AbstractVisibilities
    from autoarray.mask.abstract_mask import Mask

    if isinstance(array, AbstractVisibilities):
        return np.asarray(array.in_array), None, None
    if isinstance(array, Mask):
        header = array.header_dict if hasattr(array, "header_dict") else None
        return np.asarray(array.astype("float")), header, "mask"
    if hasattr(array, "native"):
        try:
            header = array.mask.header_dict
        except (AttributeError, TypeError):
            header = None
        return np.asarray(array.native.array).astype("float"), header, None

    return np.asarray(array), None, None


def fits_array(array, file_path, overwrite=False, ext_name=None):
    """Write an autoarray ``Array2D``, ``Mask2D``, or array-like to a ``.fits`` file.

    Handles header metadata (pixel scales, origin) automatically for
    autoarray objects.

    Parameters
    ----------
    array
        The data to write.
    file_path : str or Path
        Full path including filename and ``.fits`` extension.
    overwrite : bool
        If ``True`` an existing file at *file_path* is replaced.
    ext_name : str or None
        FITS extension name.  Auto-detected for masks (``"mask"``).
    """
    from autoconf.fitsable import output_to_fits

    values, header_dict, auto_ext_name = _fits_values_and_header(array)
    if ext_name is None:
        ext_name = auto_ext_name

    output_to_fits(
        values=values,
        file_path=file_path,
        overwrite=overwrite,
        header_dict=header_dict,
        ext_name=ext_name,
    )


def plot_grid(
    grid,
    title="",
    output_path=None,
    output_filename="grid",
    output_format=None,
    lines=None,
    ax=None,
):
    """Plot an autoarray ``Grid2D`` as a scatter plot.

    Delegates to ``autoarray.plot.plot_grid`` after converting the grid to a
    plain numpy array.

    Parameters
    ----------
    grid
        The ``Grid2D`` (or grid-like) to plot.
    title : str
        Title displayed above the panel.
    output_path : str or None
        Directory in which to save the figure.  ``None`` → call
        ``plt.show()`` instead.
    output_filename : str
        Stem of the output file name.
    output_format : str
        File format, e.g. ``"png"``.
    lines : list or None
        Line coordinates to overlay on the grid plot.
    ax : matplotlib.axes.Axes or None
        Existing ``Axes`` to draw into.
    """
    from autoarray.plot import plot_grid as _aa_plot_grid

    output_format = _resolve_format(output_format)

    if ax is not None:
        _output_path = None
    else:
        _output_path = output_path if output_path is not None else "."

    _aa_plot_grid(
        grid=np.array(grid.array if hasattr(grid, "array") else grid),
        ax=ax,
        title=title or "",
        output_path=_output_path,
        output_filename=output_filename,
        output_format=output_format,
    )


def _critical_curves_method():
    """Read ``general.critical_curves_method`` from the visualize config.

    Returns ``"marching_squares"`` (the default) or ``"zero_contour"``.
    Any unrecognised value falls back to ``"marching_squares"`` with a warning.
    If ``"zero_contour"`` is requested but ``jax_zero_contour`` is not installed
    (e.g. Python <3.11), falls back to ``"marching_squares"`` with a warning.
    """
    from autoconf import conf

    try:
        method = conf.instance["visualize"]["general"]["general"]["critical_curves_method"]
    except (KeyError, TypeError):
        method = "marching_squares"

    if method not in ("zero_contour", "marching_squares"):
        logger.warning(
            f"visualize/general.yaml: unrecognised critical_curves_method "
            f"'{method}'. Falling back to 'marching_squares'."
        )
        return "marching_squares"

    if method == "zero_contour":
        try:
            import jax_zero_contour  # noqa: F401
        except ImportError:
            logger.warning(
                "critical_curves_method='zero_contour' requested, but "
                "jax_zero_contour is not installed (Python <3.11 ships without "
                "the [jax] extra). Falling back to 'marching_squares'."
            )
            return "marching_squares"

    return method


def _caustics_from(mass_obj, grid):
    """Compute tangential and radial caustics for a mass object via LensCalc.

    The algorithm used is controlled by ``general.critical_curves_method`` in
    ``visualize/general.yaml``:

    - ``"zero_contour"`` *(default)* — uses ``jax_zero_contour`` to trace the
      zero contour of each eigen value directly.  No dense evaluation grid is
      needed; a coarse 25 × 25 scan finds the seed points automatically.
    - ``"marching_squares"`` — evaluates eigen values on the full *grid* and
      uses marching squares to find the contours.

    Parameters
    ----------
    mass_obj
        Any object understood by ``LensCalc.from_mass_obj`` (e.g. a
        :class:`~autogalaxy.galaxy.galaxies.Galaxies` or autolens ``Tracer``).
    grid : aa.type.Grid2DLike
        The grid on which to evaluate the caustics (used only for the
        ``"marching_squares"`` path; ignored by ``"zero_contour"``).

    Returns
    -------
    tuple[list, list]
        ``(tangential_caustics, radial_caustics)``.
    """
    if os.environ.get("PYAUTO_FAST_PLOTS") == "1":
        return [], []

    from autogalaxy.operate.lens_calc import LensCalc

    od = LensCalc.from_mass_obj(mass_obj)
    method = _critical_curves_method()

    if method == "zero_contour":
        tan_ca = od.tangential_caustic_list_via_zero_contour_from()
        rad_ca = od.radial_caustic_list_via_zero_contour_from()
    else:
        tan_ca = od.tangential_caustic_list_from(grid=grid)
        rad_ca = od.radial_caustic_list_from(grid=grid)

    return tan_ca, rad_ca


def _critical_curves_from(mass_obj, grid, tc=None, rc=None):
    """Compute tangential and radial critical curves for a mass object.

    If *tc* is already provided it is returned unchanged (along with *rc*),
    allowing callers to cache the curves across multiple plot calls.

    The algorithm used when *tc* is ``None`` is controlled by
    ``general.critical_curves_method`` in ``visualize/general.yaml``:
/btw ok
    - ``"zero_contour"`` *(default)* — uses ``jax_zero_contour``; no dense
      grid needed, seed points found automatically via a coarse grid scan.
    - ``"marching_squares"`` — evaluates eigen values on the full *grid* and
      uses marching squares.  Radial critical curves are only computed when at
      least one radial critical-curve area exceeds the grid pixel scale.

    Parameters
    ----------
    mass_obj
        Any object understood by ``LensCalc.from_mass_obj``.
    grid : aa.type.Grid2DLike
        Evaluation grid (used only for the ``"marching_squares"`` path).
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
    from autogalaxy.operate.lens_calc import LensCalc

    if os.environ.get("PYAUTO_FAST_PLOTS") == "1":
        return [], []

    if tc is None:
        od = LensCalc.from_mass_obj(mass_obj)
        method = _critical_curves_method()

        if method == "zero_contour":
            tc = od.tangential_critical_curve_list_via_zero_contour_from()
            rc = od.radial_critical_curve_list_via_zero_contour_from()
        else:
            tc = od.tangential_critical_curve_list_from(grid=grid)
            rc_area = od.radial_critical_curve_area_list_from(grid=grid)
            if any(area > grid.pixel_scale for area in rc_area):
                rc = od.radial_critical_curve_list_from(grid=grid)

    return tc, rc
