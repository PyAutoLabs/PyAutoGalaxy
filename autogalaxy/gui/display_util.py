"""
Display-side helpers for the mask-drawing GUIs.

Everything here is a finding aid for the person drawing a mask, never a data product: the
`Scribbler` reads a mask back from brush positions, so none of these transforms can change
what a given stroke masks.

`mask_regridded_from` / `mask_2d_regridded_from` transfer a mask between two uniform grids
of different pixel scale or size (e.g. two wavebands of the same object), so a mask drawn
on one can seed the next. Its long-term home is `autoarray.mask`; it lives here for now
because the GUI is its only caller.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

import autoarray as aa


def radial_median_subtract(
    array_native: np.ndarray, exclude: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Subtract the azimuthally-averaged (median) radial profile about the array centre.

    A smooth, near-circular galaxy is almost entirely a function of radius, while lensed
    arcs, companions and other structure are not; subtracting the per-radius median
    removes the galaxy and leaves that structure standing out, including what is buried
    inside the galaxy's envelope. It is a finding aid, not a fit: real ellipticity leaves a
    quadrupole residual and bright structure biases the median at its own radius.

    Parameters
    ----------
    array_native
        The 2D image.
    exclude
        Boolean array of the same shape; `True` pixels are dropped from the per-radius
        median (but still have the profile subtracted). Use it to keep a known
        contaminant from dragging its radius' median up and stamping a dark annulus
        across the image at that radius.

    Returns
    -------
    The image minus its radial median profile, as a float array of the same shape.
    """
    a = np.asarray(array_native, dtype=float)
    n_y, n_x = a.shape
    yy, xx = np.mgrid[0:n_y, 0:n_x]
    r_bin = np.hypot(yy - (n_y - 1) / 2.0, xx - (n_x - 1) / 2.0).astype(int)
    usable = np.isfinite(a)
    if exclude is not None:
        usable &= ~np.asarray(exclude, dtype=bool)
    profile = np.zeros(r_bin.max() + 1)
    for i in range(profile.size):
        ring = a[(r_bin == i) & usable]
        if ring.size:
            profile[i] = np.median(ring)
    return a - profile[r_bin]


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    """
    The 1-pixel inner boundary of a boolean mask (`mask & ~erosion(mask)`), for outlining
    it over an image. The outline lies on masked pixels; a mask running off the array edge
    is not outlined along the edge itself.
    """
    from scipy.ndimage import binary_erosion

    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return m
    return m & ~binary_erosion(m, np.ones((3, 3), dtype=bool), border_value=1)


def composite_panels(
    panels: Sequence[np.ndarray], gap: int = 6, fill: float = 0.0
) -> np.ndarray:
    """
    Lay 2D panels of equal height side by side, separated by `gap` blank columns of value
    `fill`. The inverse of `fold_panels`.
    """
    panels = [np.asarray(p) for p in panels]
    if len(panels) == 1:
        return panels[0]
    gutter = np.full((panels[0].shape[0], gap), fill, dtype=float)
    parts: List[np.ndarray] = []
    for i, p in enumerate(panels):
        if i:
            parts.append(gutter)
        parts.append(p)
    return np.hstack(parts)


def fold_panels(
    scribbled: np.ndarray, n_panels: int, panel_n_x: int, gap: int = 6
) -> Tuple[np.ndarray, List[int]]:
    """
    Fold a boolean array drawn on a `composite_panels` display back onto one panel's grid.

    Every panel shows the same pixels, so a stroke on any of them means the same thing:
    the per-panel slices are OR-ed together.

    Returns
    -------
    The folded (n_y, panel_n_x) boolean array and the number of True pixels per panel.
    """
    scribbled = np.asarray(scribbled, dtype=bool)
    if n_panels <= 1:
        return scribbled, [int(scribbled.sum())]
    step = panel_n_x + gap
    out = np.zeros((scribbled.shape[0], panel_n_x), dtype=bool)
    per_panel = []
    for i in range(n_panels):
        s = scribbled[:, i * step : i * step + panel_n_x]
        per_panel.append(int(s.sum()))
        out |= s
    return out, per_panel


def mask_regridded_from(
    mask_native: np.ndarray,
    src_pixel_scales,
    dst_shape_native: Tuple[int, int],
    dst_pixel_scales,
    src_origin: Tuple[float, float] = (0.0, 0.0),
    dst_origin: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Transfer a boolean mask from one uniform grid to another by nearest neighbour, so it
    covers the same region in scaled (arc-second) coordinates.

    Each destination pixel centre is converted to scaled coordinates, then to the source
    pixel containing it, whose value is taken. Nearest neighbour is the right resampling
    for a boolean: there is no interpolation across the True / False boundary. Destination
    pixels outside the source footprint are `False` (unmasked).

    Parameters
    ----------
    mask_native
        The source boolean mask, in native 2D form.
    src_pixel_scales
        The source grid's (y,x) pixel scales (a float is used for both).
    dst_shape_native
        The (y,x) shape of the destination grid.
    dst_pixel_scales
        The destination grid's (y,x) pixel scales.
    src_origin, dst_origin
        The (y,x) scaled-coordinate origins of the two grids.
    """
    mask_native = np.asarray(mask_native, dtype=bool)
    src_shape = mask_native.shape
    src_ps = aa.util.geometry.convert_pixel_scales_2d(pixel_scales=src_pixel_scales)
    dst_ps = aa.util.geometry.convert_pixel_scales_2d(pixel_scales=dst_pixel_scales)
    n_y, n_x = (int(dst_shape_native[0]), int(dst_shape_native[1]))

    # Destination pixel centres in scaled coordinates. Row 0 is the most positive y
    # (autoarray's native convention), column 0 the most negative x.
    rows_d, cols_d = np.mgrid[0:n_y, 0:n_x]
    y = dst_origin[0] + (n_y / 2.0 - rows_d - 0.5) * dst_ps[0]
    x = dst_origin[1] + (cols_d + 0.5 - n_x / 2.0) * dst_ps[1]

    # The source pixel each centre falls in: measured from the source grid's top / left
    # edges and floored, so a centre exactly on a pixel edge is assigned consistently.
    y_top = src_origin[0] + (src_shape[0] / 2.0) * src_ps[0]
    x_left = src_origin[1] - (src_shape[1] / 2.0) * src_ps[1]
    rows = np.floor((y_top - y) / src_ps[0]).astype(int)
    cols = np.floor((x - x_left) / src_ps[1]).astype(int)

    inside = (rows >= 0) & (rows < src_shape[0]) & (cols >= 0) & (cols < src_shape[1])
    out = np.zeros((n_y, n_x), dtype=bool)
    out[inside] = mask_native[rows[inside], cols[inside]]
    return out


def mask_2d_regridded_from(
    mask: aa.Mask2D,
    shape_native: Tuple[int, int],
    pixel_scales,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> aa.Mask2D:
    """
    `mask_regridded_from` for an `aa.Mask2D`: returns a `Mask2D` on the new grid covering
    the same scaled-coordinate region.
    """
    regridded = mask_regridded_from(
        mask_native=np.asarray(mask),
        src_pixel_scales=mask.pixel_scales,
        dst_shape_native=shape_native,
        dst_pixel_scales=pixel_scales,
        src_origin=mask.origin,
        dst_origin=origin,
    )
    return aa.Mask2D(mask=regridded, pixel_scales=pixel_scales, origin=origin)
