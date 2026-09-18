from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np


class Scribbler:
    #: Fixed roles of the two default scribble segments. Press the key to switch brush.
    ADD_SEGMENT = "1"  # green: pixels to add to the mask
    ERASE_SEGMENT = "2"  # red: pixels to remove from the mask (or from a proposal)

    #: One-line key legend, printed when the GUI starts and shown as the axes title.
    KEY_LEGEND = (
        "'1' add (green) | '2' erase (red) | '=' bigger brush | '-' smaller brush | "
        "'z' undo | Esc done"
    )

    def __init__(
        self,
        image,
        segment_names=None,
        cmap=None,
        norm=None,
        vmin=None,
        vmax=None,
        brush_width=0.05,
        backend="TkAgg",
        mask_overlay=None,
        figsize=(15, 15),
        rgb_image=None,
        extent: Tuple[float, float, float, float] = None,
        proposal: Optional[np.ndarray] = None,
        brush_resize_factor: float = 1.4,
        min_radius: int = 1,
        block: bool = True,
    ):
        """
        A GUI for spray-painting a mask over an image with the mouse, used by the
        `data_preparation/gui` scripts of the workspaces to draw custom masks and
        noise-scaling regions by hand.

        Adapted from https://gist.github.com/brikeats/4f63f867fd8ea0f196c78e9b835150ab

        **Brushes.** Two scribble segments are available: `'1'` (green) ADDS pixels to
        the mask and `'2'` (red) ERASES them. Press the key to switch brush; the current
        brush is drawn as a blue circle following the cursor. `'='` and `'-'` grow and
        shrink the brush multiplicatively (by `brush_resize_factor` per press, never by less
        than one pixel), so a few presses take you from a 1 px detail brush to a thick
        outskirts brush. `'z'` undoes the last circle of the active brush and `Esc` / `q`
        closes the window.

        **Starting from a proposal.** Pass `proposal` (a boolean array of the image's native
        shape, True = masked) to REFINE an existing mask instead of drawing from scratch:
        its boundary is outlined in white over the image and, after the GUI closes,
        `mask_from()` returns `(proposal | added) & ~erased`. A proposal can be a mask
        drawn earlier for this image, or one drawn for another waveband of the same object.

        **Reading the result.** `mask_from()` returns the combined mask described above
        (with no proposal, simply `added & ~erased`); `show_mask()` returns the ADD segment
        alone, for backwards compatibility; `get_scribble_masks()` returns every segment as
        its own boolean array.

        Parameters
        ----------
        image
            The `Array2D` to draw over, in its native 2D form.
        segment_names
            Names of the scribble segments (default `["1", "2"]`, add and erase).
        cmap
            The colormap name, e.g. ``"jet"``. ``None`` or ``"default"`` uses
            the configured default. A legacy ``Cmap``-style object exposing
            ``norm_from`` is still accepted and takes precedence over the
            *norm* / *vmin* / *vmax* arguments below.
        norm
            ``"log"`` for a logarithmic colour scale, ``"linear"`` (or ``None``)
            otherwise.
        vmin, vmax
            Explicit colour-scale limits.
        brush_width
            The starting brush radius as a fraction of the image height.
        backend
            The matplotlib backend the window is opened with. `"Agg"` (with
            `block=False`) builds the figure without a window, for tests.
        mask_overlay
            A `Mask2D` whose edge is scattered over the image as a guide (e.g. the circular
            mask a fit will use).
        figsize
            The matplotlib figure size in inches.
        rgb_image
            An optional RGB image shown beside the data.
        extent
            The (x0, x1, y0, y1) scaled-coordinate extent to zoom the display to.
        proposal
            An existing boolean mask (True = masked) to outline and refine, see above.
        brush_resize_factor
            The factor the brush radius is multiplied / divided by per `'='` / `'-'` press.
        min_radius
            The smallest brush radius in pixels.
        block
            If `True` (the default) the constructor opens the window and blocks until it is
            closed. If `False` the figure is built but the event loop is not started; call
            `start()` to open it, or drive the callbacks directly (tests).
        """

        if extent is not None:
            central_pixel_coordinates = image.geometry.central_pixel_coordinates
            origin = image.geometry.origin
            pixel_scales = image.geometry.pixel_scales

            x0_pix = int(
                (extent[0] - origin[1]) / pixel_scales[1]
                + central_pixel_coordinates[1]
                + 0.5
            )

            x1_pix = int(
                (extent[1] - origin[1]) / pixel_scales[1]
                + central_pixel_coordinates[1]
                + 0.5
            )

            y0_pix = int(
                (extent[2] - origin[0]) / pixel_scales[0]
                + central_pixel_coordinates[0]
                + 0.5
            )

            y1_pix = int(
                (extent[3] - origin[0]) / pixel_scales[0]
                + central_pixel_coordinates[0]
                + 0.5
            )

            extent = (x0_pix, x1_pix, y0_pix, y1_pix)

        self.im = image
        self.backend = backend
        self.figsize = figsize
        self.extent = extent

        if proposal is not None:
            proposal = np.asarray(proposal, dtype=bool)
            if proposal.shape != tuple(image.shape_native):
                raise ValueError(
                    f"proposal shape {proposal.shape} does not match the image's native "
                    f"shape {tuple(image.shape_native)}"
                )
        self.proposal = proposal

        # brush
        self.brush_radius = max(int(image.shape_native[0] * brush_width), min_radius)
        self.min_radius = min_radius
        self.brush_resize_factor = brush_resize_factor
        self.brush_color = "b"
        self.brush = None

        # scribbles
        if not segment_names:
            segment_names = [self.ADD_SEGMENT, self.ERASE_SEGMENT]
        self.scribble_colors = "gr"
        self.scribbles = OrderedDict()
        for name in segment_names:
            self.scribbles[name] = []
        self.active_scribble = self.scribbles[segment_names[0]]
        self.active_scribble_color = self.scribble_colors[0]
        self.mouse_is_down = False
        self.num_patches = 0

        self._build_figure(
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            mask_overlay=mask_overlay,
            rgb_image=rgb_image,
        )

        if block:
            self.start()

    def _build_figure(self, cmap, norm, vmin, vmax, mask_overlay, rgb_image):
        """
        Build the matplotlib figure, draw the image (and any proposal outline, mask edge or
        RGB companion) and connect the mouse / keyboard callbacks. Does not open a window
        or start the event loop -- see `start()`.
        """
        import matplotlib
        import matplotlib.pyplot as plt
        from autoarray.plot.utils import _conf_imshow_origin

        matplotlib.use(self.backend)
        image = self.im

        # create initial plot
        self.figure = plt.figure(figsize=self.figsize)
        mng = plt.get_current_fig_manager()
        if hasattr(mng, "window") and hasattr(mng.window, "wm_geometry"):
            mng.window.wm_geometry("+50+50")  # For TkAgg backend
        if rgb_image is not None:
            self.ax = self.figure.add_subplot(121)
            plt.axis(self.extent)
            plt.axis("off")
            plt.imshow(rgb_image, origin=_conf_imshow_origin())
        self.ax = self.figure.add_subplot(111)

        if cmap is None and norm is None and vmin is None and vmax is None:
            plt.imshow(image, interpolation="none", origin=_conf_imshow_origin())
        elif hasattr(cmap, "norm_from"):
            # Legacy `Cmap`-style object. The public plot namespaces no longer
            # export one, but a caller holding an instance still works.
            mpl_norm = cmap.norm_from(array=image)
            cmap_name = getattr(cmap, "cmap_name", None) or cmap.config_dict.get(
                "cmap", "viridis"
            )
            plt.imshow(
                image, cmap=cmap_name, norm=mpl_norm, origin=_conf_imshow_origin()
            )
        else:
            from autogalaxy.util.plot_utils import _resolve_colormap, norm_from

            mpl_norm = norm_from(
                array=image, use_log10=norm == "log", vmin=vmin, vmax=vmax
            )
            plt.imshow(
                image,
                cmap=_resolve_colormap(cmap),
                norm=mpl_norm,
                origin=_conf_imshow_origin(),
            )

        if mask_overlay is not None:
            grid = mask_overlay.derive_grid.edge
            grid = mask_overlay.geometry.grid_pixel_centres_2d_from(grid_scaled_2d=grid)
            plt.scatter(y=grid[:, 0], x=grid[:, 1], c="k", marker="x", s=10)

        if self.proposal is not None and self.proposal.any():
            # Outline only: filling the proposal would hide the very pixels its boundary
            # is being judged against.
            self.ax.contour(
                self.proposal.astype(float),
                levels=[0.5],
                colors="w",
                linewidths=1.0,
                origin=_conf_imshow_origin(),
            )

        self.ax.set_title(self.KEY_LEGEND, fontsize=10)
        plt.axis(self.extent)
        plt.axis("off")

        # disable default keybindings
        manager, canvas = self.figure.canvas.manager, self.figure.canvas
        if manager is not None and getattr(manager, "key_press_handler_id", None):
            canvas.mpl_disconnect(manager.key_press_handler_id)

        # callbacks
        self.figure.canvas.mpl_connect("key_press_event", self.on_keypress)
        self.figure.canvas.mpl_connect("motion_notify_event", self.on_mouse_motion)
        self.figure.canvas.mpl_connect("button_press_event", self.on_mouse_down)
        self.figure.canvas.mpl_connect("button_release_event", self.on_mouse_up)

        plt.subplots_adjust(wspace=0, hspace=0)

    def start(self):
        """
        Open the window and block until the user closes it (Esc / q).
        """
        import matplotlib.pyplot as plt

        print(f"Scribbler keys: {self.KEY_LEGEND}")
        print(f"  brush radius = {self.brush_radius} px")

        plt.ion()
        plt.show()
        self.figure.canvas.start_event_loop(timeout=-1)

    def on_mouse_up(self, event):
        self.mouse_is_down = False

    def on_mouse_down(self, event):
        self.mouse_is_down = True
        if event.inaxes != self.ax:
            return

        center = event.xdata, event.ydata
        self.add_circle_to_scribble(center)

    def on_mouse_motion(self, event):
        # Outside the image axes (figure padding, off-window) the data coordinates are
        # None; moving the brush there would raise inside matplotlib.
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        import matplotlib

        center = (event.xdata, event.ydata)

        # draw the brush circle
        if self.brush:
            self.brush.center = center
        else:
            self.brush = matplotlib.patches.Circle(
                center,
                radius=self.brush_radius,
                edgecolor=self.brush_color,
                facecolor="none",
                zorder=1e6,
            )  # always on top
            self.ax.add_patch(self.brush)

        # add to the scribble, if mouse is down
        if self.mouse_is_down:
            self.add_circle_to_scribble(center)

        # Repositioning the patch does not itself trigger a redraw; without this the brush
        # circle stops following the cursor once no other event forces a draw (e.g. on the
        # second Scribbler opened in one process). draw_idle coalesces to one paint per
        # event-loop idle, so it stays cheap.
        self.figure.canvas.draw_idle()

    def on_keypress(self, event):
        if event.key in ["q", "Q", "escape"]:
            self.quit_()
        elif event.key in ["=", "super+="]:
            self.enlarge_brush()
        elif event.key in ["-", "super+-"]:
            self.shrink_brush()
        elif event.key == "z":
            self.remove_circle_from_scribble()
        elif event.key == "v":
            self.show_mask()
        elif event.key in [str(num + 1) for num in range(len(self.scribbles))]:
            self.set_active_segment(int(event.key) - 1)

    def set_active_segment(self, index: int):
        """
        Make the `index`-th scribble segment the active brush (0 = add, 1 = erase).
        """
        name = list(self.scribbles.keys())[index]
        self.active_scribble = self.scribbles[name]
        self.active_scribble_color = self.scribble_colors[index]

    def add_circle_to_scribble(self, center):
        import matplotlib

        circle = matplotlib.patches.Circle(
            center,
            radius=self.brush_radius,
            edgecolor="none",
            facecolor=self.active_scribble_color,
        )
        self.ax.add_patch(circle)
        self.active_scribble.append(circle)
        self.num_patches += 1
        self.figure.canvas.draw()

    def remove_circle_from_scribble(self):
        if self.active_scribble:
            last_circle = self.active_scribble.pop()
            last_circle.remove()
            self.num_patches -= 1
            self.figure.canvas.draw()

    def _set_brush_radius(self, radius: float):
        self.brush_radius = max(self.min_radius, int(round(radius)))
        if self.brush:
            self.brush.radius = self.brush_radius
            self.figure.canvas.draw()
        print(f"  brush radius = {self.brush_radius} px")

    def enlarge_brush(self):
        """
        Grow the brush by `brush_resize_factor`, and by at least one pixel.
        """
        self._set_brush_radius(
            max(self.brush_radius + 1, self.brush_radius * self.brush_resize_factor)
        )

    def shrink_brush(self):
        """
        Shrink the brush by `brush_resize_factor`, and by at least one pixel, never below
        `min_radius`.
        """
        self._set_brush_radius(
            min(self.brush_radius - 1, self.brush_radius / self.brush_resize_factor)
        )

    def quit_(self):
        import matplotlib.pyplot as plt

        plt.close()
        self.figure.canvas.stop_event_loop()

    def show_mask(self):
        """
        Returns the ADD segment's mask alone (the original single-brush behaviour). Prefer
        `mask_from()`, which also applies the erase brush and any proposal.
        """
        import matplotlib.pyplot as plt

        masks = self.get_scribble_masks()
        plt.ioff()
        return masks[list(self.scribbles.keys())[0]]

    def mask_from(self, proposal: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns the mask the user drew: `(proposal | added) & ~erased`, where `added` and
        `erased` are the `'1'` and `'2'` scribble segments.

        Parameters
        ----------
        proposal
            The boolean mask (True = masked) the scribbles refine. Defaults to the
            `proposal` the GUI was opened with, or to nothing.
        """
        masks = self.get_scribble_masks()
        names = list(self.scribbles.keys())
        added = masks[names[0]]
        erased = masks[names[1]] if len(names) > 1 else np.zeros_like(added)
        if proposal is None:
            proposal = self.proposal
        if proposal is None:
            proposal = np.zeros_like(added)
        return (np.asarray(proposal, dtype=bool) | added) & ~erased

    def add_circle_to_mask(self, center, radius, mask):
        if center[0] is None or center[1] is None:
            return
        xx, yy = np.mgrid[: self.im.shape[0], : self.im.shape[1]]
        circle_mask = (xx - center[1]) ** 2 + (yy - center[0]) ** 2 <= radius**2
        mask[circle_mask] = 1

    def circles_to_mask(self, centers, radii):
        mask = np.zeros(self.im.shape[:2], dtype=bool)
        for center, radius in zip(centers, radii):
            self.add_circle_to_mask(center, radius, mask)
        return mask

    def get_scribble_masks(self):
        masks = {}
        for name, scribble in self.scribbles.items():
            if len(scribble) == 0:
                masks[name] = np.zeros(self.im.shape, dtype=bool)
            else:
                centers = [circle.center for circle in scribble]
                radii = [circle.radius for circle in scribble]
                masks[name] = self.circles_to_mask(centers, radii)
        return masks
