from collections import OrderedDict
import numpy as np
from typing import Tuple


class Scribbler:
    def __init__(
        self,
        image,
        segment_names=None,
        cmap=None,
        brush_width=0.05,
        backend="TkAgg",
        mask_overlay=None,
        figsize=(15, 15),
        rgb_image=None,
        extent: Tuple[float, float, float, float] = None,
    ):
        """

        This Scribbler tool is used for drawing custom masks and noise-maps via a GUI to images, see the other scripts in
        the gui folder for a description.

        This script is Adapted from https://gist.github.com/brikeats/4f63f867fd8ea0f196c78e9b835150ab
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

        import matplotlib
        import matplotlib.pyplot as plt
        from autoarray.plot.utils import _conf_imshow_origin

        matplotlib.use(backend)
        self.im = image

        # create initial plot
        self.figure = plt.figure(figsize=figsize)
        mng = plt.get_current_fig_manager()
        mng.window.wm_geometry("+50+50")  # For TkAgg backend
        if rgb_image is not None:
            self.ax = self.figure.add_subplot(121)
            plt.axis(extent)
            plt.axis("off")
            plt.imshow(rgb_image, origin=_conf_imshow_origin())
        self.ax = self.figure.add_subplot(111)

        if cmap is None:
            plt.imshow(image, interpolation="none", origin=_conf_imshow_origin())
        else:
            norm = cmap.norm_from(array=image)
            cmap_name = getattr(cmap, "cmap_name", None) or cmap.config_dict.get("cmap", "viridis")
            plt.imshow(image, cmap=cmap_name, norm=norm, origin=_conf_imshow_origin())

        if mask_overlay is not None:
            grid = mask_overlay.derive_grid.edge
            grid = mask_overlay.geometry.grid_pixel_centres_2d_from(grid_scaled_2d=grid)
            plt.scatter(y=grid[:, 0], x=grid[:, 1], c="k", marker="x", s=10)

        plt.axis(extent)
        plt.axis("off")

        # disable default keybindings
        manager, canvas = self.figure.canvas.manager, self.figure.canvas
        canvas.mpl_disconnect(manager.key_press_handler_id)

        # callbacks
        self.figure.canvas.mpl_connect("key_press_event", self.on_keypress)
        self.figure.canvas.mpl_connect("motion_notify_event", self.on_mouse_motion)
        self.figure.canvas.mpl_connect("button_press_event", self.on_mouse_down)
        self.figure.canvas.mpl_connect("button_release_event", self.on_mouse_up)

        # brush
        self.brush_radius = int(image.shape_native[0] * brush_width)
        self.min_radius = 4
        self.radius_increment = int(image.shape_native[0] * 0.01)
        self.brush_color = "b"
        self.brush = None

        # scribbles
        if not segment_names:
            segment_names = [str(num + 1) for num in range(2)]
        self.scribble_colors = "gr"
        self.scribbles = OrderedDict()
        for name in segment_names:
            self.scribbles[name] = []
        self.active_scribble = self.scribbles[segment_names[0]]
        self.active_scribble_color = self.scribble_colors[0]
        self.mouse_is_down = False
        self.num_patches = 0

        plt.subplots_adjust(wspace=0, hspace=0)
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

        import matplotlib

        center = (event.xdata, event.ydata)

        # draw the bush circle
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
            num = int(event.key) - 1
            name = list(self.scribbles.keys())[num]
            self.active_scribble = self.scribbles[name]
            self.active_scribble_color = self.scribble_colors[num]

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

    def enlarge_brush(self):
        self.brush_radius += self.radius_increment
        if self.brush:
            self.brush.radius = self.brush_radius
            self.figure.canvas.draw()

    def shrink_brush(self):
        self.brush_radius -= self.radius_increment
        self.brush_radius = max([self.brush_radius, self.min_radius])
        if self.brush:
            self.brush.radius = self.brush_radius
            self.figure.canvas.draw()

    def quit_(self):
        import matplotlib.pyplot as plt

        plt.close()
        self.figure.canvas.stop_event_loop()

    def show_mask(self):
        import matplotlib.pyplot as plt

        masks = self.get_scribble_masks()
        junk_mask = masks["1"]
        feature_mask = masks["2"]
        plt.ioff()
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(junk_mask.astype("int"), cmap="gray")
        # plt.title("Junk mask")
        # plt.subplot(122)
        # plt.imshow(feature_mask.astype("int"), cmap="gray")
        # plt.title("Feature mask")
        # plt.show()
        return junk_mask

    def add_circle_to_mask(self, center, radius, mask):
        if not center[0] or not center[1]:
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
                if centers:
                    masks[name] = self.circles_to_mask(centers, radii)
        return masks
