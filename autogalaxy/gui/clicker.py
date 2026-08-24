import numpy as np

import autoarray as aa

from autogalaxy import exc


class Clicker:
    def __init__(self, image, pixel_scales, search_box_size, in_pixels: bool = False):
        self.image = image

        pixel_scales = aa.util.geometry.convert_pixel_scales_2d(
            pixel_scales=pixel_scales
        )

        self.pixel_scales = pixel_scales
        self.search_box_size = search_box_size

        self.click_list = []
        self.in_pixels = in_pixels

    def start(self, data, pixel_scales):
        from matplotlib import pyplot as plt
        from autoarray.plot.utils import _conf_imshow_origin
        from autogalaxy.util.plot_utils import norm_from

        n_y, n_x = data.shape_native
        hw = int(n_x / 2) * pixel_scales
        ext = [-hw, hw, -hw, hw]
        fig = plt.figure(figsize=(14, 14))
        norm = norm_from(
            array=data, use_log10=True, vmin=1.0e-3, vmax=np.max(data) / 3.0
        )
        plt.imshow(data.native, cmap="jet", norm=norm, extent=ext, origin=_conf_imshow_origin())
        if not data.mask.is_all_false:
            grid = data.mask.derive_grid.edge
            plt.scatter(y=grid[:, 0], x=grid[:, 1], c="k", marker="x", s=10)
        plt.colorbar()
        cid = fig.canvas.mpl_connect("button_press_event", self.onclick)
        plt.show()
        fig.canvas.mpl_disconnect(cid)
        plt.close(fig)

        return aa.Grid2DIrregular(values=self.click_list)

    def onclick(self, event):
        if event.dblclick:
            y_arcsec = (
                np.rint(event.ydata / self.pixel_scales[0]) * self.pixel_scales[0]
            )
            x_arcsec = (
                np.rint(event.xdata / self.pixel_scales[1]) * self.pixel_scales[1]
            )

            (y_pixels, x_pixels) = self.image.geometry.pixel_coordinates_2d_from(
                scaled_coordinates_2d=(y_arcsec, x_arcsec)
            )

            flux = -np.inf

            for y in range(
                y_pixels - self.search_box_size, y_pixels + self.search_box_size
            ):
                for x in range(
                    x_pixels - self.search_box_size, x_pixels + self.search_box_size
                ):
                    flux_new = self.image.native[y, x]

                    if flux_new > flux:
                        flux = flux_new
                        y_pixels_max = y
                        x_pixels_max = x

                print("clicked on the pixel:", y_pixels, x_pixels)
                print("Max flux pixel:", y_pixels_max, x_pixels_max)

            if self.in_pixels:
                self.click_list.append((y_pixels_max, x_pixels_max))
            else:
                grid_arcsec = self.image.geometry.grid_scaled_2d_from(
                    grid_pixels_2d=aa.Grid2D.no_mask(
                        values=[[[y_pixels_max + 0.5, x_pixels_max + 0.5]]],
                        pixel_scales=self.pixel_scales,
                    )
                )
                y_arcsec = grid_arcsec[0, 0]
                x_arcsec = grid_arcsec[0, 1]

                print("Arc-sec Coordinate", y_arcsec, x_arcsec)

                self.click_list.append((y_arcsec, x_arcsec))
