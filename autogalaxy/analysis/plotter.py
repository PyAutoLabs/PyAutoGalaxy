from __future__ import annotations
import os
from typing import List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from autoarray.plot.output import Output

from autonerves import conf

import autoarray as aa

from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.galaxy.galaxies import Galaxies
from autogalaxy.galaxy.plot import galaxies_plots
from autogalaxy.galaxy.plot import adapt_plots


def setting(section: Union[List[str], str], name: str):
    if isinstance(section, str):
        return conf.instance["visualize"]["plots"][section][name]

    for sect in reversed(section):
        try:
            return conf.instance["visualize"]["plots"][sect][name]
        except KeyError:
            continue

    return conf.instance["visualize"]["plots"][section[0]][name]


def plot_setting(section: Union[List[str], str], name: str) -> bool:
    return setting(section, name)


class Plotter:
    def __init__(self, image_path: Union[Path, str], title_prefix: str = None):
        """
        Base class for all plotters, which output visualizations during a model fit.

        The methods of the `Plotter` are called throughout a non-linear search via the
        `Analysis` class `visualize` method.

        The images output by the `Plotter` are customized using the file
        `config/visualize/plots.yaml`.

        Parameters
        ----------
        image_path
            The path on the hard-disk to the `image` folder of the non-linear search
            results where all visualizations are saved.
        title_prefix
            An optional string prefixed to every plot title.
        """
        from pathlib import Path

        self.image_path = Path(image_path)
        self.title_prefix = title_prefix

        os.makedirs(image_path, exist_ok=True)

    @property
    def fmt(self) -> List[str]:
        """The output file format(s) read from ``config/visualize/plots.yaml``."""
        try:
            return conf.instance["visualize"]["plots"]["subplot_format"]
        except KeyError:
            return conf.instance["visualize"]["plots"]["format"]

    def output_from(self) -> "Output":
        """Return an ``autoarray`` ``Output`` object pointed at ``image_path``."""
        from autoarray.plot.output import Output

        return Output(path=self.image_path, format=self.fmt)

    def galaxies(
        self,
        galaxies: List[Galaxy],
        grid: aa.type.Grid2DLike,
    ):
        """
        Output visualization of a list of galaxies.

        Controlled by the ``[galaxies]`` section of ``config/visualize/plots.yaml``.
        Outputs include galaxy image subplots and, when enabled, a FITS file of each
        galaxy image.

        Parameters
        ----------
        galaxies
            The list of galaxies to visualize.
        grid
            A 2D grid of (y, x) arc-second coordinates used to evaluate each galaxy
            image.
        """
        galaxies = Galaxies(galaxies=galaxies)

        def should_plot(name):
            return plot_setting(section="galaxies", name=name)

        if should_plot("subplot_galaxy_images"):
            galaxies_plots.subplot_galaxy_images(
                galaxies=galaxies,
                grid=grid,
                output_path=self.image_path,
                output_format=self.fmt,
                title_prefix=self.title_prefix,
            )

        if should_plot("subplot_galaxies"):
            galaxies_plots.subplot_galaxies(
                galaxies=galaxies,
                grid=grid,
                output_path=self.image_path,
                output_format=self.fmt,
                title_prefix=self.title_prefix,
            )

        if should_plot("fits_galaxy_images"):
            galaxies_plots.fits_galaxy_images(
                galaxies=galaxies, grid=grid, output_path=self.image_path
            )

    def inversion(self, inversion: aa.Inversion):
        """
        Output visualization of an ``Inversion``.

        Controlled by the ``[inversion]`` section of ``config/visualize/plots.yaml``.
        When enabled, outputs a scatter-plot of each mapper's source-plane
        reconstruction and a CSV of the reconstruction values and noise map.

        Parameters
        ----------
        inversion
            The inversion whose reconstruction is visualized.
        """
        def should_plot(name):
            return plot_setting(section="inversion", name=name)

        output = self.output_from()

        if should_plot("subplot_inversion"):
            from autoarray.inversion.plot.inversion_plots import subplot_of_mapper

            mapper_list = inversion.cls_list_from(cls=aa.Mapper)
            fmt = output.format_list[0] if output.format_list else "png"

            for i in range(len(mapper_list)):
                subplot_of_mapper(
                    inversion=inversion,
                    mapper_index=i,
                    output_path=output.path,
                    output_filename=f"inversion_{i}",
                    output_format=fmt,
                    title_prefix=self.title_prefix,
                )

        if should_plot("csv_reconstruction"):
            from autoarray.inversion.plot.inversion_plots import save_reconstruction_csv
            save_reconstruction_csv(inversion=inversion, output_path=self.image_path)

    def adapt_images(self, adapt_images: AdaptImages):
        """
        Output visualization of adapt images from a previous model-fit search.

        Controlled by the ``[adapt]`` section of ``config/visualize/plots.yaml``.
        Outputs a subplot of the per-galaxy adapt images and, when enabled, FITS files
        of the adapt images and image-plane mesh grids.

        Parameters
        ----------
        adapt_images
            The adapt images containing per-galaxy images used to drive adaptive mesh
            and regularization schemes.
        """
        def should_plot(name):
            return plot_setting(section="adapt", name=name)

        if adapt_images.galaxy_name_image_dict is not None:
            if should_plot("subplot_adapt_images"):
                adapt_plots.subplot_adapt_images(
                    adapt_galaxy_name_image_dict=adapt_images.galaxy_name_image_dict,
                    output_path=self.image_path,
                    output_format=self.fmt,
                    title_prefix=self.title_prefix,
                )

        if should_plot("fits_adapt_images"):
            adapt_plots.fits_adapt_images(
                adapt_images=adapt_images, output_path=self.image_path
            )
