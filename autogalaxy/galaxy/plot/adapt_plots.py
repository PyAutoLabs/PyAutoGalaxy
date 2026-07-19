import numpy as np
from typing import Dict

import autoarray as aa
from autoarray.plot.utils import subplots, conf_subplot_figsize, tight_layout

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.util.plot_utils import plot_array, _save_subplot


def subplot_adapt_images(
    adapt_galaxy_name_image_dict: Dict[Galaxy, aa.Array2D],
    output_path=None,
    output_format=None,
    colormap="default",
    use_log10=False,
    title_prefix: str = None,
):
    """Create a subplot showing the adapt (model) image for each galaxy.

    Adapt images are per-galaxy model images produced during a previous
    non-linear search.  They are used to drive adaptive mesh and
    regularisation schemes in subsequent searches.  This function lays out
    one panel per entry in *adapt_galaxy_name_image_dict*, arranged in rows
    of up to three columns.

    If *adapt_galaxy_name_image_dict* is ``None`` the function returns
    immediately without producing any output.

    Parameters
    ----------
    adapt_galaxy_name_image_dict : dict[Galaxy, aa.Array2D] or None
        Mapping from galaxy (used as a label) to its adapt image array.
    output_path : str or None
        Directory in which to save the figure.  ``None`` → ``plt.show()``.
    output_format : str
        File format, e.g. ``"png"``.
    colormap : str
        Matplotlib colormap name, or ``"default"``.
    use_log10 : bool
        Apply a log₁₀ stretch to the image values.
    """
    if adapt_galaxy_name_image_dict is None:
        return

    n = len(adapt_galaxy_name_image_dict)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = subplots(rows, cols, figsize=conf_subplot_figsize(rows, cols))
    axes_list = [axes] if n == 1 else list(np.array(axes).flatten())

    _pf = (lambda t: f"{title_prefix.rstrip()} {t}") if title_prefix else (lambda t: t)
    for i, (_, galaxy_image) in enumerate(adapt_galaxy_name_image_dict.items()):
        plot_array(
            array=galaxy_image,
            title=_pf("Galaxy Image"),
            colormap=colormap,
            use_log10=use_log10,
            ax=axes_list[i],
        )

    tight_layout()
    _save_subplot(fig, output_path, "adapt_images", output_format)


def fits_adapt_images(adapt_images, output_path) -> None:
    """Write FITS files for the adapt images and image-plane mesh grids.

    Writes up to two FITS files into *output_path*:

    * ``adapt_images.fits`` — one HDU per galaxy adapt image, plus a
      ``mask`` extension, written when
      ``adapt_images.galaxy_name_image_dict`` is not ``None``.
    * ``adapt_image_plane_mesh_grids.fits`` — one HDU per galaxy
      image-plane mesh grid, written when
      ``adapt_images.galaxy_name_image_plane_mesh_grid_dict`` is not
      ``None``.

    Parameters
    ----------
    adapt_images : AdaptImages
        The adapt images container holding per-galaxy image and mesh-grid
        dictionaries.
    output_path : str or Path
        Directory in which to write the FITS files.
    """
    import numpy as np
    from pathlib import Path
    from autonerves.fitsable import hdu_list_for_output_from

    output_path = Path(output_path)

    if adapt_images.galaxy_name_image_dict is not None:
        image_list = [
            adapt_images.galaxy_name_image_dict[name].native_for_fits
            for name in adapt_images.galaxy_name_image_dict
        ]
        hdu_list = hdu_list_for_output_from(
            values_list=[image_list[0].mask.astype("float")] + image_list,
            ext_name_list=["mask"] + list(adapt_images.galaxy_name_image_dict.keys()),
            header_dict=adapt_images.mask.header_dict,
        )
        hdu_list.writeto(output_path / "adapt_images.fits", overwrite=True)

    if adapt_images.galaxy_name_image_plane_mesh_grid_dict is not None:
        mesh_grid_list = [
            adapt_images.galaxy_name_image_plane_mesh_grid_dict[name].native
            for name in adapt_images.galaxy_name_image_plane_mesh_grid_dict
        ]
        hdu_list = hdu_list_for_output_from(
            values_list=[np.array([1])] + mesh_grid_list,
            ext_name_list=[""] + list(adapt_images.galaxy_name_image_plane_mesh_grid_dict.keys()),
        )
        hdu_list.writeto(output_path / "adapt_image_plane_mesh_grids.fits", overwrite=True)
