"""The public Galaxy plotting API forwards stretch and exposes the compositor."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm
import autogalaxy.plot as aplt
import autoarray as aa
from autoarray.dataset.plot.imaging_plots import subplot_imaging_dataset_intro


def test_intro_export_and_stretch_forwarding():
    assert aplt.subplot_imaging_dataset_intro is subplot_imaging_dataset_intro
    fig, ax = plt.subplots()
    aplt.plot_array(aa.Array2D.no_mask(np.eye(3), pixel_scales=.1), ax=ax, stretch=.01)
    assert isinstance(ax.images[0].norm, AsinhNorm)
    plt.close(fig)
