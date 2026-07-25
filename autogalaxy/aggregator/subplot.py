from enum import Enum


class FITSFit(Enum):
    """
    The HDUs that can be extracted from the fit.fits file.
    """

    model_data = "MODEL_DATA"
    residual_map = "RESIDUAL_MAP"
    normalized_residual_map = "NORMALIZED_RESIDUAL_MAP"
    chi_squared_map = "CHI_SQUARED_MAP"


class Dataset(Enum):
    """
    The subplots that can be extracted from the fit image.

    The values correspond to the position of the subplot in the 4x3 grid.
    """

    data = (0, 0)
    data_log_10 = (1, 0)
    noise_map = (2, 0)
    psf = (0, 1)
    psf_log_10 = (1, 1)
    signal_to_noise_map = (2, 1)
    over_sample_size_lp = (0, 2)
    over_sample_size_pixelization = (1, 2)


class Fit(Enum):
    """
    The subplots that can be extracted from the fit image.

    The values correspond to the position of the subplot in the 4x3 grid.
    """

    data = (0, 0)
    signal_to_noise_map = (1, 0)
    model_data = (2, 0)
    normalized_residual_map = (0, 1)
    normalized_residual_map_one_sigma = (1, 1)
    chi_squared_map = (2, 1)
