from autoconf import jax_wrapper
from autoconf.dictable import register_parser
from autofit import conf

conf.instance.register(__file__)

from autoconf.dictable import from_dict, from_json, output_to_json, to_dict
from autoarray.dataset import preprocess  # noqa

from autoarray.dataset.imaging.dataset import Imaging  # noqa
from autoarray.dataset.interferometer.dataset import Interferometer  # noqa
from autoarray.dataset.dataset_model import DatasetModel
from autoarray.inversion.mesh import mesh  # noqa
from autoarray.inversion import regularization as reg  # noqa
from autoarray.inversion.mesh import image_mesh
from autoarray.inversion.mappers.abstract import Mapper  # noqa
from autoarray.settings import Settings  # noqa
from autoarray.inversion.inversion.factory import inversion_from as Inversion  # noqa
from autoarray.inversion.mesh.image_mesh.abstract import AbstractImageMesh
from autoarray.inversion.mesh.mesh.abstract import AbstractMesh
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoarray.inversion.pixelization import Pixelization  # noqa
from autoarray.inversion.mesh.border_relocator import BorderRelocator
from autoarray.mask.mask_1d import Mask1D  # noqa
from autoarray.mask.mask_2d import Mask2D  # noqa
from autoarray.mask.derive.zoom_2d import Zoom2D
from autoarray.operators.transformer import TransformerDFT  # noqa
from autoarray.operators.transformer import TransformerNUFFT  # noqa
from autoarray.layout.layout import Layout2D  # noqa
from autoarray.structures.arrays.uniform_1d import Array1D  # noqa
from autoarray.structures.arrays.uniform_2d import Array2D  # noqa
from autoarray.structures.arrays.rgb import Array2DRGB
from autoarray.structures.arrays.irregular import ArrayIrregular  # noqa
from autoarray.structures.header import Header  # noqa
from autoarray.structures.grids.uniform_1d import Grid1D  # noqa
from autoarray.structures.grids.uniform_2d import Grid2D  # noqa
from autoarray.structures.grids.irregular_2d import Grid2DIrregular  # noqa
from autoarray.operators.over_sampling.over_sampler import OverSampler  # noqa
from autoarray.operators.convolver import Convolver
from autoarray.inversion.mesh.interpolator.rectangular import (
    InterpolatorRectangular,
)  # noqa
from autoarray.inversion.mesh.interpolator.delaunay import (
    InterpolatorDelaunay,
)  # noqa
from autoarray.structures.vectors.uniform import VectorYX2D  # noqa
from autoarray.structures.vectors.irregular import VectorYX2DIrregular  # noqa
from autoarray.layout.region import Region1D  # noqa
from autoarray.layout.region import Region2D  # noqa
from autoarray.structures.visibilities import Visibilities  # noqa
from autoarray.structures.visibilities import VisibilitiesNoiseMap  # noqa
from autoarray.inversion.mesh.mesh_geometry.rectangular import (
    rectangular_edge_pixel_list_from,
)

from .analysis import model_util
from .analysis.adapt_images.adapt_images import AdaptImages
from .analysis.adapt_images.adapt_images import galaxy_name_image_dict_via_result_from
from . import aggregator as agg
from . import exc
from . import util
from .ellipse.dataset_interp import DatasetInterp
from .ellipse.ellipse.ellipse import Ellipse
from .ellipse.ellipse.ellipse_multipole import EllipseMultipole
from .ellipse.ellipse.ellipse_multipole import EllipseMultipoleScaled
from .ellipse.fit_ellipse import FitEllipse
from .ellipse.model.analysis import AnalysisEllipse
from .operate.image import OperateImage
from .operate.image import OperateImageList
from .operate.image import OperateImageGalaxies
from .operate.lens_calc import LensCalc
from .gui.scribbler import Scribbler
from .imaging.fit_imaging import FitImaging
from .imaging.model.analysis import AnalysisImaging
from .imaging.simulator import SimulatorImaging
from .interferometer.simulator import SimulatorInterferometer
from .interferometer.fit_interferometer import FitInterferometer
from .interferometer.model.analysis import AnalysisInterferometer

from .quantity.fit_quantity import FitQuantity
from .quantity.model.analysis import AnalysisQuantity
from .quantity.dataset_quantity import DatasetQuantity
from .galaxy.galaxy import Galaxy
from .galaxy.galaxies import Galaxies
from .galaxy.redshift import Redshift
from .galaxy.to_inversion import AbstractToInversion
from .galaxy.to_inversion import GalaxiesToInversion
from .profiles.geometry_profiles import EllProfile
from .profiles import (
    point_sources as ps,
    mass as mp,
    light_and_mass_profiles as lmp,
    light_linear_and_mass_profiles as lmp_linear,
    scaling_relations as sr,
)
from .profiles.light.abstract import LightProfile
from .profiles.light import standard as lp
from .profiles import basis as lp_basis
from .profiles.light.linear import LightProfileLinearObjFuncList
from .profiles.light import linear as lp_linear
from .profiles.light import operated as lp_operated
from .profiles.light import (
    linear_operated as lp_linear_operated,
)
from .profiles.light import snr as lp_snr
from . import convert
from . import mock as m  # noqa
from .util.shear_field import ShearYX2D
from .util.shear_field import ShearYX2DIrregular
from . import cosmology as cosmo
from .gui.clicker import Clicker
from .gui.scribbler import Scribbler

from autoconf import conf
from autoconf.fitsable import ndarray_via_hdu_from
from autoconf.fitsable import ndarray_via_fits_from
from autoconf.fitsable import header_obj_from
from autoconf.fitsable import output_to_fits
from autoconf.fitsable import hdu_list_for_output_from

__version__ = "2026.5.1.4"

from autoconf import check_version

check_version(__version__)


def __getattr__(name):
    if name == "plot":
        from . import plot

        globals()["plot"] = plot
        return plot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
