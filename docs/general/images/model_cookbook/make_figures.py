"""
Regenerate the committed model-figure renders for the PyAutoGalaxy docs.

Run from the PyAutoGalaxy repo root, with the worktree environment active::

    python docs/general/images/model_cookbook/make_figures.py

Every PNG is written at ``width=14.0`` inches and DPI 100, so none exceeds the
1400 px width budget of the ``model-figures`` epic.

One figure is written per ``print(model.info)`` stage of
``autogalaxy_workspace/scripts/guides/modeling/cookbook.py`` -- the models below
are that script's compositions -- and
``docs/overview/images/overview_3/mge_model.png`` is the multi-Gaussian
expansion model of ``scripts/imaging/features/multi_gaussian_expansion/modeling.py``.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

import autofit as af
import autogalaxy as ag

OUTPUT_PATH = Path(__file__).resolve().parent
OVERVIEW_PATH = REPO_ROOT / "docs" / "overview" / "images" / "overview_3"
WIDTH = 14.0


def write(model, name: str, path: Path = OUTPUT_PATH, **kwargs):
    """Render one model to ``<name>.png`` in ``path``."""
    path.mkdir(parents=True, exist_ok=True)

    af.ModelPlotter(model).figure(
        path=str(path),
        filename=name,
        format="png",
        width=WIDTH,
        **kwargs,
    )
    print(f"wrote {path.name}/{name}.png")


# ---------------------------------------------------------------------------
# the cookbook stages -- `scripts/guides/modeling/cookbook.py`
# ---------------------------------------------------------------------------


def simple():
    """``__Simple Model__`` -- cookbook.py:44-48."""
    bulge = af.Model(ag.lp_linear.Sersic)

    galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

    return af.Collection(galaxies=af.Collection(galaxy=galaxy))


def complex_model():
    """``__More Complex Models__`` -- cookbook.py:68-76."""
    bulge = af.Model(ag.lp_linear.Sersic)
    disk = af.Model(ag.lp_linear.Exponential)
    bar = af.Model(ag.lp_linear.Sersic)

    galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk, bar=bar)

    return af.Collection(galaxies=af.Collection(galaxy=galaxy))


def two_galaxies():
    """``__More Complex Models__``, two galaxies -- cookbook.py:83-107."""
    galaxy_0 = af.Model(ag.Galaxy, redshift=0.5, bulge=af.Model(ag.lp_linear.Sersic))
    galaxy_1 = af.Model(ag.Galaxy, redshift=0.5, bulge=af.Model(ag.lp_linear.Sersic))

    return af.Collection(galaxies=af.Collection(galaxy_0=galaxy_0, galaxy_1=galaxy_1))


def concise():
    """``__Concise API__`` -- cookbook.py:118-127."""
    galaxy = af.Model(
        ag.Galaxy,
        redshift=0.5,
        bulge=ag.lp_linear.Sersic,
        disk=ag.lp_linear.Exponential,
        bar=ag.lp_linear.Sersic,
    )

    return af.Collection(galaxies=af.Collection(galaxy=galaxy))


def prior_custom():
    """``__Prior Customization__`` -- cookbook.py:134-149."""
    bulge = af.Model(ag.lp_linear.Sersic)
    bulge.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    bulge.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    bulge.sersic_index = af.TruncatedGaussianPrior(
        mean=4.0, sigma=1.0, lower_limit=1.0, upper_limit=8.0
    )

    galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

    return af.Collection(galaxies=af.Collection(galaxy=galaxy))


def model_custom():
    """``__Model Customization__`` -- cookbook.py:156-194."""
    bulge = af.Model(ag.lp_linear.Sersic)
    disk = af.Model(ag.lp_linear.Exponential)

    bulge.centre = disk.centre
    bulge.sersic_index = 4.0
    bulge.effective_radius = disk.effective_radius + 0.1

    galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

    model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

    model.add_assertion(
        model.galaxies.galaxy.bulge.effective_radius
        > model.galaxies.galaxy.disk.effective_radius
    )
    model.add_assertion(model.galaxies.galaxy.bulge.effective_radius < 3.0)

    return model


# ---------------------------------------------------------------------------
# the overview_3 MGE figure
# ---------------------------------------------------------------------------


def mge_overview(total_gaussians: int = 30, gaussian_per_basis: int = 2):
    """
    The MGE model of ``features/multi_gaussian_expansion/modeling.py``.

    ``gaussian_per_basis`` bases of ``total_gaussians`` linear Gaussians, all
    sharing one ``centre``, each basis sharing its own ``ell_comps`` and every
    ``sigma`` fixed to its own log-spaced value.
    """
    mask_radius = 3.0
    pixel_scale = 0.1

    log10_sigma_list = np.linspace(
        np.log10(pixel_scale / 10.0), np.log10(mask_radius), total_gaussians
    )

    centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
    centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

    bulge_gaussian_list = []

    for _ in range(gaussian_per_basis):
        gaussian_list = af.Collection(
            af.Model(ag.lp_linear.Gaussian) for _ in range(total_gaussians)
        )

        for i, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = centre_0
            gaussian.centre.centre_1 = centre_1
            gaussian.ell_comps = gaussian_list[0].ell_comps
            gaussian.sigma = 10 ** log10_sigma_list[i]

        bulge_gaussian_list += gaussian_list

    bulge = af.Model(ag.lp_basis.Basis, profile_list=bulge_gaussian_list)

    galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

    return af.Collection(galaxies=af.Collection(galaxy=galaxy))


def main():
    write(simple(), "simple")
    write(complex_model(), "complex")
    write(two_galaxies(), "two_galaxies")
    write(concise(), "concise")
    write(prior_custom(), "prior_custom")
    write(model_custom(), "model_custom")

    write(mge_overview(), "mge_model", path=OVERVIEW_PATH)


if __name__ == "__main__":
    main()
