import numpy as np
import pytest

import autogalaxy as ag


def test__deflections_yx_2d_from():
    cnfw = ag.mp.cNFWSph(
        centre=(0.0, 0.0),
        kappa_s=0.01591814312464436,
        scale_radius=0.36,
        core_radius=0.036,
    )

    deflection_2d = cnfw.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflection_r = np.sqrt(deflection_2d[0, 0] ** 2 + deflection_2d[0, 1] ** 2)

    assert deflection_r == pytest.approx(0.006034319441107217, 1.0e-8)


def test__convergence_func__matches_convergence_2d_from_and_is_q_independent():
    """Regression: the cNFW family overrides the abstract `convergence_func` (reached by
    radial mass integration / Einstein radius) by reusing the MGE-of-3D-density
    decomposition evaluated radially. It is q-independent (like `NFW.convergence_func`):
    ellipticity is re-introduced by the MGE machinery elsewhere, so no `sigmas_factor`
    rescale is applied."""

    kappa_s, scale_radius, core_radius = 0.2, 2.0, 0.1
    radii = np.array([0.5, 1.0, 2.0, 3.0])

    sph = ag.mp.cNFWSph(
        centre=(0.0, 0.0),
        kappa_s=kappa_s,
        scale_radius=scale_radius,
        core_radius=core_radius,
    )

    # `convergence_2d_from` on a `Grid2DIrregular` short-circuits the `@over_sample`
    # decorator, so points at (0, r) evaluate the radial profile exactly.
    grid = ag.Grid2DIrregular([[0.0, r] for r in radii])
    truth = np.asarray(sph.convergence_2d_from(grid=grid).array)

    actual = np.asarray(sph.convergence_func(ag.ArrayIrregular(radii)))
    assert actual == pytest.approx(truth, 1e-8)

    # An elliptical cNFW returns the SAME radial (q-independent) convergence; this guards
    # the `sigmas_factor=1.0` choice (a sqrt(q) factor would make the elliptical case wrong).
    ell = ag.mp.cNFW(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.4),
        kappa_s=kappa_s,
        scale_radius=scale_radius,
        core_radius=core_radius,
    )
    actual_ell = np.asarray(ell.convergence_func(ag.ArrayIrregular(radii)))
    assert actual_ell == pytest.approx(truth, 1e-8)


def test__convergence_func__mass_integral_runs():
    """The Einstein-radius / enclosed-mass integral (`mass_angular_within_circle_from`)
    routes through `mass_integral` -> `convergence_func` via scipy.quad; before the
    override this raised NotImplementedError."""

    cnfw = ag.mp.cNFWSph(
        centre=(0.0, 0.0), kappa_s=0.2, scale_radius=2.0, core_radius=0.1
    )

    assert cnfw.mass_angular_within_circle_from(radius=1.0) > 0.0
