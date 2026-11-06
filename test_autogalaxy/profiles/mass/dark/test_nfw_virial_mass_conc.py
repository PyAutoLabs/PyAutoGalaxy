import numpy as np
import pytest

import autogalaxy as ag


def test__kappa_s_and_scale_radius__numpy_values():
    mp = ag.mp.NFWVirialMassConcSph(
        virial_mass=1e10,
        concentration=20.0,
        redshift_object=0.5,
        redshift_source=1.0,
    )

    assert mp.kappa_s == pytest.approx(0.03545305365717094, 1.0e-8)
    assert mp.scale_radius == pytest.approx(0.30064229478964527, 1.0e-8)


def test__deflections_yx_2d_from__jax_jit_over_mass_and_concentration_matches_numpy():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    grid = ag.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.1)

    kwargs = dict(centre=(0.1, -0.2), redshift_object=0.222, redshift_source=0.609)

    deflections_numpy = ag.mp.NFWVirialMassConcSph(
        virial_mass=10.0**9.5, concentration=50.0, **kwargs
    ).deflections_yx_2d_from(grid=grid)

    @jax.jit
    def deflections_from(virial_mass, concentration):
        mp = ag.mp.NFWVirialMassConcSph(
            virial_mass=virial_mass, concentration=concentration, **kwargs
        )
        return mp.deflections_yx_2d_from(grid=grid, xp=jnp).array

    deflections_jax = deflections_from(10.0**9.5, 50.0)

    assert np.asarray(deflections_jax) == pytest.approx(
        np.asarray(deflections_numpy), rel=1.0e-10
    )
