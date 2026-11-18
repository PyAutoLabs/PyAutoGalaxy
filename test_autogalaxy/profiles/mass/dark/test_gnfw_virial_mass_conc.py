import numpy as np
import pytest

import autogalaxy as ag
from autogalaxy.profiles.mass.dark.gnfw_virial_mass_conc import kappa_s_and_scale_radius

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


@pytest.mark.parametrize("inner_slope", [0.0, 0.5, 1.0, 1.5, 1.9, 1.99])
@pytest.mark.parametrize("c_2", [2.0, 5.0, 10.0, 20.0])
def test__kappa_s__numpy_matches_closed_form_hyp2f1(inner_slope, c_2):
    from scipy.special import hyp2f1

    cosmology = ag.cosmology.Planck15()

    kappa_s, scale_radius, virial_radius, overdens = kappa_s_and_scale_radius(
        cosmology=cosmology,
        virial_mass=1e12,
        c_2=c_2,
        overdens=0.0,
        redshift_object=0.5,
        redshift_source=1.0,
        inner_slope=inner_slope,
    )

    a = 3.0 - inner_slope
    c = (2.0 - inner_slope) * c_2
    de_c = (overdens / 3.0) * a * c**inner_slope / hyp2f1(a, a, a + 1.0, -c)
    kappa_s_expected = (
        cosmology.critical_density(0.5)
        * de_c
        * (virial_radius / c)
        / cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
            redshift_0=0.5, redshift_1=1.0
        )
    )

    assert kappa_s == pytest.approx(kappa_s_expected, rel=1.0e-10)


@pytest.mark.parametrize("inner_slope", [0.0, 0.5, 1.0, 1.5, 1.9, 1.99])
@pytest.mark.parametrize("c_2", [2.0, 5.0, 10.0, 20.0])
def test__kappa_s__jax_jit_matches_numpy(inner_slope, c_2):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    def kappa_s_from(log10m_vir, c_2, inner_slope):
        return ag.mp.gNFWVirialMassConcSph(
            log10m_vir=log10m_vir, c_2=c_2, inner_slope=inner_slope
        ).kappa_s

    kappa_s_numpy = kappa_s_from(12.0, c_2, inner_slope)
    kappa_s_jax = jax.jit(kappa_s_from)(
        jnp.array(12.0), jnp.array(c_2), jnp.array(inner_slope)
    )

    assert float(kappa_s_jax) == pytest.approx(kappa_s_numpy, rel=1.0e-8)
