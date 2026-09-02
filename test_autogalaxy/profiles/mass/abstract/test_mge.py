import numpy as np
import pytest

from autogalaxy.profiles.mass import MGEDecomposer

import autogalaxy as ag


def test__gnfw_deflections_yx_2d_via_mge__config_1__inner_slope_05():
    nfw = ag.mp.gNFW(
        centre=(0.0, 0.0),
        kappa_s=1.0,
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.3, angle=100.0),
        inner_slope=0.5,
        scale_radius=8.0,
    )

    radii_min = nfw.scale_radius / 2000.0
    radii_max = nfw.scale_radius * 30.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    deflections_via_integral = np.array([[0.26604, 0.58988]])

    mge_decomp = ag.mp.MGEDecomposer(mass_profile=nfw)

    deflections_via_mge = mge_decomp.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.1875, 0.1625]]),
        xp=np,
        sigma_log_list=sigmas,
        ellipticity_convention="major",
        three_D=True,
    )

    assert deflections_via_integral == pytest.approx(deflections_via_mge, 1.0e-3)


def test__gnfw_deflections_yx_2d_via_mge__config_2__inner_slope_15():
    nfw = ag.mp.gNFW(
        centre=(0.3, 0.2),
        kappa_s=2.5,
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.5, angle=100.0),
        inner_slope=1.5,
        scale_radius=4.0,
    )

    radii_min = nfw.scale_radius / 20000.0
    radii_max = nfw.scale_radius * 30.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 30)
    sigmas = np.exp(log_sigmas)

    deflections_via_integral = np.array([[-5.99032, -4.02541]])
    mge_decomp = ag.mp.MGEDecomposer(mass_profile=nfw)

    deflections_via_mge = mge_decomp.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.1875, 0.1625]]),
        xp=np,
        sigma_log_list=sigmas,
        ellipticity_convention="major",
        three_D=True,
    )

    assert deflections_via_integral == pytest.approx(deflections_via_mge, 1.0e-3)


def test__sersic_deflections_yx_2d_via_mge__sersic_index_2():
    mp = ag.mp.Sersic(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    radii_min = mp.effective_radius / 100.0
    radii_max = mp.effective_radius * 20.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    deflections_via_integral = np.array([[1.1446, 0.79374]])
    mge_decomp = ag.mp.MGEDecomposer(mass_profile=mp)

    deflections_via_mge = mge_decomp.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]]),
        xp=np,
        sigma_log_list=sigmas,
        ellipticity_convention="circularised",
        three_D=False,
    )

    assert deflections_via_integral == pytest.approx(deflections_via_mge, 1.0e-3)


def test__sersic_deflections_yx_2d_via_mge__sersic_index_3():
    mp = ag.mp.Sersic(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=10.0,
        effective_radius=0.2,
        sersic_index=3.0,
        mass_to_light_ratio=1.0,
    )

    radii_min = mp.effective_radius / 100.0
    radii_max = mp.effective_radius * 20.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    deflections_via_integral = np.array([[2.6134, 1.80719]])

    mge_decomp = ag.mp.MGEDecomposer(mass_profile=mp)

    deflections_via_mge = mge_decomp.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]]),
        xp=np,
        sigma_log_list=sigmas,
        ellipticity_convention="circularised",
        three_D=False,
    )

    assert deflections_via_integral == pytest.approx(deflections_via_mge, 1.0e-3)


def test__cnfw_deflections_yx_2d_via_mge():
    cnfw = ag.mp.cNFWSph(
        centre=(0.3, 0.2), kappa_s=0.05, scale_radius=1.1, core_radius=0.01
    )

    radii_min = cnfw.scale_radius / 1000.0
    radii_max = cnfw.scale_radius * 200.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    deflections_analytic = cnfw.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.1875, 0.1625]]), xp=np
    )

    mge_decomp = ag.mp.MGEDecomposer(mass_profile=cnfw)

    deflections_via_mge = mge_decomp.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.1875, 0.1625]]),
        xp=np,
        sigma_log_list=sigmas,
        ellipticity_convention="major",
        three_D=True,
    )

    assert deflections_analytic == pytest.approx(deflections_via_mge, 1.0e-3)


def test__powerlaw_deflections_yx_2d_via_mge():
    mp = ag.mp.PowerLaw(
        centre=(-0.4, -0.2),
        ell_comps=(0.17142, -0.285116),
        einstein_radius=0.2,
        slope=1.8,
    )

    radii_min = mp.einstein_radius / 10000.0
    radii_max = mp.einstein_radius * 1000.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 30)
    sigmas = np.exp(log_sigmas)

    deflections_analytic = mp.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.1875, 0.1625]]), xp=np
    )

    mge_decomp = ag.mp.MGEDecomposer(mass_profile=mp)

    deflections_via_mge = mge_decomp.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.1875, 0.1625]]),
        xp=np,
        sigma_log_list=sigmas,
        ellipticity_convention="major",
        three_D=False,
    )

    assert deflections_analytic == pytest.approx(deflections_via_mge, 1.0e-2)


def test__chameleon_deflections_yx_2d_via_mge():
    mp = ag.mp.Chameleon(
        centre=(-0.4, -0.2), ell_comps=(0.17142, -0.285116), intensity=2.8
    )

    radii_min = mp.core_radius_0 / 10000.0
    radii_max = mp.core_radius_0 * 10000.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 30)
    sigmas = np.exp(log_sigmas)

    deflections_analytic = mp.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[-0.1875, 0.1625]]), xp=np
    )

    mge_decomp = ag.mp.MGEDecomposer(mass_profile=mp)

    deflections_via_mge = mge_decomp.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[-0.1875, 0.1625]]),
        xp=np,
        sigma_log_list=sigmas,
        ellipticity_convention="major",
        three_D=False,
    )

    assert deflections_analytic == pytest.approx(deflections_via_mge, 1.0e-3)


def test__DevVaucouleurs_convergence_2d_via_mge_from__config_1():
    mp = ag.mp.DevVaucouleurs(
        ell_comps=(0.0, 0.333333),
        intensity=3.0,
        effective_radius=2.0,
        mass_to_light_ratio=1.0,
    )

    radii_min = mp.effective_radius / 100.0
    radii_max = mp.effective_radius * 20.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    mge_decomp = MGEDecomposer(mass_profile=mp)

    convergence = mge_decomp.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]]),
        sigma_log_list=sigmas,
        ellipticity_convention="circularised",
        three_D=False,
    )

    assert convergence == pytest.approx(5.6697, 1e-3)


def test__DevVaucouleurs_convergence_2d_via_mge_from__config_2():
    mp = ag.mp.DevVaucouleurs(
        ell_comps=(0.0, -0.333333),
        intensity=2.0,
        effective_radius=3.0,
        mass_to_light_ratio=1.0,
    )

    radii_min = mp.effective_radius / 100.0
    radii_max = mp.effective_radius * 20.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    mge_decomp = MGEDecomposer(mass_profile=mp)

    convergence = mge_decomp.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.0, 1.0]]),
        sigma_log_list=sigmas,
        ellipticity_convention="circularised",
        three_D=False,
    )

    assert convergence == pytest.approx(7.4455, 1e-3)


def test__DevVaucouleurs_convergence_2d_via_mge_from__intensity_4():
    mp = ag.mp.DevVaucouleurs(
        ell_comps=(0.0, -0.333333),
        intensity=4.0,
        effective_radius=3.0,
        mass_to_light_ratio=1.0,
    )

    radii_min = mp.effective_radius / 100.0
    radii_max = mp.effective_radius * 20.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    mge_decomp = MGEDecomposer(mass_profile=mp)

    convergence = mge_decomp.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.0, 1.0]]),
        sigma_log_list=sigmas,
        ellipticity_convention="circularised",
        three_D=False,
    )

    assert convergence == pytest.approx(2.0 * 7.4455, 1e-3)


def test__DevVaucouleurs_convergence_2d_via_mge_from__mass_to_light_2():
    mp = ag.mp.DevVaucouleurs(
        ell_comps=(0.0, -0.333333),
        intensity=2.0,
        effective_radius=3.0,
        mass_to_light_ratio=2.0,
    )

    radii_min = mp.effective_radius / 100.0
    radii_max = mp.effective_radius * 20.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    mge_decomp = MGEDecomposer(mass_profile=mp)

    convergence = mge_decomp.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.0, 1.0]]),
        sigma_log_list=sigmas,
        ellipticity_convention="circularised",
        three_D=False,
    )

    assert convergence == pytest.approx(2.0 * 7.4455, 1e-3)


def test__DevVaucouleurs_convergence_2d_via_mge_from__small_effective_radius():
    mp = ag.mp.DevVaucouleurs(
        centre=(0.0, 0.0),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=1.0,
    )

    radii_min = mp.effective_radius / 100.0
    radii_max = mp.effective_radius * 20.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    mge_decomp = MGEDecomposer(mass_profile=mp)

    convergence = mge_decomp.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.0, 1.0]]),
        sigma_log_list=sigmas,
        ellipticity_convention="circularised",
        three_D=False,
    )

    assert convergence == pytest.approx(0.351797, 1e-3)


def test__exponential_convergence_2d_via_mge_from__config_1():
    mp = ag.mp.Exponential(
        ell_comps=(0.0, 0.333333),
        intensity=3.0,
        effective_radius=2.0,
        mass_to_light_ratio=1.0,
    )

    radii_min = mp.effective_radius / 100.0
    radii_max = mp.effective_radius * 20.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    mge_decomp = MGEDecomposer(mass_profile=mp)

    convergence = mge_decomp.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]]),
        sigma_log_list=sigmas,
        ellipticity_convention="circularised",
        three_D=False,
    )

    assert convergence == pytest.approx(4.9047, 1e-3)


def test__exponential_convergence_2d_via_mge_from__config_2():
    mp = ag.mp.Exponential(
        ell_comps=(0.0, -0.333333),
        intensity=2.0,
        effective_radius=3.0,
        mass_to_light_ratio=1.0,
    )

    radii_min = mp.effective_radius / 100.0
    radii_max = mp.effective_radius * 20.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    mge_decomp = MGEDecomposer(mass_profile=mp)

    convergence = mge_decomp.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.0, 1.0]]),
        sigma_log_list=sigmas,
        ellipticity_convention="circularised",
        three_D=False,
    )

    assert convergence == pytest.approx(4.8566, 1e-3)


def test__exponential_convergence_2d_via_mge_from__intensity_4():
    mp = ag.mp.Exponential(
        ell_comps=(0.0, -0.333333),
        intensity=4.0,
        effective_radius=3.0,
        mass_to_light_ratio=1.0,
    )

    radii_min = mp.effective_radius / 100.0
    radii_max = mp.effective_radius * 20.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    mge_decomp = MGEDecomposer(mass_profile=mp)

    convergence = mge_decomp.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.0, 1.0]]),
        sigma_log_list=sigmas,
        ellipticity_convention="circularised",
        three_D=False,
    )

    assert convergence == pytest.approx(2.0 * 4.8566, 1e-3)


def test__exponential_convergence_2d_via_mge_from__mass_to_light_2():
    mp = ag.mp.Exponential(
        ell_comps=(0.0, -0.333333),
        intensity=2.0,
        effective_radius=3.0,
        mass_to_light_ratio=2.0,
    )

    radii_min = mp.effective_radius / 100.0
    radii_max = mp.effective_radius * 20.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    mge_decomp = MGEDecomposer(mass_profile=mp)

    convergence = mge_decomp.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.0, 1.0]]),
        sigma_log_list=sigmas,
        ellipticity_convention="circularised",
        three_D=False,
    )

    assert convergence == pytest.approx(2.0 * 4.8566, 1e-3)


def test__exponential_convergence_2d_via_mge_from__mass_to_light_1():
    mp = ag.mp.Exponential(
        ell_comps=(0.0, -0.333333),
        intensity=2.0,
        effective_radius=3.0,
        mass_to_light_ratio=1.0,
    )

    radii_min = mp.effective_radius / 100.0
    radii_max = mp.effective_radius * 20.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    mge_decomp = MGEDecomposer(mass_profile=mp)

    convergence = mge_decomp.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.0, 1.0]]),
        sigma_log_list=sigmas,
        ellipticity_convention="circularised",
        three_D=False,
    )

    assert convergence == pytest.approx(4.8566, 1e-3)


def test__nfw_convergence_2d_via_mge_from__grid_2():
    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    radii_min = nfw.scale_radius / 2000.0
    radii_max = nfw.scale_radius * 30.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    mge_decomp = MGEDecomposer(mass_profile=nfw)

    convergence = mge_decomp.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[2.0, 0.0]]),
        sigma_log_list=sigmas,
        ellipticity_convention="major",
        three_D=True,
    )

    assert convergence == pytest.approx(0.263600141, 1e-2)


def test__nfw_convergence_2d_via_mge_from__grid_05():
    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    radii_min = nfw.scale_radius / 2000.0
    radii_max = nfw.scale_radius * 30.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    mge_decomp = MGEDecomposer(mass_profile=nfw)

    convergence = mge_decomp.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.5, 0.0]]),
        sigma_log_list=sigmas,
        ellipticity_convention="major",
        three_D=True,
    )

    assert convergence == pytest.approx(1.388511, 1e-2)


def test__nfw_convergence_2d_via_mge_from__kappa_s_2():
    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=1.0)

    radii_min = nfw.scale_radius / 2000.0
    radii_max = nfw.scale_radius * 30.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    mge_decomp = MGEDecomposer(mass_profile=nfw)

    convergence = mge_decomp.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.5, 0.0]]),
        sigma_log_list=sigmas,
        ellipticity_convention="major",
        three_D=True,
    )

    assert convergence == pytest.approx(2.0 * 1.388511, 1e-2)


def test__nfw_convergence_2d_via_mge_from__scale_radius_2():
    nfw = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=2.0)

    radii_min = nfw.scale_radius / 2000.0
    radii_max = nfw.scale_radius * 30.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    mge_decomp = MGEDecomposer(mass_profile=nfw)

    convergence = mge_decomp.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]]),
        sigma_log_list=sigmas,
        ellipticity_convention="major",
        three_D=True,
    )

    assert convergence == pytest.approx(1.388511, 1e-2)


def test__nfw_convergence_2d_via_mge_from__elliptical():
    nfw = ag.mp.NFW(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        kappa_s=1.0,
        scale_radius=1.0,
    )

    radii_min = nfw.scale_radius / 2000.0
    radii_max = nfw.scale_radius * 30.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    mge_decomp = MGEDecomposer(mass_profile=nfw)

    convergence = mge_decomp.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.25, 0.0]]),
        sigma_log_list=sigmas,
        ellipticity_convention="major",
        three_D=True,
    )

    assert convergence == pytest.approx(1.388511, 1e-3)


def test__sersic_convergence_2d_via_mge_from__intensity_3():
    mp = ag.mp.Sersic(
        centre=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    radii_min = mp.effective_radius / 100.0
    radii_max = mp.effective_radius * 20.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    mge_decomp = MGEDecomposer(mass_profile=mp)

    convergence = mge_decomp.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.0, 1.5]]),
        sigma_log_list=sigmas,
        three_D=False,
        ellipticity_convention="circularised",
    )

    assert convergence == pytest.approx(4.90657319276, 1e-3)


def test__sersic_convergence_2d_via_mge_from__intensity_6():
    mp = ag.mp.Sersic(
        centre=(0.0, 0.0),
        intensity=6.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    radii_min = mp.effective_radius / 100.0
    radii_max = mp.effective_radius * 20.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    mge_decomp = MGEDecomposer(mass_profile=mp)

    convergence = mge_decomp.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.0, 1.5]]),
        sigma_log_list=sigmas,
        three_D=False,
        ellipticity_convention="circularised",
    )

    assert convergence == pytest.approx(2.0 * 4.90657319276, 1e-3)


def test__sersic_convergence_2d_via_mge_from__mass_to_light_2():
    mp = ag.mp.Sersic(
        centre=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
    )

    radii_min = mp.effective_radius / 100.0
    radii_max = mp.effective_radius * 20.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    mge_decomp = MGEDecomposer(mass_profile=mp)

    convergence = mge_decomp.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.0, 1.5]]),
        sigma_log_list=sigmas,
        three_D=False,
        ellipticity_convention="circularised",
    )

    assert convergence == pytest.approx(2.0 * 4.90657319276, 1e-3)


def test__sersic_convergence_2d_via_mge_from__elliptical():
    mp = ag.mp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

    radii_min = mp.effective_radius / 100.0
    radii_max = mp.effective_radius * 20.0
    log_sigmas = np.linspace(np.log(radii_min), np.log(radii_max), 20)
    sigmas = np.exp(log_sigmas)

    mge_decomp = MGEDecomposer(mass_profile=mp)

    convergence = mge_decomp.convergence_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]]),
        sigma_log_list=sigmas,
        three_D=False,
        ellipticity_convention="circularised",
    )

    assert convergence == pytest.approx(5.38066670129, 1e-3)


def test__wofz__numpy_path_is_scipy__rational_path_agrees_to_1e_4():
    """The numpy path is `scipy.special.wofz` itself (max relative error 1.3e-14 against
    an mpmath reference at dps 40); the hand-rolled rational routine kept for JAX is
    accurate to ~6 significant digits (3.0e-6 against the same reference)."""
    from scipy.special import wofz

    from autogalaxy.profiles.mass.abstract.mge import _wofz_rational

    z_list = [
        20.0 + 1j * 0.001,
        7.0 + 1j * 0.1,
        7.0 + 1j * 1e-11,
        2.0 + 1j * 0.001,
        2.0 + 1j * 1.0,
        1.0 + 1j * 0.001,
    ]

    for z in z_list:
        assert MGEDecomposer.wofz(z) == pytest.approx(wofz(z), rel=1.0e-12)
        assert _wofz_rational(z) == pytest.approx(wofz(z), rel=1.0e-4)


def test__deflections_2d_via_mge_from__spherical_case__is_radial_and_matches_elliptical_limit(
    monkeypatch,
):
    """A profile whose *unclamped* axis ratio is exactly 1 takes the exact radial closed
    form on the numpy path, instead of an elliptical Faddeeva sum at the q = 0.9999 clamp.

    The deflection of a circular profile is exactly radial (no cross-axis component on
    the axes; the clamped elliptical path leaves a spurious ~1e-8 one), and it matches
    the elliptical form in its q -> 1 limit. Reaching that limit needs the 0.9999 clamp
    lifted -- the clamp *is* the ~6e-5 bias the branch removes -- so it is monkeypatched
    away for the comparison only.
    """
    gnfw_sph = ag.mp.gNFWSph(
        centre=(0.0, 0.0), kappa_s=0.2, inner_slope=1.5, scale_radius=10.0
    )

    grid_axes = ag.Grid2DIrregular([[1.0, 0.0], [0.0, 2.5], [-1.7, 0.0], [0.0, 0.0]])

    deflections = gnfw_sph.deflections_yx_2d_from(grid=grid_axes)

    assert deflections[0, 1] == 0.0
    assert deflections[1, 0] == 0.0
    assert deflections[2, 1] == 0.0
    assert deflections[3, 0] == 0.0 and deflections[3, 1] == 0.0

    monkeypatch.setattr(
        MGEDecomposer,
        "axis_ratio",
        lambda self, xp=np: self.mass_profile.axis_ratio(xp=xp),
    )

    ell_comps = ag.convert.ell_comps_from(axis_ratio=0.999999, angle=0.0)

    gnfw_ell = ag.mp.gNFW(
        centre=(0.0, 0.0),
        ell_comps=(float(ell_comps[0]), float(ell_comps[1])),
        kappa_s=0.2,
        inner_slope=1.5,
        scale_radius=10.0,
    )

    grid = ag.Grid2DIrregular([[0.5, 1.0], [-2.0, 0.3], [1.5, -1.5], [0.05, 0.05]])

    deflections_sph = np.asarray(gnfw_sph.deflections_yx_2d_from(grid=grid).array)
    deflections_ell = np.asarray(gnfw_ell.deflections_yx_2d_from(grid=grid).array)

    assert deflections_sph == pytest.approx(deflections_ell, rel=1.0e-5)
