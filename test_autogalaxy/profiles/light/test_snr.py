import autogalaxy as ag


def test__signal_to_noise_via_simulator__no_psf__snr_within_expected_range():
    background_sky_level = 10.0
    exposure_time = 300.0

    grid = ag.Grid2D.uniform(shape_native=(21, 21), pixel_scales=1.0)

    sersic = ag.lp_snr.Sersic(signal_to_noise_ratio=10.0)

    simulator = ag.SimulatorImaging(
        exposure_time=exposure_time,
        background_sky_level=background_sky_level,
        noise_seed=5,
    )

    dataset = simulator.via_galaxies_from(
        grid=grid, galaxies=[ag.Galaxy(redshift=0.5, light=sersic)]
    )

    assert 8.0 < dataset.signal_to_noise_max < 12.5


def test__signal_to_noise_via_simulator__with_psf__snr_within_expected_range():
    background_sky_level = 10.0
    exposure_time = 300.0

    grid = ag.Grid2D.uniform(shape_native=(21, 21), pixel_scales=1.0)

    sersic = ag.lp_snr.Sersic(signal_to_noise_ratio=10.0)

    psf = ag.Convolver.from_gaussian(
        shape_native=(3, 3), sigma=2.0, pixel_scales=0.2, normalize=True
    )

    simulator = ag.SimulatorImaging(
        psf=psf,
        exposure_time=exposure_time,
        background_sky_level=background_sky_level,
        noise_seed=5,
    )

    dataset = simulator.via_galaxies_from(
        grid=grid, galaxies=[ag.Galaxy(redshift=0.5, light=sersic)]
    )

    assert 8.5 < dataset.signal_to_noise_max < 11.5
