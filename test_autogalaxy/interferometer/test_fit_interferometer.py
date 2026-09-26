import numpy as np
import pytest

import autoarray as aa
import autogalaxy as ag


def test__model_visibilities__real_component__correct_value(interferometer_7):
    g0 = ag.Galaxy(redshift=0.5, bulge=ag.m.MockLightProfile(image_2d=np.ones(9)))

    fit = ag.FitInterferometer(dataset=interferometer_7, galaxies=[g0])

    assert fit.model_data.slim[0].real == pytest.approx(1.48496, abs=1.0e-4)


def test__model_visibilities__imaginary_component__correct_value(interferometer_7):
    g0 = ag.Galaxy(redshift=0.5, bulge=ag.m.MockLightProfile(image_2d=np.ones(9)))

    fit = ag.FitInterferometer(dataset=interferometer_7, galaxies=[g0])

    assert fit.model_data.slim[0].imag == pytest.approx(0.0, abs=1.0e-4)


def test__model_visibilities__log_likelihood__correct_value(interferometer_7):
    g0 = ag.Galaxy(redshift=0.5, bulge=ag.m.MockLightProfile(image_2d=np.ones(9)))

    fit = ag.FitInterferometer(dataset=interferometer_7, galaxies=[g0])

    assert fit.log_likelihood == pytest.approx(-34.1685958, abs=1.0e-4)


@pytest.mark.parametrize(
    "galaxies_factory, expected_fom, expect_inversion",
    [
        ("two_sersic_galaxies", -1994.35383952, False),
        ("basis_of_sersics", -1994.3538395, False),
        ("pixelization_only", -71.770448724198, True),
        ("light_plus_pixelization", -196.15073725528504, True),
        ("two_linear_light_profiles", -23.44419, True),
        ("basis_of_linear_light_profiles", -23.44419235, True),
    ],
)
def test__fit_figure_of_merit__various_galaxy_configs__correct_value_and_inversion_flag(
    interferometer_7, galaxies_factory, expected_fom, expect_inversion
):
    pixelization_01 = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=0.01),
    )
    pixelization_10 = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )
    galaxy_pix_01 = ag.Galaxy(redshift=0.5, pixelization=pixelization_01)
    galaxy_pix_10 = ag.Galaxy(redshift=0.5, pixelization=pixelization_10)

    g0_linear_light = ag.Galaxy(
        redshift=0.5, bulge=ag.lp_linear.Sersic(sersic_index=1.0, centre=(0.05, 0.05))
    )
    g1_linear_light = ag.Galaxy(
        redshift=0.5, bulge=ag.lp_linear.Sersic(sersic_index=4.0, centre=(0.05, 0.05))
    )

    factories = {
        "two_sersic_galaxies": [
            ag.Galaxy(
                redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0, centre=(0.05, 0.05))
            ),
            ag.Galaxy(
                redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0, centre=(0.05, 0.05))
            ),
        ],
        "basis_of_sersics": [
            ag.Galaxy(
                redshift=0.5,
                bulge=ag.lp_basis.Basis(
                    profile_list=[
                        ag.lp.Sersic(intensity=1.0, centre=(0.05, 0.05)),
                        ag.lp.Sersic(intensity=1.0, centre=(0.05, 0.05)),
                    ]
                ),
            )
        ],
        "pixelization_only": [ag.Galaxy(redshift=0.5), galaxy_pix_01],
        "light_plus_pixelization": [
            ag.Galaxy(
                redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0, centre=(0.05, 0.05))
            ),
            galaxy_pix_10,
        ],
        "two_linear_light_profiles": [g0_linear_light, g1_linear_light],
        "basis_of_linear_light_profiles": [
            ag.Galaxy(
                redshift=0.5,
                bulge=ag.lp_basis.Basis(
                    profile_list=[
                        ag.lp_linear.Sersic(sersic_index=1.0, centre=(0.05, 0.05)),
                        ag.lp_linear.Sersic(sersic_index=4.0, centre=(0.05, 0.05)),
                    ]
                ),
            )
        ],
    }

    fit = ag.FitInterferometer(
        dataset=interferometer_7, galaxies=factories[galaxies_factory]
    )

    assert fit.perform_inversion is expect_inversion
    assert fit.figure_of_merit == pytest.approx(expected_fom, 1.0e-4)


def test__fit_figure_of_merit__linear_light_plus_pixelization__log_evidence_correct(
    interferometer_7,
):
    g0_linear_light = ag.Galaxy(
        redshift=0.5, bulge=ag.lp_linear.Sersic(sersic_index=1.0, centre=(0.05, 0.05))
    )

    pixelization = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    galaxy_pix = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=interferometer_7, galaxies=[g0_linear_light, galaxy_pix]
    )

    assert fit.log_evidence == pytest.approx(-37.4081355120388, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-37.4081355120388, 1.0e-4)


def test__galaxy_image_dict__normal_light_profiles__individual_galaxies_match(
    interferometer_7,
):
    g0 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0, centre=(0.05, 0.05)))
    g1 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=2.0, centre=(0.05, 0.05)))
    g2 = ag.Galaxy(
        redshift=0.5,
        light_profile_0=ag.lp.Sersic(intensity=1.0, centre=(0.05, 0.05)),
        light_profile_1=ag.lp.Sersic(intensity=2.0, centre=(0.05, 0.05)),
    )

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0, g1, g2],
    )

    g0_image = g0.image_2d_from(grid=interferometer_7.grids.lp)
    g1_image = g1.image_2d_from(grid=interferometer_7.grids.lp)

    assert fit.galaxy_image_dict[g0] == pytest.approx(g0_image.array, 1.0e-4)
    assert fit.galaxy_image_dict[g1] == pytest.approx(g1_image.array, 1.0e-4)
    assert fit.galaxy_image_dict[g2] == pytest.approx(
        g0_image.array + g1_image.array, 1.0e-4
    )


def test__galaxy_image_dict__linear_light_profile_only__correct_pixel_value(
    interferometer_7,
):
    g0_linear = ag.Galaxy(redshift=0.5, bulge=ag.lp_linear.Sersic(centre=(0.05, 0.05)))

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0_linear],
    )

    assert fit.galaxy_image_dict[g0_linear][4] == pytest.approx(0.9876689631, 1.0e-4)


def test__galaxy_image_dict__pixelization_only__no_light_galaxy_returns_zeros(
    interferometer_7,
):
    mesh = ag.mesh.RectangularUniform(shape=(3, 3))

    pixelization = ag.Pixelization(
        mesh=mesh,
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    g0 = ag.Galaxy(redshift=0.5)
    galaxy_pix_0 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0, galaxy_pix_0],
        settings=ag.Settings(use_border_relocator=True),
    )

    assert (fit.galaxy_image_dict[g0].native == 0.0 + 0.0j * np.zeros((7,))).all()


def test__galaxy_image_dict__pixelization_only__matches_inversion_mapped_reconstructed_data(
    interferometer_7,
):
    mesh = ag.mesh.RectangularUniform(shape=(3, 3))

    pixelization = ag.Pixelization(
        mesh=mesh,
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    g0 = ag.Galaxy(redshift=0.5)
    galaxy_pix_0 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0, galaxy_pix_0],
        settings=ag.Settings(use_border_relocator=True),
    )

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=interferometer_7.grids.lp,
        border_relocator=interferometer_7.grids.border_relocator,
        source_plane_mesh_grid=None,
    )

    mapper = ag.Mapper(
        interpolator=interpolator,
        regularization=pixelization.regularization,
    )

    inversion = ag.Inversion(
        dataset=interferometer_7,
        linear_obj_list=[mapper],
    )

    assert fit.galaxy_image_dict[galaxy_pix_0].array == pytest.approx(
        inversion.mapped_reconstructed_data.slim.array, 1.0e-4
    )


def test__galaxy_image_dict__linear_and_pixelization__correct_pixel_values(
    interferometer_7,
):
    g1 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=2.0, centre=(0.05, 0.05)))
    g0_linear = ag.Galaxy(redshift=0.5, bulge=ag.lp_linear.Sersic(centre=(0.05, 0.05)))

    pixelization_10 = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )
    pixelization_20 = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=2.0),
    )

    galaxy_pix_0 = ag.Galaxy(redshift=0.5, pixelization=pixelization_10)
    galaxy_pix_1 = ag.Galaxy(redshift=0.5, pixelization=pixelization_20)

    g1_image = g1.image_2d_from(grid=interferometer_7.grids.lp)

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0_linear, g1, galaxy_pix_0, galaxy_pix_1],
    )

    assert fit.galaxy_image_dict[g0_linear][4] == pytest.approx(-46.8820117, 1.0e-2)
    assert fit.galaxy_image_dict[g1] == pytest.approx(g1_image.array, 1.0e-4)
    assert fit.galaxy_image_dict[galaxy_pix_0][4] == pytest.approx(-0.00541699, 1.0e-2)
    assert fit.galaxy_image_dict[galaxy_pix_1][4] == pytest.approx(-0.00563034, 1.0e-2)


def test__galaxy_image_dict__linear_and_pixelization__sum_matches_inversion_mapped_data(
    interferometer_7,
):
    g1 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=2.0, centre=(0.05, 0.05)))
    g0_linear = ag.Galaxy(redshift=0.5, bulge=ag.lp_linear.Sersic(centre=(0.05, 0.05)))

    pixelization_10 = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )
    pixelization_20 = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=2.0),
    )

    galaxy_pix_0 = ag.Galaxy(redshift=0.5, pixelization=pixelization_10)
    galaxy_pix_1 = ag.Galaxy(redshift=0.5, pixelization=pixelization_20)

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0_linear, g1, galaxy_pix_0, galaxy_pix_1],
    )

    mapped_reconstructed_data = (
        fit.galaxy_image_dict[g0_linear]
        + fit.galaxy_image_dict[galaxy_pix_0]
        + fit.galaxy_image_dict[galaxy_pix_1]
    )

    assert mapped_reconstructed_data.array == pytest.approx(
        fit.inversion.mapped_reconstructed_data.array, 1.0e-4
    )


def test__galaxy_model_visibilities_dict__normal_light_profiles__individual_galaxies_match(
    interferometer_7,
):
    g0 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0, centre=(0.05, 0.05)))
    g1 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=2.0, centre=(0.05, 0.05)))
    g2 = ag.Galaxy(
        redshift=0.5,
        light_profile_0=ag.lp.Sersic(intensity=1.0, centre=(0.05, 0.05)),
        light_profile_1=ag.lp.Sersic(intensity=2.0, centre=(0.05, 0.05)),
    )

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0, g1, g2],
    )

    g0_visibilities = g0.visibilities_from(
        grid=interferometer_7.grids.lp, transformer=interferometer_7.transformer
    )
    g1_visibilities = g1.visibilities_from(
        grid=interferometer_7.grids.lp, transformer=interferometer_7.transformer
    )

    assert fit.galaxy_model_visibilities_dict[g0].array == pytest.approx(
        g0_visibilities.array, 1.0e-4
    )
    assert fit.galaxy_model_visibilities_dict[g1].array == pytest.approx(
        g1_visibilities.array, 1.0e-4
    )
    assert fit.galaxy_model_visibilities_dict[g2].array == pytest.approx(
        g0_visibilities.array + g1_visibilities.array, 1.0e-4
    )


def test__galaxy_model_visibilities_dict__linear_light_profile_only__correct_first_visibility(
    interferometer_7,
):
    g0_linear = ag.Galaxy(redshift=0.5, bulge=ag.lp_linear.Sersic(centre=(0.05, 0.05)))

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0_linear],
    )

    assert fit.galaxy_model_visibilities_dict[g0_linear][0] == pytest.approx(
        0.9965209248910107 + 0.00648675263899049j, 1.0e-4
    )


def test__galaxy_model_visibilities_dict__pixelization_only__no_light_galaxy_returns_zeros(
    interferometer_7,
):
    mesh = ag.mesh.RectangularUniform(shape=(3, 3))

    pixelization = ag.Pixelization(
        mesh=mesh,
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    g0 = ag.Galaxy(redshift=0.5)
    galaxy_pix_0 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0, galaxy_pix_0],
        settings=ag.Settings(use_border_relocator=True),
    )

    assert (fit.galaxy_model_visibilities_dict[g0] == 0.0 + 0.0j * np.zeros((7,))).all()


def test__galaxy_model_visibilities_dict__pixelization_only__matches_inversion_reconstructed_operated_data(
    interferometer_7,
):
    mesh = ag.mesh.RectangularUniform(shape=(3, 3))

    pixelization = ag.Pixelization(
        mesh=mesh,
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    g0 = ag.Galaxy(redshift=0.5)
    galaxy_pix_0 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0, galaxy_pix_0],
        settings=ag.Settings(use_border_relocator=True),
    )

    interpolator = mesh.interpolator_from(
        source_plane_data_grid=interferometer_7.grids.lp,
        border_relocator=interferometer_7.grids.border_relocator,
        source_plane_mesh_grid=None,
    )

    mapper = ag.Mapper(
        interpolator=interpolator,
        regularization=pixelization.regularization,
    )

    inversion = ag.Inversion(
        dataset=interferometer_7,
        linear_obj_list=[mapper],
    )

    assert fit.galaxy_model_visibilities_dict[galaxy_pix_0].array == pytest.approx(
        inversion.mapped_reconstructed_operated_data.array, 1.0e-4
    )


def test__galaxy_model_visibilities_dict__linear_and_pixelization__correct_first_visibility_values(
    interferometer_7,
):
    g1 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=2.0, centre=(0.05, 0.05)))
    g0_linear = ag.Galaxy(redshift=0.5, bulge=ag.lp_linear.Sersic(centre=(0.05, 0.05)))

    pixelization_10 = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )
    pixelization_20 = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=2.0),
    )

    galaxy_pix_0 = ag.Galaxy(redshift=0.5, pixelization=pixelization_10)
    galaxy_pix_1 = ag.Galaxy(redshift=0.5, pixelization=pixelization_20)

    g1_visibilities = g1.visibilities_from(
        grid=interferometer_7.grids.lp, transformer=interferometer_7.transformer
    )

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0_linear, g1, galaxy_pix_0, galaxy_pix_1],
    )

    assert fit.galaxy_model_visibilities_dict[g0_linear][0] == pytest.approx(
        -47.30219078770512 - 0.3079088489343429j, 1.0e-4
    )
    assert fit.galaxy_model_visibilities_dict[g1].array == pytest.approx(
        g1_visibilities.array, 1.0e-4
    )
    assert fit.galaxy_model_visibilities_dict[galaxy_pix_0][0] == pytest.approx(
        -0.00889895 + 0.22151583j, 1.0e-4
    )
    assert fit.galaxy_model_visibilities_dict[galaxy_pix_1][0] == pytest.approx(
        -0.00857457 + 0.05537896j, 1.0e-4
    )


def test__galaxy_model_visibilities_dict__linear_and_pixelization__sum_matches_inversion_operated_data(
    interferometer_7,
):
    g1 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=2.0, centre=(0.05, 0.05)))
    g0_linear = ag.Galaxy(redshift=0.5, bulge=ag.lp_linear.Sersic(centre=(0.05, 0.05)))

    pixelization_10 = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )
    pixelization_20 = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=2.0),
    )

    galaxy_pix_0 = ag.Galaxy(redshift=0.5, pixelization=pixelization_10)
    galaxy_pix_1 = ag.Galaxy(redshift=0.5, pixelization=pixelization_20)

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0_linear, g1, galaxy_pix_0, galaxy_pix_1],
    )

    mapped_reconstructed_visibilities = (
        fit.galaxy_model_visibilities_dict[g0_linear]
        + fit.galaxy_model_visibilities_dict[galaxy_pix_0]
        + fit.galaxy_model_visibilities_dict[galaxy_pix_1]
    )

    assert mapped_reconstructed_visibilities.array == pytest.approx(
        fit.inversion.mapped_reconstructed_operated_data.array, 1.0e-4
    )


def test__model_visibilities_of_galaxies_list__matches_galaxy_model_visibilities_dict(
    interferometer_7,
):
    galaxy_light = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))
    galaxy_linear = ag.Galaxy(redshift=0.5, bulge=ag.lp_linear.Sersic())

    pixelization = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    galaxy_pix = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=interferometer_7, galaxies=[galaxy_light, galaxy_linear, galaxy_pix]
    )

    assert fit.model_visibilities_of_galaxies_list[0].array == pytest.approx(
        fit.galaxy_model_visibilities_dict[galaxy_light].array, 1.0e-4
    )
    assert fit.model_visibilities_of_galaxies_list[1].array == pytest.approx(
        fit.galaxy_model_visibilities_dict[galaxy_linear].array, 1.0e-4
    )
    assert fit.model_visibilities_of_galaxies_list[2].array == pytest.approx(
        fit.galaxy_model_visibilities_dict[galaxy_pix].array, 1.0e-4
    )


def _assert_sparse_fit_matches_dense(dataset, dataset_sparse, galaxies):
    fit = ag.FitInterferometer(dataset=dataset, galaxies=galaxies)
    fit_sparse = ag.FitInterferometer(dataset=dataset_sparse, galaxies=galaxies)

    assert isinstance(fit.inversion, aa.InversionInterferometerMapping)
    assert isinstance(fit_sparse.inversion, aa.InversionInterferometerSparse)

    assert fit_sparse.log_likelihood == pytest.approx(fit.log_likelihood, rel=1.0e-8)
    assert fit_sparse.log_evidence == pytest.approx(fit.log_evidence, rel=1.0e-8)

    # The image `i_p` the sparse dirty image is corrected with (`d~ - W~ i_p`) must be exactly the image the
    # fit's `profile_visibilities` are the Fourier transform of.
    profile_visibilities = ag.Galaxies(galaxies=galaxies).visibilities_from(
        grid=dataset_sparse.grids.lp, transformer=dataset_sparse.transformer
    )

    np.testing.assert_allclose(
        dataset_sparse.transformer.visibilities_from(
            image=fit_sparse.profile_image
        ).array,
        profile_visibilities.array,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        fit_sparse.profile_visibilities.array,
        profile_visibilities.array,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test__fit_figure_of_merit__sparse_operator__linear_light_only__matches_dense(
    interferometer_7,
):
    """
    A fit whose light is entirely linear (e.g. an MGE) inverts the unsubtracted visibilities, so the dirty
    image cached on the sparse operator is used as-is and the sparse fit reproduces the dense one.
    """
    dataset_sparse = interferometer_7.apply_sparse_operator(use_jax=False)

    galaxies = [
        ag.Galaxy(
            redshift=0.5,
            bulge=ag.lp_linear.Gaussian(sigma=0.5, centre=(0.05, 0.05)),
            disk=ag.lp_linear.Gaussian(sigma=1.5, centre=(0.05, 0.05)),
        )
    ]

    galaxies_basis = [
        ag.Galaxy(
            redshift=0.5,
            bulge=ag.lp_basis.Basis(
                profile_list=[
                    ag.lp_linear.Gaussian(sigma=0.5, centre=(0.05, 0.05)),
                    ag.lp_linear.Gaussian(sigma=1.5, centre=(0.05, 0.05)),
                ]
            ),
        )
    ]

    for galaxies in (galaxies, galaxies_basis):
        fit_sparse = ag.FitInterferometer(dataset=dataset_sparse, galaxies=galaxies)

        assert fit_sparse.inversion.dataset.sparse_dirty_image is None

        _assert_sparse_fit_matches_dense(
            dataset=interferometer_7, dataset_sparse=dataset_sparse, galaxies=galaxies
        )


def test__fit_figure_of_merit__sparse_operator__light_profile_and_linear_light__matches_dense(
    interferometer_7,
):
    """
    An ordinary light profile's visibilities are subtracted before the inversion, so the sparse data vector
    must be formed from the dirty image of the profile-subtracted visibilities rather than the one cached on
    the sparse operator (which is of the unsubtracted visibilities).
    """
    dataset_sparse = interferometer_7.apply_sparse_operator(use_jax=False)

    galaxies = [
        ag.Galaxy(
            redshift=0.5,
            bulge=ag.lp.Sersic(intensity=0.1, centre=(0.05, 0.05)),
            disk=ag.lp_linear.Gaussian(sigma=1.0, centre=(0.05, 0.05)),
        )
    ]

    # The same ordinary light profile inside a `Basis` alongside linear ones.
    galaxies_basis = [
        ag.Galaxy(
            redshift=0.5,
            bulge=ag.lp_basis.Basis(
                profile_list=[
                    ag.lp.Sersic(intensity=0.1, centre=(0.05, 0.05)),
                    ag.lp_linear.Gaussian(sigma=1.0, centre=(0.05, 0.05)),
                ]
            ),
        )
    ]

    for galaxies in (galaxies, galaxies_basis):
        _assert_sparse_fit_matches_dense(
            dataset=interferometer_7, dataset_sparse=dataset_sparse, galaxies=galaxies
        )

        fit_sparse = ag.FitInterferometer(dataset=dataset_sparse, galaxies=galaxies)

        assert fit_sparse.inversion.dataset.sparse_dirty_image is not None
