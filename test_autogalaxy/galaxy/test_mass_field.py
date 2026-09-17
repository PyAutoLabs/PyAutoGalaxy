import numpy as np
import pytest

from autonerves.dictable import from_dict

import autofit as af
import autogalaxy as ag

from autogalaxy import exc


def test__construction__external_shear__stored_as_named_attribute():
    shear = ag.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)

    field = ag.MassField(redshift=0.5, shear=shear)

    assert field.redshift == 0.5
    assert field.shear is shear


def test__construction__mass_sheet__stored_as_named_attribute():
    mass_sheet = ag.mp.MassSheet(kappa=0.1)

    field = ag.MassField(redshift=1.0, mass_sheet=mass_sheet)

    assert field.redshift == 1.0
    assert field.mass_sheet is mass_sheet


def test__construction__external_potential__stored_as_named_attribute():
    potential = ag.mp.ExternalPotential(gamma_1=0.05, gamma_2=0.05, tau_1=0.01)

    field = ag.MassField(redshift=0.5, potential=potential)

    assert field.potential is potential


def test__construction__all_three_components__all_accessible_by_name():
    shear = ag.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)
    mass_sheet = ag.mp.MassSheet(kappa=0.1)
    potential = ag.mp.ExternalPotential(delta_1=0.01, delta_2=0.02)

    field = ag.MassField(
        redshift=0.5, shear=shear, mass_sheet=mass_sheet, potential=potential
    )

    assert field.shear is shear
    assert field.mass_sheet is mass_sheet
    assert field.potential is potential


def test__construction__light_profile__raises_exception_naming_the_component():
    with pytest.raises(exc.GalaxyException) as e:
        ag.MassField(redshift=0.5, bulge=ag.lp.Sersic())

    assert "'bulge'" in str(e.value)


def test__construction__pixelization__raises_exception_naming_the_component():
    pixelization = ag.Pixelization(
        mesh=ag.mesh.Delaunay(pixels=9),
        regularization=ag.reg.Constant(),
    )

    with pytest.raises(exc.GalaxyException) as e:
        ag.MassField(redshift=0.5, pixelization=pixelization)

    assert "'pixelization'" in str(e.value)


def test__construction__float__raises_exception():
    with pytest.raises(exc.GalaxyException):
        ag.MassField(redshift=0.5, shear=1.0)


def test__construction__list__raises_exception():
    with pytest.raises(exc.GalaxyException):
        ag.MassField(
            redshift=0.5, shear=[ag.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)]
        )


def test__is_not_a_galaxy__but_has_the_mass_profile_interface(mp_0):
    field = ag.MassField(redshift=0.5, shear=mp_0)

    assert isinstance(field, ag.Galaxy) is False
    assert isinstance(field, ag.MassField) is True


def test__has__light_profile_false__mass_profile_true():
    field = ag.MassField(
        redshift=0.5,
        shear=ag.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
        mass_sheet=ag.mp.MassSheet(kappa=0.1),
    )

    assert field.has(cls=ag.LightProfile) is False
    assert field.has(cls=ag.mp.MassProfile) is True
    assert field.has(cls=ag.mp.ExternalShear) is True


def test__cls_list_from__returns_every_mass_profile():
    shear = ag.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)
    mass_sheet = ag.mp.MassSheet(kappa=0.1)

    field = ag.MassField(redshift=0.5, shear=shear, mass_sheet=mass_sheet)

    assert len(field.cls_list_from(cls=ag.mp.MassProfile)) == 2
    assert field.cls_list_from(cls=ag.mp.ExternalShear) == [shear]
    assert field.cls_list_from(cls=ag.LightProfile) == []


def test__deflections_convergence_potential__match_the_equivalent_galaxy(grid_2d_7x7):
    shear = ag.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)
    mass_sheet = ag.mp.MassSheet(kappa=0.1)

    field = ag.MassField(redshift=0.5, shear=shear, mass_sheet=mass_sheet)
    galaxy = ag.Galaxy(redshift=0.5, shear=shear, mass_sheet=mass_sheet)

    np.testing.assert_array_equal(
        np.asarray(field.deflections_yx_2d_from(grid=grid_2d_7x7)),
        np.asarray(galaxy.deflections_yx_2d_from(grid=grid_2d_7x7)),
    )
    np.testing.assert_array_equal(
        np.asarray(field.convergence_2d_from(grid=grid_2d_7x7)),
        np.asarray(galaxy.convergence_2d_from(grid=grid_2d_7x7)),
    )
    np.testing.assert_array_equal(
        np.asarray(field.potential_2d_from(grid=grid_2d_7x7)),
        np.asarray(galaxy.potential_2d_from(grid=grid_2d_7x7)),
    )


def test__no_components__deflections_convergence_potential_are_zeros(grid_2d_7x7):
    field = ag.MassField(redshift=0.5)

    deflections = np.asarray(field.deflections_yx_2d_from(grid=grid_2d_7x7))
    convergence = np.asarray(field.convergence_2d_from(grid=grid_2d_7x7))
    potential = np.asarray(field.potential_2d_from(grid=grid_2d_7x7))

    assert deflections.shape == (grid_2d_7x7.shape[0], 2)
    assert convergence.shape == (grid_2d_7x7.shape[0],)
    assert potential.shape == (grid_2d_7x7.shape[0],)

    assert (deflections == 0.0).all()
    assert (convergence == 0.0).all()
    assert (potential == 0.0).all()


def test__image_2d_from__is_all_zeros_on_the_slim_grid(grid_2d_7x7):
    field = ag.MassField(
        redshift=0.5, shear=ag.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)
    )

    image = field.image_2d_from(grid=grid_2d_7x7)

    assert np.asarray(image).shape == (grid_2d_7x7.shape[0],)
    assert (np.asarray(image) == 0.0).all()


def test__image_2d_list_from__is_empty(grid_2d_7x7):
    field = ag.MassField(
        redshift=0.5, shear=ag.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)
    )

    assert field.image_2d_list_from(grid=grid_2d_7x7) == []
    assert field.image_2d_list_unbinned_from(grid=grid_2d_7x7) == []


def test__image_2d_unbinned_from__is_all_zeros_on_the_over_sampled_grid(grid_2d_7x7):
    field = ag.MassField(
        redshift=0.5, shear=ag.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)
    )

    image = field.image_2d_unbinned_from(grid=grid_2d_7x7)

    assert np.asarray(image).shape == (grid_2d_7x7.over_sampled.shape[0],)
    assert (np.asarray(image) == 0.0).all()


def test__galaxies_with_a_field__mass_quantities_are_the_sum(grid_2d_7x7):
    galaxy = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.Sersic(intensity=1.0),
        mass=ag.mp.Isothermal(einstein_radius=1.0),
    )
    field = ag.MassField(
        redshift=0.5, shear=ag.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)
    )

    galaxies = ag.Galaxies([galaxy, field])

    assert np.asarray(
        galaxies.deflections_yx_2d_from(grid=grid_2d_7x7)
    ) == pytest.approx(
        np.asarray(galaxy.deflections_yx_2d_from(grid=grid_2d_7x7))
        + np.asarray(field.deflections_yx_2d_from(grid=grid_2d_7x7)),
        1.0e-8,
    )
    assert np.asarray(galaxies.convergence_2d_from(grid=grid_2d_7x7)) == pytest.approx(
        np.asarray(galaxy.convergence_2d_from(grid=grid_2d_7x7))
        + np.asarray(field.convergence_2d_from(grid=grid_2d_7x7)),
        1.0e-8,
    )


def test__galaxies_with_a_field__image_is_the_galaxy_alone(grid_2d_7x7):
    galaxy = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))
    field = ag.MassField(
        redshift=0.5, shear=ag.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)
    )

    galaxies = ag.Galaxies([galaxy, field])

    np.testing.assert_array_equal(
        np.asarray(galaxies.image_2d_from(grid=grid_2d_7x7)),
        np.asarray(galaxy.image_2d_from(grid=grid_2d_7x7)),
    )


def test__galaxies_with_a_field__has_and_cls_list_span_both_members(grid_2d_7x7):
    shear = ag.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)

    galaxy = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.Sersic(intensity=1.0),
        mass=ag.mp.Isothermal(einstein_radius=1.0),
    )
    field = ag.MassField(redshift=0.5, shear=shear)

    galaxies = ag.Galaxies([galaxy, field])

    assert galaxies.has(cls=ag.LightProfile) is True
    assert galaxies.has(cls=ag.mp.ExternalShear) is True
    assert shear in galaxies.cls_list_from(cls=ag.mp.MassProfile)
    assert galaxies.galaxies_with_cls_list_from(cls=ag.mp.ExternalShear) == [field]
    assert galaxies.perform_inversion is False


def test__galaxies_with_a_field__galaxy_image_2d_dict__field_entry_is_zeros(
    grid_2d_7x7,
):
    galaxy = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))
    field = ag.MassField(
        redshift=0.5, shear=ag.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05)
    )

    galaxies = ag.Galaxies([galaxy, field])

    galaxy_image_2d_dict = galaxies.galaxy_image_2d_dict_from(grid=grid_2d_7x7)

    assert (np.asarray(galaxy_image_2d_dict[field]) == 0.0).all()
    assert (np.asarray(galaxy_image_2d_dict[galaxy]) != 0.0).any()


def test__dict_round_trip__rebuilds_an_equal_mass_field():
    field = ag.MassField(
        redshift=0.5,
        shear=ag.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
        mass_sheet=ag.mp.MassSheet(kappa=0.1),
    )

    field_from_dict = from_dict(field.dict())

    assert type(field_from_dict) is ag.MassField
    assert field_from_dict == field
    assert field_from_dict.redshift == 0.5
    assert field_from_dict.shear.gamma_1 == 0.05
    assert field_from_dict.mass_sheet.kappa == 0.1


def test__repr__names_the_redshift_and_every_component():
    field = ag.MassField(
        redshift=0.5,
        shear=ag.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
        mass_sheet=ag.mp.MassSheet(kappa=0.1),
    )

    assert repr(field) == "MassField(redshift=0.5, shear, mass_sheet)"
    assert repr(ag.MassField(redshift=0.5)) == "MassField(redshift=0.5)"


def test__galaxy_has_resolves_from_model_object__not_from_the_mixin():
    # `MassProfileAggregate` must not define `has`: `Galaxy` and `MassField` both take it from
    # `af.ModelObject`, which checks `self.__dict__`. `OperateImage.has` raises NotImplementedError
    # and is only ever shadowed, never reached.
    assert ag.Galaxy.has is af.ModelObject.has
    assert ag.MassField.has is af.ModelObject.has

    mro_names = [cls.__name__ for cls in ag.Galaxy.__mro__]

    assert mro_names.index("ModelObject") < mro_names.index("MassProfileAggregate")


def test__galaxy_identifier_pin__unchanged_by_mass_field_refactor():
    # Captured on `main` at commit 67f6ca6, before the `MassProfileAggregate` extraction, for exactly
    # this model. It exists to prove the extraction moved nothing that PyAutoFit hashes into a model
    # identifier: if this fails, `Galaxy` changed in a way PyAutoFit sees and the change must be undone
    # rather than the expected value updated.
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                ag.Galaxy,
                redshift=0.5,
                mass=af.Model(ag.mp.Isothermal),
                shear=af.Model(ag.mp.ExternalShear),
            )
        )
    )

    assert model.identifier == "7b251058cdae562470abab6203bacf72"
