import autofit as af
import autogalaxy as ag
import pytest

from autogalaxy.analysis import chaining_util


def test_mass_and_fields_from_carries_flat_field_with_updated_mass():
    mass = af.Model(ag.mp.PowerLaw)
    mass_result = af.Model(ag.mp.Isothermal)
    field_result = af.Model(
        ag.MassField, redshift=0.5, shear=af.Model(ag.mp.ExternalShear)
    )

    chained_mass, chained_field = chaining_util.mass_and_fields_from(
        mass=mass,
        mass_result=mass_result,
        fields_result=field_result,
    )

    model = af.Collection(
        galaxies=af.Collection(lens=af.Model(ag.Galaxy, redshift=0.5, mass=chained_mass)),
        fields=chained_field,
    )

    assert chained_mass is mass
    assert chained_field is field_result
    assert ("fields", "shear", "gamma_1") in model.unique_prior_paths


def test_mass_and_fields_from_rejects_missing_field():
    with pytest.raises(ValueError, match="fields_result"):
        chaining_util.mass_and_fields_from(
            mass=af.Model(ag.mp.PowerLaw),
            mass_result=af.Model(ag.mp.Isothermal),
            fields_result=None,
        )
