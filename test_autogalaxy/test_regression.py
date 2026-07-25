import pytest

from autogalaxy import Galaxy
from autogalaxy.abstract_fit import AbstractFitInversion
import autofit as af
import autoarray as aa
from autogalaxy.profiles.light.linear import LightProfileLinear, Sersic


class FitInversion(AbstractFitInversion):
    def __init__(self, model_obj, settings: aa.Settings, light_profiles):
        super().__init__(model_obj=model_obj, settings=settings)
        self.light_profiles = light_profiles

    @property
    def linear_light_profile_intensity_dict(self):
        return {light_profile: 1.0 for light_profile in self.light_profiles}


@pytest.fixture(name="light_profile")
def make_light_profile():
    return Sersic()


def test_no_modify_state(light_profile):
    model_obj = af.ModelInstance(
        {
            "galaxies": af.ModelInstance(
                {
                    "galaxy": Galaxy(
                        redshift=0.5,
                        light_profile=light_profile,
                    )
                }
            )
        }
    )

    fit_inversion = FitInversion(
        model_obj=model_obj,
        light_profiles=[light_profile],
        settings=aa.Settings(),
    )
    result = fit_inversion.model_obj_linear_light_profiles_to_light_profiles

    assert result.galaxies.galaxy.light_profile is not light_profile
    assert model_obj.galaxies.galaxy.light_profile is light_profile


@pytest.fixture(name="path")
def make_path():
    return "galaxies", 0, "light_profile"


@pytest.fixture(name="model_obj")
def make_model_obj(light_profile):
    return af.ModelInstance(
        {
            "galaxies": [
                Galaxy(
                    redshift=0.5,
                    light_profile=light_profile,
                )
            ]
        }
    )


def test_list_galaxies(model_obj, path):
    assert model_obj.path_instance_tuples_for_class((LightProfileLinear,))[0][0] == path


def test_object_from_path(model_obj, path, light_profile):
    assert model_obj.object_for_path(path) == light_profile
