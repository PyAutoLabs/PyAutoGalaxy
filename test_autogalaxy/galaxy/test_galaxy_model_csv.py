import csv

import pytest

import autofit as af
import autogalaxy as ag


def _read_raw_rows(file_path):
    with open(file_path, newline="") as f:
        return list(csv.DictReader(f))


def test__mass_single_class__round_trip(tmp_path):
    file_path = tmp_path / "main_lens_mass.csv"

    profiles_by_galaxy = {
        "lens_0": {
            "mass": ag.mp.dPIEMassSph(centre=(0.0, 0.0), ra=8.0, rs=20.0, b0=3.0)
        },
        "lens_1": {
            "mass": ag.mp.dPIEMassSph(centre=(10.0, 8.0), ra=5.0, rs=12.0, b0=1.2)
        },
    }

    ag.galaxy_models_to_csv(
        profiles_by_galaxy=profiles_by_galaxy,
        file_path=file_path,
        family="mass",
        redshifts={"lens_0": 0.5, "lens_1": 0.5},
    )

    table = ag.galaxy_models_from_csv(file_path, family="mass")

    assert table.family == "mass"
    assert [r.galaxy for r in table.rows] == ["lens_0", "lens_1"]
    assert [r.attr_name for r in table.rows] == ["mass", "mass"]
    assert all(r.profile_class is ag.mp.dPIEMassSph for r in table.rows)
    assert table.rows[0].params == {
        "centre": (0.0, 0.0),
        "ra": 8.0,
        "rs": 20.0,
        "b0": 3.0,
    }
    assert table.rows[0].redshift == 0.5


def test__mass_sparse_columns__dpie_plus_nfw__round_trip(tmp_path):
    file_path = tmp_path / "main_lens_mass.csv"

    profiles_by_galaxy = {
        "lens_0": {
            "mass": ag.mp.dPIEMassSph(centre=(0.0, 0.0), ra=8.0, rs=20.0, b0=3.0)
        },
        "host_halo": {
            "dark": ag.mp.NFWMCRLudlowSph(
                centre=(0.0, 0.0),
                mass_at_200=1.99e15,
                redshift_object=0.5,
                redshift_source=2.0,
            )
        },
    }

    ag.galaxy_models_to_csv(
        profiles_by_galaxy=profiles_by_galaxy,
        file_path=file_path,
        family="mass",
        redshifts={"lens_0": 0.5, "host_halo": 0.5},
    )

    raw_rows = _read_raw_rows(file_path)
    assert set(raw_rows[0].keys()).issuperset(
        {
            "galaxy",
            "attr_name",
            "profile_class",
            "y",
            "x",
            "ra",
            "rs",
            "b0",
            "mass_at_200",
            "redshift_object",
            "redshift_source",
            "redshift",
        }
    )
    # dPIE row leaves NFW-only columns blank.
    assert raw_rows[0]["galaxy"] == "lens_0"
    assert raw_rows[0]["mass_at_200"] == ""
    assert raw_rows[0]["redshift_object"] == ""
    # NFW row leaves dPIE-only columns blank.
    assert raw_rows[1]["galaxy"] == "host_halo"
    assert raw_rows[1]["ra"] == ""
    assert raw_rows[1]["b0"] == ""

    table = ag.galaxy_models_from_csv(file_path, family="mass")
    assert table.rows[0].profile_class is ag.mp.dPIEMassSph
    assert table.rows[0].params == {
        "centre": (0.0, 0.0),
        "ra": 8.0,
        "rs": 20.0,
        "b0": 3.0,
    }
    assert table.rows[1].profile_class is ag.mp.NFWMCRLudlowSph
    assert table.rows[1].params == {
        "centre": (0.0, 0.0),
        "mass_at_200": 1.99e15,
        "redshift_object": 0.5,
        "redshift_source": 2.0,
    }


def test__light_family__sersic_sph_plus_sersic_core__round_trip(tmp_path):
    file_path = tmp_path / "light.csv"

    profiles_by_galaxy = {
        "lens_0": {
            "bulge": ag.lp.SersicSph(
                centre=(0.0, 0.0), intensity=1.5, effective_radius=3.0, sersic_index=4.0
            )
        },
        "source_0": {
            "bulge": ag.lp.SersicCore(
                centre=(0.3, 0.5),
                ell_comps=(0.1, -0.2),
                intensity=2.0,
                effective_radius=0.3,
                sersic_index=1.0,
            )
        },
    }

    ag.galaxy_models_to_csv(
        profiles_by_galaxy=profiles_by_galaxy,
        file_path=file_path,
        family="light",
    )

    table = ag.galaxy_models_from_csv(file_path, family="light")
    assert table.rows[0].profile_class is ag.lp.SersicSph
    assert table.rows[0].params["centre"] == (0.0, 0.0)
    assert "ell_comps" not in table.rows[0].params  # SersicSph has no ell_comps
    assert table.rows[1].profile_class is ag.lp.SersicCore
    assert table.rows[1].params["ell_comps"] == (0.1, -0.2)


def test__point_family__round_trip(tmp_path):
    file_path = tmp_path / "source_point.csv"

    profiles_by_galaxy = {
        "source_0": {"point_0": ag.ps.Point(centre=(0.3, 0.5))},
        "source_1": {"point_1": ag.ps.Point(centre=(-0.8, 1.2))},
    }

    ag.galaxy_models_to_csv(
        profiles_by_galaxy=profiles_by_galaxy,
        file_path=file_path,
        family="point",
        redshifts={"source_0": 1.0, "source_1": 2.0},
    )

    table = ag.galaxy_models_from_csv(file_path, family="point")
    assert [r.attr_name for r in table.rows] == ["point_0", "point_1"]
    assert all(r.profile_class is ag.ps.Point for r in table.rows)
    assert table.rows[0].params["centre"] == (0.3, 0.5)
    assert table.rows[1].redshift == 2.0


def test__cross_family_join__builds_named_galaxies(tmp_path):
    mass_csv = tmp_path / "mass.csv"
    light_csv = tmp_path / "light.csv"
    point_csv = tmp_path / "point.csv"

    ag.galaxy_models_to_csv(
        profiles_by_galaxy={
            "lens_0": {
                "mass": ag.mp.dPIEMassSph(centre=(0.0, 0.0), ra=8.0, rs=20.0, b0=3.0)
            },
        },
        file_path=mass_csv,
        family="mass",
        redshifts={"lens_0": 0.5},
    )
    ag.galaxy_models_to_csv(
        profiles_by_galaxy={
            "lens_0": {
                "bulge": ag.lp.SersicSph(
                    centre=(0.0, 0.0),
                    intensity=1.5,
                    effective_radius=3.0,
                    sersic_index=4.0,
                )
            },
            "source_0": {
                "bulge": ag.lp.SersicCore(
                    centre=(0.3, 0.5),
                    ell_comps=(0.1, -0.2),
                    intensity=2.0,
                    effective_radius=0.3,
                    sersic_index=1.0,
                )
            },
        },
        file_path=light_csv,
        family="light",
        redshifts={"lens_0": 0.5, "source_0": 1.0},
    )
    ag.galaxy_models_to_csv(
        profiles_by_galaxy={
            "source_0": {"point_0": ag.ps.Point(centre=(0.3, 0.5))},
        },
        file_path=point_csv,
        family="point",
        redshifts={"source_0": 1.0},
    )

    galaxies = ag.galaxies_from_csv_tables(
        ag.galaxy_models_from_csv(mass_csv, family="mass"),
        ag.galaxy_models_from_csv(light_csv, family="light"),
        ag.galaxy_models_from_csv(point_csv, family="point"),
    )

    assert set(galaxies.keys()) == {"lens_0", "source_0"}
    assert galaxies["lens_0"].redshift == 0.5
    assert isinstance(galaxies["lens_0"].mass, ag.mp.dPIEMassSph)
    assert isinstance(galaxies["lens_0"].bulge, ag.lp.SersicSph)
    assert galaxies["source_0"].redshift == 1.0
    assert isinstance(galaxies["source_0"].bulge, ag.lp.SersicCore)
    assert isinstance(galaxies["source_0"].point_0, ag.ps.Point)


def test__af_models_round_trip(tmp_path):
    mass_csv = tmp_path / "mass.csv"

    ag.galaxy_models_to_csv(
        profiles_by_galaxy={
            "lens_0": {
                "mass": ag.mp.dPIEMassSph(centre=(0.0, 0.0), ra=8.0, rs=20.0, b0=3.0)
            },
        },
        file_path=mass_csv,
        family="mass",
        redshifts={"lens_0": 0.5},
    )

    galaxy_models = ag.galaxy_af_models_from_csv_tables(
        ag.galaxy_models_from_csv(mass_csv, family="mass"),
    )

    assert "lens_0" in galaxy_models
    galaxy_model = galaxy_models["lens_0"]
    assert isinstance(galaxy_model, af.Model)
    assert galaxy_model.cls is ag.Galaxy
    assert galaxy_model.redshift == 0.5
    assert galaxy_model.mass.cls is ag.mp.dPIEMassSph
    assert galaxy_model.mass.ra == 8.0
    assert galaxy_model.mass.rs == 20.0
    assert galaxy_model.mass.b0 == 3.0


def test__af_models__tuple_param_supports_prior_override(tmp_path):
    point_csv = tmp_path / "point.csv"

    ag.galaxy_models_to_csv(
        profiles_by_galaxy={"source_0": {"point_0": ag.ps.Point(centre=(0.3, 0.5))}},
        file_path=point_csv,
        family="point",
        redshifts={"source_0": 1.0},
    )

    galaxy_models = ag.galaxy_af_models_from_csv_tables(
        ag.galaxy_models_from_csv(point_csv, family="point"),
    )

    point_attr = galaxy_models["source_0"].point_0
    point_attr.centre_0 = af.GaussianPrior(mean=0.3, sigma=3.0)
    point_attr.centre_1 = af.GaussianPrior(mean=0.5, sigma=3.0)

    instance = galaxy_models["source_0"].instance_from_unit_vector([0.5, 0.5])
    assert isinstance(instance.point_0, ag.ps.Point)
    assert isinstance(instance.point_0.centre, tuple)
    assert len(instance.point_0.centre) == 2


def test__redshift_consistency_check__raises(tmp_path):
    mass_csv = tmp_path / "mass.csv"
    light_csv = tmp_path / "light.csv"

    ag.galaxy_models_to_csv(
        profiles_by_galaxy={
            "lens_0": {
                "mass": ag.mp.dPIEMassSph(centre=(0.0, 0.0), ra=8.0, rs=20.0, b0=3.0)
            }
        },
        file_path=mass_csv,
        family="mass",
        redshifts={"lens_0": 0.5},
    )
    ag.galaxy_models_to_csv(
        profiles_by_galaxy={
            "lens_0": {
                "bulge": ag.lp.SersicSph(
                    centre=(0.0, 0.0),
                    intensity=1.5,
                    effective_radius=3.0,
                    sersic_index=4.0,
                )
            }
        },
        file_path=light_csv,
        family="light",
        redshifts={"lens_0": 0.7},
    )

    with pytest.raises(ValueError, match="inconsistent redshifts"):
        ag.galaxies_from_csv_tables(
            ag.galaxy_models_from_csv(mass_csv, family="mass"),
            ag.galaxy_models_from_csv(light_csv, family="light"),
        )


def test__writer_rejects_profile_from_wrong_family(tmp_path):
    """Writing a SersicSph (light) under family='mass' must raise immediately."""
    file_path = tmp_path / "mass.csv"

    with pytest.raises(ValueError, match="not exposed in family namespace"):
        ag.galaxy_models_to_csv(
            profiles_by_galaxy={
                "lens_0": {
                    "bulge": ag.lp.SersicSph(
                        centre=(0.0, 0.0),
                        intensity=1.0,
                        effective_radius=1.0,
                        sersic_index=2.0,
                    )
                },
            },
            file_path=file_path,
            family="mass",
        )


def test__class_not_found_in_namespace__raises(tmp_path):
    file_path = tmp_path / "mass.csv"
    file_path.write_text(
        "galaxy,attr_name,profile_class,y,x\n" "lens_0,mass,NotARealClass,0.0,0.0\n"
    )

    with pytest.raises(ValueError, match="profile_class 'NotARealClass' not found"):
        ag.galaxy_models_from_csv(file_path, family="mass")


def test__light_variant_namespaces__qualified_class_names_round_trip(tmp_path):
    import autogalaxy as ag
    from autogalaxy.galaxy.galaxy_model_csv import (
        galaxy_models_to_csv,
        galaxy_models_from_csv,
        galaxies_from_csv_tables,
    )

    lp_linear = ag.lp_linear.Sersic(
        centre=(0.0, 0.0), ell_comps=(0.1, 0.0), effective_radius=0.8, sersic_index=2.5
    )
    lp_operated = ag.lp_operated.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=1.0, sigma=0.3
    )

    path = tmp_path / "light.csv"
    galaxy_models_to_csv(
        {"gal": {"bulge": lp_linear, "psf": lp_operated}}, path, family="light"
    )

    text = path.read_text()
    assert "linear.Sersic" in text
    assert "operated.Gaussian" in text

    galaxies = galaxies_from_csv_tables(galaxy_models_from_csv(path, family="light"))

    assert type(galaxies["gal"].bulge) is ag.lp_linear.Sersic
    assert type(galaxies["gal"].psf) is ag.lp_operated.Gaussian


def test__lenstool_parameterized_mass_row__round_trips(tmp_path):
    import autogalaxy as ag
    from autogalaxy.galaxy.galaxy_model_csv import (
        galaxy_models_to_csv,
        galaxy_models_from_csv,
        galaxies_from_csv_tables,
    )

    profile = ag.mp.dPIEMassLenstool(
        centre=(1.5, -3.0),
        ellipticity=0.68,
        angle_pos=8.97,
        sigma=987.3,
        r_core=18.96,
        r_cut=283.5,
        redshift_object=0.39,
        redshift_source=11.76,
    )

    path = tmp_path / "mass.csv"
    galaxy_models_to_csv(
        {"O1": {"mass": profile}}, path, family="mass", redshifts={"O1": 0.39}
    )
    galaxies = galaxies_from_csv_tables(galaxy_models_from_csv(path, family="mass"))

    assert galaxies["O1"].mass.b0 == pytest.approx(profile.b0, rel=1e-12)
    assert galaxies["O1"].redshift == 0.39


def test__duplicate_galaxy_attr_rows__raise(tmp_path):
    from autogalaxy.galaxy.galaxy_model_csv import (
        galaxy_models_from_csv,
        galaxies_from_csv_tables,
    )

    path = tmp_path / "dup.csv"
    path.write_text(
        "galaxy,attr_name,profile_class,y,x,einstein_radius\n"
        "lens,mass,IsothermalSph,0.0,0.0,1.0\n"
        "lens,mass,IsothermalSph,0.0,0.0,2.0\n"
    )

    with pytest.raises(ValueError, match="Duplicate CSV rows"):
        galaxies_from_csv_tables(galaxy_models_from_csv(path, family="mass"))


def test__unknown_column__raises_instead_of_silent_default(tmp_path):
    from autogalaxy.galaxy.galaxy_model_csv import galaxy_models_from_csv

    path = tmp_path / "typo.csv"
    path.write_text(
        "galaxy,attr_name,profile_class,y,x,einstien_radius\n"
        "lens,mass,IsothermalSph,0.0,0.0,1.5\n"
    )

    with pytest.raises(ValueError, match="einstien_radius"):
        galaxy_models_from_csv(path, family="mass")
