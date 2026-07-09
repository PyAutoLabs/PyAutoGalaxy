import csv

import pytest

import autogalaxy as ag
from autoarray.structures.grids.irregular_2d import Grid2DIrregular


def _read_raw_rows(file_path):
    with open(file_path, newline="") as f:
        return list(csv.DictReader(f))


def test__round_trip__no_redshift(tmp_path):
    centres = [(3.5, 2.5), (-4.4, -5.0), (1.0, 0.0)]
    luminosities = [0.9, 0.45, 1.2]
    file_path = tmp_path / "galaxies.csv"

    ag.galaxy_table_to_csv(centres, luminosities, file_path)

    table = ag.galaxy_table_from_csv(file_path)

    assert isinstance(table.centres, Grid2DIrregular)
    assert table.luminosities == luminosities
    assert table.redshifts is None
    assert [tuple(c) for c in table.centres.in_list] == centres


def test__round_trip__with_redshift(tmp_path):
    centres = [(3.5, 2.5), (-4.4, -5.0)]
    luminosities = [0.9, 0.45]
    redshifts = [0.5, 0.7]
    file_path = tmp_path / "galaxies.csv"

    ag.galaxy_table_to_csv(centres, luminosities, file_path, redshifts=redshifts)

    table = ag.galaxy_table_from_csv(file_path)

    assert table.luminosities == luminosities
    assert table.redshifts == redshifts
    assert [tuple(c) for c in table.centres.in_list] == centres


def test__missing_redshift_column__redshifts_none(tmp_path):
    file_path = tmp_path / "galaxies.csv"
    file_path.write_text("y,x,luminosity\n" "3.5,2.5,0.9\n" "-4.4,-5.0,0.45\n")

    table = ag.galaxy_table_from_csv(file_path)

    assert table.redshifts is None
    assert table.luminosities == [0.9, 0.45]
    assert [tuple(c) for c in table.centres.in_list] == [(3.5, 2.5), (-4.4, -5.0)]


def test__partial_redshift__raises(tmp_path):
    file_path = tmp_path / "galaxies.csv"
    file_path.write_text(
        "y,x,luminosity,redshift\n" "3.5,2.5,0.9,0.5\n" "-4.4,-5.0,0.45,\n"
    )

    with pytest.raises(ValueError, match="partially populated 'redshift'"):
        ag.galaxy_table_from_csv(file_path)


def test__extra_columns_loaded_as_properties(tmp_path):
    file_path = tmp_path / "galaxies.csv"
    file_path.write_text(
        "name,y,x,luminosity,notes\n"
        "g0,3.5,2.5,0.9,bright\n"
        "g1,-4.4,-5.0,0.45,faint\n"
    )

    table = ag.galaxy_table_from_csv(file_path)

    assert table.luminosities == [0.9, 0.45]
    assert table.redshifts is None
    assert table.properties["name"] == ["g0", "g1"]
    assert table.properties["notes"] == ["bright", "faint"]


def test__row_order_preserved(tmp_path):
    centres = [(0.5, 0.0), (-2.0, 3.0), (4.0, -1.0), (1.5, 1.5)]
    luminosities = [0.1, 0.2, 0.3, 0.4]
    file_path = tmp_path / "galaxies.csv"

    ag.galaxy_table_to_csv(centres, luminosities, file_path)

    table = ag.galaxy_table_from_csv(file_path)

    assert table.luminosities == luminosities
    assert [tuple(c) for c in table.centres.in_list] == centres


def test__empty_csv__empty_population(tmp_path):
    file_path = tmp_path / "galaxies.csv"
    file_path.write_text("")

    table = ag.galaxy_table_from_csv(file_path)

    assert list(table.centres.in_list) == []
    assert table.luminosities == []
    assert table.redshifts is None


def test__header_only_csv__empty_population(tmp_path):
    file_path = tmp_path / "galaxies.csv"
    file_path.write_text("y,x,luminosity\n")

    table = ag.galaxy_table_from_csv(file_path)

    assert list(table.centres.in_list) == []
    assert table.luminosities == []
    assert table.redshifts is None


def test__mismatched_lengths_to_csv__raises(tmp_path):
    file_path = tmp_path / "galaxies.csv"

    with pytest.raises(ValueError, match="matching length"):
        ag.galaxy_table_to_csv(
            centres=[(0.0, 0.0), (1.0, 1.0)],
            luminosities=[0.5],
            file_path=file_path,
        )

    with pytest.raises(ValueError, match="match centres"):
        ag.galaxy_table_to_csv(
            centres=[(0.0, 0.0), (1.0, 1.0)],
            luminosities=[0.5, 0.6],
            file_path=file_path,
            redshifts=[0.5],
        )


def test__to_csv__redshift_column_only_when_provided(tmp_path):
    file_path = tmp_path / "galaxies.csv"

    ag.galaxy_table_to_csv(
        centres=[(0.0, 0.0)],
        luminosities=[0.5],
        file_path=file_path,
    )

    rows = _read_raw_rows(file_path)
    assert "redshift" not in rows[0]

    ag.galaxy_table_to_csv(
        centres=[(0.0, 0.0)],
        luminosities=[0.5],
        file_path=file_path,
        redshifts=[0.7],
    )

    rows = _read_raw_rows(file_path)
    assert rows[0]["redshift"] == "0.7"


def test__extra_property_columns__loaded_and_round_tripped(tmp_path):
    from autogalaxy.galaxy.galaxy_table import (
        galaxy_table_from_csv,
        galaxy_table_to_csv,
    )

    path = tmp_path / "members.csv"
    galaxy_table_to_csv(
        centres=[(1.0, 2.0), (3.0, 4.0)],
        luminosities=[0.4, 0.2],
        file_path=path,
        properties={
            "ellipticity": [0.3, 0.1],
            "angle_pos": [45.0, -10.0],
            "mag": [19.5, 21.0],
        },
    )

    table = galaxy_table_from_csv(file_path=path)

    assert table.properties["ellipticity"] == [0.3, 0.1]
    assert table.properties["angle_pos"] == [45.0, -10.0]
    assert table.properties["mag"] == [19.5, 21.0]
    assert table.luminosities == [0.4, 0.2]


def test__legacy_three_column_schema__properties_empty(tmp_path):
    from autogalaxy.galaxy.galaxy_table import galaxy_table_from_csv

    path = tmp_path / "legacy.csv"
    path.write_text("y,x,luminosity\n1.0,2.0,0.4\n")

    table = galaxy_table_from_csv(file_path=path)

    assert table.properties == {}
