"""
CSV reader/writer for galaxy populations.

A *galaxy population table* describes a set of galaxies via their on-sky centres, an
observationally measured property (typically luminosity), and an optional redshift per
galaxy. Workflows that fit many companion galaxies via a shared scaling relation
(``einstein_radius = scaling_factor * luminosity ** scaling_exponent``) need to read the
centres + luminosities for every galaxy from a single file rather than maintaining parallel
lists hardcoded in the modeling script.

This module provides the typed schema layer for that file format. The expected CSV columns
are:

    y, x, luminosity, redshift?, <property>...

The ``redshift`` column is optional. Any further numeric columns (e.g. ``ellipticity``,
``angle_pos``, ``mag`` for a Lenstool-style member catalogue) are loaded into
``GalaxyTable.properties`` keyed by column name — nothing is silently dropped. Row order
is preserved on read and on write.

The actual CSV I/O is delegated to :mod:`autoconf.csvable`; this module only owns the
column-name conventions and the typed return value.

The mirror schema for point-source datasets lives in :mod:`autolens.point.dataset` (see its
``output_to_csv`` / ``list_from_csv`` functions). The two formats deliberately do not share
infrastructure — the column conventions differ, and coupling them would be premature.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from autoconf import csvable

from autoarray.structures.grids.irregular_2d import Grid2DIrregular


@dataclass
class GalaxyTable:
    """
    A typed view onto a galaxy-population CSV.

    Parameters
    ----------
    centres
        ``Grid2DIrregular`` of (y, x) coordinates, one per galaxy.
    luminosities
        Per-galaxy luminosities, in the same order as ``centres``.
    redshifts
        Per-galaxy redshifts in the same order, or ``None`` if the input did not carry a
        ``redshift`` column.
    properties
        Any additional numeric columns, keyed by column name (e.g.
        ``properties["ellipticity"]``), each a per-galaxy list in row order. Empty dict
        when the CSV has no extra columns.
    """

    centres: Grid2DIrregular
    luminosities: List[float]
    redshifts: Optional[List[float]] = field(default=None)
    properties: Dict[str, List[float]] = field(default_factory=dict)


def galaxy_table_from_csv(file_path: Union[str, Path]) -> GalaxyTable:
    """
    Load a galaxy population from a CSV with columns ``y, x, luminosity, redshift?``.

    The ``redshift`` column is optional. If every row in the file populates it, the values
    are loaded into ``GalaxyTable.redshifts``; if the column is absent or every row leaves
    it blank, ``GalaxyTable.redshifts`` is ``None``. Partial population (some rows have a
    redshift, others do not) is rejected with ``ValueError`` — the partial-population
    convention mirrors :func:`autolens.point.dataset.list_from_csv`.

    Additional columns are loaded into ``GalaxyTable.properties`` keyed by column name —
    numeric columns as per-galaxy floats, non-numeric ones (names, notes) as strings.
    Nothing is silently dropped. Row order is preserved.

    Parameters
    ----------
    file_path
        Path to the CSV file. An empty CSV (no header line) and a header-only CSV both
        return an empty population.
    """
    rows = csvable.list_from_csv(file_path)

    if not rows:
        return GalaxyTable(centres=Grid2DIrregular([]), luminosities=[])

    centres = [(float(r["y"]), float(r["x"])) for r in rows]
    luminosities = [float(r["luminosity"]) for r in rows]

    populated = [r.get("redshift") not in ("", None) for r in rows]

    if all(populated):
        redshifts: Optional[List[float]] = [float(r["redshift"]) for r in rows]
    elif any(populated):
        raise ValueError(
            "galaxy_table CSV has partially populated 'redshift' column; every row "
            "must populate it or every row must leave it blank."
        )
    else:
        redshifts = None

    reserved = {"y", "x", "luminosity", "redshift"}
    properties: Dict[str, List] = {}
    for column in rows[0]:
        if column in reserved:
            continue
        try:
            properties[column] = [float(r[column]) for r in rows]
        except (TypeError, ValueError):
            # Non-numeric catalogue columns (names, notes, flags) ride along as strings.
            properties[column] = [r[column] for r in rows]

    return GalaxyTable(
        centres=Grid2DIrregular(centres),
        luminosities=luminosities,
        redshifts=redshifts,
        properties=properties,
    )


def galaxy_table_to_csv(
    centres: Sequence[Tuple[float, float]],
    luminosities: Sequence[float],
    file_path: Union[str, Path],
    redshifts: Optional[Sequence[float]] = None,
    properties: Optional[Dict[str, Sequence[float]]] = None,
) -> None:
    """
    Write a galaxy population to ``file_path`` as a CSV with columns
    ``y, x, luminosity, redshift?``.

    The ``redshift`` column is only emitted when ``redshifts`` is not None. ``centres``
    and ``luminosities`` (and ``redshifts`` when provided) must all have the same length;
    a ``ValueError`` is raised otherwise.

    Parameters
    ----------
    centres
        Sequence of (y, x) coordinates.
    luminosities
        Per-galaxy luminosities.
    file_path
        Destination CSV path. Parent directories are created if missing.
    redshifts
        Optional per-galaxy redshifts.
    properties
        Optional extra per-galaxy numeric columns, keyed by column name (each the same
        length as ``centres``) — e.g. ``{"ellipticity": [...], "angle_pos": [...],
        "mag": [...]}`` for a member catalogue carrying shape + magnitude.
    """
    if len(centres) != len(luminosities):
        raise ValueError(
            f"centres ({len(centres)}) and luminosities ({len(luminosities)}) must "
            f"have matching length."
        )
    if redshifts is not None and len(redshifts) != len(centres):
        raise ValueError(
            f"redshifts ({len(redshifts)}) must match centres ({len(centres)}) length "
            f"when provided."
        )

    if properties is not None:
        for name, values in properties.items():
            if len(values) != len(centres):
                raise ValueError(
                    f"properties['{name}'] ({len(values)}) must match centres "
                    f"({len(centres)}) length."
                )

    headers = ["y", "x", "luminosity"]
    if redshifts is not None:
        headers.append("redshift")
    if properties is not None:
        headers.extend(properties.keys())

    rows = []
    for i, (yx, lum) in enumerate(zip(centres, luminosities)):
        row = {
            "y": float(yx[0]),
            "x": float(yx[1]),
            "luminosity": float(lum),
        }
        if redshifts is not None:
            row["redshift"] = float(redshifts[i])
        if properties is not None:
            for name, values in properties.items():
                row[name] = float(values[i])
        rows.append(row)

    csvable.output_to_csv(rows, file_path, headers=headers)
