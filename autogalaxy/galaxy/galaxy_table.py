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

    y, x, luminosity, redshift?

The ``redshift`` column is optional. Extra columns are silently ignored. Row order is
preserved on read and on write.

The actual CSV I/O is delegated to :mod:`autoconf.csvable`; this module only owns the
column-name conventions and the typed return value.

The mirror schema for point-source datasets lives in :mod:`autolens.point.dataset` (see its
``output_to_csv`` / ``list_from_csv`` functions). The two formats deliberately do not share
infrastructure — the column conventions differ, and coupling them would be premature.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

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
    """

    centres: Grid2DIrregular
    luminosities: List[float]
    redshifts: Optional[List[float]] = field(default=None)


def galaxy_table_from_csv(file_path: Union[str, Path]) -> GalaxyTable:
    """
    Load a galaxy population from a CSV with columns ``y, x, luminosity, redshift?``.

    The ``redshift`` column is optional. If every row in the file populates it, the values
    are loaded into ``GalaxyTable.redshifts``; if the column is absent or every row leaves
    it blank, ``GalaxyTable.redshifts`` is ``None``. Partial population (some rows have a
    redshift, others do not) is rejected with ``ValueError`` — the partial-population
    convention mirrors :func:`autolens.point.dataset.list_from_csv`.

    Extra columns are silently ignored. Row order is preserved.

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

    return GalaxyTable(
        centres=Grid2DIrregular(centres),
        luminosities=luminosities,
        redshifts=redshifts,
    )


def galaxy_table_to_csv(
    centres: Sequence[Tuple[float, float]],
    luminosities: Sequence[float],
    file_path: Union[str, Path],
    redshifts: Optional[Sequence[float]] = None,
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

    headers = ["y", "x", "luminosity"]
    if redshifts is not None:
        headers.append("redshift")

    rows = []
    for i, (yx, lum) in enumerate(zip(centres, luminosities)):
        row = {
            "y": float(yx[0]),
            "x": float(yx[1]),
            "luminosity": float(lum),
        }
        if redshifts is not None:
            row["redshift"] = float(redshifts[i])
        rows.append(row)

    csvable.output_to_csv(rows, file_path, headers=headers)
