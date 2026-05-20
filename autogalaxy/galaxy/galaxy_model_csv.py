"""
Named-galaxy CSV reader/writer for the PyAuto ecosystem.

Where ``galaxy_table.py`` covers a flat ``y, x, luminosity`` schema for the
scaling-relation tier, this module covers the *full model* — each row carries a
galaxy name, an attribute name (e.g. ``mass``, ``bulge``, ``point_0``), a
``profile_class`` string, and the constructor parameters of that profile.
Multiple family-level CSVs (mass / light / point) can be joined on the
``galaxy`` column to compose a full ``Galaxy`` (or ``af.Model[Galaxy]``).

The schema is sparse: a single CSV can carry rows of different profile classes
with disjoint constructor parameters; cells for parameters the row's class
does not use are blank.

Profile-class dispatch goes through three family namespaces inside
:mod:`autogalaxy`:

  - ``mass``  → :mod:`autogalaxy.profiles.mass`
  - ``light`` → :mod:`autogalaxy.profiles.light.standard`
  - ``point`` → :mod:`autogalaxy.profiles.point_sources`

The module stays inside :mod:`autogalaxy`; PyAutoLens re-exports the public
helpers under ``al.*`` so workspace scripts can use the canonical namespace.
"""
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from autoconf import csvable

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.profiles import mass as _mp_module
from autogalaxy.profiles.light import standard as _lp_module
from autogalaxy.profiles import point_sources as _ps_module


_FAMILY_NAMESPACES = {
    "mass": _mp_module,
    "light": _lp_module,
    "point": _ps_module,
}


_TUPLE_PARAM_COL_NAMES = {
    "centre": ("y", "x"),
}


_FIXED_HEADERS = ("galaxy", "attr_name", "profile_class")
_REDSHIFT_HEADER = "redshift"


@dataclass
class GalaxyModelRow:
    """
    A single CSV row resolved to its concrete profile class + constructor kwargs.

    ``params`` only contains the columns the row's ``profile_class`` actually
    consumes; blank cells are dropped so the profile class falls back to its
    own default values.
    """

    galaxy: str
    attr_name: str
    profile_class: type
    params: Dict[str, Any] = field(default_factory=dict)
    redshift: Optional[float] = None


@dataclass
class GalaxyModelTable:
    """
    A typed view onto one family CSV (mass / light / point).
    """

    rows: List[GalaxyModelRow]
    family: str


def _profile_init_params(cls):
    """Yield ``(name, default)`` for each constructor parameter of ``cls`` except ``self``."""
    sig = inspect.signature(cls)
    for name, parameter in sig.parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        default = (
            parameter.default
            if parameter.default is not inspect.Parameter.empty
            else None
        )
        yield name, default


def _is_tuple_param(default) -> bool:
    return isinstance(default, tuple)


def _tuple_col_names(param_name: str):
    if param_name in _TUPLE_PARAM_COL_NAMES:
        return _TUPLE_PARAM_COL_NAMES[param_name]
    return (f"{param_name}_0", f"{param_name}_1")


def _coerce_scalar(value):
    """Coerce a CSV cell string to float when possible, else return as-is."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def galaxy_models_to_csv(
    profiles_by_galaxy: Dict[str, Dict[str, Any]],
    file_path: Union[str, Path],
    family: str,
    redshifts: Optional[Dict[str, float]] = None,
) -> None:
    """
    Write a family of profiles to ``file_path`` as a CSV.

    Parameters
    ----------
    profiles_by_galaxy
        Mapping ``{galaxy_name: {attr_name: profile_instance}}``. Each
        ``profile_instance`` is a concrete ``LightProfile`` / ``MassProfile`` /
        ``PointSource``; its ``__init__`` constructor arguments are introspected
        via :func:`inspect.signature` and emitted as columns.
    file_path
        Destination CSV path. Parent directories are created if missing.
    family
        One of ``"mass"``, ``"light"``, ``"point"`` — used as a label only;
        no validation against the class namespaces happens on write.
    redshifts
        Optional ``{galaxy_name: redshift}`` mapping. When provided, every row
        for that galaxy carries the redshift in the ``redshift`` column.

    Notes
    -----
    Sparse columns are supported: if galaxies use different profile classes
    with non-overlapping constructor parameters, the header line is the union
    of all parameter columns seen, with blank cells where a row's class does
    not use a column.
    """
    if family not in _FAMILY_NAMESPACES:
        raise ValueError(
            f"family must be one of {sorted(_FAMILY_NAMESPACES)}, got '{family}'."
        )

    namespace = _FAMILY_NAMESPACES[family]

    rows: List[dict] = []
    for galaxy_name, attr_dict in profiles_by_galaxy.items():
        for attr_name, profile in attr_dict.items():
            cls = type(profile)
            if getattr(namespace, cls.__name__, None) is not cls:
                raise ValueError(
                    f"Profile {galaxy_name!r}.{attr_name!r} has class "
                    f"{cls.__name__!r} which is not exposed in family "
                    f"namespace '{namespace.__name__}' (family '{family}'). "
                    f"Check that the profile belongs to the family declared "
                    f"on this CSV — mass profiles go in 'mass', light profiles "
                    f"in 'light', point sources in 'point'."
                )
            row: Dict[str, Any] = {
                "galaxy": galaxy_name,
                "attr_name": attr_name,
                "profile_class": cls.__name__,
            }
            for param_name, default in _profile_init_params(cls):
                value = getattr(profile, param_name, None)
                if value is None:
                    continue
                if _is_tuple_param(default):
                    col0, col1 = _tuple_col_names(param_name)
                    row[col0] = float(value[0])
                    row[col1] = float(value[1])
                else:
                    row[param_name] = float(value) if isinstance(value, (int, float)) else value
            if redshifts is not None and galaxy_name in redshifts:
                row[_REDSHIFT_HEADER] = float(redshifts[galaxy_name])
            rows.append(row)

    headers = list(_FIXED_HEADERS)
    seen = set(headers)
    for row in rows:
        for key in row:
            if key not in seen and key != _REDSHIFT_HEADER:
                seen.add(key)
                headers.append(key)
    if any(_REDSHIFT_HEADER in r for r in rows):
        headers.append(_REDSHIFT_HEADER)

    csvable.output_to_csv(rows, file_path, headers=headers)


def galaxy_models_from_csv(
    file_path: Union[str, Path],
    family: str,
) -> GalaxyModelTable:
    """
    Read a family CSV produced by :func:`galaxy_models_to_csv`.

    Parameters
    ----------
    file_path
        Path to the CSV file. An empty CSV (no header line) and a header-only
        CSV both return an empty :class:`GalaxyModelTable`.
    family
        One of ``"mass"``, ``"light"``, ``"point"``. Selects the namespace
        the ``profile_class`` column is looked up against.

    Raises
    ------
    ValueError
        If ``family`` is not one of the supported names, or if a row's
        ``profile_class`` does not exist in the family namespace.
    """
    if family not in _FAMILY_NAMESPACES:
        raise ValueError(
            f"family must be one of {sorted(_FAMILY_NAMESPACES)}, got '{family}'."
        )

    namespace = _FAMILY_NAMESPACES[family]
    raw_rows = csvable.list_from_csv(file_path)

    parsed: List[GalaxyModelRow] = []
    for raw in raw_rows:
        cls_name = raw["profile_class"]
        cls = getattr(namespace, cls_name, None)
        if cls is None:
            raise ValueError(
                f"profile_class '{cls_name}' not found in namespace "
                f"'{namespace.__name__}' (family '{family}'). Verify the class "
                f"name is spelled correctly and exposed in that namespace."
            )

        params: Dict[str, Any] = {}
        for param_name, default in _profile_init_params(cls):
            if _is_tuple_param(default):
                col0, col1 = _tuple_col_names(param_name)
                v0, v1 = raw.get(col0), raw.get(col1)
                if v0 not in ("", None) and v1 not in ("", None):
                    params[param_name] = (float(v0), float(v1))
            else:
                v = raw.get(param_name)
                if v not in ("", None):
                    params[param_name] = _coerce_scalar(v)

        rz_raw = raw.get(_REDSHIFT_HEADER)
        redshift = float(rz_raw) if rz_raw not in ("", None) else None

        parsed.append(
            GalaxyModelRow(
                galaxy=raw["galaxy"],
                attr_name=raw["attr_name"],
                profile_class=cls,
                params=params,
                redshift=redshift,
            )
        )

    return GalaxyModelTable(rows=parsed, family=family)


def _group_rows_by_galaxy(*tables: GalaxyModelTable) -> Dict[str, List[GalaxyModelRow]]:
    by_galaxy: Dict[str, List[GalaxyModelRow]] = {}
    for table in tables:
        for row in table.rows:
            by_galaxy.setdefault(row.galaxy, []).append(row)
    return by_galaxy


def _resolve_redshift(galaxy_name: str, rows: List[GalaxyModelRow]) -> Optional[float]:
    redshifts = {row.redshift for row in rows if row.redshift is not None}
    if len(redshifts) > 1:
        raise ValueError(
            f"Galaxy '{galaxy_name}' has inconsistent redshifts across CSV rows: "
            f"{sorted(redshifts)}. All rows for a given galaxy must share the same "
            f"redshift (or all leave it blank)."
        )
    return next(iter(redshifts)) if redshifts else None


def galaxies_from_csv_tables(*tables: GalaxyModelTable) -> Dict[str, Galaxy]:
    """
    Combine one or more family tables into concrete ``Galaxy`` instances.

    Rows are joined on the ``galaxy`` column; each galaxy's profile attributes
    are attached under their ``attr_name``. Per-galaxy redshifts must be
    consistent across every CSV row for that galaxy (or all blank).

    Returns
    -------
    Mapping ``{galaxy_name: Galaxy}`` preserving the first-seen order across
    the input tables.
    """
    by_galaxy = _group_rows_by_galaxy(*tables)

    galaxies: Dict[str, Galaxy] = {}
    for galaxy_name, rows in by_galaxy.items():
        redshift = _resolve_redshift(galaxy_name, rows)
        attrs = {row.attr_name: row.profile_class(**row.params) for row in rows}
        galaxies[galaxy_name] = Galaxy(redshift=redshift, **attrs)

    return galaxies


def galaxy_af_models_from_csv_tables(*tables: GalaxyModelTable) -> Dict[str, Any]:
    """
    Combine one or more family tables into ``af.Model[Galaxy]`` instances.

    Each profile becomes an ``af.Model(profile_class, **params)`` with the
    CSV's concrete values as fixed defaults; the returned ``af.Model[Galaxy]``
    instances are ready to mutate (set priors on selected params) and
    assemble into an ``af.Collection``.

    Returns
    -------
    Mapping ``{galaxy_name: af.Model[Galaxy]}`` preserving the first-seen
    order across the input tables.
    """
    import autofit as af

    by_galaxy = _group_rows_by_galaxy(*tables)

    galaxy_models: Dict[str, Any] = {}
    for galaxy_name, rows in by_galaxy.items():
        redshift = _resolve_redshift(galaxy_name, rows)
        attrs: Dict[str, Any] = {}
        for row in rows:
            model = af.Model(row.profile_class)
            for name, value in row.params.items():
                if isinstance(value, tuple):
                    for i, component in enumerate(value):
                        setattr(model, f"{name}_{i}", component)
                else:
                    setattr(model, name, value)
            attrs[row.attr_name] = model
        galaxy_models[galaxy_name] = af.Model(Galaxy, redshift=redshift, **attrs)

    return galaxy_models
