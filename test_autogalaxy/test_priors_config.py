"""
Tests that guard the packaged `autogalaxy/config/priors` yaml files against drift.

`autonerves.JSONPriorConfig` turns every yaml file under `config/priors` into a
dotted path built from the file's location relative to `priors`, and looks a
prior up by asking whether `f"{cls.__module__}.{cls.__name__}"` ends with that
path. A yaml file in the wrong folder, a mistyped class key or a parameter that
is no longer an `__init__` argument is therefore silently dead: no error is
raised, the default prior is simply never used.

These tests walk the packaged yaml files directly (not via `conf.instance`,
which the test suite pushes to `test_autogalaxy/config`) and assert that every
key resolves to a real class and every parameter is a real `__init__` argument.
"""

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import yaml

import autofit as af
import autogalaxy as ag

PRIORS_PATH = Path(ag.__file__).parent / "config" / "priors"

CENSUS_PACKAGES = ("autogalaxy", "autoarray", "autofit")

# Class-path suffixes that are known not to resolve, mapped to the reason. The
# goal is for this to stay empty -- every entry is prior configuration that is
# silently dead.
ALLOWLIST: Dict[str, str] = {}

_CENSUS: Dict[str, type] = {}
_CENSUS_SKIPPED: List[Tuple[str, str]] = []


def class_census() -> Dict[str, type]:
    """
    Every class defined by the PyAuto packages, keyed by its fully qualified
    `module.Name` path. Built once and cached, as walking the packages imports
    every submodule.
    """
    if _CENSUS:
        return _CENSUS

    for package_name in CENSUS_PACKAGES:
        package = importlib.import_module(package_name)
        modules = [package]

        walker = pkgutil.walk_packages(
            package.__path__,
            prefix=f"{package.__name__}.",
            onerror=lambda name: _CENSUS_SKIPPED.append((name, "walk_packages")),
        )

        while True:
            try:
                info = next(walker)
            except StopIteration:
                break
            except Exception as e:
                _CENSUS_SKIPPED.append(("<walk>", repr(e)))
                continue

            if info.name.split(".")[-1] == "__main__":
                # `__main__` modules are command line entry points which run on
                # import (e.g. `autofit.mcp.__main__` starts a stdio server and
                # re-pins `conf.instance`), so they must never be imported here.
                continue

            try:
                modules.append(importlib.import_module(info.name))
            except Exception as e:
                _CENSUS_SKIPPED.append((info.name, repr(e)))

        for module in modules:
            # `vars` rather than `inspect.getmembers`, as the top level packages
            # define a lazy `__getattr__` which would import optional
            # dependencies on attribute access.
            for obj in list(vars(module).values()):
                if (
                    inspect.isclass(obj)
                    and getattr(obj, "__module__", None) == module.__name__
                ):
                    _CENSUS[f"{obj.__module__}.{obj.__qualname__}"] = obj

    return _CENSUS


def prior_blocks() -> Iterator[Tuple[Path, str, str, dict]]:
    """
    Yield `(path, relative_dotted_path, top_level_key, block)` for every top
    level key of every yaml file under the packaged priors folder.
    """
    for path in sorted(PRIORS_PATH.rglob("*.yaml")):
        relative = ".".join(path.relative_to(PRIORS_PATH).with_suffix("").parts)
        config = yaml.safe_load(path.read_text()) or {}

        for key, block in config.items():
            yield path, relative, key, block


def class_path_suffix(relative: str, key: str) -> str:
    """
    The class path a `JSONPriorConfig` lookup must end with for this key, e.g.
    `mass.dark.nfw.NFWSph`. Keys may themselves be dotted, e.g.
    `model.FlatLambdaCDM` in `cosmology.yaml`.
    """
    return f"{relative}.{key}"


def matching_classes(suffix: str) -> List[str]:
    return [
        name for name in class_census() if name == suffix or name.endswith(f".{suffix}")
    ]


def init_parameters(cls: type) -> Tuple[set, bool]:
    """
    The `__init__` argument names of `cls` (excluding `self`) and whether the
    signature accepts arbitrary keyword arguments.
    """
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return set(), True

    names = set()
    has_var_keyword = False

    for name, parameter in signature.parameters.items():
        if parameter.kind is parameter.VAR_KEYWORD:
            has_var_keyword = True
        names.add(name)

    names.discard("self")

    return names, has_var_keyword


def parameter_base_name(parameter: str) -> str:
    """
    Tuple arguments are flattened in the yaml with an index suffix, e.g.
    `centre` is configured as `centre_0` and `centre_1`.
    """
    head, _, tail = parameter.rpartition("_")

    if head and tail.isdigit():
        return head

    return parameter


def test__priors_path_is_the_packaged_config():
    assert PRIORS_PATH.is_dir()
    assert any(PRIORS_PATH.rglob("*.yaml"))


def test__every_prior_key_resolves_to_a_class():
    misses = []

    for path, relative, key, _ in prior_blocks():
        suffix = class_path_suffix(relative, key)

        if suffix in ALLOWLIST:
            continue

        if not matching_classes(suffix):
            misses.append(
                f"{path.relative_to(PRIORS_PATH)} :: {key} "
                f"(no class path ends with '{suffix}')"
            )

    assert not misses, (
        "Prior configuration keys which do not resolve to any class, and are "
        "therefore silently dead:\n  " + "\n  ".join(misses)
    )


def test__every_prior_parameter_is_an_init_argument():
    misses = []

    for path, relative, key, block in prior_blocks():
        suffix = class_path_suffix(relative, key)

        if suffix in ALLOWLIST or not isinstance(block, dict):
            continue

        matches = matching_classes(suffix)

        if not matches:
            continue

        names = set()
        has_var_keyword = False

        for name in matches:
            class_names, class_var_keyword = init_parameters(class_census()[name])
            names |= class_names
            has_var_keyword |= class_var_keyword

        if has_var_keyword:
            continue

        for parameter in block:
            if parameter in names or parameter_base_name(parameter) in names:
                continue

            misses.append(
                f"{path.relative_to(PRIORS_PATH)} :: {key} :: {parameter} "
                f"(__init__ arguments are {sorted(names)})"
            )

    assert not misses, (
        "Prior configuration parameters which are not `__init__` arguments of "
        "their class, and are therefore silently dead:\n  " + "\n  ".join(misses)
    )


def test__nfw_truncated_mcr_scatter_ludlow_sph_model_resolves():
    model = af.Model(ag.mp.NFWTruncatedMCRScatterLudlowSph)

    assert model.prior_count == 6

    model.instance_from_prior_medians()


def test__ellipse_multipole_scaled_model_resolves():
    # `m` and `major_axis` have no configured prior, exactly as for the parent
    # `EllipseMultipole`, so they are passed as fixed values here.
    model = af.Model(ag.EllipseMultipoleScaled, m=4, major_axis=1.0)

    assert model.prior_count == 2

    model.instance_from_prior_medians()
