"""Lazy JAX pytree registration for `Galaxies` and the classes it contains.

When a user wraps a PyAutoGalaxy call in their own ``@jax.jit`` and the call
receives a ``Galaxies`` as a traced argument, every concrete class reachable
from it must be registered as a JAX pytree node so JAX can flatten and
unflatten across the JIT boundary.

This module is the counterpart of ``AnalysisImaging._register_fit_imaging_pytrees``
(and its interferometer equivalent) for code paths that do not go through an
``Analysis`` — custom forward models and hand-built likelihood functions.

Unlike PyAutoLens's ``autolens.jax.register_tracer_classes``, which
``PointSolver`` and ``Simulator`` call automatically, **nothing inside
autogalaxy calls this function**: both ``Analysis`` classes already register
their pytrees inline from ``fit_from``. It exists purely as the public entry
point for user code, so do not remove it as unused.

Mirrors PyAutoFit's ``autofit/jax/pytrees.py`` layout. Idempotent:
re-registration of a class is a silent no-op.
"""
from typing import Iterable


def register_galaxies_classes(galaxies) -> bool:
    """Register every concrete class reachable from ``galaxies`` as a JAX pytree.

    Registers ``Galaxies`` itself (via the shared
    ``register_galaxies_pytree``, which installs the custom flatten a ``list``
    subclass needs), then walks each galaxy and registers ``Galaxy`` plus every
    light / mass profile class encountered.

    Both halves are required. Registering ``Galaxies`` alone still fails with
    ``TypeError: ... problematic value is of type Galaxy ... at path
    galaxies[0]`` the first time a jitted function is handed a ``Galaxies``.

    Returns ``True`` if registration ran (or was already complete), ``False``
    if JAX is not installed (in which case the call is a silent no-op).
    """
    try:
        import jax  # noqa: F401
    except ImportError:
        return False

    from autogalaxy.analysis.jax_pytrees import register_galaxies_pytree

    register_galaxies_pytree()

    for galaxy in galaxies:
        _register_object_classes(galaxy)

    return True


def _register_object_classes(obj) -> None:
    """Walk an object recursively and register each non-builtin class it carries.

    Used to register ``Galaxy`` plus every concrete profile class
    (``Sersic``, ``Isothermal``, ``NFW``, ...) the galaxy holds. Skips builtin
    types (numbers, strings, sequences) since those are not user classes that
    JAX needs flatten/unflatten functions for.
    """
    from autoarray.abstract_ndarray import register_instance_pytree

    cls = type(obj)
    if _is_builtin(cls):
        return

    register_instance_pytree(cls)

    for value in _iter_attribute_values(obj):
        _register_object_classes(value)


def _iter_attribute_values(obj) -> Iterable:
    """Yield each attribute value of ``obj`` worth recursing into.

    Walks ``obj.__dict__`` (if present) and recurses one level into list /
    tuple / dict containers to reach profile objects held in collections.
    """
    if not hasattr(obj, "__dict__"):
        return

    for value in vars(obj).values():
        if isinstance(value, (list, tuple)):
            for item in value:
                yield item
        elif isinstance(value, dict):
            for item in value.values():
                yield item
        else:
            yield value


def _is_builtin(cls) -> bool:
    """True for primitive / container / standard-library / numerical-backend
    types that should not be registered as JAX pytrees.

    Catches the JAX tracer types (``DynamicJaxprTracer`` et al.) explicitly:
    if a walker happens to recurse into JAX-traced state (because it ran
    inside a ``jax.jit`` trace), registering ``type(tracer)`` would make every
    subsequent tracer-flatten call route into our dict-based flatten, which
    fails because tracers do not have ``__dict__``.
    """
    if cls is type(None):
        return True
    module = cls.__module__
    if module == "builtins":
        return True
    if module.startswith(("numpy", "jax", "jaxlib")):
        return True
    return False
