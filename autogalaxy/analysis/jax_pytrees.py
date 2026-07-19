"""Shared JAX pytree registrations for autogalaxy analysis classes.

Each ``Analysis*`` class registers its own ``Fit*`` and per-analysis
constants inline (so the call site stays self-documenting), but the
``Galaxies`` registration is shared from here because the custom
flatten/unflatten logic is non-trivial and identical across all
analyses that hold a ``Galaxies`` aggregate.
"""


def register_galaxies_pytree() -> None:
    """Register ``Galaxies`` as a JAX pytree.

    ``Galaxies`` is a ``list`` subclass — the generic ``__dict__`` flatten
    in ``register_instance_pytree`` would drop the list contents. This
    registers a custom flatten that carries the list items as dynamic
    children and the ``__dict__`` entries as aux.

    Idempotent — guarded by ``_pytree_registered_classes`` so repeated
    calls (e.g. from each ``Analysis*.fit_from``) are cheap.
    """
    from autoarray.abstract_ndarray import _pytree_registered_classes
    from autonerves.jax_wrapper import register_pytree_node
    from autogalaxy.galaxy.galaxies import Galaxies

    if Galaxies in _pytree_registered_classes:
        return

    def _flatten_galaxies(galaxies):
        dict_items = tuple(sorted(galaxies.__dict__.items()))
        return tuple(galaxies), dict_items

    def _unflatten_galaxies(aux, children):
        new = Galaxies.__new__(Galaxies)
        list.__init__(new, children)
        for key, value in aux:
            setattr(new, key, value)
        return new

    register_pytree_node(Galaxies, _flatten_galaxies, _unflatten_galaxies)
    _pytree_registered_classes.add(Galaxies)
