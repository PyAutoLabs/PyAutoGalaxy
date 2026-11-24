"""
The one shared resolution path of an analysis's compute `setup`.

`AnalysisSetupMixin` holds everything every analysis does with its `setup` input: store it
(`_init_setup`), resolve it against the model in `modify_before_fit` (`resolve_setup`), log what
`setup="auto"` would choose when no setup was given (`_log_setup_auto_would_choose`) and write the
resolved setup to `files/setup.json` (`save_setup`). The pure helpers it uses live in
`autogalaxy.analysis.setup_util`.

A subclass supplies two things:

- `setup_key_and_note_from(model)`: the `(dataset_type, variant)` rule key of `aa.Setup.auto` for
  the analysis and model, plus an optional note for the banner (e.g. that a fallback rule was used).
- `_apply_resolved_setup(setup)`: what applying the resolved setup means beyond the backend and
  threads, e.g. `AnalysisDataset` attaches the operator it names to the dataset and checks it. It
  runs after `_use_jax` is set, so the check sees the resolved backend. The default does nothing,
  for analyses whose dataset has no operator to attach.

The mixin has no `__init__` (the subclass calls `_init_setup(setup)` in its constructor) and is not
on `autogalaxy.Analysis`, so analyses that take no `setup` are unaffected. It goes before the
`Analysis` base in a subclass's bases.
"""

import logging
from typing import Optional, Tuple

import autofit as af
import autoarray as aa

from autogalaxy.analysis import setup_util

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class AnalysisSetupMixin:
    """
    Mixin resolving the compute `setup` of an analysis (backend, device, operator, precision,
    transformer, threads).

    The subclass calls `_init_setup(setup)` in its constructor, `resolve_setup(model)` first in its
    `modify_before_fit` and `save_setup(paths)` in its `save_attributes`, and implements
    `setup_key_and_note_from(model)`; it overrides `_apply_resolved_setup(setup)` if applying a setup
    changes its dataset.
    """

    def _init_setup(self, setup):
        """
        Store the normalised `setup` input (`None`, `"auto"` or an `aa.Setup`, see
        `setup_util.setup_input_from`); the resolved setup is set by `resolve_setup`.
        """
        self.setup_input = setup_util.setup_input_from(setup)
        self.setup = None
        self.setup_chosen_by = None

    def setup_key_and_note_from(
        self, model: af.Collection
    ) -> Tuple[Tuple[str, str], Optional[str]]:
        """
        The `(dataset_type, variant)` rule key of `aa.Setup.auto` for this analysis and model, plus a
        note for the setup banner (or `None`).
        """
        raise NotImplementedError

    def _apply_resolved_setup(self, setup: aa.Setup):
        """
        Apply the resolved setup to the analysis beyond its backend and threads (e.g. attach the
        operator it names to the dataset). Called by `resolve_setup` after `_use_jax` is set. The
        default does nothing.
        """

    def resolve_setup(self, model: af.Collection):
        """
        Resolve the `setup` input of the analysis against the model and apply it.

        With no `setup` input nothing changes: one log line reports what `setup="auto"` would choose (any
        error working that out is ignored).

        With `setup="auto"` the setup is chosen by `aa.Setup.auto` from the rule key of the analysis and
        model (`setup_key_and_note_from`); `PYAUTO_SETUP=<setup_id>` overrides it. An explicit `aa.Setup`
        or setup_id is used as given. The resolved setup then:

        - sets the analysis backend (`_use_jax = setup.backend == "jax"`, falling back to numpy when JAX is
          disabled or not installed, as `af.Analysis` does);
        - is applied by `_apply_resolved_setup` (e.g. `AnalysisDataset` attaches the operator it names to
          the dataset and checks it against the backend, so a mismatch raises here rather than rejecting
          every sample);
        - pins the thread pools (`setup.apply_threads()`);
        - is logged as the setup banner line and stored as `self.setup`.

        Parameters
        ----------
        model
            The model fitted by the search, which decides the rule variant (e.g. pixelization mesh or MGE).
        """
        setup_input = getattr(self, "setup_input", None)

        if setup_input is None:
            self._log_setup_auto_would_choose(model=model)
            return

        note = None

        environment = setup_util.setup_environment()

        if setup_input == setup_util.SETUP_AUTO:
            key, note = self.setup_key_and_note_from(model)
            setup = aa.Setup.auto(*key, environment=environment)
            chosen_by = setup_util.chosen_by_from(setup=setup, auto=True)
        else:
            setup = setup_input
            chosen_by = setup_util.chosen_by_from(setup=setup, auto=False)

        self._use_jax = setup_util.use_jax_from(setup.backend == "jax")

        self._apply_resolved_setup(setup)

        setup.apply_threads()

        self.setup = setup
        self.setup_chosen_by = chosen_by

        logger.info(
            setup_util.setup_banner_from(
                setup=setup,
                chosen_by=chosen_by,
                environment=environment,
                note=note,
            )
        )

    def _log_setup_auto_would_choose(self, model: af.Collection):
        """
        Log one line reporting the setup `setup="auto"` would choose, applying nothing.
        """
        try:
            key, _ = self.setup_key_and_note_from(model)
            setup = aa.Setup.auto(*key, environment=setup_util.setup_environment())
        except Exception as e:
            logger.debug(
                f"Setup: could not work out what setup='auto' would choose: {e}"
            )
            return

        logger.info(
            f"Setup: auto would choose {setup.setup_id} (rule {setup.rule_id}); "
            f'pass setup="auto" to use it'
        )

    def save_setup(self, paths: af.DirectoryPaths):
        """
        Write the resolved setup to `files/setup.json`, when a setup was resolved.
        """
        setup: Optional[aa.Setup] = getattr(self, "setup", None)

        if setup is not None:
            paths.save_json(
                name="setup",
                object_dict=setup_util.setup_dict_from(
                    setup=setup, chosen_by=self.setup_chosen_by
                ),
            )
