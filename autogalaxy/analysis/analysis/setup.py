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

    def _init_setup(self, setup, setup_defaulted: bool = False):
        """
        Store the normalised `setup` input (`None`, `"auto"` or an `aa.Setup`, see
        `setup_util.setup_input_from`); the resolved setup is set by `resolve_setup`.

        `setup_defaulted` records that the input is the implicit default `"auto"` (neither `setup` nor
        `use_jax` was passed, see `setup_util.setup_and_use_jax_from`).
        """
        self.setup_input = setup_util.setup_input_from(setup)
        self.setup_defaulted = bool(setup_defaulted) and self.setup_input is not None
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

    def prepare_for_model(self, model: af.Collection):
        """
        Resolve the setup from `model` before the fit starts, when the analysis has a `setup` input
        that is not yet resolved (`af.Analysis.prepare_for_model`).

        Pre-fit likelihood use, such as `print_vram_use(model=...)`, calls this so a `setup="auto"`
        analysis knows its backend before its likelihood is evaluated. The banner is logged now; the
        `modify_before_fit` of a later fit with the same model re-applies the same setup without
        logging it again (see `resolve_setup`).
        """
        if getattr(self, "setup_input", None) is None:
            return
        if getattr(self, "setup", None) is not None:
            return
        self.resolve_setup(model=model)

    def resolve_setup(self, model: af.Collection, log_banner: bool = True):
        """
        Resolve the `setup` input of the analysis against the model and apply it.

        With no `setup` input (`setup=None`, or an explicit `use_jax` with no `setup`) nothing changes: one
        log line reports what `setup="auto"` would choose (any error working that out is ignored).

        With `setup="auto"` the setup is chosen by `aa.Setup.auto` from the rule key of the analysis and
        model (`setup_key_and_note_from`); `PYAUTO_SETUP=<setup_id>` overrides it. When `"auto"` is the
        implicit default and no setup can be chosen (e.g. no rule for the dataset), a warning is logged and
        the constructor backend is kept. An explicit `aa.Setup` or setup_id is used as given. The resolved setup then:

        - sets the analysis backend (`_use_jax = setup.backend == "jax"`, falling back to numpy when JAX is
          disabled or not installed, as `af.Analysis` does);
        - is applied by `_apply_resolved_setup` (e.g. `AnalysisDataset` attaches the operator it names to
          the dataset and checks it against the backend, so a mismatch raises here rather than rejecting
          every sample);
        - pins the thread pools (`setup.apply_threads()`);
        - is logged as the header and `Setup:` lines (two records) and stored as `self.setup`.

        It is safe to call more than once (e.g. by `prepare_for_model` before the fit and by
        `modify_before_fit` when it starts): the setup is re-resolved and re-applied each time, but when
        it is the one already resolved (same `setup_id`, chosen the same way) the banner is not logged a
        second time. A different setup is applied and logged.

        Parameters
        ----------
        model
            The model fitted by the search, which decides the rule variant (e.g. pixelization mesh or MGE).
        log_banner
            Whether to log the setup lines now. `False` keeps them for `log_setup_banner`, so a caller that
            does more work after resolving (e.g. `AnalysisDataset` preloading its grids, which can warn) logs
            them last, next to the fit's `Parallel:` / `Estimate:` lines.
        """
        setup_input = getattr(self, "setup_input", None)

        self._setup_banner_lines = []

        if setup_input is None:
            self._log_setup_auto_would_choose(model=model)
            if log_banner:
                self.log_setup_banner()
            return

        note = None

        environment = setup_util.setup_environment()

        if setup_input == setup_util.SETUP_AUTO:
            try:
                key, note = self.setup_key_and_note_from(model)
                setup = aa.Setup.auto(*key, environment=environment)
            except Exception as e:
                if not getattr(self, "setup_defaulted", False):
                    raise
                # The default `"auto"` never stops a fit that ran before it was the default: when no
                # setup can be chosen, the analysis keeps its constructor backend (as `setup=None`).
                logger.warning(
                    f"Setup: setup='auto' (the default) could not choose a setup ({e}); the fit "
                    f"runs with the constructor backend (use_jax={bool(self._use_jax)})."
                )
                return
            chosen_by = setup_util.chosen_by_from(setup=setup, auto=True)
        else:
            setup = setup_input
            chosen_by = setup_util.chosen_by_from(setup=setup, auto=False)

        previous = getattr(self, "setup", None)
        already_logged = (
            previous is not None
            and previous.setup_id == setup.setup_id
            and getattr(self, "setup_chosen_by", None) == chosen_by
        )

        self._use_jax = setup_util.use_jax_from(setup.backend == "jax")

        # `apply_setup` is idempotent, so re-applying the same setup rebuilds nothing.
        self._apply_resolved_setup(setup)

        setup.apply_threads()

        self.setup = setup
        self.setup_chosen_by = chosen_by

        if already_logged:
            return

        # The header and `Setup:` line are logged after the operator build and checks above (or
        # later still, by `log_setup_banner`), so they sit with the fit's `Parallel:` / `Estimate:` lines.
        # They report the backend that runs (`_use_jax`), which differs from the designated one when
        # JAX is disabled or not installed.
        self._setup_banner_lines = setup_util.setup_banner_lines_from(
            setup=setup,
            chosen_by=chosen_by,
            environment=environment,
            note=note,
            use_jax=self._use_jax,
        )

        if log_banner:
            self.log_setup_banner()

    def log_setup_banner(self):
        """
        Log the setup lines `resolve_setup` prepared (the header and `Setup:` line, or the one "not set"
        line), each as its own record, once.
        """
        for line in getattr(self, "_setup_banner_lines", None) or []:
            logger.info(line)
        self._setup_banner_lines = []

    def _log_setup_auto_would_choose(self, model: af.Collection):
        """
        Prepare the one line reporting the setup `setup="auto"` would choose (logged by
        `log_setup_banner`), applying nothing.
        """
        try:
            key, _ = self.setup_key_and_note_from(model)
            setup = aa.Setup.auto(*key, environment=setup_util.setup_environment())
        except Exception as e:
            logger.debug(
                f"Setup: could not work out what setup='auto' would choose: {e}"
            )
            return

        self._setup_banner_lines = [setup_util.setup_not_set_line_from(setup)]

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
