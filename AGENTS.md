# PyAutoGalaxy — Agent Instructions

Canonical, agent-agnostic instructions for this repo. `CLAUDE.md` imports this
file; any tool that does not process `@`-imports should read this directly.

## What this repo is

**PyAutoGalaxy** (package `autogalaxy`) is a Bayesian galaxy-morphology fitting
library: light/mass profiles (incl. MGE/linear/operated), `Galaxy`/`Galaxies`,
per-dataset `Fit*` and `Analysis*` classes (imaging, interferometer, ellipse),
inversions for linear profiles/pixelizations, and adapt-image multi-stage
fitting.

Dependency direction: autogalaxy may import **autoarray** (data structures),
**autofit** (model-fitting), and **autoconf** (config). It must **never**
import `autolens` — lensing lives one layer up.

## Related repos

- **Source siblings:** PyAutoConf, PyAutoArray, PyAutoFit (upstream);
  PyAutoLens (downstream — builds multi-plane lensing on autogalaxy).
- **autogalaxy_workspace** — runnable tutorials/examples (`../autogalaxy_workspace`).
- **autogalaxy_workspace_test** — integration + JAX/likelihood parity scripts.
- **HowToGalaxy** — the lecture-style tutorial series (`../HowToGalaxy`).
- **docs/** — Sphinx source; published to ReadTheDocs.
- **Science context:** the lensing-focused knowledge wiki at
  `autolens_assistant/wiki/literature/` (concepts, entities, sources) covers
  source reconstruction, regularization, bulge/halo decomposition, kinematics,
  and multipoles useful to galaxy modelling.

## Quick commands

```bash
pip install -e ".[dev]"                                    # install with dev/test extras
python -m pytest test_autogalaxy/                          # full test suite
python -m pytest test_autogalaxy/galaxy/test_galaxy.py     # one focused test (add -s for output)
black autogalaxy/                                          # formatter (advisory — not gated)
```

In a sandboxed / restricted environment, point numba and matplotlib at
writable caches:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python -m pytest test_autogalaxy/
```

## CI / definition of green

PRs must pass `pytest --cov` on the CI matrix (Python 3.12 **and** 3.13). There
is no black/ruff/flake8 gate — formatting is advisory. (`requires-python` in
`pyproject.toml` is `>=3.9`.)

## Configuration & defaults

autoconf supplies the packaged defaults under `autogalaxy/config/`. Workspaces
override them via their own `config/` directory; the test suite pushes a local
config dir via `conf.instance.push(...)` in `test_autogalaxy/conftest.py`. When
a change adds a new config key, mirror it into the packaged defaults so
downstream workspaces inherit it.

## JAX & `xp`

NumPy is the default everywhere; JAX is opt-in and never imported at module
level. `xp=np` (default) selects NumPy; `xp=jnp` selects JAX (imported locally).
Thread `xp` through **every** nested call — a missed site silently defaults to
`xp=np` and fails when a tracer hits an `np.*` op. Two patterns cross the
`jax.jit` boundary: the `if xp is np:` **guard** for raw `jax.Array` returns
(used by all `LensCalc` hessian methods in `operate/lens_calc.py`), and
**pytree registration** for functions returning a real wrapper/structured
object.

**Unit tests are NumPy-only.** A JAX/`xp` change is validated only by the
parity scripts in `autogalaxy_workspace_test` (`jax.jit` round-trip +
`fitness._vmap` batch eval) — never by `test_autogalaxy/`.

Full detail lives in PyAutoArray:
**[`PyAutoArray/docs/agents/jax_and_decorators.md`](../PyAutoArray/docs/agents/jax_and_decorators.md)**.

## Public API

The public surface is defined authoritatively in `autogalaxy/__init__.py` —
read it rather than trusting a hand-maintained namespace table. Canonical
import:

```python
import autogalaxy as ag
```

Profiles are namespaced there (`ag.lp.*`, `ag.lp_linear.*`, `ag.mp.*`,
`ag.lmp.*`, …) alongside `ag.Galaxy`, `ag.FitImaging`, `ag.AnalysisImaging`.

## Key rules / footguns

- Import direction: autoarray / autofit / autoconf only — **never** `autolens`.
- Operate mixins are `OperateImage` (`operate/image.py`) and `LensCalc`
  (`operate/lens_calc.py`). There is no `OperateDeflections` / `operate/
  deflections.py`.
- Grid-decorated profile methods return a **raw array** (the decorator wraps
  it); write `aa.decorators.*` and read coordinates via `grid.array[:, 0]`.
- All files use Unix line endings (LF, `\n`) — never `\r\n`.

## Working on issues

1. Read the issue description and any linked plan.
2. Identify affected files and make the change.
3. Run the full suite: `python -m pytest test_autogalaxy/`.
4. If you changed public API, say so explicitly — PyAutoLens and the workspaces
   may need updates.
5. Ensure all tests pass before opening a PR.

## Deep dives

- [`PyAutoArray/docs/agents/jax_and_decorators.md`](../PyAutoArray/docs/agents/jax_and_decorators.md)
  — decorator system, `xp` backend pattern, and the `jax.jit` boundary.

<!-- repos_sync:history:begin -->
## Never rewrite history

Never rewrite pushed history on any repo with a remote — no `git init` over a
tracked repo, no force-push to `main`, no fresh-start "Initial commit", no
`filter-repo` / `filter-branch` / `rebase -i` on pushed branches. To get a
clean tree: `git fetch origin && git reset --hard origin/main && git clean -fd`.
<!-- repos_sync:history:end -->
