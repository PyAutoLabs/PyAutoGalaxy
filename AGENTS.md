# PyAutoGalaxy — Agent Instructions

**PyAutoGalaxy** is a Bayesian galaxy morphology fitting library. It depends on `autoarray` (data structures) and `autofit` (model-fitting framework).

## Setup

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
python -m pytest test_autogalaxy/
python -m pytest test_autogalaxy/galaxy/test_galaxy.py
python -m pytest test_autogalaxy/galaxy/test_galaxy.py::TestGalaxy::test_name
```

### Sandboxed / Codex runs

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python -m pytest test_autogalaxy/
```

## Key Architecture

- **Profiles**: `LightProfile` (`lp.*`), `MassProfile` (`mp.*`), `LightProfileLinear` (`lp_linear.*`)
- **Galaxy** (`galaxy/galaxy.py`): holds light/mass profiles, pixelizations
- **Fit classes**: `FitImaging`, `FitInterferometer`, `FitEllipse`
- **Analysis classes**: `AnalysisImaging`, `AnalysisInterferometer` — implement `log_likelihood_function`
- **Decorator system** (from autoarray): `@to_array`, `@to_grid`, `@to_vector_yx`, `@transform`
- **Operate mixins**: `OperateImage`, `OperateDeflections`, `LensCalc`

## Key Rules

- The `xp` parameter controls NumPy vs JAX: `xp=np` (default) or `xp=jnp`
- Functions inside `jax.jit` must guard autoarray wrapping with `if xp is np:`
- Decorated functions return **raw arrays** — the decorator wraps them
- Use `grid.array[:, 0]` to access grid coordinates (not `grid[:, 0]`)
- All files must use Unix line endings (LF)
- Format with `black autogalaxy/`

## Working on Issues

1. Read the issue description and any linked plan.
2. Identify affected files and write your changes.
3. Run the full test suite: `python -m pytest test_autogalaxy/`
4. Ensure all tests pass before opening a PR.
5. If changing public API, note the change in your PR description — downstream packages (PyAutoLens) and workspaces may need updates.
## Never rewrite history

NEVER perform these operations on any repo with a remote:

- `git init` in a directory already tracked by git
- `rm -rf .git && git init`
- Commit with subject "Initial commit", "Fresh start", "Start fresh", "Reset
  for AI workflow", or any equivalent message on a branch with a remote
- `git push --force` to `main` (or any branch tracked as `origin/HEAD`)
- `git filter-repo` / `git filter-branch` on shared branches
- `git rebase -i` rewriting commits already pushed to a shared branch

If the working tree needs a clean state, the **only** correct sequence is:

    git fetch origin
    git reset --hard origin/main
    git clean -fd

This applies equally to humans, local Claude Code, cloud Claude agents, Codex,
and any other agent. The "Initial commit — fresh start for AI workflow" pattern
that appeared independently on origin and local for three workspace repos is
exactly what this rule prevents — it costs ~40 commits of redundant local work
every time it happens.
