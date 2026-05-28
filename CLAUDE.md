# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scientific Context

Adjacent strong-lensing science — including topics that overlap galaxy
modelling (light profiles + MGE, mass models, regularisation,
pixelisation, kinematics, lens-galaxy structure) — lives in the lensing
sub-wiki at
[`PyAutoLabs/PyAutoPaper`](https://github.com/PyAutoLabs/PyAutoPaper),
locally at `../PyAutoPaper/lensing_wiki/`. The wiki is lensing-focused
but galaxy-modelling readers will find `concepts/source-reconstruction.md`,
`concepts/regularization.md`, `concepts/bulge-halo-decomposition.md`,
`concepts/kinematics-and-lensing.md`, and `concepts/multipoles.md`
directly useful. Start at `../PyAutoPaper/lensing_wiki/index.md`.

## Commands

### Install
```bash
pip install -e ".[dev]"
```

### Run Tests
```bash
# All tests
python -m pytest test_autogalaxy/

# Single test file
python -m pytest test_autogalaxy/galaxy/test_galaxy.py

# Single test
python -m pytest test_autogalaxy/galaxy/test_galaxy.py::TestGalaxy::test_name

# With output
python -m pytest test_autogalaxy/imaging/test_fit_imaging.py -s
```

### Codex / sandboxed runs

When running Python from Codex or any restricted environment, set writable cache directories so `numba` and `matplotlib` do not fail on unwritable home or source-tree paths:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python -m pytest test_autogalaxy/
```

This workspace is often imported from `/mnt/c/...` and Codex may not be able to write to module `__pycache__` directories or `/home/jammy/.cache`, which can cause import-time `numba` caching failures without this override.

### Formatting
```bash
black autogalaxy/
```

### Plot Output Mode

Set `PYAUTO_OUTPUT_MODE=1` to capture every figure produced by a script into numbered PNG files in `./output_mode/<script_name>/`. This is useful for visually inspecting all plots from an integration test without needing a display.

```bash
PYAUTO_OUTPUT_MODE=1 python scripts/my_script.py
# -> ./output_mode/my_script/0_fit.png, 1_tracer.png, ...
```

When this env var is set, all `save_figure`, `subplot_save`, and `_save_subplot` calls are intercepted — the normal output path is bypassed and figures are written sequentially to the output_mode directory instead.

## Architecture

**PyAutoGalaxy** is a Bayesian galaxy morphology fitting library. It depends on two sibling packages:
- **`autoarray`** – low-level data structures (grids, masks, arrays, imaging/interferometer datasets, inversions/pixelizations)
- **`autofit`** – non-linear search / model-fitting framework (defines `af.Analysis`, `af.ModelInstance`, `af.Result`)

### Core Class Hierarchy

```
GeometryProfile (geometry_profiles.py)
├── SphProfile
└── EllProfile
    ├── LightProfile (profiles/light/abstract.py) + OperateImage mixin
    │   ├── lp.Sersic, lp.Exponential, lp.Gaussian, etc. (profiles/light/standard/)
    │   ├── LightProfileLinear (profiles/light/linear/) - intensity solved via inversion
    │   ├── LightProfileOperated (profiles/light/operated/) - PSF already applied
    │   └── Basis (profiles/basis.py) - collection of profiles for MGE / shapelets
    └── MassProfile (profiles/mass/abstract/abstract.py) + OperateDeflections mixin
        ├── mp.Isothermal, mp.NFW, mp.PowerLaw, etc.
        └── stellar, dark, total subdirectories

Galaxy (galaxy/galaxy.py) - inherits af.ModelObject, OperateImageList, OperateDeflections
  └── holds arbitrary kwargs: light profiles, mass profiles, pixelizations, etc.

Galaxies (galaxy/galaxies.py) - a List[Galaxy] with group operations
```

### Fit Classes

Each dataset type has a `Fit*` class that orchestrates the full fitting pipeline:

- `FitImaging` (`imaging/fit_imaging.py`) – CCD imaging
- `FitInterferometer` (`interferometer/fit_interferometer.py`) – ALMA/interferometry
- `FitEllipse` (`ellipse/fit_ellipse.py`) – isophote/ellipse fitting

All inherit from `AbstractFitInversion` (`abstract_fit.py`), which handles the linear algebra inversion step when `LightProfileLinear` or pixelization-based profiles are present.

### Analysis Classes (autofit integration)

Each dataset type has an `Analysis*` class that implements `log_likelihood_function`:

- `AnalysisImaging` (`imaging/model/analysis.py`)
- `AnalysisInterferometer` (`interferometer/model/analysis.py`)
- `AnalysisEllipse` (`ellipse/model/analysis.py`)

These inherit from `AnalysisDataset` → `Analysis` (in `analysis/analysis/`), which inherits `af.Analysis`. The `log_likelihood_function` builds a `Fit*` object from the `af.ModelInstance` and returns its `figure_of_merit`.

### Decorator System (from autoarray)

Profile methods that consume a grid and return an array, grid, or vector use decorators from `autoarray.structures.decorators`. These ensure the **output type matches the input grid type**:

| Decorator | `Grid2D` input | `Grid2DIrregular` input |
|---|---|---|
| `@aa.grid_dec.to_array` | `Array2D` | `ArrayIrregular` |
| `@aa.grid_dec.to_grid` | `Grid2D` | `Grid2DIrregular` |
| `@aa.grid_dec.to_vector_yx` | `VectorYX2D` | `VectorYX2DIrregular` |

The `@aa.grid_dec.transform` decorator (always stacked below the output decorator) shifts and rotates the grid to the profile's reference frame before passing it to the function body.

The canonical stacking order is:
```python
@aa.grid_dec.to_array      # outermost: wraps output
@aa.grid_dec.transform     # innermost: transforms grid
def convergence_2d_from(self, grid, xp=np, **kwargs):
    y = grid.array[:, 0]   # use .array to get raw numpy/jax array
    x = grid.array[:, 1]
    return ...             # return raw array; decorator wraps it
```

**Key rule**: the function body must return a **raw array** (not an autoarray). The decorator handles wrapping. Access grid coordinates via `grid.array[:, 0]` / `grid.array[:, 1]` (not `grid[:, 0]`), because after `@transform` the grid is still an autoarray object and `.array` is the safe way to extract the underlying data for both numpy and jax backends.

See PyAutoArray's `CLAUDE.md` for full details on the decorator internals.

### JAX Support

The codebase is designed so that **NumPy is the default everywhere and JAX is opt-in**. JAX is never imported at module level — it is only imported locally inside functions when explicitly requested.

The `xp` parameter pattern is the single point of control:
- `xp=np` (default throughout) — pure NumPy path, no JAX dependency at runtime
- `xp=jnp` — JAX path, imports `jax` / `jax.numpy` locally inside the function

This means:
- **Unit tests** (`test_autogalaxy/`) always run on the NumPy path. No test should import JAX or pass `xp=jnp` unless it is explicitly testing the JAX path.
- **Integration tests** (in `autogalaxy_workspace_test/`) are where the JAX path is exercised, typically wrapped in `jax.jit` to test both correctness and compilation.
- `conftest.py` forces JAX backend initialisation before the test suite runs, but this only ensures JAX is available — it does not switch the default backend.

`AbstractFitInversion.use_jax` tracks whether a fit was constructed with JAX. `AnalysisImaging` has `use_jax: bool = True` to opt into the JAX path for model-fitting.

When adding a new function that should support JAX:
1. Default the parameter to `xp=np`
2. Guard any JAX imports with `if xp is not np:` and import `jax` / `jax.numpy` locally inside that branch
3. Add the NumPy implementation as the default path (finite-difference, `np.*` calls, etc.)
4. Add a JAX implementation in the guarded branch (e.g. `jax.jacfwd`, `jnp.vectorize`)
5. Verify correctness by comparing both paths in `autogalaxy_workspace_test/scripts/`

### Threading `xp` through inherited / property / convert-helper calls

Adding `xp=np` to a method body and swapping `np.*` for `xp.*` is **only half the work**. Every nested call inside that body — whether to `self.X()`, `obj.X()`, a helper in `convert.py`, an inherited `@property`, or a sibling method — must also receive `xp=xp` if it can route to numpy operations on what would otherwise be JAX tracers. Otherwise the inner call quietly defaults to `xp=np` and fails when a tracer reaches an `np.*` op.

Concrete sites that have bitten this codebase (all fixed during prompt 7 of `ellipse_fitting_jax`):

- **`@property` chains that hardcode `np`.** `Ellipse.ellipticity` and `Ellipse.minor_axis` are properties (no kwargs possible), so a caller in an xp-aware method must either inline the computation under `if xp is not np:` or convert the property to a method. Pattern: read every `@property` you call from xp-aware code; if it does `np.sqrt(...)` on `self.ell_comps`, it's a hazard.
- **Inherited methods.** `Ellipse.angle(xp=np)` and `Ellipse.angle_radians(xp=np)` accept `xp` (they're defined on `EllProfile`), but call sites in `EllipseMultipole.points_perturbed_from` used `ellipse.angle()` without passing it. Pattern: grep within xp-aware functions for `self.X(` and `obj.X(`; for each, verify `xp=xp` is passed.
- **`convert.py` helpers.** `multipole_comps_from`, `multipole_k_m_and_phi_m_from`, `axis_ratio_and_angle_from`, `angle_from` etc. all take `xp=np`; call sites must thread it. They also use Python `&` on JAX bool tracers, which silently calls `__array__()` — replace with `xp.logical_and`.
- **`@cached_property` on traced arrays.** Caches a tracer in `self.__dict__` which is invalid across `vmap` batch elements (different batches share the cache). Use plain `@property` for any property whose value depends on JAX-traced inputs.

**Validation: `jax.jit(fn)(concrete_instance)` is NOT a sufficient JAX trace check.** A `ModelInstance` with concrete float `ell_comps` propagates as floats through `np.*` ops without raising — the bug stays hidden. **Use `jax.vmap(fitness)(jnp.array(params))` instead** (or `Fitness._vmap` on autofit's wrapper). Vmap forces tracer propagation through every leaf and exposes un-threaded `xp` sites.

When adding a JAX path to an Analysis class, the workspace_test parity script must include both a `jax.jit(analysis.fit_from)(instance)` round-trip AND a `fitness._vmap(parameters)` batch evaluation. See `autogalaxy_workspace_test/scripts/jax_likelihood_functions/imaging/lp.py` and the ellipse counterpart for the template.

### JAX and autoarray wrappers at the `jax.jit` boundary

Autoarray types (`Array2D`, `ArrayIrregular`, `VectorYX2DIrregular`, etc.) are **not registered as JAX pytrees**. This means:

- Constructing them **inside** a JIT trace is fine (Python code runs normally during tracing)
- **Returning** them as the output of a `jax.jit`-compiled function **fails** with `TypeError: ... is not a valid JAX type`

Functions decorated with `@aa.grid_dec.to_array` / `@to_vector_yx` wrap their return value in an autoarray type. This wrapping is safe for intermediate calls (the autoarray object is consumed by downstream Python code). However, if such a function is the **outermost call** inside a `jax.jit` lambda, its return value will fail at the JIT boundary.

The solution is the **`if xp is np:` guard** in the function body:

```python
def convergence_2d_via_hessian_from(self, grid, xp=np):
    convergence = 0.5 * (hessian_yy + hessian_xx)

    if xp is np:
        return aa.ArrayIrregular(values=convergence)  # numpy: wrapped
    return convergence                                  # jax: raw jax.Array
```

This pattern is applied throughout `autogalaxy/operate/lens_calc.py`. Functions that are only ever called as intermediate steps (e.g. `deflections_yx_2d_from`) do NOT need this guard — their autoarray wrappers are never the JIT output.

### Linear Light Profiles & Inversions

`LightProfileLinear` subclasses do not take an `intensity` parameter—it is solved via a linear inversion (provided by `autoarray`). The `GalaxiesToInversion` class (`galaxy/to_inversion.py`) handles converting galaxies with linear profiles or pixelizations into the inversion objects needed by `autoarray`.

### Adapt Images (Multi-stage fitting)

`AdaptImages` (`analysis/adapt_images/`) stores per-galaxy model images from a previous search. These are passed to subsequent searches to drive adaptive mesh/regularization schemes. `galaxy_name_image_dict_via_result_from` extracts adapt images from a `Result`.

### Configuration

Default priors, visualization settings, and general config live in `autogalaxy/config/`. Tests push a local config directory via `conf.instance.push(...)` in `test_autogalaxy/conftest.py`.

### Operate Mixins

- `OperateImage` (`operate/image.py`) – provides `blurred_image_2d_from`, `visibilities_from`, etc. on light objects
- `OperateDeflections` (`operate/deflections.py`) – provides deflection-related operations on mass objects

Both are mixin classes inherited by `LightProfile`, `MassProfile`, `Galaxy`, and `Galaxies`.

### Workspace Script Style

Scripts in `autogalaxy_workspace` and `autogalaxy_workspace_test` use `"""..."""` docstring blocks as prose commentary throughout — **not** `#` comments. Every script opens with a module-level docstring (title + underline + description), and each logical section of code is preceded by a `"""..."""` block with a `__Section Name__` header explaining what follows. See any script in `autogalaxy_workspace/scripts/` for examples of this style.

### Workspace (Examples & Notebooks)

The `autogalaxy_workspace` at `/mnt/c/Users/Jammy/Code/PyAutoLabs/autogalaxy_workspace` contains runnable examples and tutorials. Key locations:

- `start_here.ipynb` / `start_here.py` – entry point overview of the API
- `scripts/imaging/` – end-to-end scripts: `simulator.py`, `fit.py`, `modeling.py`, `likelihood_function.py`, `features/`
- `scripts/interferometer/`, `scripts/ellipse/`, `scripts/multi/` – same structure for other dataset types
- `notebooks/` – Jupyter notebook equivalents of all `scripts/`
- `scripts/guides/` – topic guides (e.g. linear profiles, pixelizations, chaining)
- `config/` – workspace-level config that overrides package defaults when running workspace scripts

The **HowToGalaxy** tutorial series lives in its own repository at `PyAutoLabs/HowToGalaxy` (no longer under the workspace).

### Namespace Conventions

When importing `autogalaxy as ag`:
- `ag.lp.*` – standard light profiles
- `ag.lp_linear.*` – linear light profiles
- `ag.lp_operated.*` – operated light profiles
- `ag.lp_basis.*` / `ag.lp_snr.*` – basis and SNR profiles
- `ag.mp.*` – mass profiles
- `ag.lmp.*` – light+mass profiles
- `ag.ps.*` – point sources
- `ag.Galaxy`, `ag.Galaxies`
- `ag.FitImaging`, `ag.AnalysisImaging`, `ag.SimulatorImaging`

## Line Endings — Always Unix (LF)

All files in this project **must use Unix line endings (LF, `\n`)**. Windows/DOS line endings (CRLF, `\r\n`) will break Python files on HPC systems.

**When writing or editing any file**, always produce Unix line endings. Never write `\r\n` line endings.

After creating or copying files, verify and convert if needed:

```bash
# Check for DOS line endings
file autogalaxy/galaxy/galaxy.py   # should say "ASCII text", not "CRLF"

# Convert all Python files in the project
find . -type f -name "*.py" | xargs dos2unix
```

Prefer simple shell commands.
Avoid chaining with && or pipes.
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
