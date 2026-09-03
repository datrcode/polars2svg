# Contributing to polars2svg

`polars2svg` is currently a single-maintainer project, but issues, bug reports,
and pull requests are welcome. This document covers dev environment setup,
test conventions, and the style conventions an outside contributor needs to
know before opening a PR.

## Dev setup

The project uses [`uv`](https://docs.astral.sh/uv/) with a local virtual
environment at `.venv/`.

```bash
git clone https://github.com/datrcode/polars2svg.git
cd polars2svg
uv venv --python 3.13
uv pip install -e . --group dev
```

`--group dev` pulls in every optional extra (`interactive`, `layouts`,
`export`, `mlx`) plus test-only tooling, so the full test suite — including
`linkp`/`chordp`/interactive-variant tests — runs unmodified. Use
`.venv/bin/python`, never a system `python3`.

The `mlx` extra installs MLX with the Metal backend on Apple silicon. **On Linux
it installs no backend at all** — PyPI's `mlx` ships none there, so the install
succeeds and `import mlx.core` fails with `ImportError: libmlx.so: cannot open
shared object file`. Pick one explicitly, and only one: `uv pip install -e
'.[mlx-cuda13]'` (CUDA 13; `.[mlx-cuda]` for a CUDA 12 toolkit) on an NVIDIA box,
`uv pip install -e '.[mlx-cpu]'` otherwise. Two backend extras in one environment
overwrite each other's `libmlx.so` silently. With a working backend the
`TFDPLayout` tests — which otherwise skip — will run against it. On Windows no
backend distribution exists, so those tests stay skipped. `tests/test_tfdp_backend.py` exercises each
individual MLX op the layout depends on, so a backend gap fails by name there
rather than as a mystery result inside the layout loop.

After modifying any framework file (anything under `polars2svg/`), reinstall
before running tests:

```bash
uv pip install -e .
```

## Running tests

```bash
.venv/bin/python -m pytest tests/
```

A few test groups need extra care:

- **Golden-image tests** (`test_*_golden.py`) compare a fresh render against a
  checked-in reference file. A *missing* golden fails the test — it is never
  written for you. When a golden is genuinely new, or a change is intentional,
  regenerate with:

  ```bash
  UPDATE_GOLDEN=1 .venv/bin/python -m pytest tests/
  ```

  Review the diff before committing regenerated goldens — an unreviewed
  `UPDATE_GOLDEN=1` run will silently rubber-stamp a regression as the new
  baseline. If a golden is missing because your checkout lacks it, restore the
  file rather than regenerating: a fresh baseline records whatever *your*
  machine produces, which is not necessarily what the golden was capturing. Adding or changing a golden test also means updating
  `notebooks/golden_images.ipynb` (shows every golden SVG side-by-side with a
  fresh render) so reviewers can see the visual diff without running pytest.

- **Performance baseline** (`tests/test_performance.py`) times each render
  component (3-iteration median) against a machine-local, gitignored
  `tests/perf_baseline.json`. Regenerate it after an intentional performance
  change, or when adding a new component (add its workload to
  `_make_workloads()` first):

  ```bash
  UPDATE_PERF_BASELINE=1 .venv/bin/python -m pytest tests/test_performance.py
  ```

- **Color-mode tests** — adding a new color enum, component color coverage, or
  cross-component consistency check means updating
  `notebooks/color_modes.ipynb` (shows all color mode combinations across
  components) alongside the test.

- A few t-field tests pull data via `kagglehub` and need network access /
  credentials; these are excluded in CI (see `.github/workflows/ci.yml`) and
  can be skipped locally with `-k 'not tfield'` if you don't have Kaggle set up.

### Test file conventions

Every `test_*.py` file must end with:

```python
if __name__ == '__main__':
    unittest.main()
```

Import `unittest` at the top of every test file, even if no
`unittest.TestCase` classes are used directly (some tests are pytest-style
functions) — this keeps every test file independently runnable as a script.

## Style conventions

The codebase does not run a full linter/formatter — CI only runs `ruff check`
with the minimal `E9,F` ruleset (syntax errors and pyflakes: undefined names,
unused imports/variables, duplicate dict keys). This is deliberate: the
project has a distinctive, consistent style that a stricter ruleset (import
sorting, line length, complexity) would fight rather than support. Match the
surrounding code rather than reflowing it to a generic style guide:

- **Aligned assignments** — consecutive related assignments/dict entries are
  column-aligned with extra spaces, not single-spaced (see any component
  `__init__` or kwargs table for examples).
- **`_underscore_`-wrapped locals** for internal/derived variables (e.g.
  `_row_count_`), reserving plain `snake_case` for parameters and public
  attributes.
- **Dynamic `setattr()`-bound enums** — most `*p=`/`*P=` constant tables
  (color modes, size modes, etc.) are bound onto instances rather than
  declared as class attributes. The binding stays dynamic, but every bound
  member is now *declared* in a class-level annotation block (see
  `polars2svg.py`, just above `__init__`) so type checkers can see it. Add a
  member to an enum and you must add it to that block —
  `tests/test_typing_surface.py` fails until you do.

## Type annotations

**New and modified functions must be fully annotated** — a return type and
every parameter (`self`/`cls` excepted). This applies to internals, not just
the public API.

This reversed a previous policy. The codebase was deliberately untyped
internally on the grounds that annotating it would be prohibitively large; a
2026-09-02 audit measured that instead, and found ~72% of what mypy reports
comes from a handful of undeclared dynamic attribute surfaces rather than from
missing signatures. Declaring those is cheap, so the annotation work is now
tractable and is proceeding module by module.

Two ratchets enforce it, and both only move one way:

- `[[tool.mypy.overrides]]` in `pyproject.toml` lists modules that are fully
  annotated and holds them to `disallow_untyped_defs` with every relaxed error
  code re-enabled. Add a module when you finish it; never remove one.
- `TestAnnotationCoverageRatchet` in `tests/test_typing_surface.py` caps the
  number of unannotated functions per module. Adding an untyped function fails
  the suite; annotating one means lowering that module's number in the same
  commit (a ceiling left above the real count also fails).

So: to annotate a module, type its functions, lower its ceiling to the new
count, and — if it reaches 0 — add it to the mypy override list. Expect the
promotion to surface latent findings; that is the point of it.

The migration itself is finished: every module except `interactive_controller.py`
is fully annotated, and that one is *permanently relaxed* by decision — it builds
its widget classes at runtime and lives behind the `interactive` extra, so there
is nothing static to check. Don't "finish" it. Its ceiling still applies, and new
code in it is still expected to be typed.

### Adding a component parameter

A parameter appears in **four** places, and the test suite checks all four
against each other, so a partial addition fails rather than shipping:

1. `_VALID_KWARGS` on the component class — what the constructor accepts at
   runtime (an unlisted name raises `TypeError`).
2. `_defaults_` in `__init__` — its default value.
3. The class-level declaration block — so `self.<param>` type-checks inside the
   component (the values are assigned by `setattr`, which no checker can follow).
4. The `<Component>Kwargs` TypedDict — so `p2s.xyp(..., <param>=...)` type-checks
   at the *call site*, and editors complete it.

Keep the TypedDict's value type conservative. Most parameters are data-drivable
(a literal *or* a column name *or* a `(field, enum)` spec) and are typed `Any` on
purpose: a too-narrow type here rejects valid user code, which is worse than no
type at all. The precise ones were each confirmed against how the test suite
actually calls them — do the same before tightening one.

The public API surface (`Polars2SVG.__init__`, the component factory methods,
`tField`, `panelize`, the exported layout classes) has always been typed and
must stay that way — `uvx mypy polars2svg` runs in CI. Never weaken a public
annotation to satisfy the checker.

One trap worth knowing: do **not** add `from __future__ import annotations` to
`polars2svg.py`. It stringifies annotations, and `test_typing_surface.py`
asserts on the evaluated objects (`__init__`'s return annotation must *be*
`None`, not `'None'`). The `>=3.12` floor means it is rarely needed anyway.

## Adding a new component

Following the repo's own `CLAUDE.md` rules when adding a new render component:

1. Add a perf workload for it in `tests/test_performance.py::_make_workloads()`
   and regenerate the baseline (`UPDATE_PERF_BASELINE=1 ...`).
2. Add golden-image test(s) and update `notebooks/golden_images.ipynb`.
3. If it introduces a new color enum or color-comparable behavior, add
   coverage to the color-mode tests and update `notebooks/color_modes.ipynb`.
4. If it needs a heavy optional dependency, guard the import the way
   `p2s_graph_mixin.py`/`chordp.py`/`xyp.py` do and add it to the right extra
   in `pyproject.toml` (`layouts`, `interactive`, `export`, or a new one) —
   see `tests/test_optional_dependency_extras.py` for the pattern used to
   verify a missing extra fails with a clear `ImportError` rather than a bare
   `ModuleNotFoundError`.

## CI

`.github/workflows/ci.yml` runs on every push to `main` and every PR:

- `mypy` on the public surface
- `bandit` (static security scan — see `SECURITY.md` for the project's
  threat model and how findings are annotated with `# nosec <code> - <reason>`
  rather than blanket-suppressed)
- `pip-audit` against the full resolved dependency graph
- `ruff check` (the `E9,F` ruleset described above)
- A Linux clean-room job: builds the wheel, installs it into a stock
  `python:3.13-slim` container, and runs the test suite against the installed
  wheel (excluding the machine-local perf baseline and network-dependent
  Kaggle t-field tests)

### Running CI's checks locally

The first two jobs are plain CLI invocations with no GitHub-specific context, so
they reproduce natively (~12s):

```bash
./tools/preflight.sh
```

That runs mypy, bandit, pip-audit and ruff exactly as `ci.yml` does, and reports
all four rather than stopping at the first failure. Run it before pushing —
`pytest` passing locally does **not** mean CI is green, since none of these four
checks are part of the test suite.

The Linux clean-room job is deliberately not covered: it exists to exercise
linux/amd64 inside a stock `python:3.13-slim` container, which is precisely what
a local macOS run cannot reproduce. Reproducing it needs Docker plus
[`act`](https://github.com/nektos/act), and each run recompiles pycairo from the
sdist inside the container — leave that one to CI.

`.github/workflows/release.yml` is separate — it only fires on a `v*` tag push
and publishes to PyPI via Trusted Publishing. Contributors don't need to touch
it; version bumps and tagging are a maintainer action.

## Pull requests

- Keep PRs focused — one fix or feature per PR.
- Update `CHANGELOG.md`'s `[Unreleased]` section for any user-visible change.
- Make sure `.venv/bin/python -m pytest tests/` passes locally before opening
  the PR (CI will also run it, but the golden-image and color-mode notebook
  updates are not enforced by CI and are easy to forget).
- Run `./tools/preflight.sh` too — the test suite does not cover mypy, bandit,
  pip-audit or ruff, so a green `pytest` can still land a red CI.
