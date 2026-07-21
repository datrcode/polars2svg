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

The `mlx` extra installs MLX with whatever backend suits the platform (Metal on
Apple silicon, CPU elsewhere). On a Linux + NVIDIA box, `uv pip install -e
'.[mlx-cuda]'` gets the CUDA backend instead, and the `TFDPLayout` tests — which
otherwise skip — will run against it. `tests/test_tfdp_backend.py` exercises each
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
  checked-in reference file. On first run, or when a change is intentional,
  regenerate with:

  ```bash
  UPDATE_GOLDEN=1 .venv/bin/python -m pytest tests/
  ```

  Review the diff before committing regenerated goldens — an unreviewed
  `UPDATE_GOLDEN=1` run will silently rubber-stamp a regression as the new
  baseline. Adding or changing a golden test also means updating
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
  declared as class attributes; this is intentional (see the `[tool.mypy]`
  comment in `pyproject.toml` for why the public surface is fully typed but
  these internals are not).
- Public API surface (`Polars2SVG.__init__`, the component factory methods,
  `tField`, `panelize`, the exported layout classes) should stay fully typed —
  `uvx mypy polars2svg` runs in CI and only checks these signatures
  (internals are exempted via `disable_error_code` in `pyproject.toml`).

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
