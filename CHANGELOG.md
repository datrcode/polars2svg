# Changelog

All notable changes to **polars2svg** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`TFDPLayout` now runs on NVIDIA GPUs via MLX's CUDA backend.** The layout was
  previously documented and packaged as Apple-silicon-only. MLX ships an official
  CUDA backend, and the eight t-FDP compute kernels are backend-agnostic
  `mlx.core` — so the same code runs on Metal or CUDA with **no change to the
  math**. New `[mlx-cuda]` / `[mlx-cuda13]` extras install it (Linux only; NVIDIA
  SM ≥ 7.5, driver ≥ 550.54.14, glibc ≥ 2.35).
- **`polars2svg.gpu_backend()`** — reports which backend `TFDPLayout` resolved to:
  `'metal'`, `'cuda'`, or `'cpu'`.
- **`tests/test_tfdp_backend.py`** — exercises each MLX op t-FDP depends on
  (broadcast all-pairs diff, gather, `.at[].add()` scatter-add with duplicate
  indices, keyed RNG) directly on the resolved device, so a backend gap fails by
  name. Plus GPU-vs-CPU cross-check, convergence, and finiteness tests in
  `test_tfdp_layout.py`.

- **Legends and colorbars: opt-in `legend=` on every rendered component**
  (`xyp`, `histop`, `timep`, `piep`, `linkp`, `chordp`, `spreadlinesp`;
  `smallp` panels inherit it from their template component). Layered value:
  `legend=True` (≡ `'right'`), a position string
  (`'right' | 'left' | 'top' | 'bottom'`), or a dict
  (`{'pos', 'title', 'fmt', 'max_items', 'order'}`). The legend **kind is
  auto-selected** from the resolved color mode — a categorical swatch list
  for `CSETp`/bare-categorical color, a colorbar for the spectrum modes —
  and the strip is reserved **from** `wxh` (the plot region shrinks; the
  physical output size is unchanged). A truthy `legend` with nothing to
  legend (flat/fixed color) silently renders nothing, so
  `set_defaults(legend=True)` is safe as a global default. Legends are
  recorded into the shared `DisplayList`, so SVG **and** WebGPU outputs both
  carry them, and the captured scale/category metadata is exposed as
  `component.legend_info` (a `polars2svg.LegendInfo`). v1 scope is the color
  encoding only (no size legends yet). The default `legend=False` renders
  byte-identically to previous output.

### Fixed

- **`pip install polars2svg[mlx]` was broken.** `tfdp_layout.py` imports
  `scipy`/`networkx`/`scikit-learn`, but the `mlx` extra did not chain `[layouts]`
  (as `interactive` does), so the extra resolved to an install whose only module
  could not import. `mlx` now chains `polars2svg[layouts]`.

### Changed

- `TFDPLayout` no longer hardcodes `device=mx.gpu`. It probes the GPU once and
  falls back to the CPU device with a one-time warning if none is usable — the
  plain Linux MLX wheel has no GPU backend, which would otherwise have failed at
  the first kernel.

- **Slimmed the core install; added `[layouts]` and `[interactive]` extras.**
  `pip install polars2svg` now pulls only `polars`, `numpy`, `pyarrow`,
  `pillow`, and `platformdirs` — enough for `xyp`, `histop`, `timep`, `piep`,
  `smallp`, and `spreadlinesp`. `networkx`, `scikit-learn`, `scipy`, `shapely`,
  and `squarify` move to the new `polars2svg[layouts]` extra (needed by
  `linkp`'s/`chordp`'s pluggable layouts, the graph-layout mixin, and
  shapely-typed `background=` shapes on `xyp`/`linkp`); `panel`,
  `jupyter_bokeh`, and `param` move to `polars2svg[interactive]` (which
  includes `layouts`, since the interactive graph views need it too).
  `polars2svg[all]` restores the previous "batteries included" install.
  Calling a component that needs a missing extra (`chordp()`, `panelize()`,
  `xypi()`, a graph-layout mixin method, `background=` shapes) now raises a
  clear `ImportError` naming the extra to install, rather than either
  succeeding silently or failing with a bare `ModuleNotFoundError` — this
  breaks nothing at runtime as long as the extra is installed, but is a
  **breaking change to the default install** if you were relying on
  `linkp`/`chordp`/interactive variants working out of the box with a bare
  `pip install polars2svg`.

## [0.1.0] — 2026-07-10

Initial public release: Polars-native DataFrame → SVG visualizations for Jupyter.
Pass a Polars `DataFrame` straight to a component and get a crisp, self-contained
SVG back. Every component also has a linked, interactive variant (built on Panel)
for brushing and cross-filtering in a notebook, plus optional WebGPU rendering for
large frames.

### Added

- **Component roster** — eight rendered components, each a method on a
  `Polars2SVG` instance:
  - `xyp` — scatter / distribution plot (numeric or categorical x/y).
  - `histop` — horizontal histogram bars, one per category/bin.
  - `timep` — temporal bar chart with linear or periodic (day-of-week, month, …)
    time modes.
  - `linkp` — node-link graph / network with pluggable layouts.
  - `chordp` — chord diagram of weighted flows around a circle.
  - `piep` — pie chart.
  - `spreadlinesp` — egocentric radial "spread" rings for influence/propagation.
  - `smallp` — small multiples: a grid of one template component faceted by a field.
- **Interactive, cross-linked variants** — `xypi`, `histopi`, `timepi`, `linkpi`,
  `smallpi` (and peers) sharing the static signatures, composed into a dashboard
  with `panelize(layout)` for brushing and cross-filtering in a notebook.
- **WebGPU rendering** — pass `use_webgpu=True` to render through WebGPU for large
  frames; a `webgpu()` / `gpuDisplayList()` path exists across all components.
- **Orthogonal `count=` / `color=` encoding** — a shared aggregation rule for
  `count=` (row count, sum, distinct-count, struct distinct-count) and a
  dtype-keyed `color=` inference (numeric → magnitude spectrum, otherwise
  categorical), with enums (`SCALARp`, `SETp`, `CSETp`, `CMAGNITUDE_SUMp`,
  `CROW_MAGNITUDEp`, `CROW_STRETCHEDp`, …) to pin intent explicitly.
- **T-fields** — `p2s.tField(column, enum)` for time-transformation fields
  (returns a frozen `TField` `str` subclass); the legacy `'column|suffix'` string
  form still works with a one-time deprecation warning.
- **Pluggable layout classes** exported from the package: `PolarsForceDirectedLayout`,
  `ConveyProximityLayout`, `LandmarkMDSLayout`, `PivotMDSLayout`, `TFDPLayout`
  (all satisfy the `LayoutAlgorithm` protocol via `.results()`).
- **Export API** — `save(path)` / `savePNG(path)` on every rendered component;
  SVG save has no extra dependency, PNG rasterization is behind the `[export]`
  extra (`svglib`, `reportlab`, `rlPyCairo`).
- **Typed package** — ships a `py.typed` marker with type hints on the public
  surface (constructor, component factories, `tField`, `panelize`, layout classes).
- **Exception hierarchy** — `Polars2SVGError` base with `InvalidSpecError` and
  `DataError` subclasses, all exported from the package.
- **`__version__`** attribute, read from installed package metadata.
- **Optional extras** — `[mlx]` (Apple-silicon MLX-accelerated force-directed
  layout) and `[export]` (PNG rasterization).
- **Diagnostic INFO logging** through the `polars2svg_logger` for dtype-keyed
  `count=`/`color=` inference choices (off by default).
- **SECURITY.md** documenting the SVG-injection threat model (row-data label text
  is HTML-escaped; component configuration is trusted).

[Unreleased]: https://github.com/datrcode/polars2svg/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/datrcode/polars2svg/releases/tag/v0.1.0
