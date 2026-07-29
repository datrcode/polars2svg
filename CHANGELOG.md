# Changelog

All notable changes to **polars2svg** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Interactive time-axis navigation on `xypi` and `timepi` — `u` / `e`
  keyboard shortcuts.** When the interactive view's time axis represents linear
  time, three shortcuts refilter the interaction stack against the base
  (bottom-of-stack) dataframe by time window. `u` unfilters every base row
  within the currently visible timeframe — the visible rows plus any that were
  filtered out of it — and is a no-op at the base of the stack. `e` expands the
  timeframe in **both** directions; `shift+e` expands backward (earlier events)
  only and `ctrl+e` forward (later events) only. Each expansion pulls in a chunk
  equal to `x_time_expand_perc` of the visible time span per direction and
  pushes the widened dataframe onto the stack (a no-op when there are no further
  rows to add). New **`x_time_expand_perc=` on `xyp` and `timep`** (default
  `0.1`) sets the expansion amount and is carried through `template=` clones.
  The shortcuts are inert on non-time / non-linear axes.

  The two components differ in how expansion snaps, matching their axis type:
  `xypi` has a continuous axis, so it grows by a **raw time slice** equal to
  `x_time_expand_perc` of the visible span. `timepi` is **binned**, so its
  visible timeframe spans the currently visible bins (start of the first to end
  of the last) and expansion snaps to **whole bins** — each side grows by
  `max(1, round(x_time_expand_perc × visible_bin_count))` bins, so a day/week/
  month view always gains whole day/week/month bars rather than a partial edge
  bar. `timepi`'s shortcuts are additionally inert on a **periodic** time axis.

## [0.1.2] — 2026-07-25

### Added

- **Force-directed origin-destination flow maps — `link_shape='flowmap'` on
  `linkp`.** An implementation of Jenny et al. (IJGIS 2017): each flow is drawn
  as a quadratic Bezier whose single control point is placed by an iterative
  equilibrium of five forces (flows-against-flow, nodes-against-flow,
  anti-torsion, spring, angular resolution), plus per-flow/canvas constraint
  rectangles, intersection reduction for flows sharing a node, and clearance
  around unconnected nodes. The layout runs **once** over all aggregated flows
  (`ODFlowLayout` in `polars2svg/od_flow_layout.py`; the paper's tuning
  constants are exposed as keyword arguments at their paper defaults) and is
  deterministic. Runtime is quadratic in the flow count — the method targets
  flow maps of roughly 100–200 aggregated flows, and `linkp` logs a warning
  above 200.
- **`link_arrows=` on `linkp`** — draw arrowheads at link destinations. Under
  `link_shape='flowmap'` the arrowheads are fed to the layout as obstacles
  (paper section 3.2.3), so curves route around them.
- **`time=` / `timing_marks_length=` on `linkp` — timing marks along each
  edge.** When a date/datetime column is supplied, every event is drawn as a
  short colored tick on its edge: position along the edge encodes when the
  event occurred (spectrum color), and side of the edge plus a slight slant
  encode the direction of activity. `time` accepts the same three forms as
  `timep`'s time field (a column-name str, a `TField`, or a `(field,
  TimeLinearTypeP|TimePeriodicTypeP)` tuple); unlike `timep`, `time=None`
  means the feature is off rather than auto-detect. Works under all three
  `link_shape`s (`line`, `curve`, `flowmap`), reusing the same curve/flowmap
  control-point math as link rendering so the two never drift. Interactively,
  the `a` key now cycles all four `link_arrows` × timing-marks combinations
  when exactly one date/datetime column is present on the data (or `time=`
  was set explicitly); with zero or several date columns it falls back to
  toggling arrows only, as before. SVG output only for now — the WebGPU
  display list does not yet emit timing marks.
- **`timing_marks_spacing=` on `linkp`** — the minimum on-screen spacing
  between timing marks, in pixels. It sets the decimation resolution: marks
  landing within `timing_marks_spacing` pixels of each other along an edge
  collapse into one (carrying the bin's mean time/color), so larger values
  render marks proportionally sparser — roughly one mark per that many pixels
  of edge. Defaults to `1.0` (the previous per-pixel behavior) and is clamped
  to `>= 1`, since sub-pixel marks are visually indistinguishable and would
  re-inflate the mark count the decimation exists to bound. Useful for keeping
  netflow-scale renders legible (and their SVG small) without dropping `time=`.
  Interactively, `shift-a` opens a pixel-grid spacing picker (1–32 px) that
  cycles forward and `ctrl-a` backward, mirroring the `shift-l`/`o`/`p` size
  pickers; the committed value re-renders the marks live.
- **MLX acceleration for the flow layout.** `ODFlowLayout`'s O(N²) force
  kernels run on NumPy by default and automatically move to the GPU in float32
  when the optional `[mlx]` extra is installed and a device is available.
  Output stays deterministic for a given machine/backend; the float32 path
  differs from the float64 NumPy path by far less than one pixel. Intersection
  reduction, obstacle clearance, and the per-flow scalar forces always run on
  NumPy/Python.
- **Spectral (Fiedler) seriation for categorical axes on `xyp`** —
  `x_order`/`y_order='spectral'` orders an axis by the Fiedler vector of a
  category × category affinity matrix, so similar categories land adjacent and
  block structure lines up along the axis. Tunable with `spectral_by` (the
  signal column(s) defining similarity; defaults to the opposite axis),
  `spectral_weight`, `spectral_similarity` (`'cosine' | 'linear' |
  'correlation'`), and `spectral_normalize`. Under small multiples, a
  `'spectral'` order on a **shared** axis (`SM_X`/`SM_Y` in `sm_shared`) is
  computed once over the full dataset and applied identically to every tile so
  panels stay comparable; an unshared axis is seriated per tile. The ordering is
  defined up to reflection, and a non-categorical axis raises `ValueError`.
- **MLX / CUDA availability indicator in the interactive stack control** — two
  header rows showing whether MLX and a CUDA device are usable, reusing the
  flow layout's cached GPU probe rather than resolving the device a second time.
- **Neighborhood-preserving circle packing — the `ncp pack` layout operation
  (`NCPLayout`, `[P]` in the shift-W layout picker).** An implementation of Li
  et al. (Computational Visual Media 2026): a compaction pass that takes the
  current layout, gives each node a circle sized by its flow volume, and packs
  the nodes into a tight, non-overlapping arrangement that preserves the
  layout's spatial neighbourhoods — reclaiming the empty space a spring layout
  leaves between nodes. Runs the paper's three-stage continuation pipeline
  (Delaunay planar-graph init → power-diagram compaction → force-directed
  refinement) on the currently visible graph; with a selection only the
  selected nodes are packed, and exactly-coincident nodes are collapsed to one
  circle beforehand like every other layout op. Node radius is `log` of the
  node's count (its total incident edge weight), falling back to the number of
  neighbours when the graph is unweighted. Deterministic; pure NumPy/SciPy
  (`ncp_layout.py`).
- **Interactive community detection in `linkpi` — the `d` key.** Runs Louvain
  community detection (`networkx`, resolution 1.0, fixed seed) over the graph at
  the current stack level and recolors the nodes one hue per community. Exactly
  coincident nodes are contracted first so a stacked group counts as a single
  member, and each community's color is hashed off its canonical (lexicographically
  smallest) member so re-running `d` keeps colors stable rather than reshuffling
  them with louvain's ordering. The new colors are pushed across every stack
  level; `shift-d` restores the node coloring the `LinkP` was created with. An
  algorithm failure leaves the view untouched rather than killing the callback.
- **`websocket_max_message_size=` on `panelize()`.** The Bokeh WebSocket message
  limit (in bytes) you intend to serve the dashboard under. `panelize()` measures
  the composed SVG document and, when it would exceed the limit, logs a warning
  naming the measured size — a large `linkp` (e.g. timing marks over a
  netflow-sized frame) can push the document past Bokeh's 20 MB default and make
  the browser fail with "Unexpected end of JSON input". Pass the same value to
  your `show()`/serve call to actually raise the limit (and silence the warning).
  Defaults to `None` (no check), so existing calls are unaffected.

### Fixed

- **Interactive `linkpi` state now stays consistent across the whole dataframe
  stack.** Layout operations (`w`), even-out-distribution (`E`), and layouts
  loaded from a file now propagate their node positions to every level of the
  stack, and background / label / node-color state is written down onto every
  layer — including `dfs_layout[0]`, the template that pushed layers clone from.
  Previously these applied only to the currently active level, so pushing or
  popping the stack could surface a stale layout, background, or node coloring.
- **`shift-Q` (select common neighbors) did not reach the other components.**
  `linkpi` assigned `selected_entities` directly instead of going through
  `setSelectedEntitiesAndNotifyOthers()`, so the new selection never
  cross-filtered the rest of the dashboard. An empty intersection now clears the
  selection instead of leaving the previous one in place.
- **Browser shortcuts stole `linkpi` keys.** `ctrl-c`, `ctrl-shift-C`,
  `ctrl-e`, `ctrl-s`, and `ctrl-shift-S` now call `preventDefault()`, so the
  native copy/search-bar/Save-Page-As actions can no longer clobber the
  component's own clipboard write and label-mode operations. Verified on macOS;
  conflicts on Windows and Linux browsers are not yet fully resolved.
- **Untrusted label text could break out of an SVG `id` in `xyp` line
  rendering.** The `line_by` value was interpolated into `id="..."` and its
  matching `url(#...)` reference unescaped. It is now passed through an
  allowlist (every character outside `[A-Za-z0-9]` becomes `-`), so XML-special
  characters cannot escape the attribute; per-row uniqueness is unaffected
  because it comes from a separate row-index suffix.
- **`spreadlinesp` node and label text was not HTML-escaped** on the way into
  `<text>` elements, unlike the other components. Focal-node, cloud, and bin
  labels are now escaped, matching the threat model in `SECURITY.md`.
- **The stack control mis-measured its available height** once the MLX/CUDA
  status rows were added, and label centering was off.

### Changed

- **Stack control rendering reworked** — new layout pass for the rows/index
  readout with corrected label centering, and the indicator font is capped at
  12px so it no longer scales past its row.
- **Internal refactors, no behavior change.** Background rendering shared by
  `xyp`/`linkp`/`chordp` moved into `p2s_background_mixin`; node/edge color
  resolution shared by `linkp`/`chordp` into `p2s_component_color_mixin`; and
  the color logic shared by `histop`/`timep` into `p2s_bin_component_mixin`
  (~1,100 lines of duplication removed).
- The `[mlx]` extra now serves `ODFlowLayout` in addition to `TFDPLayout`.
- **`linkp`'s `use_pos_for_bounds` now defaults to `False`** (was `True`). By
  default the view bounds fit only the nodes present in the dataframe at each
  stack level, instead of stretching to every key in `pos` (including nodes not
  drawn at that level). Set it back to `True` to include all `pos` keys in the
  bounds; when `True` it still overrides `SM_X` / `SM_Y` under small multiples.

## [0.1.1] — 2026-07-16

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

- **Labels containing dotted numbers were silently corrupted in rendered SVG.**
  `roundSvgFloats()` trimmed float precision by regex over the finished SVG
  string, so it matched any digit-dot-digit run *anywhere* — including inside
  `<text>`/`<tspan>` element content, not just numeric attribute values. Any
  label that merely looked like a float was rewritten: the node label
  `1.172.32.1` rendered as `1.17.32.1`. The pass is now disabled pending a
  rewrite that only touches attribute-value floats; `tests/test_svg_float_precision.py`
  guards the behavior. Coordinates are now emitted at full precision, so SVG
  output is byte-different from 0.1.0 (golden images updated) and somewhat
  larger — rendering is unchanged apart from the corrected labels.

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

[Unreleased]: https://github.com/datrcode/polars2svg/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/datrcode/polars2svg/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/datrcode/polars2svg/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/datrcode/polars2svg/releases/tag/v0.1.0
