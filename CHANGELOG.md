# Changelog

All notable changes to **polars2svg** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`draw_link_labels=` on `linkp`** — label each drawn edge with the third element
  of its relationship tuple (`[('fm','to','predicate')]`), or, for a two-part tuple,
  with the field driving `color=` (rtsvg `rt_linknode_mixin.py:1413-1425`
  parity, where that fallback was `color_by`). An edge whose rows carry more than
  one value is labeled `*` rather than left blank, so a collision never reads as
  an edge with no data. A label whose field is also the link color field takes the
  link's color; otherwise it uses the default label foreground. Text is cropped to
  the edge length and dropped entirely when not even one character fits.
  - **`link_labels=`** is the link-side counterpart of `node_labels=`: a
    `{label_value: display_str}` dict that renames values for display, and — like
    `node_labels=` — leaves an edge unlabeled when its value is absent from the
    dict. `'*'` is a value like any other, so a collision marker can be renamed too.
  - **`label_only=` now gates both channels** from one set of names: a node by its
    node name, an edge by its label value. A `*` edge survives when any of the
    values behind it is in the set (rtsvg `rt_linknode_mixin.py:1419-1422`). Both
    tests run against the raw value, before `node_labels=`/`link_labels=` rename it
    for display. Consequence worth knowing: a `label_only=` naming only nodes now
    silences every edge label.
  - **Bidirectional edges are labeled twice**, once per direction, on opposite
    sides of the edge: both directions are canonicalized onto one baseline
    (ordered by node name), so `a→b` and `b→a` never overprint.
  - **Clearance is measured to the ink, not the baseline.** A label whose glyphs
    grow away from its edge is offset by its descent; one whose glyphs grow back
    over the edge is offset by its ascent — so both sides of a pair sit the same
    distance off the edge whatever the strings are. Ascent/descent come from a
    character-class approximation (`_labelInk_`), because the emitted markup names
    no font-family and so carries no real metrics. Measured against WebKit, every
    string shape now clears by 2.5px ±0.3 at `txt_h=12`; a single flat offset put
    x-height-only labels (`cow`) ~3px too far out and let a descender (`dog`)
    touch the edge on the other side.
  - **`link_shape='line'`** rotates the text onto the chord and
    **`link_shape='curve'`** runs it along the drawn Bezier via an SVG
    `<textPath>` (one invisible per-edge path in `<defs>`, id-scoped per
    instance). Rotations stay within ±90° so text is never upside down.
    **`'flowmap'` is not labeled** — its edges are force-routed around each other
    and text threaded along them fights the routing; setting `draw_link_labels=`
    with that shape warns.
  - The **WebGPU** path draws the same labels as rotated glyph runs; having no
    text-on-path primitive, it approximates the curve case as a straight run along
    the curve's midpoint tangent (same anchor, same side).
  - Independent of `draw_node_labels=`. The third tuple element is validated only
    when `draw_link_labels=` is on, so renders that carry an unused third element
    keep working.

- **`linkpi`'s label cycle (`ctrl-shift-s`) covers the link channel.** The walk is
  now `none → node → node+link → link → sticky → none`, so edge labels can be
  brought up on their own or alongside node labels. The two link states are only
  offered when the graph has something to put in them — some relationship naming a
  label field (a third tuple element, or a `color=` field for the two-part
  fallback) on a `line`/`curve` shape; without one the cycle stays the three-state
  walk it was before. `LinkP.linkLabelsAvailable()` is the check, and
  `labelModeCycle()` / `nextLabelMode()` expose the walk. The sticky state keeps
  link labels off: sticky holds selected *nodes*, and `label_only=` now gates both
  channels off the same names, so leaving them on would filter every edge label
  away. A mode that becomes unreachable (link shape switched to `flowmap`) restarts
  the cycle rather than sticking.

- **`linkpi` layout mode `circle (color)` (shift-G → `C`).** A color-grouped
  counterpart to `circle`, in the same spirit as `grid (color)` is to `grid`. It
  uses the identical circular drag shape (press-point = center, drag distance =
  radius), but hands out the circumference slots one color at a time, so every
  node color owns a single contiguous arc, with an extra slot-width of empty
  space at each color boundary to make the switch visible. Arcs are ordered — and
  the whole ring rotated — by the direction of each color's off-circle neighbors,
  and nodes within an arc are ordered the same way, so a color lands near the
  nodes it connects to. When the grouping carries no information (every node one
  color / uncolored, or every node its own color) it falls back to the plain
  `circle` layout. Backed by the new `Polars2SVG.circularNodeColorLayout()`.

- **Stack control widget — `c` / `ctrl+shift+c` collapse shortcuts and an `h`
  help overlay.** Two keyboard shortcuts prune the interaction stack directly
  from the stack control widget. `c` *collapses* the stack to just the base and
  the currently visible frame, leaving `[base, current]` (or just `[base]` when
  the base itself is the visible frame). `ctrl+shift+c` *rebases* — it discards
  the entire stack and makes the visible dataframe the new base, leaving
  `[current]` (routed through `replaceStack`, so peers that track a base
  dataframe — e.g. the graph view — reset accordingly). Both ops propagate to
  every view sharing the stack. A new `h` **help overlay** lists the shortcuts in
  the same translucent-box style as the other interactive components' help
  menus, and the widget now takes keyboard focus on hover so it receives keys.

- **Interactive structure navigation on `linkpi` — `f` / `F` keyboard
  shortcuts.** Two graph-structure counterparts to the `xypi`/`timepi` time
  shortcuts: instead of a time window, they recover rows from the base
  (bottom-of-stack) dataframe by graph adjacency and **add them on top of the
  current view**. `f` *edge-unfilters* — for the currently-visible edges it adds
  back every base row lying on those edges (restoring rows thinned by
  color/brush/other filters). `shift+f` *node-expands* — for the
  currently-visible nodes it adds back every base row incident to them (source or
  destination), pulling previously-filtered neighbors and their edges in. Both
  keep the entire current view; a targeted edge is refilled to its full base set
  while every other visible row is preserved. When nodes are selected, the
  operation is scoped to the selected subgraph — `f` refills only the edges among
  the selected nodes and `F` only the rows incident to them, leaving the rest of
  the view untouched. Each op mines the pristine base dataframe, pushes onto the
  stack (undoable via `X`), and is a no-op when nothing new can be added.

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

### Fixed

- **`hyperTreeLayout` drew crossing edges.** The radial layout gave each node an
  angular sector sized by its leaf count and placed the node at the sector's
  midpoint, but never constrained where in that sector the children could go.
  When a sector was wide, the straight segment from a node down to a child on
  the far side of it dipped back inside the node's own circle and cut through a
  sibling subtree. The layout now also clips the children's range to the tangent
  cone of the node's circle, `|θ − θ_node| ≤ acos(r_depth / r_depth+1)` — the
  *annulus wedge* constraint from Eades' "Drawing free trees" (1992) — which
  pins every parent-to-child segment inside both its annulus band and its own
  sector, so no two edges of the drawn tree can cross. Separately, components of
  four or fewer nodes were placed at hard-coded square corners, which drew a
  4-node path as a crossed `X`; they now go through the same radial algorithm as
  every other component. The guarantee covers the spanning tree the layout
  draws; edges of the input graph outside that tree (any cycle) are not placed
  by the layout and may still cross. `hyperTreeDonutLayout` is unchanged — its
  leaves are packed into a 2-D ring band by design, so its leaf edges cross
  regardless of the sector math.

- **Interactive render caches could serve a stale render after an `id()`
  collision.** Each interactive view memoizes rendered frames in a cache keyed on
  `id(df)`, but a dropped frame's id can be reused by a later dataframe, so a
  cache hit was not guaranteed to belong to the requested frame — occasionally
  painting a since-freed frame's render. Cache entries now carry their source
  dataframe and every hit is identity-checked, re-rendering on a mismatch.
  Affects the stack control widget, the generic interactive views
  (`xypi` / `timepi` / `histopi` / `chordpi` / `piepi`), `smallpi`, and
  `spreadlinepi`.

### Changed

- **BREAKING — `linkp`'s `draw_labels=` is now `draw_node_labels=`.** With edges
  labelable too (see `draw_link_labels=` above), one `draw_labels=` could no
  longer say which channel it meant. `linkp` gains the matched pair
  `draw_node_labels=` / `node_labels=` and `draw_link_labels=` / `link_labels=`.
  The rename is **linkp-only** — `chordp`, `histop`, `piep`, `smallp` and the
  udist tiles each have exactly one kind of entity to label and keep plain
  `draw_labels=`. Passing `draw_labels=` to `linkp` raises `TypeError` naming the
  replacement rather than being silently ignored, matching how `node_shape=` /
  `draw_context=` were retired from the component. The interactive setter
  `LinkP.drawLabels()` becomes `drawNodeLabels()`, joined by `drawLinkLabels()`.

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
