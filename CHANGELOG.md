# Changelog

All notable changes to **polars2svg** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Typed keyword arguments: `p2s.xyp(...)` and friends now catch a misspelled
  parameter before the call runs.** Every component takes `**kwargs` validated at
  runtime against `_VALID_KWARGS`, which meant a downstream user got a correct
  return type and *nothing at all* for the ~30 parameters that actually matter --
  no completion, no checking. Each factory method is now typed
  `**kwargs: Unpack[<Component>Kwargs]` ([PEP 692], available on the `>=3.12`
  floor), covering **269 parameters across the nine components**:

  ```python
  p2s.xyp(df, x='a', y='b', dot_sze=6)
  # error: Unexpected keyword argument "dot_sze" for "xyp"; did you mean "dot_size"?
  ```

  The nine TypedDicts (`XYpKwargs`, `LinkPKwargs`, `ChPKwargs`, `HistopKwargs`,
  `TimepKwargs`, `SpreadLinesPKwargs`, `PiepKwargs`, `SmallpKwargs`, `TileKwargs`)
  are exported from the package root so callers can annotate their own dicts:

  ```python
  opts: p2s.XYpKwargs = {'dot_size': 6, 'wxh': (400, 300)}
  p2s.xyp(df, x='a', y='b', **opts)
  ```

  `ChPKwargs` sits behind the same optional-extra guard as `ChP` itself, since
  chordp imports scipy at module level.

  **Value types are evidence-based, not guessed.** 77 of the 269 carry a precise
  type; the rest are `Any` because they are data-drivable (a literal *or* a column
  name *or* a `(field, enum)` spec) and a narrower guess would reject valid calls.
  Each precise type was confirmed against how the test suite actually calls it,
  which vetoed 15 that a reading of the defaults would have gotten wrong -- most
  usefully `legend` (defaults to `False`, but also takes `'top'` and a dict) and
  `wxh` (defaults to a tuple, but also takes a list, and an int on histop/smallp).
  Positional `*args` stays `Any`: it dispatches on argument *type*, not position.

  A parameter now appears in four places -- `_VALID_KWARGS`, `_defaults_`, the
  class declaration block, and the `Kwargs` TypedDict -- and all four are
  test-enforced against each other, so drift fails the suite rather than shipping.

  [PEP 692]: https://peps.python.org/pep-0692/

- **`tile(svg_list, ...)`** — composes already-rendered SVGs into one document: any
  component, any SVG string, or another `tile()`, mixed freely. It is the only method on
  `Polars2SVG` that takes renderings rather than a DataFrame, and it is the counterpart to
  `smallp` — small multiples faceting *one* template over a field, `tile` arranging
  *different* components side by side. Ported from rtsvg, where the strip (`tile`), its
  `horz=False` column and the grid (`table`) took two methods and two parameters; here
  `per_row=` alone says which one you get.

  | Parameter | Meaning |
  |---|---|
  | `per_row=None` *(default)* | a single row — every tile side by side |
  | `per_row=<n>` | a grid, filled row by row, `n` tiles per row (last row may be short) |
  | `per_row=1` | a single column — every tile stacked |
  | `spacer=0` *(default)* | that much separation both horizontally and vertically |
  | `spacer=(h, v)` | separate horizontal / vertical gaps; a one-row layout has no vertical gaps to draw, and `per_row=1` no horizontal ones |
  | `wxh=None` *(default)* | the canvas is exactly as large as the tiling |
  | `wxh=(w, h)` | scale the whole tiling to fit that canvas — uniformly and centered, so nothing is cropped or distorted; one side may be `None` to follow the aspect ratio |
  | `bg_color=None` *(default)* | canvas color (the framework background); shows through the spacer gaps and any viewport margin |

  Tiles are top-aligned within their row and rows are left-aligned, so a row is as tall as
  its tallest tile and the canvas as wide as its widest row. Sizes are read from each
  child's root `<svg>` tag, so a component whose rendered size differs from the `wxh` it
  was asked for (`smallp`) still measures correctly. The rendered size is `.wxh_actual`;
  the unscaled size of the tiling itself is `.content_wxh`.

  The `wxh=` viewport is emitted as a `translate`/`scale` transform rather than a nested
  `viewBox`: svglib — the rasterizer behind `save('.png')` — does not honor a `viewBox` on
  a nested `<svg>`, and scaled each axis independently, so a tiling that looked right in a
  browser exported distorted and clipped.

- **`xyp(aspect=...)`** — locks the ratio between the two axis scales, for data whose shape
  does not match the width/height ratio of the component. Geospatial (longitude/latitude)
  scatters were the motivating case: with each axis independently stretched to fill its own
  pixel extent, a map on a component that is not shaped like its extent comes out distorted.

  | Value | Meaning |
  |---|---|
  | `None` (default) | unchanged — each axis independently fills its own pixel extent |
  | `'equal'` | one world unit is the same length on both axes |
  | `'geo'` | x is degrees longitude, y is degrees latitude — a degree of longitude is given `cos(latitude)` as many pixels |
  | *number* | pixels-per-y-unit / pixels-per-x-unit, matching matplotlib's `set_aspect()` |

  The ratio is pinned by **widening** the over-magnified axis about its center, so the plot
  gains empty margin rather than losing data. `x_range`/`y_range` become a floor for the
  visible window rather than a ceiling, and rows that the widening brings into view are
  drawn instead of leaving a band that is inside the frame but empty by construction.

  `'geo'` is a plate carrée centered on the data, not a reprojection: it is the usual
  good-enough correction for a regional extent and will not stay true across a latitude span
  wide enough for the curvature to matter.

  Requires numeric x and y axes — a categorical axis draws its grid lines from the category
  count and the temporal renderers work in world-unit datetimes, so neither would follow a
  widened window; both raise rather than render a plot whose dots and grid lines disagree.

- **README section on the fabricated `display_svg` snippet.** Search results and
  AI-generated answers circulate a `from polars2svg import display_svg` /
  `display_svg(df)` sample that has never been API in any released version. The name
  looks borrowed from `IPython.display.display_svg`, which takes SVG data rather than a
  DataFrame, and its premise — render a DataFrame as a table — has no entry point to
  map onto, since every component encodes named fields into a chart. The Quickstart now
  names the snippet, says plainly that it does not work and why, and points at
  `Polars2SVG()` plus a component call, so there is correct indexable text competing
  with the invented version.

### Changed

- **The typing migration is finished: strict mypy is the package-wide floor and no
  error code is disabled.** The strict-promotion backlog (PLANNING.md §11) is empty —
  the last **11 modules and 67 errors** are cleared — so `[tool.mypy]` drops the
  seven-code `disable_error_code` list, moves `disallow_untyped_defs` /
  `disallow_incomplete_defs` / `check_untyped_defs` up into the global block, and
  reduces `[[tool.mypy.overrides]]` to the single `interactive_controller` exemption.
  `uvx mypy polars2svg` now means what it says. It previously passed with
  `attr-defined`, `var-annotated`, `assignment`, `has-type`, `misc`, `dict-item` and
  `return-value` switched off for every unpromoted module — the categories that catch
  a typo'd attribute, a wrong assignment and a wrong return — which is a green that
  reports on the checks that were left running rather than on the code.

  Two thirds of the 67 were mechanical: empty containers that needed an annotation
  (`X = {}` → `X: dict = {}`, 19 of them) and int-inferred locals that later hold a
  float (14). The rest were per-site, and four are worth reading as findings rather
  than as config work:

  - **`xyp.__toScreen__()` returns `None` on a zero-width axis range, and not every
    caller knew.** The linear renderer's `__line__` already guarded it
    (`if _offset_ is None: return`); the periodic-time renderer's did not, and its
    `__distanceBetweenLines__` computed `abs(None - None)`. Both degenerate paths
    raised `TypeError`. They now skip the line and report zero spacing respectively,
    the latter falling through to the coarsest tick granularity.
  - **`chordp._kmeans_()` initialised its cluster accumulator to `{}` and replaced it
    with a list of lists on the first iteration.** At `iterations <= 0` the trailing
    `assign[i]` indexed the dict and raised `KeyError`; it is now seeded with the
    empty clusters it means and typed `list`.
  - **`spreadlinesp`'s connector shim forwarded `*args, **kwargs` into a call that
    already passes `dl=`.** All three call sites pass four positional coordinates, so
    nothing collided at runtime, but the signature admitted it could. `_connect_` now
    takes the four coordinates it actually accepts.
  - **`isHexColor()` is now a `TypeGuard[str]`.** `isinstance(x, HexColorString)` is a
    metaclass predicate over strings, not a base-class test, so a checker narrowed the
    value to `HexColorString` and then refused to return it where `str` was declared.
    The two `spreadlinesp` sites that needed narrowing call the predicate directly;
    the other 24 `isinstance(…, HexColorString)` sites are unchanged and can adopt it
    as they need to.

  Three `p2s_graph_mixin` layouts that return `(pos, extra)` when asked
  (`ipSubnetTreeMapLayout`, `ipSubnetForceDirectedLayout`, `neighborhoodLayout`) are
  annotated `-> dict | tuple`, matching `hyperTreeDonutLayout`, which already was; and
  six helpers that could return `None` (`__shapelyToSVGPath__` on xyp and linkp,
  `linkp.nodeColor`, `piep.__fitLabel__`, `xyp.__timeColumn__`,
  `p2s_geometry_mixin`'s two intersection helpers) say so. `TestRatchetConfigConsistency`
  was rewritten for the inverted config — it now asserts that strict checking is
  global, that nothing is disabled tree-wide, and that the relaxed set is exactly
  `PERMANENTLY_RELAXED` — and `TestAnnotationSanity`'s Optional-return check learned
  two more real None paths (a one-argument `.get()`, and `best = None` … `return best`)
  that it had been reporting as annotation artifacts.

- **`render_with()` and `legendInfoColorbar()` now return their real types, not `Any`.**
  All seven components' `render_with()` returns a new instance of that component and was
  annotated `-> Any`, which silently switched checking off for everything downstream of
  `p2s.xyp(df, ...).render_with(df2)`. Now `-> 'XYp'` and so on, so a typo on the result is
  caught at the call site. `P2SLegendMixin.legendInfoColorbar()` was the same shape and now
  returns `LegendInfo`.

- **Typed 14 component parameters from their own documented vocabularies.** The factory
  docstrings already spell out what several parameters accept; the `Kwargs` TypedDicts said
  `Any` anyway. Applied where the documentation and the suite's actual usage agree:

  | Parameter | Was | Now |
  |---|---|---|
  | `background_label_color`, `background_stroke` (xyp, linkp) | `Any` | `str \| dict \| None` |
  | `background_opacity`, `background_stroke_w` (xyp, linkp) | `Any` | `float \| dict \| None` |
  | `spectral_similarity` (xyp), `link_shape` (linkp), `skeleton_algorithm` (chordp) | `Any` | `str` |
  | `link_size` (linkp) | `Any` | `str \| float \| None` |
  | `bundle_rings` (chordp) | `Any` | `int` |
  | `donut_ratio` (piep) | `Any` | `float` |

  Every documented form was checked to still pass — and to still *run*; three that a reading
  of the docstring alone would have gotten wrong were caught by that check:

  - **`background_stroke_w` — the two documents disagree.** The class docstring says
    `1.0 | {key: float, ...}`; the `_defaults_` comment says `None / number / dict`. The
    union (`float | dict | None`) is what shipped.
  - **`smallp(sm_template=...)` — the docstring is stale.** It lists `XYp | Timep | Histop`,
    but *seven* components implement `renderSmallMultiples()`. Typing it as documented would
    have rejected working code, so it stays `Any`.
  - **`histop(bin_by=)` / `timep(time=)` were false positives** — the "vocabulary" is prose
    containing a slash ("can be specified as a string / i.e., not a keyword argument").

  Bare `Any` is now 799 of 3,906 annotations (20.5%), from 967 (24.8%) when the review
  started. What remains is either correct (mixin host attributes, `*args`/`**kwargs`, the
  genuinely data-drivable `color=`/`count=`/`dot_size=`) or blocked behind guard work
  recorded in `PLANNING.md`.

- **Narrowed 114 component attribute declarations from `Any` to their real types.**
  A review of the finished migration flagged the `Any` count — 967 bare `Any`, 24.8% of
  all annotations — as higher than it should be. Most of the component class declarations
  turned out to be lazy rather than principled: they were generated in bulk keyed on
  "initialised to `None`", and the whole batch got one type.

  Instrumenting the suite to record what is actually assigned gave a clean single type
  (plus `None`) for most of them — `df_orig: pl.DataFrame | None`,
  `template: 'XYp | None'`, `legend_info: LegendInfo | None`,
  `_gpu_dl_: DisplayList | None`, `_gpu_payload_: dict | None`. 114 were applied, keeping
  only those that cost **zero** new checker errors. Bare `Any` is down to 823 (21.1%).

  It also **removed four errors from the strict-promotion backlog** (PLANNING.md §11,
  71 → 67) and cleared one module out of it entirely: once the components declared
  `_color_stat_min_`/`_max_` and `_legend_stat_min_`/`_max_` as `float | None`, the
  conflict with `P2SComponentColorMixin` — which had been *inferring* `float` for them
  from an assignment while its own guards test them against `None` — became visible. They
  are now declared in the mixin that owns the concept.

  What deliberately stayed `Any`: **`df` on every component**, where `pl.DataFrame | None`
  is the true type but produced 137 errors on its own (every unguarded `self.df.…` in the
  render paths — the §11 T4 cascade in concentrated form); `Piep._fixed_color_`, which is
  assigned a `HexColorString` that no checker can see as a `str` until the T5 virtual-class
  item is fixed; and the genuinely data-drivable parameters (`color=`, `count=`,
  `dot_size=`), which accept a literal *or* a column name *or* a `(field, enum)` tuple by
  design.

- **Phase 3 is complete: wave E, and the whole package is annotated.**
  `spreadlinepi`, `stack_control` and `p2s_interactive_mixin` — 53 functions, all
  three promoted to the strict ratchet. **43 of 44 modules are at zero unannotated
  functions**, and every one of them is strictly checked: the ratchet now covers
  29 modules, up from 3 when the migration started.

  Overall coverage is **86.5%** (926 of 1,071 functions) — **100% of everything
  outside `interactive_controller.py`**, which is now *permanently relaxed* by
  decision rather than left pending. It builds its widget classes at runtime with
  `type('LINKPI', (ReactiveHTML,), {...})` and keeps its state in Panel/param
  descriptors, so there is nothing static to check; it also sits behind the
  `interactive` extra and is invisible to the public typed surface. Annotating it
  would mean a wall of `Any` and `type: ignore` for no checkable benefit. Its
  ceiling still applies, so it cannot get worse, and new code in it is still
  expected to be typed. `TestPermanentlyRelaxedModule` keeps that decision
  bounded and visible, and asserts every *other* module reached zero.

  The migration's arc, for the record: 4.9% annotated and 2,423 errors under an
  exploratory `check_untyped_defs` run at the start; 88 errors now, none of them
  `attr-defined`.

- **Phase 3 wave D: the eight render components.** `xyp`, `linkp`, `chordp`,
  `spreadlinesp`, `piep`, `timep`, `histop`, `smallp` — 396 functions, the largest
  wave and the one the golden-image tests cover. Annotation coverage is now
  **81.5%** (873 of 1,071 functions) with **40 of 44 modules at zero**; `histop`,
  `timep` and `smallp` joined the strict ratchet (26 modules).

  Most of wave D's 203 initial errors traced to one root, and it was the same one
  phase 1 solved for `setattr`-assigned parameters: **120 attributes across the
  eight components are initialised to `None` in `__init__` and filled during
  render**, so a checker infers `... | None` and flags every later use. Declaring
  them cleared 152 errors. Among them were `SpreadLinesP`'s `vx0/vy0/vx1/vy1`,
  the unguarded-`None` arithmetic left open since phase 1.

  `xyp`, `linkp`, `chordp`, `piep` and `spreadlinesp` are fully annotated but not
  yet promoted — 19 strict errors between them, mostly return types that need
  widening plus the caller guards widening then demands. That work is deliberately
  left rather than half-applied: it changes `None`-handling in the render paths.

- **Phase 3 wave C: the core class and the layout algorithms.** `polars2svg.py`,
  `ncp_layout`, `flow_field_background`, `od_flow_layout` and
  `udist_scatterplots_via_sectors_tile_opt` — 163 functions. All five were
  promoted to the strict ratchet, which now covers **23 of 44 modules**; overall
  annotation coverage is **44.5%** (477 of 1,071 functions), with 32 modules at
  zero unannotated.

  Wave C exposed a third blind spot in trace-seeded annotation, and this one had
  already shipped in waves A and B: **a parameter the tests only ever pass `None`
  to comes back annotated `None`**, which is not a type — nothing but `None` can
  be passed. 29 parameters across nine modules were affected. All are now `Any`,
  and a fourth invariant in `TestAnnotationSanity` rejects the annotation outright.

- **Phase 3 (waves A and B) of the typing migration: 259 functions annotated,
  annotation coverage 4.9% -> 29.3%.** The twelve leaf modules (wave A, 61
  functions) and the ten mixins (wave B, 198) are now fully annotated -- 27 of
  44 modules are at zero unannotated functions, and the strict `[[tool.mypy.overrides]]`
  ratchet grew from 3 modules to 18.

  Annotations were **seeded from a runtime trace of the test suite** rather than
  guessed: a profile hook recorded the actual argument and return types of every
  call into the target modules across all 3,108 tests. That is evidence, not
  inference, but it has two blind spots, and both produced wrong annotations that
  a type checker could not have caught:

  - **It never sees a branch the tests do not take.** `optimalInscriptionCircle()`
    and `__approximateInscribedCircle__()` came back as `-> None` while both
    actually return tuples.
  - **`sys.setprofile` reports a return of `None` when a call leaves via an
    exception**, so a function whose failure paths all `raise` looks Optional.
    `_svgRootWxh_()` came back `-> tuple | None` though every failure path raises.

  Three new whole-package invariants in `tests/test_typing_surface.py::TestAnnotationSanity`
  close both blind spots permanently, checking the annotations against the AST
  rather than against usage: a parameter defaulting to `None` must be Optional; a
  function returning Optional must have a path that yields `None`; a function
  annotated `-> None` must not return a value. The first caught 11 wrong
  annotations (8 of them pre-existing, from before this migration).

- **Phase 1 of the typing migration: the whole dynamic attribute surface is now
  declared, and `attr-defined` is at zero.** Under an exploratory
  `check_untyped_defs` run the package reported 2,423 errors, 1,825 of them
  "has no attribute" from just 261 distinct names. Those came from three
  setattr()-driven surfaces, all now carrying class-level declaration blocks:

  | Surface | Declarations | Errors cleared |
  |---|---:|---:|
  | `Polars2SVG` enum members (phase 1a) | 92 | 636 |
  | Component parameters from `_defaults_` (phase 1b) | 197 | 907 |
  | Mixin host attributes + `__slots__`/`__dict__` records (phase 1c) | 111 | 324 |

  2,423 -> 556 errors (-77%), with **no `attr-defined` left anywhere**. Every
  block is bare annotations, so nothing exists at runtime and no instance state
  is shadowed -- a test asserts that for each one.

  Component parameter types are deliberately conservative: `Any` wherever a
  parameter is data-drivable (accepts a literal *or* a column name), which the
  generator detected by looking for `isinstance(self.X, ...)` dispatch in the
  code. Mixin host attributes are `Any` throughout, since a mixin genuinely does
  not know its host's concrete types. Precise caller-facing parameter types are
  phase 2's `Unpack[TypedDict]` work, which is where they belong.

  New guards in `tests/test_typing_surface.py`: `TestComponentAttrDeclarations`
  closes the triangle with the existing `assertParamSpecMatches()` runtime check
  (`_VALID_KWARGS` == `_defaults_` == declarations), and
  `TestEnumMemberDeclarations` does the same for the enum block.

- **Type annotations are now required on new code, reversing the previous policy.**
  `pyproject.toml` used to state that the internals were dynamically typed on purpose
  and that annotating them would be "boiling the ocean for no downstream benefit". An
  audit measured that claim: 4.9% of the package's 1,071 functions were fully annotated,
  but ~72% of everything mypy reports came from three *undeclared dynamic attribute
  surfaces* -- not from missing signatures. Declaring an attribute surface is cheap, so
  the annotation work is tractable after all and is now underway module by module.

  The first of those surfaces is done: the 92 enum members that `Polars2SVG.__init__`
  flattens onto the instance with `setattr()` loops are now declared in a class-level
  annotation block, which removed 631 spurious "has no attribute" errors. They are bare
  annotations, so nothing changes at runtime -- no class attribute is created and no
  instance state is shadowed.

  Two one-way ratchets keep it moving: `[[tool.mypy.overrides]]` holds finished modules
  to `disallow_untyped_defs` with every relaxed error code re-enabled (`laguerre_voronoi`,
  `layout_budget`, `layout_protocol` to start), and `TestAnnotationCoverageRatchet` caps
  the unannotated-function count per module so adding an untyped function fails the suite.
  `index` and `func-returns-value` were dropped from `disable_error_code` after measuring
  zero occurrences of either. See CONTRIBUTING.md for the workflow.

- **Golden coverage for `draw_link_labels=`, which previously had none.** Two new goldens --
  `linkp_link_labels_curve` (`<textPath>` along the drawn Bezier) and `linkp_link_labels_line`
  (text rotated onto the chord) -- cover the two placement mechanisms and the bidirectional
  pair whose labels sit on opposite sides of one shared baseline. `normalize_svg()` now
  tokenizes the random half of LinkP's `p2sll{id}_{n}` `<textPath>` path ids, without which an
  exact-string golden for the curve case could not be stable; no existing golden contains
  them, so none moved.

  Known limitation this surfaced: **svglib implements no `<textPath>`**, so curve-shaped link
  labels do not appear in `save('.png')` / `savePNG()` output (the SVG is correct, and
  `link_shape='line'` labels rasterize normally).

- **Internal refactor, no behavior change: one cloud-icon definition, not three.** The
  collapsed-node cloud `<defs>` block was three hand-maintained copies of the same 644-byte
  literal -- one in `linkp` (`#cloud`) and two in `spreadlinesp` (`#cloud` and the unstroked
  `#cloud_outline`). They now come from `cloudIconDef(id, stroke)` beside the existing
  `CLOUD_ICON_*` metrics in `p2s_displaylist`, which reproduces all three byte for byte, so
  no golden moved. The attribution (SVGRepo / Ankush Syal, CC-BY) travels with the constant
  instead of being repeated per copy.

- **`LinkP` no longer emits geometry that falls entirely outside the canvas.** Links, nodes,
  node labels, link labels (with the `<textPath>` `<defs>` path each one needs) and timing
  marks are rejected before string assembly when their bounding box misses the viewport.
  A render that fits on the canvas is byte-identical to before; a zoomed one is a fraction
  of its former size:

  | fixture (1024x768, 120 nodes / 600 edges) | before | after | |
  |---|---:|---:|---:|
  | curve, 1x | 212,188 | 212,188 | *identical* |
  | curve, 10x | 226,400 | 84,120 | -62.8% |
  | curve, 100x | 236,808 | 58,475 | -75.3% |
  | curve, 10,000x | 256,457 | 63,022 | -75.4% |
  | line, 100x | 139,855 | 10,831 | -92.3% |

  This is the size problem rounding cannot touch. Zoom moves a coordinate's bytes from the
  fractional tail into the integer part -- the widest token in that fixture grows from
  `1003.03` to `-4906591.73` -- so a fractional-digit rounder recovers less and less exactly
  as the file gets worse, while the elements carrying those magnitudes are overwhelmingly
  the ones nobody can see. Dropping the element removes the token instead of shortening it.

  Only *wholly* outside is dropped. An element that straddles the boundary is emitted whole,
  and the bounds are conservative in every direction: a cubic is bounded by the convex hull
  of its control points (so a curve that bows back into view survives both endpoints being
  off-canvas), a node by its radius and stroke, a collapsed node by the cloud glyph's own
  box, and a label by its measured ink -- a node label whose *baseline* sits below the canvas
  still has glyphs rising into it, and is kept.

  Nothing about selection changes. `recordsAt()`, `entitiesAtPoint()`, `overlappingEntities()`
  and `__moveSelectedEntities__()` hit-test the `__sx__`/`__sy__` screen columns and never the
  emitted string, and `df`, `df_link` and `df_node` still carry every row, so an off-canvas
  node is still selectable at its screen position and comes back the moment the view moves.
  The GPU display list is built from those same retained tables; the two label lists it does
  draw from (`_node_label_info_`, `_link_label_info_`) and the timing-mark mirror are culled
  in step with the SVG, so the two renderings of a view agree.

  No existing golden changes — they all fit on their canvas, and a render that fits is
  byte-identical. `linkp_off_canvas_cull` is new, and is the one that pins a zoomed render:
  a node culled while its edges still cross the frame, and a label whose glyphs no longer
  reach back into view.

- **`LinkP` emits the cloud `<defs>` block only when a node actually collapses** — 808 bytes
  off every ordinary linkp render, or 41% of a small one. Only collapsed nodes draw
  `<use href="#cloud">`, but the 808-byte icon definition was written into every render
  regardless; the nine linkp goldens all carried it and none referenced it.

  Whether a cloud exists is a property of the *current render*, not of the instance:
  collapsing is decided by grouping on screen coordinates, so a graph that collapses at one
  zoom separates at another. The condition is therefore evaluated in `__renderSVG__` on every
  render — an interactive zoom moves the block in and out — and `<defs>` itself survives for
  the `<textPath>` link-label paths, which are an independent payload.

  The icon's `d=` constant is also pre-rounded from 5 fractional digits to 2 (a further 151
  bytes; ~0.03px at the icon's scale), in `LinkP` and in both of `SpreadLinesP`'s copies.
  Ten linkp goldens change; `linkp_collapsed_nodes` is new and is the one that keeps the
  block, so the "cloud present" path stays covered.

### Fixed

- **histop cropped its bin labels at half the bar's width.** The per-bin label is
  drawn inside the plot at the left edge, so the whole bar row is available to it,
  but `_lbl_max_w_` was `self._plot_w_ * 0.5` -- half the row was thrown away and
  names with ample room came out as `'cinderell...'`.  The budget is now the full
  plot width less a 2px pad on each side.  Because `cropText()` overhangs the width
  it is given by the ellipsis it appends, the fit test and the crop take different
  budgets: the untruncated label is measured against the whole span, and only when
  it does not fit is it cropped against the span less the ellipsis -- so a cropped
  label lands inside the plot border instead of over it, and a label that fits the
  row is never truncated.  No golden changed (none carried a cropped label), so two
  tests in `test_histop_basic.py` now cover both halves.

- **A tuple relationship endpoint rendered fine on its own but broke the moment it
  was interactive.** `p2s.linkp(df, [(('module','class'),'param')])` drew correctly,
  then `panelize()` raised `ColumnNotFoundError: unable to find column "__fm0__"`.
  A tuple endpoint is expanded into a synthetic `'|'`-joined `__fm{i}__` column and
  the component rewrites its own `relationships` to name it — but that column exists
  only on the private render frame (`linkp.df`), never on `df_orig`. `linkpi()` built
  its graph, and stocked `ln_params`, from `df_orig` paired with the *rewritten*
  spec. The pristine spec was already being saved as `relationships_orig` and simply
  had no reader; it does now, on both the graph build and every stack operation
  (`pushStack`, `replaceBaseDataframe`, edge filter / collapse / unfilter). Node
  names still line up because `createNetworkXGraph()` joins with the same separator.
  All-string relationships were never affected, which is why this went unnoticed.
- **`_extractNodes_()` returned an empty set for a tuple endpoint** — no exception,
  just every node looking absent, which reads downstream as "nothing is visible". It
  now joins tuple endpoints the way the graph does before matching.
- **`render_with()` was broken for a tuple endpoint on linkp, chordp and
  spreadlinesp**, independently of `panelize()`: a template clone inherited the
  template's already-rewritten `relationships`, whose endpoints are plain
  `'__fm0__'` strings by then, so re-running the expansion was a no-op and the
  clone's fresh frame never got the column — `field "__fm0__" not found in
  DataFrame`. Clones now re-expand from the pristine spec (an explicit
  `relationships=` override still wins).
- **`spreadlinepi`'s `'x'` (filter out selection) had the same pairing error** in
  `_filter_out_nodes()`: rewritten relationships against `_spread_.df_orig`. It also
  leaked the scratch join column into the pushed frame; the frame is now returned
  with the caller's columns.
- **Four `ChP.__validateInput__()` errors identified themselves as `LinkP`.**
- **`stack_control.sketchHtml()` was annotated `-> str` but returns
  `self.mod_inner or None`** — and `None` is the documented "defer to
  `_repr_svg_()`" signal in the `SketchRepresentable` protocol.
- **The runtime-built widget classes hid their own `self`.** `stack_control` and
  `spreadlinepi` define the methods of their `type()`-constructed classes as
  module-level functions, where `self` is an ordinary parameter; 18 of them were
  unannotated and invisible to a per-method coverage count.
- **`render_with(df, **overrides)` could not take `Unpack[<Component>Kwargs]`** —
  PEP 692 rejects a `**kwargs` TypedDict that repeats a named parameter, and `df`
  is both. It stays `Any`, with a comment saying why.
- **`Polars2SVG._apply_defaults()` typed its mapping as `dict`**, which a
  `TypedDict` is not assignable to — the same phase-2 incompatibility already
  fixed in `assignKwargOverrides()`. Now `Mapping[str, Any]`.
- **`webgpu()` returns `None` when the component has no display list** but was
  annotated `-> dict` on all eight components.
- **`xyp.__constructOrderColumns__()` computed `_is_dict_` and then branched on
  it** in three places, which no checker can narrow through; now an inline
  `isinstance`, and the now-unused variable is gone.
- **`udist_scatterplots_via_sectors_tile_opt` built and discarded a tuple per
  statement in 21 places** (`_lu_['a0'].append(x), _lu_['a0u'].append(y)`), inside
  the per-sector loops; `od_flow_layout` had one more in its flow-indexing loop.
  Rewritten with `;`.
- **`ncp_layout._stage2_crossing_edge()` returns `None` when no edges cross** but
  was annotated `-> tuple`.
- **`Polars2SVG.assignKwargOverrides()` / `assignKwargsWithDefaults()` typed their
  mapping as `dict`**, which a `TypedDict` is not assignable to — the phase-2
  `Unpack` work made components pass exactly that. Now `Mapping[str, Any]`.
- **Eleven parameters annotated non-optional while defaulting to `None`** --
  `chordp` (2), `p2s_graph_mixin` (3), `tfdp_layout` (2), `circle_packer`,
  `interactive_controller`, `p2s_interactive_mixin`, `p2s_legend_mixin`,
  `p2s_render_mixin`. Eight predate this migration.
- **`p2s_graph_mixin.rectangularLayout()` used `isinstance(nodes, list) == False`**
  in three places; rewritten as `not isinstance(...)`, which reads better and lets
  a checker narrow the type.
- **`circle_packer` held a tuple expression used to sequence two statements**
  (`seen.add(next), seen.add(prev)`), building and discarding a tuple per iteration
  of the pack loop.
- **`_svgRootWxh_()`, `optimalInscriptionCircle()`, `__approximateInscribedCircle__()`
  and `Tile.gatherMetrics()` had return types contradicted by their own code** --
  see the trace blind spots above.
- **`chordp` / `piep` small-multiple kwargs dicts.** `_kwargs_` is a heterogeneous
  dispatch dict seeded with `sm_shared` (a `set`), which pinned its value type and
  made every later `_kwargs_['...'] = <list/tuple>` an error. Now annotated
  `dict[str, Any]`, and `renderSmallMultiples()` is annotated with it.
- **`chordp` `node_gap` is a float, not an int.** Its default is the literal `2`,
  but the ring layout assigns `0.0` to it when the nodes do not fit.
- **`circle_packer`'s host reference was typed `object`.** It calls
  `overlappingCirclesIntersections()`, `circlesOverlap()` and three other geometry
  helpers on `rt_self`, so `object` was too narrow to be true.

- **`layout_budget.Budget` attribute types.** `stopped_at`, `note`, `_t0_` and `_total_`
  were initialized to `None` with no annotation, so a checker inferred their type as
  `None` and every later assignment in `start()`/`_stop_()` was an error. Now annotated
  `Optional[...]`.
- **`laguerre_voronoi` suppression that suppressed nothing.** The quadtree insert loop
  carried `# type: ignore[union-attr]` on a line whose actual error code is `attr-defined`
  (mypy narrows `self.children` to exactly `None` inside the enclosing `is None` test, and
  `_subdivide()` reassigning it is invisible), so the ignore covered nothing. The code list
  now matches. No runtime assert was added -- that line is the insert hot loop.

  Both were found by promoting those modules into the typing ratchet, which is what the
  promotion step is for.

- **A label map is no longer mistakable for the flag that turns labels on.** `node_labels=`
  supplies the `{name: display_str}` map; `draw_node_labels=` turns the labels on. One prefix
  apart, so the value meant for the flag lands on the map -- and a bool there failed in
  whichever way the parameter happened to be consumed: stored and never read with the draw
  flag off, or `object of type 'bool' has no len()` from the middle of the render with it on.
  Neither names the parameter, and neither names the fix. linkp had been catching exactly
  this for `link_labels=` since edge labels landed; the guard simply never reached the
  parameters beside it.

  That guard now lives in one place -- `Polars2SVG.rejectBoolParam()` -- and covers
  `node_labels=`, `link_labels=` and `label_only=` on linkp and `node_labels=` /
  `label_only=` on chordp, which had the identical defect and failed just as deep
  (`'bool' object has no attribute 'items'`). It raises at parse time with a message naming
  the flag to use instead. Bool only, deliberately: these parameters accept several shapes
  each, so this catches the one wrong value that is a predictable confusion rather than a
  typo -- the same line `normalizeWxh()` already draws between a bool and the other ints.

  This is a behavior change: code passing `node_labels=True` today renders no labels and now
  raises instead. That is the point -- it is how the mistake went unnoticed. The linkp
  special-character test in `test_edge_case_inputs.py` made exactly it, rendered zero `<text>`
  elements, and asserted nothing about labels for as long as it existed, hiding the entity
  escaping bug above. `tests/test_label_param_flag_confusion.py` covers every label map and
  filter on both components, and checks that the shapes they really accept still work.

  spreadlinesp also declares `node_labels=`, and it has a *different* defect -- the parameter
  is never read at all, bool or not -- so it is deliberately left alone here.

- **XML escaping has one door, and it is the last step.** Escaping was spelled five
  different ways -- `svgText()` in the text mixin, the same standard-library call inline in
  `chordp` and `spreadlinesp`, two hand-rolled `.replace()` chains in `linkp` (one each
  way), and nothing in the interactive chrome, whose strings are its own. Nothing tied them
  together, and the convention broke where two of them met: `linkp` escaped the node-label
  column and *then* cut the escaped string by character count. `_wrap_label_()` breaks at
  `label_line_width` characters and the ellipsis path slices the last line, so a `&` or `<`
  landing near a cut split its entity down the middle --
  `<tspan ...>aaaaaaaaa&</tspan><tspan ...>amp;bbbbb</tspan>` -- which is not well-formed
  XML. Browsers tolerate it through `innerHTML`, so it read as display corruption rather
  than script execution, but a strict consumer rejects the whole document: `save('.png')`
  and any `<img>` tag. At the default `label_line_width=32` it took a label longer than 32
  characters with a special character at the wrong offset; at narrower widths, much less.

  There is now one function, `p2s_text_mixin.svgEscape()` (with `svgUnescape()` for the
  parse-back direction), reachable as a module import or as a `self.p2s` method, and one
  rule stated in its header: **escape last**. Everything that measures, crops, wraps,
  truncates or slices works on the raw string; the escape happens immediately before the
  text is interpolated into markup. Every call site in the package now goes through it,
  including `p2s_displaylist.svgToDisplayList()` and the WebGPU error overlay.

  Two things fall out of the ordering fix. `linkp`'s node-label wrap now breaks at
  `label_line_width` characters *of the label*, not of its escaped spelling, so a label
  containing `&` wraps where a caller asking for 10 characters per line would expect rather
  than five characters early; and off-canvas culling, which measured `textLength()` on the
  escaped string, no longer charges five characters for a `&` that draws as one -- the same
  correction applies to link labels. `_node_label_info_`, which feeds the glyph atlas, now
  carries the raw label, so the GPU path no longer has to un-escape what the SVG path just
  escaped. No golden changed.

  `tests/test_svg_escaping.py` covers all of it: the door's own contract, the wrap and
  ellipsis paths swept across every offset a special character can occupy, a round-trip
  check through every component that puts row data into markup (`histop`, `piep`, `xyp`,
  `chordp`, `linkp` node/link labels and legend, `spreadlinesp`, `smallp`), and a source
  scan that fails if a sixth escaping mechanism appears. The test meant to be watching this,
  `test_edge_case_inputs.py::test_linkp_labels_special_chars`, passed `node_labels=` (the
  `{name: display}` map) where it meant `draw_node_labels=`, rendered zero `<text>` elements
  and so asserted nothing about labels; it now names the flag and asserts the label text.

- **`LinkP` curve geometry no longer overflows Int32 at high zoom.** Screen coordinates are
  `Int32`, and `__curveControlPointColumns__` squared the endpoint deltas in that dtype to
  get the chord length. Past ~46,341 pixels of separation -- routine once a `view_window`
  magnifies the layout -- the square wraps: negative, and `sqrt()` returns NaN that rides
  into the Bezier control points and out into `d="M ... C NaN NaN NaN NaN ..."`, which no
  renderer draws; or positive, and the curve is silently bowed to a wrong length. On the
  §3 fixture at 100x, 196 of 587 edges had NaN control points, and a 557,200px chord
  measured 35,133px. `__arrowColumns__` had the same defect on the `line` branch, where
  both terms are `Int32`. Both now lift to `Float64` before squaring, exactly as
  `__segmentDistSqExpr__` already did deliberately. No effect on an unzoomed render --
  every golden is unchanged -- and found while verifying the off-canvas cull above.

- **A genuinely broken `spreadlinepi` import is no longer swallowed.** `interactive_controller`
  registers the `SpreadLinesP` wrapper behind `except ImportError: pass`, which is right for the
  module cycle it exists for — `spreadlinepi` imports the controller first and registers itself
  at the end of its own body, so whichever import wins completes the registry — but it also
  caught every unrelated `ImportError`. A real failure inside `spreadlinepi` therefore dropped
  `SpreadLinesP` from `_PLOT_TYPE_TO_WRAPPER_` silently, and `panelize()` then reported `no
  wrapper for component type "SpreadLinesP" (register it in _PLOT_TYPE_TO_WRAPPER_ …)` —
  instructions to do the thing the code already does — while hiding the actual error. The
  handler now re-raises unless the module is genuinely mid-import: a partially initialized
  module is still in `sys.modules`, a broken one has been removed from it. Latent rather than
  live; both import orders resolved correctly before and after, and the registry entry is now
  pinned by `test_spreadlinepi.py::TestWrapperRegistration`.

- **`TestLabelModeCycle` no longer draws its node positions from the unseeded global RNG.**
  The fixture passed no `pos=`, so `LinkP` placed all five nodes randomly; when two adjacent
  nodes landed close together the edge became too short to carry its label (the text is
  cropped to the edge length) and `test_link_states_actually_render_edge_labels` failed. The
  test is about the label-mode cycle, not about layout, so the positions are now pinned.
  Latent since the test was written — every `LinkP` construction and every `renderSVG()`
  draws from `random`, so any change to how many renders precede it reshuffles the outcome.

- **`XYp` resolves the visible world window in one place** (`__resolveRanges__`), which the dot
  transforms, the numeric grid lines, the axis end labels, and everything downstream of
  `x_transform_vars`/`y_transform_vars` (`wxToSx`, background shapes, `filterByRectangle`,
  `recordsAt`) now all read. `__renderContext__` previously recomputed the axis minima and
  maxima from the post-filter dataframe, independently of the window the dots were
  transformed with; the two agreed for every shipped configuration, so this is a no-op for
  existing plots, but it is what lets `aspect=` move the dots and the grid lines together.

### Removed

- **`PolarsForceDirectedLayout` and `ConveyProximityLayout`** — deprecated in 0.2.0 with a
  warning naming this release, and now deleted along with their modules
  (`polars_force_directed_layout.py`, `convey_proximity_layout.py`). Importing either name
  from `polars2svg` now raises `ImportError`.

  The migrations are unchanged from the 0.2.0 deprecation notice:

  | Was | Use |
  |---|---|
  | `PolarsForceDirectedLayout(g).results()` | `networkx.spring_layout(g)` |
  | `PolarsForceDirectedLayout(g, pos=pos, static_nodes=fixed)` | `networkx.spring_layout(g, pos=pos, fixed=list(fixed))` |
  | `ConveyProximityLayout(g)` — for a distance-preserving layout | `LandmarkMDSLayout(g)` or `PivotMDSLayout(g)` |

  Both caveats from that notice still apply. `networkx.spring_layout` is
  Fruchterman-Reingold and is **not** distance-preserving — if screen distance tracking
  graph distance is what you wanted, the MDS pair is the replacement, not spring. And
  **resistive (effective-resistance) distances have no replacement**:
  `ConveyProximityLayout(use_resistive_distances=True)` was their only source in the
  framework.

  With both gone, the framework no longer implements Cohen (ACM ToCHI 1997), "Drawing
  Graphs to Convey Proximity"; its entry has been dropped from the README's references.
  The `layouts` extra is unaffected — both modules used only networkx and numpy, which
  many other layouts still need.

## [0.2.0] — 2026-08-28

### Removed

- **`linkp`'s `force directed` and `convey proximity` interactive layout operations.** Both
  were unbounded at interactive sizes and neither was the best available implementation of
  what it did. Measured on `barabasi_albert_graph(n, 3)`: `force directed` took 4.28 s /
  36.34 s / 114.59 s / **784.20 s** at n=500/1000/2000/4000 — at n=4000 that is ~75x the
  next-slowest layout operation and ~7,800x the median one. `convey proximity` took
  **12.03 s** at n=2000 with a 2.77 GB peak RSS, scaling as n^2.

  `spring nx` now covers both jobs (see Changed). The classes themselves remain importable
  this release under a deprecation warning — see Deprecated.

### Deprecated

- **`PolarsForceDirectedLayout` and `ConveyProximityLayout`** — both emit a one-time
  deprecation warning on instantiation and will be **removed in 0.3.0**. They are two
  implementations of the same paper (Cohen, ACM ToCHI 1997), and after their removal the
  framework does not implement that method.

  Migration:

  | Was | Use |
  |---|---|
  | `PolarsForceDirectedLayout(g).results()` | `networkx.spring_layout(g)` |
  | `PolarsForceDirectedLayout(g, pos=pos, static_nodes=fixed)` | `networkx.spring_layout(g, pos=pos, fixed=list(fixed))` |
  | `ConveyProximityLayout(g)` — for a distance-preserving layout | `LandmarkMDSLayout(g)` or `PivotMDSLayout(g)` |

  Two caveats worth stating plainly. `networkx.spring_layout` is Fruchterman-Reingold and is
  **not** distance-preserving, so layouts will differ from Cohen's stress majorization — if
  screen distance tracking graph distance is what you wanted, the MDS pair is the
  replacement, not spring. And **resistive (effective-resistance) distances have no
  replacement**: `ConveyProximityLayout(use_resistive_distances=True)` was their only source
  in the framework.

### Added

- **`flowmap_max_flows=`, `flowmap_iterations=` and `flowmap_samples=` on `linkp`** — the
  quality-for-time knobs for `link_shape='flowmap'`. `ODFlowLayout` has always accepted the
  latter two; `linkp` hardwired the paper's defaults, so the only way to make an expensive
  flow map cheaper was not to draw one.

  `flowmap_max_flows` is the new one, and the one with real leverage: it lays out only that
  many flows, largest count first, and draws the rest as plain curves. Because the layout's
  cost grows faster than the square of the flow count, decimating the input beats trimming
  the iteration budget by a wide margin. On a 560-flow render measuring **65.2 s** at the
  defaults:

  | | time | speedup |
  |---|---:|---:|
  | defaults (all flows, 100 iterations, 15 samples) | 65.2 s | — |
  | `flowmap_samples=8` | 32.6 s | 2.0x |
  | `flowmap_iterations=30` | 19.7 s | 3.3x |
  | `flowmap_max_flows=100` | 4.2 s | **15.7x** |
  | `flowmap_max_flows=100, flowmap_samples=8` | 3.0 s | **21.9x** |

  The flows that get dropped are the ones carrying least, which are also the ones a reader
  is least likely to be tracing — and the paper's method targets ~100 flows precisely
  because that is what stays readable. Every link is still drawn either way; decimation
  changes how a flow is drawn, never whether it is.

- **`backend=` on `ODFlowLayout` and `flowmap_backend=` on `linkp`** — `'auto'` (default,
  unchanged behaviour), `'numpy'` or `'mlx'`. The two are not numerically identical: MLX
  computes the force kernels in float32 where NumPy uses float64, and the force iteration
  compounds that into control points differing by pixels rather than by rounding. Pin it
  when a flow map is going to be compared against a stored expectation, so the comparison
  stays a property of the code rather than of whichever extras happen to be installed.

- **`linkpi`'s picker menus say which choices will stop to ask, and stop committing those
  by accident.** An operation that will ask a question at a large enough graph now reads
  `spring nx  (asks >8,000 nodes)` in the picker, and `flowmap  (asks >200 edges)` in the
  shape picker, so the cost is visible at the moment of choosing rather than after
  committing. The annotation comes from the operation's declaration, so it stays true
  whatever is on screen.

  Those items are also exempt from the two ways these menus used to commit without an
  Enter: the single-character mnemonic, and the 2.5-second inactivity timeout. Pressing
  `l` then `3` previously started the force layout in two keystrokes with nothing in
  between, and walking away from an open menu could commit whatever happened to be
  highlighted. Cheap operations are unchanged — annotating everything would make the
  annotation worth nothing.

- **Escape cancels a running layout in `linkpi`, keeping its best-so-far result**, and
  `flowmap_time_budget=` on `linkp` gives the same treatment a wall clock. The iterative
  layouts polars2svg owns — `ODFlowLayout`, `NCPLayout`, `TFDPLayout` — now accept a
  `layout_budget.Budget`, checked once per iteration, which stops on a time limit or on a
  caller-supplied signal and returns what it has. Force layouts truncate gracefully: fewer
  iterations is a less-relaxed layout, not a broken one, and every node or flow still comes
  back with a finite position. What it cost is reported in the interactive info line.

  Opt-in by design — with no budget these layouts behave exactly as before, which is what
  keeps their determinism meaningful, since a wall-clock limit cannot be reproducible. For
  the same reason a truncated flow map is **not** cached: it reflects how fast the machine
  happened to be at that moment, and caching it would make one slow instant permanent for
  the rest of the session.

  This is the treatment that only works on loops polars2svg owns. `spring nx` and louvain
  are imported and expose no per-iteration hook, so they keep the confirmation gate and the
  before-the-call levers instead.

- **`linkpi` keeps working while a slow operation runs.** Layout operations, community
  detection, background producers and the switch to `link_shape='flowmap'` now run on a
  worker thread instead of on the event loop, so the widget keeps repainting, the status
  line stays visible, and a long operation looks like a long operation rather than like a
  hang. A short note naming the running operation is painted before the work starts.

  Being plain about what this is not: it does **not** make anything interruptible. Python
  cannot stop a thread parked inside numpy or networkx, so the confirmation gate and the
  quality-for-time levers remain the things that bound the cost. What this changes is
  whether the browser is alive while you wait.

  A user action that arrives while an operation is running is now **dropped rather than
  queued**, with a brief `busy` notice. Queueing them meant a burst of stale operations
  replaying against a view that had since moved — for a widget driven by held-down keys,
  worse than losing the keystrokes. Updates arriving from *linked views* are not dropped:
  those are peer-driven, and skipping one would leave this view showing something the
  others are not.

- **`linkpi` asks before running an operation that is declared to need asking at the
  current graph size.** Each layout and background operation now carries a declaration of
  what can be done about it if it runs long — whether it has a loop that could take a
  deadline, whether it is pure enough to move to a subprocess, which parameters trade
  quality for time, and the node count above which it should ask first. Over that count
  the operation refuses once and says so on the canvas; repeating the same operation runs
  it, and asking for anything else in between cancels the pending confirmation. Switching
  `link_shape` to `'flowmap'` is gated the same way.

  The declarations are deliberately *not* a cost model. Measured growth is too unstable to
  fit: the same algorithm on the same graph family produced per-doubling exponents of 3.1,
  1.7 and 2.8, and the flow-map layout varies 12x in runtime at a *fixed* flow count
  depending only on how clustered the flows are. So the registry records properties of the
  code — readable and checkable — and the confirmation message names the size and the
  threshold, never a predicted duration.

### Changed

- **`linkpi`'s `spring nx` spends fewer iterations on large graphs.** networkx's default of
  50 costs 10.4 s at 4,000 nodes and 43.2 s at 8,000. Above 2,000 nodes the count now scales
  down proportionally (floored at 10), which brings 8,000 nodes to 10.6 s — a 4.1x saving.
  networkx exposes no per-iteration hook, so unlike the layouts polars2svg owns, the choice
  has to be made before the call rather than by interrupting it. What was spent is reported
  in the interactive info line. Note this bounds the growth without flattening it: the
  per-iteration cost still rises with the graph.

- **`linkp`'s `spring nx` layout operation now runs with a selection**, laying out the
  selected nodes while pinning the rest through `networkx.spring_layout`'s `fixed=`. It
  previously declined outright when anything was selected, which is why a second
  force-directed implementation existed. Measured 3–13x faster than the implementation it
  replaces on the selection path (0.180 s vs 2.30 s at n=500; 1.799 s vs 7.31 s at n=2000),
  holding pinned nodes to within float32 rounding. `fixed=` also suppresses networkx's
  rescale, so pinned nodes keep their exact coordinates.

- **`linkp` caches the `link_shape='flowmap'` force layout** instead of recomputing it on
  every render. The layout depends only on the flow endpoints and the obstacle geometry, so
  renders that change neither — a label toggle, a selection, a brush, a stack step — now
  reuse the previous result. Pan and zoom still recompute: the layout runs in screen
  coordinates, so they genuinely change its input.

- **`linkp`'s flowmap warning is more accurate.** It described the layout as quadratic in the
  flow count; measured growth is faster than that (exponent 3.0–4.0 on both the NumPy and MLX
  paths), and the count is not the whole story — the intersection-reduction pass works on
  flows sharing a node, so a dispersed set of flows costs far more than a hub-heavy set of
  the same size.

### Fixed

- **A missing golden-image reference was silently created instead of failing the test.**
  The helper wrote the file whenever it did not exist, which is a convenient default
  exactly once — on the very first run — and a trap afterwards: any checkout without the
  reference files re-baselined the whole suite against whatever that machine produced and
  reported green. Writing now requires `UPDATE_GOLDEN=1`; a missing golden fails and says
  so. (Test-suite only; no effect on the published package.)

- **A `linkpi` layout operation that declined to run still consumed an undo level.** Any
  operation that returned without moving anything — a global layout asked for with a
  selection in place, or one that raised — left a no-change snapshot on the undo stack, so
  the next `u` was a silent no-op the user had to press twice.

- **A `linkpi` layout operation or community detection that ran out of memory looked like a
  key press that did nothing.** Both call sites caught bare `Exception`, which swallowed
  `MemoryError` (and `KeyboardInterrupt`) and left the view untouched with no message. Those
  two now propagate, and every other failure is logged through the package logger rather
  than discarded.

- **`LandmarkMDSLayout` computed √n full-graph Dijkstras and discarded the result.** The
  block recomputed the adjacency matrix and landmark distance array after the coordinates
  were already assigned, and nothing read it. Removing it cuts 18–27% off every call
  (0.04 s → 0.024 s at n=2000).

### Added

- **`FlowFieldBackground` — a layered flow-field background for an existing layout**
  (PLANNING.md **§9.2 / B4**). Given the positions a `linkp` is already drawing and the
  individual edge records behind it, it returns `{name: BackgroundShape}` cells that
  linkp draws under the links to show which way traffic is moving. A vector field holds
  one vector per point, so opposing and crossing flows cancel where they overlap; the
  edges are split into `k_layers` internally coherent layers instead, each drawn as its
  own field. `k_layers=1` is the plain (cancelling) net field, `2` is the dominant flow
  plus what fights it, `3+` peels a further crossing stream. Every edge lands in exactly
  one layer, so K is a hard count rather than a cap. `glyph='arrow'` draws filled darts
  anchored on the grid; `glyph='streamline'` draws evenly-spaced integral curves with
  filled head markers. NumPy and Polars only — unlike the graph layouts it needs no
  optional extra. Exported as `p2s.FlowFieldBackground` and
  `from polars2svg import FlowFieldBackground`.
  - **Not a layout.** It moves no nodes and has no `results()`, so it does not satisfy
    `LayoutAlgorithm` and cannot be mistaken for one. Its cells carry their own
    appearance, so `background=` is the only argument a caller passes.
  - **Cost is bounded by two limits, because there are two costs.** The kernel footprint
    is memory and grows with edge *length* (`support_budget=`, default 6M grid cells);
    the layer assignment is time and grows with edge *count* (`max_edges=`, default
    25 000). Over budget, the grid is coarsened toward `min_grid_res` first — every edge
    kept, spatial detail lost — and only then are the lightest edges dropped.
    `summary()` reports whatever happened. A 5 000-node graph of 400 000 canvas-spanning
    edges went from exhausting memory to 1.9 s.
- **Background operations in `linkpi` (shift-b)** — a picker of background *producers*,
  separate from the layout-operation picker. Committing an entry runs it immediately
  (these are one-shot actions, not a mode). Ships `flow field (2 layers)`,
  `flow field (3 layers)`, `flow field (streamlines)` and `clear background`; `b` still
  cycles background visibility. A background operation deliberately does **not** go
  through `apply_layout_operation()`: it costs no undo slot (undo restores positions, and
  none moved), does not reset the view window, and skips `__contractCollapsedGraph__`.

### Changed

- **`neighborhood (spatial)` moved from the layout picker to the background picker**
  (shift-G → shift-b, mnemonic `n` in both). It never was a layout: it clusters the
  positions already on screen and outlines each neighbourhood, leaving every node
  exactly where it was (measured: 0 of 40 moved, against 40 of 40 for `hyper tree
  donut`, `circle pack` and `neighborhood (graph)`). As a layout operation it cost an
  undo slot and reset the view window for a change it never made, ran through
  `__contractCollapsedGraph__` (which hands a non-moving producer a `.pos` that
  disagrees with its `.df`), and its cells were treated as layout-owned — so any later
  layout wiped neighbourhoods that still described the unchanged positions. As a
  background operation it does none of that and its cells are contextual. It is
  **removed from the layout picker**, not listed in both. `neighborhood (graph)` stays a
  layout operation: it repositions communities, so its cells really are its own output.
  Its selection guard is also gone — a background describes the whole embedding, and
  `neighborhoodLayout` has no way to scope the clustering anyway.

- **A background now has a lifetime determined by its provenance** (PLANNING.md **B4**).
  A background that *is* a layout's own output (`hyper tree donut`, `circle pack`, the
  `neighborhood` pair) still dies when a later layout produces none. A background
  produced *for* a layout — anything from the shift-b picker — is contextual: it survives
  node drags and later layouts, and is replaced only by another background or an explicit
  clear. Moving nodes in response to what a background showed is the intended workflow,
  so doing it no longer erases the background. Previously any layout operation without a
  background of its own cleared whatever was showing.

- **Background records — `background=` entries can carry their own appearance**
  (PLANNING.md **§9.1 / B1–B3**). An entry in `background=` is still a bare shape
  descriptor (shapely geometry, `[(x, y), ...]`, `'<circle .../>'`, or an `M/L/C/Z`
  path string), but it may now instead be a **record** built with
  `p2s.bgShape(shape, **fields)` — or written as a plain `{'shape': ..., 'fill': ...}`
  dict — that carries `fill`, `fill_opacity`, `stroke`, `stroke_opacity`,
  `stroke_width`, `dash`, `stroke_linecap`, `stroke_linejoin`, `label` and
  `label_color`. Both forms mix freely in one dict. Shared by `xyp` and `linkp`
  through `P2SBackgroundMixin`. Reachable both ways: `p2s.bgShape(...)`,
  `p2s.INHERIT` and `p2s.BackgroundShape` on the instance, and
  `from polars2svg import BackgroundShape, INHERIT` at module level.
  - **Style was keyed by name alongside the geometry rather than attached to it.**
    Appearance lived in five parallel `background_*` dicts, so every new capability
    cost a new top-level parameter × its scalar/dict forms × a fallback chain. The
    five parameters remain and are unchanged — they are now what a record's
    `p2s.INHERIT` fields defer to — but **new appearance goes on the record**:
    `stroke_opacity=`, `dash=`, `stroke_linecap=`, `stroke_linejoin=` and `label=`
    arrived without a sixth, seventh or eighth parameter.
  - **`INHERIT` vs `None` is the load-bearing distinction.** Every field defaults to
    `p2s.INHERIT` ("use the component's `background_*` parameter"), so a bare
    descriptor is exactly a record with every field `INHERIT` — back-compatibility by
    construction rather than by maintenance. `None` means *explicitly off*, which the
    parameters could not express: a name missing from a `background_fill` dict was
    filled with the axis colour at full opacity, so callers had to smuggle
    suppression through values (`fill-opacity` 0 for shapes that must not be filled,
    `stroke-width` 0 for shapes that must not be stroked).
  - **`stroke-opacity` is emitted for the first time.** A stroked background could not
    be made translucent, so it competed with the links it sat under; the workaround
    was pre-lightening the stroke colour toward the background. `dash` →
    `stroke-dasharray` is new for the same reason. Cap and join reach the SVG only —
    the GPU line primitive carries no cap/join field.
  - **`label=` decouples the drawn label from the dict key**, and `label=None`
    suppresses one entirely. `background_label_color` still governs whether labels are
    drawn at all (it is what the interactive `b` cycle drives), so a record supplies
    its own `label_color` to force one on.
  - **Draw order is now documented contract**: dict insertion order, all shapes then
    all labels, so a later entry lands over an earlier one. Previously true, but
    incidental; a producer emitting several layers depends on it.
  - `p2s.bgShape()` records are immutable — template cloning shares non-container
    leaves by reference, so a mutable record would leak edits from a clone back into
    its template and every sibling.

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
    distance off the edge whatever the strings are. Ascent and descent come from
    the bundled font's own glyph outlines (`_labelInk_` → `textInk()`, see below),
    so every string shape clears its edge by exactly the intended 2 + stroke/2 px;
    a single flat offset put x-height-only labels (`cow`) ~3px too far out and let
    a descender (`dog`) touch the edge on the other side.
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

- **`null_nodes=` on `linkp`** — draw a missing relationship endpoint instead of
  silently leaving the entity beside it as an unconnected dot. Each entity gets its
  **own** null partner, `p2s.NULL_NODE_PREFIX + entity`, so a row with a null `to`
  renders as a short stub edge from the entity to a node of its own; two records
  with a missing endpoint are never asserted to point at the same thing. (A single
  shared null node would make every such entity one connected component — neighbor
  expansion from it would select all of them at once, community detection would
  group them, and force layout would ball them up.) Off by default, since it
  changes what is rendered.
  - The sentinel is built from the non-printable ASCII US (`0x1f`), like
    `MULTI_FIELD_SEP`, because `'None'`, `'null'` and `''` are all plausible values
    in real data and a readable sentinel would collide with them. Node labels show
    it as `(null)` via `nullNodeDisplay()`.
  - The substitution happens once, in `linkp.__validateInput__()`, and fills both
    `df` and `df_orig` — the render reads the first, `linkpi()`'s stack and graph
    read the second, and filling only one would recreate the very split the
    parameter exists to close. `p2s.nullFillEndpoints()` is idempotent, so a
    template clone of an already-filled frame is a no-op. Rows with **both**
    endpoints null have no entity to anchor to and are left alone.

- **`linkp` timing marks now render on the WebGPU path.** `time=` marks shipped
  SVG-only in 0.1.2, so a `use_webgpu=True` render — including a `linkp` drawn as a
  `smallp` GPU cell — silently dropped them. `__renderTimingMarks__()` now retains the
  per-mark segment geometry it already computes as a numeric table, and
  `gpuDisplayList()` emits it as line primitives between the links and the nodes; the
  GPU never re-derives the geometry, so the two paths cannot drift. The per-pixel
  decimation and the exact-duplicate collapse are inherited from that same table, so the
  GPU mark count equals the SVG mark count. Marks are guarded independently of link
  drawing, matching SVG: `link_size=None` draws no links but still draws its marks.

- **`p2s.chordpi()`, `p2s.piepi()` and `p2s.spreadlinepi()`** — three of the eight
  interactive wrappers existed only as module-level functions. All eight are now
  reachable as `p2s.` methods, like the other five.

- **`p2s.panelizeSketch(layout, use_webgpu=)`** — the method dropped the argument the
  underlying `interactive_controller.panelizeSketch()` has always accepted, so a sketch
  could not be asked for GPU rendering through `p2s.`.

- **`spreadlinesp` is instrumented for WebGPU** instead of reaching the GPU by re-parsing
  its own finished SVG. It was the last component on the `svgToDisplayList()` route; every
  component now records primitives as it renders, so both representations come from one set
  of numbers. What this changes for a render:
  - **The ego-cloud selection ring is correct again.** The parse route lost it in both
    directions — a *fully* selected cloud stroked `<use href="#cloud_outline">`, which the
    parser filled at alpha 0 (invisible, so it read as unselected), and a *partially*
    selected cloud had its `<clipPath>` window ignored, so the whole ring was stroked and it
    read as fully selected. The clip is now a `DisplayList` scissor.
  - Curves (bin outlines, cross-connects, channel pills, discontinuity zigzags) are recorded
    by handing their `d` string to the new shared `pathToDL()` — the same flattener the
    parser uses, so a curve cannot land in two different places.
  - Alter circles keep their translucent fill *and* opaque ring: `DisplayList.circle()` now
    takes `stroke_opacity=` separately from `opacity=`.
  - **`DisplayList.applyTransform(scale, tx, ty)`** is new and is what made this possible.
    `spreadlinesp` sizes its viewBox from the bins it has just placed, so the world→canvas
    mapping is not known until the render is over; the body records in world coordinates and
    is mapped once at the end, using the same triple `_rootViewBoxTransform_()` derives. The
    legend is recorded in screen pixels and spliced on afterwards, never transformed twice.
  - Recording is unconditional, matching the other seven components, so an SVG-only render
    now pays for GPU primitives it may not use: ~21 ms → ~31 ms on a 300-row / 800×400
    fixture. Regenerate `tests/perf_baseline.json` (machine-local) if you track it.
  - `svgToDisplayList()` stays as the universal fallback for markup that arrives as a
    finished string — it is no longer any component's primary route.

### Changed

- **An unfilled background shape emits `fill="none"`** instead of `fill-opacity="0.0"`
  with no `fill` attribute. Both render the same, but only because SVG's initial fill
  (black) was being drawn at zero alpha; background shapes sit directly under `<svg>`
  with no ancestor carrying a fill, so "no fill" now says so. Applies to
  `background_fill=None` / `background_opacity=None` as well as a record's
  `fill=None`. String-matching that attribute in an emitted SVG is the only thing
  affected.

### Fixed

- **A shapely `LineString` / `MultiLineString` background was filled with the axis
  colour** despite the documented "fill forced to `'none'`". The coercion assigned the
  string `'none'` to the fill parameter, which then fell past every branch of the
  colour ladder to the axis-inner fallback — so an open path was painted with the
  interior SVG implies when it closes a subpath in order to fill it. Now genuinely
  unfilled, unless the entry's record sets `fill=` explicitly.
- **A name missing from a `background_stroke_w` dict emitted the dict itself** —
  `stroke-width="{'flow 1': 1.0, ...}"`, invalid SVG, and a `ValueError` when the GPU
  path parsed it back. It falls back to `1.0`.
- **The GPU display-list path re-parsed the SVG the writer had just produced.**
  `__backgroundShapeToDL__` regexed `fill`, `fill-opacity`, `stroke` and
  `stroke-width` back out of the emitted string, and `__backgroundLabelToDL__` did the
  same to the `<text>` element — the same class of defect as the `svgToDisplayList()`
  re-parse route removed from `spreadlinesp` in 0.1.2. Both writers now read the same
  resolved record, which is what lets `stroke-opacity` and `dash` reach the GPU path
  at all. Both methods also moved to `P2SBackgroundMixin`: they were duplicated in
  `xyp` and `linkp`, differing by one dead local.

- **A partial `order=` silently misrepresented the data, in two different ways.**
  Both components took a caller's incomplete ordering and quietly disposed of
  whatever it left out (PLANNING.md **C1**, **C2**).
  - **`chordp` dropped every unlisted node** — its arc *and* its ribbons. `df_node`
    is built from `self.order`, so a 5-node graph rendered with `order=['a','b','c']`
    lost two arcs and five of eleven chords with no warning. (The audit note blamed
    an inner join in `__calculateGeometry_NONPOLARS__`; that method has no call site
    and never runs. The live path drops the nodes in *both* the `node_size='vary'`
    and fixed-size branches.)
  - **`xyp` collapsed every unlisted value onto one shared row/column** — the list
    form via `fill_null(len(order))` and the dict form via `fill_null(max+1)`. The
    categories overplotted each other on a single slot, and because an axis label is
    read from the `__x__`/`__y__` value at `arg_min()`/`arg_max()` of the index, that
    slot was then labelled with whichever collided value happened to sort first. The
    audit note recorded only the dict form; both were affected.

  Unlisted values are now **appended in sorted order, each keeping its own arc or
  slot**, so nothing is hidden and nothing collides — matching what `chordp`'s `pos=`
  path already did. To merge them deliberately, place **`p2s.REMAINDERp`** anywhere in
  the order: everything unlisted collapses into one bucket at the sentinel's position,
  named `remainder`.

  ```python
  p2s.chordp(df, [('fm','to')], order=['a','b','c'])                 # a b c d e …
  p2s.chordp(df, [('fm','to')], order=['a','b','c', p2s.REMAINDERp]) # a b c [remainder]
  p2s.xyp(df, x='k', y='v', x_order=[p2s.REMAINDERp, 'a','b'])       # [remainder] a b
  ```

  The bucket is a data operation, not just a relabelling: `chordp` rewrites the edge
  endpoints and re-aggregates (so edges wholly inside the remainder become self-loops
  on it, and total edge weight is conserved), and `xyp` rewrites the source value so
  the axis label, coloring and brushing all read `remainder`. Merging needs one name to
  stand for many values, so a non-String node/axis column is cast to String and the
  listed keys with it. A listed value literally named `remainder` collides with the
  bucket and raises `ValueError`; `p2s.REMAINDERp` against a struct/tuple `x_order=`
  raises `NotImplementedError` (no single struct value can name the bucket) — the
  append behaviour still applies there.

- **`chordp` allocated arc gaps against the wrong node count.** `_nodes_len_` came from
  `nodes_all` while the arcs are drawn from `df_node`. The two diverge whenever `order=`
  names a node absent from the data (and now whenever a `REMAINDERp` bucket collapses
  several nodes into one arc), so the circle was divided into too few slices and the
  arcs ran past 360° — `order=['a','zz','b']` on a 5-node graph tiled to 432°. Gap
  allocation now counts the arcs actually drawn.

- **Text was measured in one font and rendered in another.** Every text-derived
  coordinate the package emits — where a label is cropped, how wide a legend is,
  how far a link label sits off its edge — is computed from the bundled
  `NotoSans-Regular-subset.ttf` via the baked table in `p2s_font_metrics.py`. The
  emitted markup never said so. `P2STextMixin.default_font` was only ever applied
  by `svgText()`; the raw `<text>` strings that `linkp` (12 of them), `spreadlinesp`
  (5) and `xyp` (1) build by hand named no `font-family` at all, so they inherited
  whatever the host page or viewer supplied, and `chordp`'s two label emitters
  asked for a bare `sans-serif`. Under a wider face a label overflows the width
  `cropText()` trimmed it to fit; under a taller one a link label touches the edge
  it was offset to clear. Neither reads as a metrics bug downstream — an
  overflowing label just looks slightly too long — and no golden could catch it,
  since the SVG goldens compare our markup to our markup and the PNG goldens
  rasterize through a third font engine.
  - **Every component's root `<svg>` now carries `font-family="{default_font}"`**,
    which CSS inheritance carries to every `<text>` beneath it — one site per
    component rather than one per emitter, so a newly added raw emitter is covered
    by construction. `chordp`'s two label emitters name the same face explicitly.
  - Still a *request*, not a guarantee: a machine without Noto Sans installed falls
    back to the generic sans and the measurement is approximate again. Embedding
    the subset as a base64 `@font-face` (~90KB per document) would close that and
    is deliberately not done.
- **`linkp` link-label clearance came from constants calibrated against a font that
  was not being used.** `_labelInk_`'s four per-character-class fractions (0.67em
  ascender, 0.56em short ascender, 0.46em x-height, 0.20em descent) were eyeballed
  against whatever face WebKit picked. Real Noto Sans ascends 0.760em, descends
  0.240em and has a 0.536em x-height, so every class of label sat ~0.5–1.1px closer
  to its edge than intended. `tools/gen_font_metrics.py` now bakes per-glyph ink
  extents (`INK_EXTENTS`, the outlines' yMin/yMax) alongside the advances, and the
  new `Polars2SVG.textInk(txt, txt_h)` — the vertical companion to `textLength()`,
  quantizing size the same way — reads them. Link labels move out by that much;
  nothing else in any component moves.

- **A node could be drawn but absent from the graph, so `x` crashed on it.**
  `createNetworkXGraph()` builds a graph purely from `add_edge()` calls over rows
  that survive `polarsFilterColumnsWithNaNs()`, which drops a whole row when
  *either* endpoint is null — or, for a three-part relationship, when the label
  field is null. `linkp` draws and hit-tests from the same columns *without* that
  row filter, so an entity that only ever appears opposite a null was on screen and
  rubber-band selectable while the graph had never heard of it. Every graph-derived
  interactive op then disagreed with the view: removing the selection (`x`) raised
  `NetworkXError: The node ... is not in the graph` and aborted the whole filter
  after deep-copying the graph, and invert-selection (`q`) could never reach those
  nodes, so they stayed stubbornly unselected. Entities like this now enter the
  graph as **isolated nodes**, restoring the "every drawn node is a graph node"
  invariant without changing what any existing chart renders. Reported against a
  ~69K-node email graph where 21,846 rows carried a null `to`, stranding 486
  entities. Use `null_nodes=True` (above) to give them a visible partner instead.
- **A selection could outlive the stack level it was made on.** `pushStack()`
  re-intersected the selection with the new level's graph but `popStack()` and
  `setStackPostion()` did not. That is harmless for a plain filter push (the parent
  has every node back), but `f`/`F` push a **superset**, so popping off one of those
  — or jumping levels outright — left entities selected that the landing level had
  never had, and the next `x` raised `NetworkXError` again. Both now re-intersect,
  without an mvc broadcast (`display()` pops in a loop to walk the stack, and one
  notification per level would be chatter). `apply_push_selected()` additionally
  uses `remove_nodes_from()` as a backstop, which ignores an unknown name instead of
  raising, and the `e` / `Q` key ops guard their neighbor lookups the way `E`
  already did.
- **`x` left orphaned nodes in the pushed graph.** `apply_push_selected()` hands
  its surgically-edited graph to `pushStack()` rather than paying to rebuild one
  from the filtered dataframe, but `filterDataFrameByGraph()` keeps only rows that
  are edges — so a leaf whose only neighbor was just removed lost all of its rows
  while staying in the graph. The pushed level then held nodes it did not draw, and
  invert-selection handed back names with nothing on screen behind them. The graph
  is now trimmed to the entities the filtered frame actually contains.
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

- **Dashed strokes restarted their dash pattern at every vertex on the WebGPU path.**
  SVG runs `stroke-dasharray` continuously along a whole `<path>`, but each flattened
  segment reached the shader as an independent instance whose dash offset began at zero
  — so a dashed polyline or curve showed a dash boundary at every vertex, most visibly
  on `xyp`'s `LINESTYLE_DOTTED` / `LINESTYLE_SPECIFIED` lines. The line instance now
  carries a twelfth float, the arc length already travelled along the parent stroke,
  which `line_vs` adds to the fragment's distance. Emitted by `xyp`'s polyline path and
  by the `svgToDisplayList()` path/polygon stroke flattener; solid strokes are unaffected
  and pay nothing for it.

- **`linkp`'s collapsed-node cloud drew as a plain circle on the WebGPU path** while
  `spreadlinesp`'s drew as a rounded rect, so the same collapsed node looked like two
  different things depending on which component rendered it. Both now use one shared
  approximation of the `<use href="#cloud">` icon (`CLOUD_ICON_W/H/RX` in
  `p2s_displaylist.py`), which is also what the SVG parser has always produced.

- **Three places still said WebGPU covered "currently xyp, histop".** All eight static
  components have had a `webgpu()` representation for some time; `panelize()`'s
  docstring, `webgpuHTML()`'s docstring and the `panelize()` dispatch comment now say so.

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

- **Coordinates are rounded where they are generated**, shrinking SVG output with
  no visible change. Three emitters interpolated raw floats and serialized 13-16
  fractional digits:
  - **`linkp` curve/flowmap control points** — the endpoints of every `<path>`
    were already whole pixels (screen coordinates are `Int32`), so the control
    points steering them carried precision that could not mean anything. Rounded
    to 2 decimals at the point of string assembly; the underlying columns keep
    full precision for arrowhead geometry, link labels and the GPU display list.
    A 120-node/600-edge curve render drops **233,863 → 215,805 bytes (-7.7%)**.
  - **`xyp`'s `r=` and `fill-opacity=`** — both come from a normalization ratio.
    Radius rounds to 2 decimals (1/100 px), opacity to 3 (alpha renders at 8
    bits, so 2 would quantize a ramp more coarsely than the renderer does).
    `dot_size='sz'` on 4,000 rows drops **254,754 → 201,199 bytes (-21.0%)**.
    These round the column itself, so the SVG string, the `group_by` keys and the
    GPU instance buffer all stay derived from one value.
  - **`xyp`'s supersampled raster coordinates** — a `dot_size/dot_size_supersample`
    step is rarely representable (4/3 → 1.333…). `dot_size=4,
    dot_size_supersample=3` drops **193,096 → 165,751 bytes (-14.2%)**. The
    `dot_size_supersample=1` path is untouched and remains byte-identical.

  Maximum coordinate movement is half a quantum (0.005 px). Note for anyone
  reading `df_pixels` directly: `__radius__` and `__fill_opacity__` are now
  quantized in the dataframe, not only in the emitted string.

### Fixed

- **`TFDPLayout`'s "mlx is required" error sent Linux users back to the install that
  had just failed.** PyPI's `mlx` is a front-end that ships no backend library: on
  macOS it pulls Metal automatically, on Linux it pulls nothing, so
  `polars2svg[mlx]` — and `polars2svg[all]`, which chains it — installs cleanly and
  then dies at `import mlx.core` with `ImportError: libmlx.so: cannot open shared
  object file`. The handler answered that with `pip install polars2svg[mlx]  #
  Apple silicon (Metal), or CPU elsewhere`, i.e. the extra already installed. It now
  separates the two cases: mlx genuinely absent gets the extras (including
  `[mlx-cuda13]`, previously unnamed), and mlx present-without-a-backend gets the
  backend distribution to add — `mlx[cpu]`, `mlx[cuda12]` or `mlx[cuda13]` — plus the
  underlying error. `README.md`, `CONTRIBUTING.md` and the `pyproject.toml` extras
  comment carried the same "CPU elsewhere" claim and now describe what Linux actually
  resolves.

  Only `TFDPLayout` is affected — `ODFlowLayout` runs on NumPy and reaches for MLX
  only when MLX works. The `[mlx-cpu]` extra below is the other half of this fix.

### Added

- **`[mlx-cpu]` extra — an MLX install that actually works on Linux.**
  `pip install polars2svg[mlx-cpu]` pulls the CPU backend distribution that PyPI's
  `mlx` does not, so a Linux host without an NVIDIA GPU gets a working `TFDPLayout`
  from one command. It completes the set: `[mlx]` (macOS/Metal), `[mlx-cpu]`,
  `[mlx-cuda]` (CUDA 12), `[mlx-cuda13]` (CUDA 13).

  **Install exactly one backend extra.** Every MLX backend ships the same
  `mlx/lib/libmlx.so`, so two in one environment is last-one-installed-wins, with no
  error from pip or uv — enough to leave a CUDA install running on the CPU backend
  after downloading a gigabyte of CUDA wheels. This is also why **`[all]` still
  chains plain `mlx` and no backend**: it keeps `polars2svg[all,mlx-cuda13]` exact
  rather than a race. On Linux, `[all]` alone therefore still has no MLX backend —
  pair it with one, e.g. `polars2svg[all,mlx-cpu]`.

  **Windows has no MLX backend distribution at all.** `mlx` publishes Windows
  front-end wheels, so `[mlx]` and `[all]` install there, but every backend is gated
  to Linux or macOS and `TFDPLayout` cannot run. Nothing else in polars2svg is
  affected.

### Removed

- **`Polars2SVG.roundSvgFloats()`**, along with the eight per-component calls to it.
  The helper trimmed float tails by regex over a *finished* SVG string — a pass that
  cannot tell a coordinate from a label, which is how the node label `1.172.32.1` came
  to render as `1.17.32.1`, taking numeric axis ticks with it. It has been an inert
  `return svg` since that was found, so **removing it changes no output**: every
  component emits the same bytes it did before. Anyone still calling it was getting
  their input back unchanged.

  **Not replaced, deliberately.** Float precision belongs where the number becomes a
  string — the rounding already applied in the emitting expressions (linkp's curve
  control points, xyp's `r=` and `fill-opacity=`) — because that is the only place that
  knows whether a run of digits is geometry or text. An attribute-only rewrite was
  measured first and rejected: it saves a flat 151–213 bytes per file, does not grow
  with row count (0.14% of a 4,000-row scatter), and would have regenerated 52 of 68
  golden images to buy it.

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

[Unreleased]: https://github.com/datrcode/polars2svg/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/datrcode/polars2svg/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/datrcode/polars2svg/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/datrcode/polars2svg/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/datrcode/polars2svg/releases/tag/v0.1.0
