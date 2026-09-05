# Security Policy

## Supported Versions

`polars2svg` is pre-1.0 (currently `0.2.x`). Only the latest released version
on PyPI is supported — there are no maintained backport branches. Please
upgrade to the latest release before reporting an issue.

| Version | Supported |
| ------- | --------- |
| 0.2.x   | ✅        |
| < 0.2   | ❌        |

## Reporting a Vulnerability

This is a single-maintainer project. Please report suspected security issues
privately by email to **dave.trimm+polars2svg@gmail.com** rather than opening
a public GitHub issue. Include a minimal reproduction (a DataFrame + component
call) if possible. Expect an initial response within a few days; there is no
formal SLA.

## Deployment Model

`polars2svg` is built for **one trusted person in one process** — a Jupyter
notebook, or a Panel view its own author opens on their own machine. Every
statement in the threat model below assumes that. Two consequences are worth
saying outright, because nothing in the code enforces them:

- **The interactive components are not a multi-tenant application.**
  `panelize()`, `linkpi()`, `xypi()` and the rest return Panel `ReactiveHTML`
  views whose parameters synchronise in **both** directions. The browser writes
  `search_str`, the picker-menu `*_choice` strings, mouse coordinates and key
  events; the Python handlers act on them as keystrokes from the person sitting
  in front of the figure. There is no notion of a user in any of it — nothing
  authenticates, authorises, or rate-limits, and any client that can write those
  parameters can do anything the keyboard can do, including navigating the
  dataframe stack and starting layout operations. Serving one of these views to
  an audience you do not trust is outside what the library is designed for.

- **`Polars2SVG()` is a process-wide singleton, and so is its configuration.**
  Every call returns the same instance — documented behaviour that component
  constructors rely on, not an implementation detail — so `set_defaults()`,
  `reset_defaults()` and `setColorOverrides()` reach every render in the
  process. In the intended single-user case that is exactly the point: configure
  once and every figure follows. Under a server handling several sessions it is
  one session's settings silently changing another session's output. The
  interactive controllers' dataframe stacks are per-view rather than per-session
  in the same way.

The limits that do exist in the interactive code are **cost** bounds, not trust
boundaries. `linkpi` bounds the search box's regex (a pattern-length cap plus a
whole-scan deadline), caps the dataframe stack's depth, asks before running an
expensive layout on a large graph, and warns when a composed document would
exceed the Bokeh WebSocket message limit. Each of those keeps an honest mistake
from wedging a session; none is a defence against a hostile client.

## Threat Model

`polars2svg` renders a Polars `DataFrame` you already have in-process into an
SVG (and optionally rasterizes that SVG to PNG, or drives a WebGPU/interactive
Jupyter widget). It does not fetch data over the network, execute code from
the DataFrame, or evaluate untrusted expressions.

**What is treated as untrusted (may contain adversarial content):**

- String *data* — column values used as text: axis labels, chord/link node
  and edge labels, histogram/pie category labels, timestamp labels, node
  labels supplied via `node_labels=`. This is the data most likely to
  originate from an external or user-controlled source (filenames, user
  handles, free-text fields, etc.).

**What is treated as trusted (caller-supplied configuration, not sanitized):**

- Component parameters: colors, fonts, dimensions (`wxh=`), enum choices,
  file paths passed to `save()`/`savePNG()`. These come from your own code,
  not from row data, and are not escaped or validated for injection —
  passing attacker-controlled strings here (e.g. a color string built from
  untrusted input) is a misuse of the API, not a vulnerability in it.

- Values the interactive widgets synchronise back from the browser:
  `search_str`, the picker `*_choice` strings, mouse coordinates, key events.
  They arrive over a WebSocket but stand for the local user's own keystrokes
  (see *Deployment Model*). The ones that select an operation —
  `layout_operation`, `background_operation`, `key_op_finished` — are looked
  up in fixed registries, so a value naming no operation is inert rather than
  dangerous; the rest are bounded for cost and otherwise used as given.

- SVG strings passed to `tile()` as `svg_list`. `tile()` composes
  already-rendered SVG by wrapping each child in a `<g transform=…>` and
  embedding it **verbatim** — that is the component, and it is what lets it
  compose renderings this library did not produce. Whatever you hand it reaches
  the output unchanged, which makes it configuration in the same sense `color=`
  is. Do not build an `svg_list` entry out of anything you would not paste into
  the page yourself.

**How untrusted string data is handled:**

- All body text rendered via `svgText()` / `svgAxisLabels()`
  (`polars2svg/p2s_text_mixin.py`) is passed through `html.escape()` before
  being embedded in an SVG `<text>` element.
- `chordp` labels (`polars2svg/chordp.py`) are escaped with `html.escape()`
  at the point they are drawn, including labels remapped through
  `node_labels=`.
- `linkp` labels (`polars2svg/linkp.py`) are escaped via Polars
  `str.replace_all()` for `&`, `<`, `>` before being wrapped/wrapped into
  `<text>` elements.
- Interactive-widget state passed to the browser (menu state, selection
  state) is serialized with `json.dumps()`, not string-concatenated into
  script bodies.
- Synthetic SVG element `id`s (used to scope inline `<style>`/`<script>` per
  figure so multiple figures can coexist in one notebook) are generated with
  `random.randint()`, not derived from row data.

`tests/test_edge_case_inputs.py` and `tests/test_p2s_text_mixin.py` lock this
behavior — they render DataFrames containing `&`, `<`, `>`, `"`, and a raw
`<script>` payload through every labeled component and assert the resulting
SVG both parses as well-formed XML and never contains an unescaped tag
originating from row data.

**Known non-goals:**

- `polars2svg` does not sanitize DataFrame values you supply as component
  *configuration* (e.g. a `color=` string). Only the render targets listed
  above receive automatic escaping.
- The rendered SVG is not evaluated against a strict allow-list of SVG
  features; it relies on well-formed, escaped text content plus the fact
  that all structural markup (paths, shapes, ids) is generated by
  `polars2svg` itself from numeric geometry, not from arbitrary string data.
- There is no session isolation, and the interactive widgets' synchronised
  parameters are not an authorisation boundary. Serving the interactive
  components to several mutually untrusting users from one process is not
  supported — see *Deployment Model* for what that costs in practice.
