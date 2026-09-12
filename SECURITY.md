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

## Supported Deployment Profiles

polars2svg supports three deployment profiles. A profile is not advice: each one
states what is **enforced**, by which mechanism, and which test proves it. A
deployment that is not one of these is unsupported — not because it is forbidden,
but because nothing here has been checked against it.

| Profile | Shape | Status |
| --- | --- | --- |
| **Notebook** | One trusted person, one process | Fully supported |
| **Profile A — Appliance** | A server renders SVG from untrusted data; the output is embedded in a page others view | Supported for the render path |
| **Profile B — Interactive, multi-user** | One process serves the live interactive widgets to several mutually untrusting people | **Out of scope** |

### Notebook (the default)

A Jupyter notebook, or a Panel view its own author opens on their own machine.
Everything in the library is available, and the threat model below describes what
is escaped and what is not. This is what polars2svg was built for.

### Profile A — Appliance

You render with polars2svg on a server, from data you do not control, and embed
the resulting SVG in a page someone else loads. What makes this supportable is
not a promise that every label is escaped — it is that the **output is
checkable**, because polars2svg generates every element and attribute in a render
itself, from numeric geometry and a fixed vocabulary.

**What the library enforces.** `polars2svg/svg_contract.py` defines an allow-list
over the rendered document: a conforming render contains only the 16 elements and
44 attributes the components actually emit, every reference (`href`,
`url(#…)`) resolves inside the same document — in `<style>` text as well as in
attributes, since CSS fetches too — no attribute is an event handler, no value
carries a `javascript:`/`vbscript:`/`data:` scheme, and the document
carries no DOCTYPE. `checkOutputContract()` returns the violations;
`assertOutputContract()` raises. Both are public API — **call
`assertOutputContract()` on anything you are about to serve.** It is the
enforcement point, and it reports rather than rewrites: a violation means the
render is not fit to serve, not that it has been made fit.

**What proves it.** `tests/test_profile_conformance.py` runs that contract over
the entire golden corpus and over every payload in `tests/injection_corpus.py`
driven through every untrusted-text surface in every component — currently 27
hostile strings × 20 surfaces. The corpus itself is checked for vacuity (a render
that silently dropped the label would otherwise pass trivially), the checker is
checked against hand-written violating documents (so the suite cannot go green
with a broken checker), and a coverage ratchet fails the suite if a new component
ships without injection cases.

**What you are still responsible for.**

- **`tile()` is outside this profile.** It embeds foreign SVG verbatim — that is
  the component, not a defect — so it makes no promise about markup it did not
  produce. Do not build an `svg_list` entry from untrusted input.

  The contract is not *blind* to it, though, and the difference matters: because
  `checkOutputContract()` reads the composed document, hostile markup embedded
  through `svg_list` is reported like any other violation. So an appliance that
  gates its responses on `assertOutputContract()` is covered even when it tiles
  markup it did not write. Being caught at the gate is not the same as being safe
  to compose, which is why the advice above stands.
- **Path parameters are inputs in this profile.** `save()`, `savePNG()`,
  `savePositions()` and `loadPositions()` take a path from the caller and are
  classed as trusted configuration below. Never forward a request field into one.
- **Serve it correctly.** Send a `Content-Security-Policy`, and do not serve a
  render from its own URL as `image/svg+xml` on a domain that matters: a
  standalone SVG document is a same-origin script context. Embed it, or serve it
  from a separate origin.
- **The contract is about markup polars2svg generates.** It is a strong, tested
  property. It is not a proof that no XSS is possible anywhere in your page.

### Profile B — Interactive, multi-user: out of scope

**Serving the interactive components to several mutually untrusting people is out
of scope.** `panelize()`, `linkpi()`, `xypi()`, `spreadlinepi()` and the
`ReactiveHTML` views they return are **not** hardened for it, are not covered by
the conformance suite, and will not be. This is a deliberate decision, not a
backlog item.

Concretely, and none of this is hypothetical:

- The views' parameters synchronise in **both** directions. The browser writes
  `search_str`, the picker `*_choice` strings, `layout_operation`, mouse
  coordinates and key events; the Python handlers act on them as keystrokes from
  the person sitting in front of the figure. Nothing authenticates, authorises,
  or rate-limits, so any client that can write those parameters can do anything
  the keyboard can do.
- Not every one of those values is validated before use, and at least one reaches
  the widget's DOM through `innerHTML`. In a notebook that is self-inflicted; with
  two people in one process it is not.
- The rendered SVG reaches the browser through a binding that is deliberately
  exempt from Panel's HTML sanitizer (it would otherwise strip the plot to
  nothing), so the interactive path has no sanitizer under it.
- The interaction controller's view registry is append-only and keyed by object
  identity, and its dataframe stacks are per-view. One process shared across
  sessions shares those.
- The limits that exist in the interactive code — the search regex bound, the
  stack depth cap, the confirm gate on expensive layouts, the WebSocket payload
  warning — are **cost** bounds that keep an honest mistake from wedging a
  session. None is a defence against a hostile client.

If you need interactive views for several users, run one process per user behind
your own authentication and treat each process as single-tenant. That is the
supported shape.

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
  (see *Supported Deployment Profiles*). The ones that select an operation —
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

  What that verbatim embedding does **not** reach is the filesystem.
  `savePNG()` / `save('*.png')` rasterize through svglib, which resolves
  `xlink:href` on `<image>` and `<use>` against the *source path* of the document
  it parsed — so rasterizing from a file path would turn an external reference in
  a tiled child SVG into a local file read whose contents land in the exported
  PNG. `polars2svg/export.py` hands svglib a stream rather than a path, which
  leaves svglib with no base path to resolve against and makes it decline
  external references outright; relative, absolute, traversal, `file://` and
  external-`<use>` spellings are all inert.
  `tests/test_export_save.py::TestRasterizeDoesNotReadLocalFiles` pins this end
  to end, so it holds as a tested property rather than an accident of how the
  rasterizer happens to be called. Trusting `svg_list` therefore means trusting
  it with your *page*, not with your disk.

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

`tests/test_profile_conformance.py` is the broader check behind Profile A: 27
hostile strings — attribute-breakout, pre-encoded entities, CDATA and comment
escapes, `javascript:`/`data:` URLs, bidi overrides — driven through all 20
untrusted-text surfaces and asserted **structurally**, against the allow-list in
`polars2svg/svg_contract.py`, rather than by substring. The distinction matters:
a substring assertion for `<script>` passes unchanged against
`<img src=x onerror=…>`, which is how a single-payload test can stay green
through a real regression.

**Known non-goals:**

- `polars2svg` does not sanitize DataFrame values you supply as component
  *configuration* (e.g. a `color=` string). Only the render targets listed
  above receive automatic escaping.
- The output allow-list (`polars2svg/svg_contract.py`) is **not applied
  automatically.** Rendering does not call it; it is a gate a caller puts in
  front of what it serves, and Profile A asks you to do exactly that. Nothing in
  the library refuses to return a non-conforming document, because a render that
  fails the contract is a bug to report, not a condition to handle at runtime.
- The contract covers markup `polars2svg` generates. `tile()` is outside it by
  construction, and no allow-list over our output says anything about the rest of
  the page you embed it in.
- **Stylesheet text is allow-listed by reference, not by grammar.** Every
  `url(…)` inside a `<style>` block is range-checked exactly like an attribute
  value — quoted or bare, it must be a same-document `#fragment` — but the
  contract does not parse CSS, so it cannot allow-list *properties*. The
  constructs that fetch or execute without naming a `url()` (`@import`,
  `expression()`, `image-set()`, a `javascript:` URL) are refused by name
  instead. That is a deny-list, and a deny-list only excludes what somebody
  thought of: a future CSS fetch primitive spelled some third way would pass.
  polars2svg's own renders emit a small fixed stylesheet, so this seam matters
  only for foreign CSS — which reaches the document solely through `tile()`, and
  `tile()` is already outside the profile.
- There is no session isolation, and the interactive widgets' synchronised
  parameters are not an authorisation boundary. Serving the interactive
  components to several mutually untrusting users from one process is out of
  scope — see *Profile B* above for the specifics.
