# polars2svg

**Fast, consistent, linkable views of event and relationship data in Jupyter —
including the network and flow views other libraries don't do — rendered as plain
SVG.** Hand a [Polars](https://pola.rs) `DataFrame` to a component and get a
self-contained SVG back. Brush any interactive view and every other view in the
dashboard follows, the network graph included.

<p align="center">
  <img src="https://raw.githubusercontent.com/datrcode/polars2svg/main/docs/images/xyp_scatter.png" alt="Scatter plot" width="32%">
  <img src="https://raw.githubusercontent.com/datrcode/polars2svg/main/docs/images/linkp_network.png" alt="Network graph" width="32%">
  <img src="https://raw.githubusercontent.com/datrcode/polars2svg/main/docs/images/chordp_flows.png" alt="Chord diagram" width="32%">
</p>

## Why polars2svg

It is built for exploratory analysis of records that have a time, a few categories
and relationships between entities: logs, network flows, transactions,
communications. The goal is to look at many small views of the same data quickly.

- **The same value gets the same color in every chart.** Categorical colors are
  derived from the value itself, so `'tcp'` is the same color in a histogram, a
  timeline and a network graph without coordinating legends, and it stays that
  color in the light and dark palettes alike. Across a dozen small views, this is
  what lets the eye connect them.
- **Size and color answer different questions.** A bar can be sized by bytes and
  colored by how many rows produced it, so "few events, lots of traffic" stands
  out without pre-aggregating by hand. See
  [Encoding data](#encoding-data-count-and-color).
- **Linked brushing that includes graphs.** Seven components have interactive
  variants that share one selection through `p2s.panelize()`. The node-link and
  chord views take part as fully as the scatter plot and the histogram.
- **Network and flow layouts from the literature.** t-FDP, neighborhood-preserving
  circle packing, force-directed origin-destination flow maps, hierarchical edge
  bundling, Pivot and Landmark MDS, and IPv4-subnet layouts. Each is cited in
  [References](#references). See [Network and flow views](#network-and-flow-views).
- **The output is a string.** Every chart is a self-contained SVG with no
  JavaScript, which you can save, diff, embed in HTML, or check against an
  [output contract](#security--deployment) before serving it.
- **A slim base install.** Polars and NumPy, plus PyArrow and Pillow for I/O.
  Everything heavier is an opt-in extra.

## When something else fits better

- **Polars input alone isn't a reason to switch.** Altair, Plotly and hvPlot accept
  a Polars `DataFrame` directly now.
- **A chart type that isn't in the [component list](#components), or a fully custom
  encoding.** The components here are fixed chart types, not a grammar of graphics.
  Altair or plotnine will get you there.
- **Publication figures with fine typographic control.** Use matplotlib.
- **Millions of marks in a static image.** An SVG grows with every mark it draws.
  Aggregate to a raster instead, for example with Datashader. The interactive
  views can render through WebGPU (`use_webgpu=True`), but the static SVG cannot.
- **A web dashboard served to many people.** The interactive layer runs one trusted
  user per process. See [Security & deployment](#security--deployment).

## Install

```bash
pip install polars2svg
```

The base install covers `xyp`, `histop`, `timep`, `piep`, `smallp`, and
`spreadlinesp`. Everything else lives behind extras:

```bash
pip install polars2svg[layouts]     # linkp, chordp, and shapely-typed background= shapes
pip install polars2svg[interactive] # panelize / xypi / histopi / ... (includes layouts)
pip install polars2svg[export]      # component.save('chart.png')
pip install polars2svg[all]         # all of the above, plus the MLX GPU layout
```

Calling a component that needs an extra you haven't installed (e.g. `chordp()`
or `panelize()`) raises a clear `ImportError` naming the extra to install.

The GPU-accelerated `TFDPLayout` has its own extras, and on Linux `[all]` alone
leaves it unable to load. See [GPU-accelerated layouts](#gpu-accelerated-layouts)
before installing on Linux or Windows.

Requires **Python ≥ 3.12** and **Polars ≥ 1.36** (1.x and 2.x).

## Quickstart

```python
import polars as pl
from polars2svg import Polars2SVG

p2s = Polars2SVG()

df = pl.DataFrame({
    "x":     [1, 2, 3, 4, 5, 6],
    "y":     [3, 1, 4, 1, 5, 9],
    "group": ["a", "b", "a", "b", "a", "b"],
})

# In a Jupyter notebook, the returned object renders itself as SVG.
p2s.xyp(df, "x", "y", color="group", dot_size=6, wxh=(400, 300))
```

Every component returns an object with a `.svg` attribute (the raw SVG string)
and a Jupyter `_repr_svg_`, so the last expression in a cell displays the chart
inline. To save it:

```python
chart = p2s.xyp(df, "x", "y", color="group", dot_size=6, wxh=(400, 300))
open("scatter.svg", "w").write(chart.svg)
```

### If you arrived here from a snippet that doesn't work

A code sample along these lines circulates in search results and AI-generated
answers. **It is not polars2svg API and never has been, in any released version:**

```python
# ✗ Does not work — there is no `display_svg` in polars2svg.
from polars2svg import display_svg
display_svg(df)
```

The name appears to be borrowed from IPython's `IPython.display.display_svg`, a
different function that takes SVG data rather than a DataFrame. polars2svg has no
"render a DataFrame as a table" entry point at all: every component encodes named
*fields* into a chart, so no single-argument function could know what you wanted
drawn. Reach for one of the [components](#components) instead:

```python
p2s.histop(pl.DataFrame({"Engine": ["Polars", "Pandas", "Spark"],
                         "Speed":  [30.5, 1.0, 12.3]}),
           "Engine", count="Speed", wxh=(400, 200))
```

## A worked example

A few network flows, viewed three ways and tiled into one SVG:

```python
from datetime import datetime

import polars as pl
from polars2svg import Polars2SVG

p2s = Polars2SVG()

flows = pl.DataFrame({
    "ts":    [datetime(2026, 9, d, h) for d, h in
              [(1, 9), (1, 14), (2, 3), (3, 22), (4, 9), (5, 11), (6, 2), (7, 16)]],
    "src":   ["10.0.0.5", "10.0.0.5", "10.0.0.7", "10.0.1.9",
              "10.0.0.7", "10.0.1.9", "10.0.0.5", "10.0.1.3"],
    "dst":   ["10.0.1.9", "8.8.8.8",  "10.0.1.9", "8.8.8.8",
              "10.0.0.5", "10.0.1.3", "8.8.8.8",  "10.0.0.7"],
    "proto": ["tcp", "udp", "tcp", "udp", "tcp", "tcp", "udp", "tcp"],
    "bytes": [1_200, 64, 980_000, 80, 3_400, 15_000, 72, 2_100],
})

# Bar length is total bytes; bar color is how many flows produced it.
by_bytes = p2s.histop(flows, "src", count="bytes", color=p2s.CROW_MAGNITUDEp, wxh=(260, 160))
# Row count per source, split by protocol.
by_proto = p2s.histop(flows, "src", color="proto", wxh=(260, 160))
# Every day folded onto one 24-hour cycle.
by_hour  = p2s.timep(flows, p2s.tField("ts", p2s.PT_Hp), color="proto", wxh=(260, 160))

p2s.tile([by_bytes, by_proto, by_hour])
```

The `proto` colors match across the second and third charts, and they will match
in the network graph below, because each color comes from the value rather than
from the chart.

## Components

Each is a method on a `Polars2SVG` instance and takes a `pl.DataFrame` plus the
fields to encode.

| Component | Call | What it draws |
|-----------|------|---------------|
| **xyp** | `p2s.xyp(df, x, y, ...)` | Scatter / distribution plot; x and y can be numeric or categorical. |
| **histop** | `p2s.histop(df, field, ...)` | Horizontal histogram bars, one per category/bin. |
| **timep** | `p2s.timep(df, ts_field, ...)` | Temporal bar chart with linear or periodic (day-of-week, month, …) time modes. |
| **linkp** | `p2s.linkp(df, relationships, ...)` | Node-link graph / network with pluggable layouts. |
| **chordp** | `p2s.chordp(df, relationships, ...)` | Chord diagram of weighted flows around a circle. |
| **piep** | `p2s.piep(df, field, ...)` | Pie chart. |
| **spreadlinesp** | `p2s.spreadlinesp(df, ...)` | Egocentric radial "spread" rings for influence/propagation. |
| **smallp** | `p2s.smallp(df, ...)` | Small multiples — a grid of one template component faceted by a field. |

Seven of the eight have an interactive, cross-linked variant — `xypi`, `histopi`,
`timepi`, `linkpi`, `chordpi`, `piepi` and `smallpi` (there is no `spreadlinespi`).
Each one **wraps a rendering rather than a DataFrame**, and `p2s.panelize()`
composes them into a dashboard as a list of rows, sharing one selection across
every view in it:

```python
df = pl.DataFrame({
    "x":     [1, 2, 3, 4, 5, 6],
    "y":     [3, 1, 4, 1, 5, 9],
    "group": ["a", "b", "a", "b", "a", "b"],
})

xi = p2s.xypi(p2s.xyp(df, "x", "y", color="group", wxh=(400, 300)))
hi = p2s.histopi(p2s.histop(df, "group", wxh=(400, 300)))

p2s.panelize([[xi, hi]])      # one row — brushing either view filters both
p2s.panelize([[xi], [hi]])    # two rows, stacked
```

`panelize()` also accepts bare static components and wraps them for you. Pass
`use_webgpu=True` to an interactive variant to render it through WebGPU.

Press `h` in any interactive view for its key bindings, and `a` for the settings
panel — a live display of how the view is drawn (and, on `linkp`, of what its layout and
background keys will do), editable in place. On `xypi`, `histopi`, `timepi`, `chordpi`
and `piepi` its rows re-render the view: legend, colour scale, style, count and order,
the marks' sizes and opacity, and on `timepi` the time granularity. A panel you leave
alone changes nothing. One of its rows is
the **hover tooltip**: rest the pointer on a mark and it says what is under it, as text
or as a whole component re-rendered against just those records.

```python
icon = p2s.xyp(df, "x", "y", wxh=(32, 32))     # any component works
xi   = p2s.xypi(p2s.xyp(df, "x", "y", color="group", wxh=(400, 300)), icon=icon)
```

Tooltips are off until you turn them on: open the panel with `a` and cycle the
`tooltip` row with `space` (`off -> text -> icon`; `icon` is offered only when you
passed one). Hovering never changes the selection, the stack or any linked view.

What you have selected comes back to Python as a DataFrame. Read it from the layout
`panelize()` returns, or from any one view:

```python
layout = p2s.panelize([[xi, hi]])
layout                                  # display it, drag in a view, then:
layout.selectedDataFrame()              # the rows the views are showing

@layout.onSelection
def picked(df):                         # called after every selection change
    print(df.height, "rows selected")
```

The selection is what the views show: the whole frame until you drag, pick or filter,
then that subset, and back as you pop the stack. On `linkpi`, selected nodes narrow it to
the rows with a selected node at either end. Brushing is a preview and changes nothing
here. The DataFrame has your input columns.

Finished renderings compose into a single static SVG with `p2s.tile(...)` — the one
method that takes renderings rather than a DataFrame:

```python
p2s.tile([chart_a, chart_b])                    # side by side
p2s.tile([chart_a, chart_b], per_row=1)         # stacked
p2s.tile(charts, per_row=4, spacer=(8, 24))     # a 4-across grid
p2s.tile(charts, per_row=4, wxh=(640, 480))     # ... scaled to fit 640x480
```

Elements may be components, SVG strings, or other tiles. `per_row=` is the whole layout
model: `None` (the default) is a single row, `<n>` fills a grid row by row, and `1` is a
single column. `spacer=` is one gap in both directions or a `(horizontal, vertical)`
pair, and `wxh=` scales the whole tiling — uniformly and centered — into a canvas of that
size.

## Network and flow views

`linkp` places nodes from `pos=`, a networkx-style `{node: (x, y)}` dict, so any
layout that produces one works. polars2svg ships several of its own, for graphs
where a generic spring layout falls short:

- **`p2s.ipSubnetTreeMapLayout()`** and **`p2s.ipSubnetForceDirectedLayout()`**
  group IPv4 nodes by subnet.
- **`p2s.hyperTreeLayout()`** draws a radial tree.
- **`PivotMDSLayout`** and **`LandmarkMDSLayout`** scale to large graphs, and
  **`TFDPLayout`** is the t-distribution force-directed layout, which runs on the GPU
  ([GPU-accelerated layouts](#gpu-accelerated-layouts)). Each takes a networkx graph
  and returns `pos` from `.results()`.
- **`link_shape='flowmap'`** bends each edge with a force-directed
  origin-destination flow layout, so that overlapping flows separate.
- **`p2s.chordp()`** draws weighted flows around a circle and can bundle its links
  with hierarchical edge bundling.

Continuing the worked example, the same flows as a graph laid out by subnet:

```python
g   = p2s.createNetworkXGraph(flows, [("src", "dst")], count="bytes")
pos = p2s.ipSubnetTreeMapLayout(g, subnet_mask=24)

p2s.linkp(flows, [("src", "dst")], pos, color="proto",
          count="bytes", link_size="vary", wxh=(360, 260))
```

These need the `layouts` extra.

## Encoding data: `count=` and `color=`

Two parameters recur across the counting/aggregating components, and they are
**orthogonal** — one controls *size/magnitude*, the other controls *color*.

### `count=` — "how big / how much is this element?"

The aggregation rule is shared by every component that takes `count=`:

| `count=` spec | Aggregation |
|---------------|-------------|
| `p2s.ROW_COUNTp` *(default)* | `pl.len()` — number of rows |
| a numeric field | `sum` of that field |
| a non-numeric field, or `('field', p2s.SETp)` | `n_unique` (distinct count) |
| a multi-field tuple | struct the fields, then `n_unique` |

What `count=` visibly does depends on the component. At default settings it is
the primary size knob for **histop** (bar length) and **timep** (bar height); it
nudges the derived node order in **chordp**; and for **linkp**/**chordp** it only
drives geometry once you opt into `node_size='vary'` / `link_size='vary'`.

**Negative counts.** A bar in histop or timep grows from zero, so a bin whose
`count=` sums below zero is drawn as zero — an empty slot, not a bar running the
other way. In a stacked bar each colour's segment is clamped on its own, so a
negative segment draws as nothing and the rest of the bar is unaffected. Either
way a warning names the component, the count field and the bins, once per case.

### `color=` — "what color is this element?"

Independent of `count=`. A bare field infers its meaning from the column dtype
(numeric → magnitude spectrum, otherwise → categorical), and enums let you pin
intent explicitly — e.g. `('field', p2s.CSETp)` forces categorical,
`('field', p2s.CMAGNITUDE_SUMp)` forces a numeric spectrum.

The `CROW_MAGNITUDEp` / `CROW_STRETCHEDp` color modes always color by **raw row
count** (`pl.len()`), regardless of what `count=` is set to — so a bar sized by
`count='bytes'` can still be colored by how many rows landed in it. Because the
two encodings are orthogonal, a tall bar can be cold-colored (many bytes, few
rows) or vice-versa — that is by design.

## Appearance: palettes and color overrides

`color=` above says how *data* is encoded. The framework's own furniture — canvas,
axes, labels, borders, indicators — comes from a **palette**, chosen per instance:

```python
p2s = Polars2SVG(palette='dark')        # 'light' (the default) or 'dark'
p2s.setPalette('light')                 # or switch an existing instance
```

A palette can be adjusted without replacing it. Keys are `(type, subtype)` slots:

```python
p2s.setPalette('dark', {('background', 'default'): '#000000'})
```

To pin a color to a specific *data value* — as opposed to the furniture — use
`setColorOverrides()`, which takes cell values rather than slots:

```python
p2s.setColorOverrides({'prod': '#cc0000', 'staging': '#888888'})
p2s.removeColorOverrides(['staging'])
```

An override outranks the palette and survives a `setPalette()` call: it is an
explicit instruction about one value, not a default for chart furniture.

Both settings are **instance-wide**, like `set_defaults()` — build a second
`Polars2SVG()` when you want a second style. Nothing is global to the process, so
two instances never interfere.

Dark tunes the framework's furniture and the grayscale distribution ramp. Data
colors — the ColorBrewer Spectral spectrum and the hash-derived categoricals — are
the same in both palettes, so the same value keeps the same color and two figures
stay comparable across palettes.

The palette reaches the **interactive** components too: the selection rectangle, the
layout guides and the status line follow it, so a drag stays visible on a dark canvas.
The tooltip and configuration panel still draw on their own light surface.

## GPU-accelerated layouts

`TFDPLayout` runs on [MLX](https://github.com/ml-explore/mlx). On macOS with Apple
silicon, `pip install polars2svg[mlx]` is all it needs. Linux and Windows need the
notes below.

**On Linux, `[mlx]` and `[all]` are not enough.** PyPI's `mlx` is a front-end that ships
no backend library; on macOS it depends on the Metal backend unconditionally, on Linux it
pulls nothing. The install succeeds and `import mlx.core` then fails with `ImportError:
libmlx.so: cannot open shared object file`. Choose the backend explicitly:

```bash
pip install polars2svg[mlx-cpu]     # Linux, no NVIDIA GPU
pip install polars2svg[mlx-cuda13]  # Linux + NVIDIA, CUDA 13 toolkit
pip install polars2svg[mlx-cuda]    # Linux + NVIDIA, CUDA 12 toolkit
```

Combine one of these with `[all]` if you want everything — `polars2svg[all,mlx-cuda13]`.
**Install exactly one backend.** They all ship the same `libmlx.so`, so asking for two
leaves you with whichever was installed last, and no warning that it happened.

**On Windows there is no MLX backend at all**, so `TFDPLayout` cannot run there. `mlx`
publishes Windows front-end wheels, which is why `[mlx]` and `[all]` still install; every
backend distribution is Linux- or macOS-only. Everything else in polars2svg works
normally.

Only `TFDPLayout` is affected by any of this: `ODFlowLayout` runs on NumPy and reaches for
MLX only when MLX works.

`TFDPLayout` runs the same MLX code on either GPU backend — Metal on Apple silicon,
CUDA on NVIDIA. Check which one you got with `polars2svg.gpu_backend()` (`'metal'`,
`'cuda'`, or `'cpu'`). That name is exported **only when MLX imports successfully**, so
on the broken-Linux install above it is absent rather than reporting `'cpu'` —
`hasattr(polars2svg, 'gpu_backend')` is itself the "did MLX load?" check, and
`import polars2svg` keeps working either way. The CUDA wheels are Linux-only and need NVIDIA SM ≥ 7.5
(Turing or newer), driver ≥ 550.54.14, and glibc ≥ 2.35. Outside that envelope MLX
falls back to the CPU device — `TFDPLayout` still works, just slower.

**Match the extra to your CUDA toolkit, not just your driver.** MLX JIT-compiles its
CUDA kernels against the system CUDA headers (`CUDA_HOME` / `CUDA_PATH`, else
`/usr/local/cuda`), so the wheel's NVRTC and those headers have to agree:

| System CUDA toolkit | Extra | Also needs |
|---|---|---|
| 12.x | `polars2svg[mlx-cuda]` | driver ≥ 550.54.14 |
| 13.x | `polars2svg[mlx-cuda13]` | driver ≥ 580 |

Mismatch it and the first GPU kernel dies in a wall of `nvcc` syntax errors from inside
the CUDA headers (e.g. NVRTC 12.9 cannot parse CUDA 13's `cuda_fp4.hpp`). polars2svg
detects this at import, logs a warning naming the cause, and falls back to CPU rather
than failing mid-layout — so a silent slowdown here means it's worth checking
`gpu_backend()`.

## Security & deployment

polars2svg supports three deployment profiles, described in full in
[SECURITY.md](SECURITY.md):

| Profile | Shape | Status |
| --- | --- | --- |
| **Notebook** | One trusted person, one process | Fully supported |
| **Appliance (A)** | A server renders from untrusted data; output embedded in a page others view | Supported for the render path |
| **Interactive, multi-user (B)** | One process serves the live widgets to mutually untrusting people | **Out of scope** |

If you render from data you do not control, gate what you serve on the output
contract. It is an allow-list over the rendered document — only the elements and
attributes the components actually emit, every reference resolving inside the same
document, no event handlers, no scripts:

```python
import polars as pl
from polars2svg import Polars2SVG, assertOutputContract

p2s   = Polars2SVG()
chart = p2s.histop(pl.DataFrame({'cat': ['a', 'b', 'a']}), 'cat')

# Raises OutputContractError if the render is not fit to serve.
assertOutputContract(chart.svg)
```

It reports; it never rewrites. `tile()` is outside the profile by construction — it
embeds foreign SVG verbatim, which is the component, not a defect. The interactive
components (`panelize()`, `linkpi()`, …) are **not** hardened for multiple
untrusting users; run one process per user behind your own authentication.

## Development

```bash
uv venv --python 3.13                               # then install exactly what uv.lock pins:
uv sync --frozen --extra layouts --extra interactive --extra export --group dev
.venv/bin/python -m pytest tests/                   # run the suite
```

After changing framework files, re-run `uv pip install -e .` before testing.

`tests/test_readme.py` executes every Python block on this page and resolves every
API name and asset path it mentions, so a snippet here cannot drift from the code.

## References

polars2svg implements algorithms from the following published work. Each
implementation file carries the full citation in its module header.

- **Circle packing** (`chordp`/`spreadlinesp` packing, `packCircles()` —
  [circle_packer.py](polars2svg/circle_packer.py)):
  W. Wang, H. Wang, G. Dai, and H. Wang, "Visualization of large hierarchical data
  by circle packing," *Proc. SIGCHI Conference on Human Factors in Computing
  Systems (CHI '06)*, 2006, pp. 517–520. doi:[10.1145/1124772.1124851](https://doi.org/10.1145/1124772.1124851)
- **Neighborhood-preserving non-uniform circle packing** (`linkp` `ncp pack`
  layout op, `NCPLayout` — [ncp_layout.py](polars2svg/ncp_layout.py)):
  D. Li, J. Yuan, X. Guo, X. Wang, Y. Liu, W. Yang, and S. Liu, "NCP:
  Neighborhood-Preserving Non-Uniform Circle Packing for Visualization,"
  *Computational Visual Media*, 2026. arXiv:[2602.00668](https://arxiv.org/abs/2602.00668)
- **Hierarchical edge bundling** (`chordp` bundled link shapes —
  [chordp.py](polars2svg/chordp.py)):
  D. Holten, "Hierarchical Edge Bundles: Visualization of Adjacency Relations in
  Hierarchical Data," *IEEE Transactions on Visualization and Computer Graphics*,
  vol. 12, no. 5, pp. 741–748, 2006. doi:[10.1109/TVCG.2006.147](https://doi.org/10.1109/TVCG.2006.147)
- **Force-directed origin-destination flow maps** (`linkp` flowmap link shape,
  `ODFlowLayout` — [od_flow_layout.py](polars2svg/od_flow_layout.py)):
  B. Jenny, D. M. Stephen, I. Muehlenhaus, B. E. Marston, R. Sharma, E. Zhang,
  and H. Jenny, "Force-directed layout of origin-destination flow maps,"
  *International Journal of Geographical Information Science*, 2017.
  doi:[10.1080/13658816.2017.1307378](https://doi.org/10.1080/13658816.2017.1307378)
- **ColorBrewer "Spectral" scheme** (the framework-wide color spectrum —
  [p2s_colors_mixin.py](polars2svg/p2s_colors_mixin.py)):
  M. Harrower and C. A. Brewer, "ColorBrewer.org: An Online Tool for Selecting
  Colour Schemes for Maps," *The Cartographic Journal*, vol. 40, no. 1,
  pp. 27–37, 2003. doi:[10.1179/000870403235002042](https://doi.org/10.1179/000870403235002042)
- **t-FDP force-directed layout** (`TFDPLayout` —
  [tfdp_layout.py](polars2svg/tfdp_layout.py)):
  F. Zhong, M. Xue, J. Zhang, F. Zhang, R. Ban, O. Deussen, and Y. Wang,
  "Force-Directed Graph Layouts Revisited: A New Force Based on the
  t-Distribution," *IEEE Transactions on Visualization and Computer Graphics*,
  2023. arXiv:[2303.03964](https://arxiv.org/abs/2303.03964)
- **Landmark MDS** (`LandmarkMDSLayout` — [mds_at_scale.py](polars2svg/mds_at_scale.py)):
  V. de Silva and J. B. Tenenbaum, "Global versus local methods in nonlinear
  dimensionality reduction," *Proc. NIPS*, 2003, pp. 721–728.
- **Pivot MDS** (`PivotMDSLayout` — [mds_at_scale.py](polars2svg/mds_at_scale.py)):
  U. Brandes and C. Pich, "Eigensolver Methods for Progressive Multidimensional
  Scaling of Large Data," *Proc. 14th Symposium on Graph Drawing (GD)*, 2006,
  pp. 42–53.
- **Radial ("hypertree") tree drawing** (`hyperTreeLayout()` —
  [p2s_graph_mixin.py](polars2svg/p2s_graph_mixin.py)):
  P. Eades, "Drawing free trees," *Bulletin of the Institute for Combinatorics
  and its Applications*, vol. 5, pp. 10–36, 1992.
- **Laguerre-Voronoi (power) diagrams** (`laguerre_voronoi()` —
  [laguerre_voronoi.py](polars2svg/laguerre_voronoi.py)):
  H. Imai, M. Iri, and K. Murota, "Voronoi Diagram in the Laguerre Geometry and
  its Applications," *SIAM Journal on Computing*, vol. 14, no. 1, pp. 93–105, 1985.
- **Scatterplot de-cluttering**
  (`uniformSampleDistributionInScatterplotsViaSectorBasedTransformation()` —
  [udist_scatterplots_via_sectors_tile_opt.py](polars2svg/udist_scatterplots_via_sectors_tile_opt.py)):
  H. Rave, V. Molchanov, and L. Linsen, "Uniform Sample Distribution in
  Scatterplots via Sector-based Transformation," *2024 IEEE Visualization and
  Visual Analytics (VIS)*, 2024, pp. 156–160. doi:[10.1109/VIS55277.2024.00039](https://doi.org/10.1109/VIS55277.2024.00039)
- **SpreadLine layout** (`spreadlinesp` — [spreadlinesp.py](polars2svg/spreadlinesp.py)):
  Y.-H. Kuo, D. Liu, and K.-L. Ma, "SpreadLine: Visualizing Egocentric Dynamic
  Influence," *IEEE Transactions on Visualization and Computer Graphics*
  (Proc. IEEE VIS 2024). arXiv:[2408.08992](https://arxiv.org/abs/2408.08992)

Several modules are vendored from the author's
[racetrack_svg_framework](https://github.com/datrcode/racetrack_svg_framework)
(Apache-2.0); each carries a provenance header noting its origin.

To cite polars2svg itself, see [CITATION.cff](CITATION.cff).

## License / Notices

polars2svg is licensed under **Apache-2.0** (see [LICENSE](LICENSE)).

The wheel bundles a subset of the **Noto Sans** font
(`polars2svg/fonts/NotoSans-Regular-subset.ttf`), © The Noto Project Authors,
distributed under the **SIL Open Font License 1.1** (see
[polars2svg/fonts/OFL.txt](polars2svg/fonts/OFL.txt)).

This product includes color specifications and designs developed by
**Cynthia Brewer** ([http://colorbrewer.org/](http://colorbrewer.org/)) —
© 2002 Cynthia Brewer, Mark Harrower, and The Pennsylvania State University,
used under the Apache-style ColorBrewer license (see [NOTICE](NOTICE)).
