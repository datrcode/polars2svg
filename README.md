# polars2svg

**Polars-native DataFrame → SVG visualizations for Jupyter.** Pass a
[Polars](https://pola.rs) `DataFrame` straight to a component and get a crisp,
self-contained SVG back — scatter plots, histograms, temporal bars, network and
chord diagrams, pie charts, and small multiples. Every component also has a
linked, interactive variant (built on [Panel](https://panel.holoviz.org)) for
brushing and cross-filtering in a notebook, plus optional WebGPU rendering for
large frames.

<p align="center">
  <img src="https://raw.githubusercontent.com/datrcode/polars2svg/main/docs/images/xyp_scatter.png" alt="Scatter plot" width="32%">
  <img src="https://raw.githubusercontent.com/datrcode/polars2svg/main/docs/images/linkp_network.png" alt="Network graph" width="32%">
  <img src="https://raw.githubusercontent.com/datrcode/polars2svg/main/docs/images/chordp_flows.png" alt="Chord diagram" width="32%">
</p>

## Install

The base install is deliberately slim — just Polars + NumPy (+ PyArrow/Pillow for
I/O) — so a "give me SVG from a DataFrame" install stays light:

```bash
pip install polars2svg
```

That covers `xyp`, `histop`, `timep`, `piep`, `smallp`, and `spreadlinesp`.
Everything else lives behind extras:

```bash
pip install polars2svg[layouts]     # linkp, chordp, and shapely-typed background= shapes
pip install polars2svg[interactive] # panelize / xypi / histopi / ... (includes layouts)
pip install polars2svg[export]      # component.save('chart.png')
pip install polars2svg[mlx]         # MLX-accelerated t-FDP layout (Apple silicon / Metal)
pip install polars2svg[mlx-cuda]    # ... the same layout on Linux + NVIDIA (CUDA 12)
pip install polars2svg[all]         # everything above (plain [mlx]; add [mlx-cuda] for NVIDIA)
```

Calling a component that needs an extra you haven't installed (e.g. `chordp()`
or `panelize()`) raises a clear `ImportError` naming the extra to install.

`TFDPLayout` runs the same MLX code on either GPU backend — Metal on Apple silicon,
CUDA on NVIDIA. Check which one you got with `polars2svg.gpu_backend()` (`'metal'`,
`'cuda'`, or `'cpu'`). The CUDA wheels are Linux-only and need NVIDIA SM ≥ 7.5
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

Requires **Python ≥ 3.12**.

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

Interactive, cross-linked variants share the same signatures via
`p2s.xypi(...)`, `p2s.linkpi(...)`, etc., and are composed into a dashboard with
`p2s.panelize(layout)`. Pass `use_webgpu=True` to render through WebGPU.

## Encoding data: `count=` and `color=`

Two parameters recur across the counting/aggregating components, and they are
**orthogonal** — one controls *size/magnitude*, the other controls *color*.

### `count=` — "how big / how much is this element?"

The aggregation rule is shared by every component that takes `count=`:

| `count=` spec | Aggregation |
|---------------|-------------|
| `ROW_COUNTp` *(default)* | `pl.len()` — number of rows |
| a numeric field | `sum` of that field |
| a non-numeric field, or `('field', SETp)` | `n_unique` (distinct count) |
| a multi-field tuple | struct the fields, then `n_unique` |

What `count=` visibly does depends on the component. At default settings it is
the primary size knob for **histop** (bar length) and **timep** (bar height); it
nudges the derived node order in **chordp**; and for **linkp**/**chordp** it only
drives geometry once you opt into `node_size='vary'` / `link_size='vary'`.

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

## Development

```bash
uv venv --python 3.13 && uv pip install -e .        # runtime deps
uv pip install -e . --group dev                     # + test/dev tooling
.venv/bin/python -m pytest tests/                   # run the suite
```

After changing framework files, re-run `uv pip install -e .` before testing.
Regenerate the README gallery images with
`.venv/bin/python docs/generate_images.py` (requires Google Chrome for headless
SVG→PNG rasterization).

## References

polars2svg implements algorithms from the following published work. Each
implementation file carries the full citation in its module header.

- **Circle packing** (`chordp`/`spreadlinesp` packing, `packCircles()` —
  [circle_packer.py](polars2svg/circle_packer.py)):
  W. Wang, H. Wang, G. Dai, and H. Wang, "Visualization of large hierarchical data
  by circle packing," *Proc. SIGCHI Conference on Human Factors in Computing
  Systems (CHI '06)*, 2006, pp. 517–520. doi:[10.1145/1124772.1124851](https://doi.org/10.1145/1124772.1124851)
- **Hierarchical edge bundling** (`chordp` bundled link shapes —
  [chordp.py](polars2svg/chordp.py)):
  D. Holten, "Hierarchical Edge Bundles: Visualization of Adjacency Relations in
  Hierarchical Data," *IEEE Transactions on Visualization and Computer Graphics*,
  vol. 12, no. 5, pp. 741–748, 2006. doi:[10.1109/TVCG.2006.147](https://doi.org/10.1109/TVCG.2006.147)
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
- **Force-directed / incremental proximity layouts** (`PolarsForceDirectedLayout`,
  `ConveyProximityLayout` — [polars_force_directed_layout.py](polars2svg/polars_force_directed_layout.py),
  [convey_proximity_layout.py](polars2svg/convey_proximity_layout.py)):
  J. D. Cohen, "Drawing Graphs to Convey Proximity: An Incremental Arrangement
  Method," *ACM Transactions on Computer-Human Interaction*, vol. 4, no. 3,
  pp. 197–229, 1997. doi:[10.1145/264645.264657](https://doi.org/10.1145/264645.264657)
- **Landmark MDS** (`LandmarkMDSLayout` — [mds_at_scale.py](polars2svg/mds_at_scale.py)):
  V. de Silva and J. B. Tenenbaum, "Global versus local methods in nonlinear
  dimensionality reduction," *Proc. NIPS*, 2003, pp. 721–728.
- **Pivot MDS** (`PivotMDSLayout` — [mds_at_scale.py](polars2svg/mds_at_scale.py)):
  U. Brandes and C. Pich, "Eigensolver Methods for Progressive Multidimensional
  Scaling of Large Data," *Proc. 14th Symposium on Graph Drawing (GD)*, 2006,
  pp. 42–53.
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
