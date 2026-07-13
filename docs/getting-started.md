# Getting started

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
pip install polars2svg[mlx]         # MLX-accelerated t-FDP layout (Apple silicon / Metal)
pip install polars2svg[mlx-cuda]    # ... the same layout on Linux + NVIDIA (CUDA 12)
pip install polars2svg[all]         # everything above
```

Calling a component that needs an extra you haven't installed (e.g. `chordp()`
or `panelize()`) raises a clear `ImportError` naming the extra to install.

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
open("scatter.svg", "w").write(chart.svg)   # raw SVG, always available
chart.save("scatter.png")                    # PNG raster — needs the [export] extra
```

## The shape of every call

Each component is a method on a `Polars2SVG` instance and takes a
`pl.DataFrame` plus the fields to encode:

```python
p2s.histop(df, "category")                             # bars by row count
p2s.timep(df, "timestamp")                             # temporal bars, auto resolution
p2s.linkp(df, [("src", "dst")], color="dept")          # network from an edge list
p2s.smallp(df, template, "region")                     # one panel per region
```

Recurring parameters:

- `wxh=(width, height)` — output size in pixels.
- `count=` — "how big / how much is this element?" (bar length, slice share, …).
- `color=` — "what color is this element?" — independent of `count=`.
- `legend=True` (or `'right' | 'left' | 'top' | 'bottom'`, or a dict) — opt-in
  legend or colorbar, auto-selected from the resolved color mode.

`count=` and `color=` are deliberately orthogonal and have precise,
enum-pinnable semantics — read [count= and color=](guides/count-color.md)
before reaching for either in anger.

## GPU note (optional)

`TFDPLayout` runs the same MLX code on either GPU backend — Metal on Apple
silicon, CUDA on NVIDIA. Check which one you got with
`polars2svg.gpu_backend()` (`'metal'`, `'cuda'`, or `'cpu'`). Outside the
supported envelope MLX falls back to the CPU device — layouts still work, just
slower. If you're on CUDA, match the extra to your CUDA toolkit
(`[mlx-cuda]` for 12.x, `[mlx-cuda13]` for 13.x): a mismatch is detected at
import and polars2svg falls back to CPU with a logged warning.

## Next steps

- Browse the [component gallery](components/index.md).
- Wire up linked, interactive views with [panelize](guides/interactivity.md).
- Bin time your way with [t-fields](guides/t-fields.md).
