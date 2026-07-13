# polars2svg

**Polars-native DataFrame → SVG visualizations for Jupyter.** Pass a
[Polars](https://pola.rs) `DataFrame` straight to a component and get a crisp,
self-contained SVG back — scatter plots, histograms, temporal bars, network and
chord diagrams, pie charts, and small multiples. Every component also has a
linked, interactive variant (built on [Panel](https://panel.holoviz.org)) for
brushing and cross-filtering in a notebook, plus optional WebGPU rendering for
large frames.

<div style="display:flex; gap:1rem; flex-wrap:wrap; align-items:flex-start;">
  <img src="examples/xyp_scatter.svg" alt="Scatter plot" style="max-width:32%; min-width:220px;">
  <img src="examples/linkp_network.svg" alt="Network graph" style="max-width:32%; min-width:220px;">
  <img src="examples/chordp_flows.svg" alt="Chord diagram" style="max-width:32%; min-width:220px;">
</div>

## Why polars2svg

- **Polars-native** — components consume `pl.DataFrame` directly and aggregate
  through lazy Polars pipelines; no pandas round-trips.
- **SVG out, no ceremony** — every component returns an object with a `.svg`
  string and a Jupyter `_repr_svg_`, so the last expression in a cell just
  displays.
- **Orthogonal encodings** — `count=` sets size/magnitude, `color=` sets color,
  and an aggregation-aware color enum matrix pins exactly what a color means
  (see [count= and color=](guides/count-color.md)).
- **Deep time model** — linear and periodic (day-of-week, month, hour, …)
  binning via [t-fields](guides/t-fields.md).
- **Interactive when you want it** — `panelize()` wires linked brushing,
  selection, and a shared undo/redo stack across every view
  (see [Interactivity](guides/interactivity.md)).

## Install

```bash
pip install polars2svg
```

The base install is deliberately slim — Polars + NumPy (+ PyArrow/Pillow for
I/O). Network/chord layouts, interactivity, export, and GPU acceleration live
behind extras — see [Getting started](getting-started.md).

## At a glance

| Component | What it draws |
|-----------|---------------|
| [xyp](components/xyp.md) | Scatter / distribution plot; numeric or categorical axes |
| [histop](components/histop.md) | Horizontal histogram bars, one per category/bin |
| [timep](components/timep.md) | Temporal bar chart with linear or periodic time modes |
| [piep](components/piep.md) | Pie / donut / waffle chart |
| [linkp](components/linkp.md) | Node-link graph / network with pluggable layouts |
| [chordp](components/chordp.md) | Chord diagram of weighted flows around a circle |
| [spreadlinesp](components/spreadlinesp.md) | Egocentric radial "spread" rings for influence/propagation |
| [smallp](components/smallp.md) | Small multiples — one template faceted by a field |

Browse the [rendered gallery](components/index.md), or start with the
[quickstart](getting-started.md).
