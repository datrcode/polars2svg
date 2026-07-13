# Interactivity

Every component has a linked, interactive variant — `xypi`, `histopi`,
`timepi`, `chordpi`, `linkpi`, `spreadlinepi`, `smallpi` — built on
[Panel](https://panel.holoviz.org) `ReactiveHTML` wrappers. They share the
static components' signatures, so promoting a chart to interactive is usually
a one-letter change.

Requires the `[interactive]` extra:
`pip install polars2svg[interactive]` (includes `[layouts]`). Interactive
views need a live Python kernel (a Jupyter notebook or a Panel server).

## `panelize()` — linked views in one call

```python
layout = [[p2s.xypi(df, "x", "y", color="group")],
          [p2s.histopi(df, "group"), p2s.timepi(df, "timestamp")]]

p2s.panelize(layout)
```

`panelize()` composes the views into a dashboard and auto-wires the
coordination hub (an MVC `InteractionController`) behind them:

- **Linked brushing** — brush a region in one view and the same rows highlight
  everywhere.
- **Linked selection & filtering** — selections propagate; filter operations
  narrow every view at once.
- **Shared undo/redo DataFrame stack** — every filter pushes onto a shared
  stack, so you can drill in and back out without losing state. The
  `stack=` argument names the stack if you want several independent groups.
- **Node-position broadcast** — drag a node in one `linkpi` and other graph
  views of the same nodes follow.

`panelizeSketch(layout)` renders a fast, non-live sketch of the same layout —
useful for iterating on the arrangement.

## `linkpi` — a full graph editor

The interactive network view goes well beyond brushing:

- drag nodes; mouse-wheel **zoom**;
- **layout pickers** to re-run layouts on the current selection or the whole
  graph;
- dozens of **keyboard shortcuts** for selection, expansion, filtering, and
  layout (press the help key in the view for the overlay listing them all);
- clipboard copy of the current view;
- **layout save/load** to persist hand-tuned node positions.

## WebGPU rendering for large frames

Every component can render through WebGPU instead of SVG when frames get
large — same visual output, GPU rasterization:

```python
p2s.panelize(layout, use_webgpu=True)   # dashboards
p2s.xyp(df, "x", "y").webgpu()          # single components
```

## Legends and export

Interactivity is independent of the opt-in `legend=` parameter (available on
all rendered components) and of the export API — `component.save('chart.png')`
with the `[export]` extra, or `component.svg` for the raw SVG string anywhere.
