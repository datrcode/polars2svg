# Interactivity

Every component has a linked, interactive variant — `xypi`, `histopi`,
`timepi`, `piepi`, `chordpi`, `linkpi`, `spreadlinepi`, `smallpi` — built on
[Panel](https://panel.holoviz.org) `ReactiveHTML` wrappers.

Each one **wraps a component you have already built**, rather than taking the
DataFrame again:

```python
xyp = p2s.xyp(df, "x", "y", color="group")   # the static component
p2s.xypi(xyp)                                # the interactive view of it
```

So the static call keeps its full signature and the `*i` call adds view options on
top. `linkpi` additionally takes `mvc=` to join an existing coordination hub.

Requires the `[interactive]` extra:
`pip install polars2svg[interactive]` (includes `[layouts]`). Interactive
views need a live Python kernel (a Jupyter notebook or a Panel server).

## `panelize()` — linked views in one call

```python
layout = [[p2s.xypi(p2s.xyp(df, "x", "y", color="group"))],
          [p2s.histopi(p2s.histop(df, "group")),
           p2s.timepi(p2s.timep(df, "timestamp"))]]

p2s.panelize(layout)
```

(`timep` needs a real date/datetime column — a string one raises `ValueError`.)

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

### Selection labels

`linkpi` draws a label overlay on the current selection, capped by
`max_selection_labels=` (default `32`). The cap is on the **selection**, not on the
labels: past it nothing is labelled rather than the first 32 being labelled, because a
partial overlay reads as a complete one and is worse than none. The info line says
`(labels capped)` when that happens, so the view tells you which regime you are in.

`linkpi()` wraps a **built `linkp`** rather than taking the DataFrame itself, and
forwards keywords to the view:

```python
lp = p2s.linkp(df, [("src", "dst")], wxh=(400, 360))

p2s.linkpi(lp, max_selection_labels=100)   # label larger selections
p2s.linkpi(lp, max_selection_labels=0)     # never label
```

Raise it when your nodes are few and named; leave it alone on a large graph, where the
overlay costs a redraw per selection change and covers the drawing it annotates.

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
