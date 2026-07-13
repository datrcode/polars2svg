# linkp — node-link graph / network

Each `relationships` pair contributes an edge and its endpoint nodes. Nodes
are placed by `pos=` (a networkx-style `{node: [x, y]}` dict) or given random
positions; layout classes like `TFDPLayout` (GPU-accelerated t-FDP) produce
`pos` dicts. Node and link size, color, opacity, shape, labels, convex hulls,
and shapely backgrounds are all configurable.

Requires the `[layouts]` extra: `pip install polars2svg[layouts]`.

![linkp network](../examples/linkp_network.svg)

```python
edges = [("api", "auth"), ("api", "db"), ("api", "cache"), ("auth", "db"),
         ("web", "api"), ("web", "cdn"), ("worker", "db"), ("worker", "queue"),
         ("queue", "worker"), ("cache", "db"), ("cdn", "web"), ("mobile", "api"),
         ("mobile", "cdn"), ("report", "db"), ("report", "cache")]
tier = {"web": "edge", "mobile": "edge", "cdn": "edge",
        "api": "service", "auth": "service", "worker": "service",
        "report": "service", "db": "data", "cache": "data", "queue": "data"}
df = pl.DataFrame({"src":  [a for a, _ in edges],
                   "dst":  [b for _, b in edges],
                   "tier": [tier[a] for a, _ in edges]})

p2s.linkp(df, [("src", "dst")], node_color="tier", color="tier",
          node_size="medium", draw_labels=True, wxh=(400, 360), legend=True)
```

## Key parameters

| Parameter | Forms | Notes |
|-----------|-------|-------|
| `relationships` | `[('from', 'to')]`, `[('from', 'to', 'predicate')]`, tuple fields | Tuple fields concatenate with `\|`. |
| `pos` | `{node: [x, y], ...}` | networkx-style; nodes absent from `pos` get random positions. |
| `color` | `'field'`, `'#rrggbb'`, `p2s.CROW_MAGNITUDEp` … | Applies to links and nodes; `node_color=` overrides for nodes. |
| `node_size`, `link_size` | `'small' / 'medium' / ... '`, `'vary'` | Fixed sizes by default. |
| `count` | `p2s.ROW_COUNTp`, `'field'`, … | **Only drives geometry once `node_size='vary'` / `link_size='vary'`** — at fixed sizes it has no visible effect (a one-time warning fires if set anyway). |
| `draw_labels` | `True` / `False` | Node labels. |
| `legend` | `True`, position string, dict | |

!!! note "count= vs CROW_* color"
    `CROW_MAGNITUDEp` / `CROW_STRETCHEDp` color by **raw row count**, not by
    `count=` — the two are independent by design. Details in
    [count= and color=](../guides/count-color.md).

Interactive variant: `p2s.linkpi(...)` is a full graph editor — drag, wheel
zoom, layout pickers, keyboard shortcuts, layout save/load. See
[Interactivity](../guides/interactivity.md).
