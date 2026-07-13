# chordp — chord diagram

Nodes are arranged as arcs around a circle and edges drawn as chords (curved
or bundled) between them. By default the node order is derived by
hierarchically clustering the edge weights — which `count=` feeds — so the
ring arranges related nodes near each other; pin an explicit order with
`order=`.

Requires the `[layouts]` extra: `pip install polars2svg[layouts]`.

![chordp flows](../examples/chordp_flows.svg)

```python
flows = [("North", "South", 8.0), ("North", "East", 5.0), ("North", "West", 3.0),
         ("South", "East", 6.0), ("South", "West", 4.0), ("East", "West", 7.0),
         ("West", "North", 5.0), ("East", "North", 2.0), ("South", "North", 4.0)]
df = pl.DataFrame({"fm":     [a for a, _, _ in flows],
                   "to":     [b for _, b, _ in flows],
                   "weight": [w for _, _, w in flows]})

p2s.chordp(df, [("fm", "to")], count="weight", node_size="vary",
           link_size="vary", color="fm", node_color=p2s.COLOR_BY_NODE_NAME,
           wxh=(360, 360))
```

## Key parameters

| Parameter | Forms | Notes |
|-----------|-------|-------|
| `relationships` | `[('from', 'to')]`, tuple fields | Tuple fields concatenate with `\|`. |
| `count` | `p2s.ROW_COUNTp` (default), `'field'`, `('field', p2s.SETp)` | Sets edge weights: feeds the derived node order always; scales arc/ribbon **geometry** only with `node_size='vary'` / `link_size='vary'`. With order pinned and fixed sizes it is inert (one-time warning). |
| `color` | `'src'`, `'dst'`, `'field'`, `('field', enum)`, `'#rrggbb'` | `'src'` / `'dst'` inherit the endpoint node's color. |
| `node_color` | `p2s.COLOR_BY_NODE_NAME`, `'#rrggbb'`, `'field'` | |
| `node_size`, `link_size` | fixed (default) or `'vary'` | `'vary'` scales by count. |
| `link_shape` | `'curve'` (default), `'bundled'` | |
| `order` | list | Pin the ring order explicitly. |
| `legend` | `True`, position string, dict | |

Interactive variant: `p2s.chordpi(...)`.
