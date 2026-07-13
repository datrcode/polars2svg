# piep — pie / donut / waffle

Mirrors [histop](histop.md) in parameters and usage: bins become slices,
`count=` sets each slice's share of the whole, and `color=` sets each slice's
color. `style=` switches between pie, donut, and waffle.

![piep donut](../examples/piep_donut.svg)

```python
share = {"solar": 34, "wind": 27, "hydro": 18, "nuclear": 13, "gas": 8}
df = pl.DataFrame({"source": [k for k, v in share.items() for _ in range(v)]})

p2s.piep(df, "source", style=p2s.DONUTp, draw_labels=True,
         color="source", wxh=(320, 320))
```

## Key parameters

| Parameter | Forms | Notes |
|-----------|-------|-------|
| `bin_by` | `'field'`, `('f1', 'f2', ...)` | Positional; multi-field slices join with `\|`. |
| `count` | `p2s.ROW_COUNTp` (default), `'field'`, `('field', p2s.SETp)` | Slice share. Numeric → sum; non-numeric → distinct count. |
| `color` | `'field'`, `('field', COLOR_ENUM)`, `'#RRGGBB'`, list of hex | Default (no color) uses five barely-distinct shades of the data color, assigned so adjacent slices differ. **To color each slice by its own category, pass `color=<bin_by field>`** (as above). |
| `style` | pie (default), `p2s.DONUTp`, waffle | |
| `draw_labels` | `True` / `False` | Slice labels. |
| `legend` | `True`, position string, dict | |

Interactive use: compose into a [panelize](../guides/interactivity.md) layout
alongside linked views.
