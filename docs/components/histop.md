# histop — horizontal histogram

One bar per category/bin of the binning field. Bar **length** is set by
`count=` (row count by default) and bar **color** by the orthogonal `color=`;
bars sort by `order=`. `style=` also renders the same binning as boxplot,
swarm, or stacked bar.

![histop bars](../examples/histop_bars.svg)

The example encodes two different measures at once: bar length is **byte
volume** (`count="bytes"`), while the colorbar shows **raw row count**
(`color=p2s.CROW_MAGNITUDEp`). `/static/js` transferred the most bytes *and*
served the most requests; a long, cold bar would instead mean few large
responses.

```python
p2s.histop(df, "endpoint", count="bytes", color=p2s.CROW_MAGNITUDEp,
           wxh=(420, 220), legend=True)
```

## Key parameters

| Parameter | Forms | Notes |
|-----------|-------|-------|
| `bin_by` | `'field'`, `('f1', 'f2', ...)` | Positional; multi-field bins join with `\|` for display. |
| `count` | `p2s.ROW_COUNTp` (default), `'field'`, `('field', p2s.SETp)` | Numeric field → sum; non-numeric or `SETp` → distinct count. **The primary size knob.** |
| `color` | `'field'`, `('field', COLOR_ENUM)`, `p2s.CROW_MAGNITUDEp` … | Bare string field → stacked categorical segments; bare numeric → whole-bar spectrum. `CROW_*` colors by raw row count regardless of `count=`. |
| `order` | field / enum | Bar ordering. |
| `style` | histogram (default), boxplot, swarm, stacked bar | |
| `count_range` | `(min, max)` | Fix the length scale — useful inside [smallp](smallp.md). |
| `legend` | `True`, position string, dict | Colorbar for spectrum modes, swatches for categorical. |

Because length and color are independent, a bar sized by one measure can be
colored by another — that's a feature, and it's explained in depth in
[count= and color=](../guides/count-color.md).

Interactive variant: `p2s.histopi(...)`.
