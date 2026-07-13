# smallp — small multiples

Trellis / facet views: each panel renders one slice of the DataFrame using a
**template** component as the blueprint. Panels come from either splitting the
data on `category_by` values (split mode) or re-rendering the full dataset
with different template parameters (`cycle_by` mode).

![smallp facets](../examples/smallp_facets.svg)

```python
# a dataless template sets the per-panel size and shares both axes
tmpl = p2s.xyp(x="x", y="y", dot_size=2, wxh=(190, 140),
               sm_shared={p2s.SM_X, p2s.SM_Y})

p2s.smallp(df, tmpl, "region", wxh=(420, 340))   # one panel per region
```

The template's `wxh` is the **panel** size; smallp's `wxh` is the overall
canvas. Panels that don't fit are collated into a single "Remainder" panel by
default (`collate_remainder=True`) — if you see only a Remainder panel, your
panels are too large for the canvas.

`sm_shared={p2s.SM_X, p2s.SM_Y}` shares axis ranges across panels so they are
directly comparable; components also accept `count_range` for shared magnitude
scales.

## Key parameters

| Parameter | Forms | Notes |
|-----------|-------|-------|
| `sm_template` | an `XYp`, `Timep`, `Histop`, `LinkP`, `ChP` … instance | Its `wxh` sets the panel size; kwargs forward to it. |
| `category_by` | `'field'`, `('f1', 'f2')`, list/dict of DataFrames | Split mode. With a two-field tuple, `grid_mode=True` arranges panels as a grid. |
| `cycle_by` | `{param: [v1, v2, ...], ...}` | Cycle mode (mutually exclusive with `category_by`): each panel is the **full** dataset with a different template parameter value, e.g. `{'color': ['country', 'region', 'city']}` or `{'time': ['date\|mp', 'date\|DoWp']}`. Lists zip in lockstep. |
| `order` | `p2s.ROW_COUNTp`, `'field'`, `('field', enum)` | Panel ordering; `descending=True` by default. |
| `include_all` | `False` (default) | Adds an "all" panel. |
| `collate_remainder` | `True` (default) | Collate non-fitting panels into one. |
| `sketch_only` | `False` (default) | Fast preview of the layout. |

`smallp` has no `count=` of its own — magnitude semantics live in the
template component.

Interactive variant: `p2s.smallpi(...)`.
