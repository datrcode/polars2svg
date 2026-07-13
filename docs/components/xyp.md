# xyp — scatter / distribution plot

Each row becomes a dot at (x, y). Both axes accept numeric **or** categorical
(set-based) fields, multiple fields (vstacked), or tuples (converted to structs
and ordered). Dot color, size, and opacity are independently data-drivable, and
optional distributions, connecting lines, and shapely background shapes layer
on top.

![xyp scatter](../examples/xyp_scatter.svg)

```python
import random
import polars as pl
from polars2svg import Polars2SVG

p2s = Polars2SVG()

random.seed(7)
n = 120
df = pl.DataFrame({
    "x":     [random.gauss(0, 1) for _ in range(n)],
    "y":     [random.gauss(0, 1) for _ in range(n)],
    "group": [random.choice(["alpha", "beta", "gamma"]) for _ in range(n)],
})

p2s.xyp(df, "x", "y", color="group", dot_size=5, wxh=(400, 300), legend=True)
```

## Key parameters

| Parameter | Forms | Notes |
|-----------|-------|-------|
| `x`, `y` | `'field'`, `('f1', 'f2', ...)`, `['f1', 'f2', ...]` | Numeric fields default to scalar axes, everything else to categorical. Pin intent with `p2s.SCALARp` / `p2s.SETp` in the spec. |
| `color` | `'field'`, `('field', COLOR_ENUM)`, `'#RRGGBB'`, `p2s.CROW_MAGNITUDEp` … | Bare numeric field → magnitude spectrum; bare non-numeric → categorical. See [count= and color=](../guides/count-color.md). |
| `dot_size` | `int`, `float`, `'field'`, `p2s.ROW_COUNTp` | An **integer** size triggers the pixel-grid pipeline; field-driven sizes scale within `dot_size_range=(0.5, 4.0)`. |
| `opacity` | `float`, `'field'`, `p2s.ROW_COUNTp` | Field-driven opacity scales within `opacity_range=(0.5, 1.0)`. |
| `x_order`, `y_order` | list or `{value: rank}` dict | Ordering for categorical axes; unlisted values sort last. |
| `legend` | `True`, `'right'/'left'/'top'/'bottom'`, dict | Swatch list for categorical color, colorbar for spectrum modes. |
| `wxh` | `(width, height)` | Output size in pixels. |

`xyp` has no `count=` parameter — its size analog is `dot_size=`, and
`p2s.ROW_COUNTp` stands in for row count wherever a field is accepted.

Interactive variant: `p2s.xypi(...)` — same signature, linked brushing via
[panelize](../guides/interactivity.md).
