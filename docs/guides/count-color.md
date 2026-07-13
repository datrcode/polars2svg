# `count=` and `color=`

Two parameters recur across the counting/aggregating components, and they are
**orthogonal by design**:

- `count=` answers *"how big / how much is this element?"*
- `color=` answers *"what color is this element?"*

Understanding what each one does — and what it deliberately does **not** do —
saves the two most common surprises in the library.

## `count=` — the aggregation rule

Every component that takes `count=` shares one aggregation rule:

| `count=` spec | Aggregation |
|---------------|-------------|
| `p2s.ROW_COUNTp` *(default)* | `pl.len()` — number of rows |
| a numeric field | `sum` of that field |
| a non-numeric field, or `('field', p2s.SETp)` | `n_unique` (distinct count) |
| a multi-field tuple | struct the fields, then `n_unique` |

## What `count=` visibly does, per component

The aggregation is uniform; the *visible effect* is not. With every other
parameter at its default:

| Component | Effect of `count=` at defaults |
|-----------|-------------------------------|
| [histop](../components/histop.md) | **Bar length** — the primary size knob. |
| [timep](../components/timep.md) | **Bar height** — the primary size knob. |
| [piep](../components/piep.md) | **Slice share** of the whole. |
| [linkp](../components/linkp.md) | **Nothing visible.** Default node/link sizes are fixed; count only drives geometry once you set `node_size='vary'` / `link_size='vary'`. A one-time warning fires if you set `count=` without either. |
| [chordp](../components/chordp.md) | Reshuffles the **derived node order** (count sets the edge weights the ring-clustering uses). Arc/ribbon geometry additionally needs `'vary'` sizing; with the order pinned and fixed sizes, count is inert (one-time warning). |
| [spreadlinesp](../components/spreadlinesp.md) | Changes the **within-bin circle sort order**, not circle size (size is packing-determined). |
| [xyp](../components/xyp.md) | *No `count=` parameter.* The size analog is `dot_size=`; `p2s.ROW_COUNTp` stands in for row count where a field is accepted. |
| [smallp](../components/smallp.md) | *No `count=` parameter.* Panel ordering uses `order=` (which accepts `p2s.ROW_COUNTp`); magnitude semantics live in the template. |

## `color=` — independent of `count=`

`color=` accepts a bare field, a `('field', COLOR_ENUM)` pair, hex constants,
or one of the row-count modes. A bare field infers its meaning from the
column **dtype**:

- numeric column → **magnitude spectrum** (colored by the sum per element)
- anything else → **categorical** (hash-derived colors per value)

The enum matrix pins intent explicitly when inference isn't what you want:

| You want | Use |
|----------|-----|
| categorical colors, even for a numeric field | `('field', p2s.CSETp)` |
| a spectrum by sum / min / median / mean / max | `('field', p2s.CMAGNITUDE_SUMp)` etc. |
| the same, but rank-normalized (stretched) | `('field', p2s.CSTRETCHED_SUMp)` etc. |
| a spectrum by how many distinct values landed together | `('field', p2s.CSET_MAGNITUDEp)` / `CSET_STRETCHEDp` |

## `CROW_MAGNITUDEp` and `CROW_STRETCHEDp` — color by row count

"Color Row Magnitude" and "Color Row Stretched" color by **raw row count**
(`pl.len()`) — *regardless of what `count=` is set to*. This is intentional:
if you size bars by byte volume with `count='bytes'`, a `CROW_*` color still
answers "how many rows landed here", not "how many bytes".

The two encodings compose into charts that say two things at once — the
[histop example](../components/histop.md) sizes bars by bytes and colors them
by request count. It can also *look* surprising: a bar can be long but
cold-colored (many bytes from few rows) or short but warm (many small rows).
That is correct behavior, not a bug — the encodings are orthogonal.

## Diagnosing dtype-keyed inference

Because a bare field spec infers sum-vs-distinct (`count=`) and
spectrum-vs-categorical (`color=`) from the column dtype, an upstream schema
change (string IDs becoming integers, say) can silently flip the
interpretation. Two tools for that:

1. **Pin intent with enums** — `('field', p2s.SCALARp)` / `('field', p2s.SETp)`
   for count; `('field', p2s.CSETp)` / `('field', p2s.CMAGNITUDE_SUMp)` etc.
   for color. An explicit enum is never re-inferred.
2. **Turn on INFO logging** — every bare-spec inference site emits a one-time
   INFO log naming the field, the interpretation chosen, and the enum that
   would override it. INFO is off by default, so normal use stays quiet:

```python
import logging
logging.getLogger('polars2svg_logger').setLevel(logging.INFO)
logging.getLogger('polars2svg_logger').addHandler(logging.StreamHandler())
```
