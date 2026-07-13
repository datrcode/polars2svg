# Time fields (t-fields)

A **t-field** asks a component to transform a time column before binning it —
"fold `timestamp` onto day-of-week", "force monthly resolution" — anywhere a
time spec is accepted (`timep`'s `time=`, `smallp`'s `cycle_by`, …).

## `p2s.tField(column, enum)`

```python
p2s.timep(df, p2s.tField("timestamp", p2s.PT_DoWp))    # counts by day-of-week
```

`tField()` returns a frozen `str` subclass, so it works everywhere a plain
string does — in tuples, sets, and dict keys — but it can never be mistaken
for a real column name.

## Periodic enums — fold time onto a cycle

| Enum | Cycle |
|------|-------|
| `PT_Qp` | Quarter |
| `PT_mp` | Month |
| `PT_m_dp` | Month + day *(leap-year day count)* |
| `PT_m_d_Hp` | Month + day + hour |
| `PT_DoYp` | Day of year *(non-leap day count)* |
| `PT_DoWp` | Day of week |
| `PT_DoW_Hp` | Day of week + hour |
| `PT_DoW_H_Mp` | Day of week + hour + minute |
| `PT_dp` | Day of month |
| `PT_d_Hp` | Day of month + hour |
| `PT_d_H_Mp` | Day of month + hour + minute |
| `PT_Hp` | Hour |
| `PT_H_Mp` | Hour + minute |
| `PT_H_M_Sp` | Hour + minute + second |
| `PT_Mp` | Minute |
| `PT_M_Sp` | Minute + second |
| `PT_Sp` | Second |

## Linear enums — force a chronological resolution

Left alone, linear time picks a resolution automatically. To force one, use a
`LT_*` enum — from `LT_Yp` (yearly) down through `LT_Y_m_d_H_M_Sp` (seconds),
including uneven bins like `LT_Y_m_d_4Hp` (4-hour) and `LT_Y_m_d_H_15Mp`
(15-minute).

```python
p2s.timep(df, p2s.tField("timestamp", p2s.LT_Y_mp))    # force monthly bins
```

## The legacy string form

Older code wrote t-fields as `'column|suffix'` strings (e.g. `'timestamp|mp'`
for a monthly fold — the suffixes match the enum names without the `PT_`/`LT_`
prefix and trailing `p`). This still works, with two caveats:

- If the literal string **is** a real column in your DataFrame (a column
  actually named `price|mp`), it is used as-is — never hijacked into a
  transform.
- Each accepted legacy string emits a one-time deprecation warning pointing at
  `p2s.tField()`.

Prefer `p2s.tField()` in new code: it is explicit, collision-proof, and
equality-compatible with the legacy string (they hash the same, so they are
interchangeable in sets and dicts).
