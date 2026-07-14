import json
import os
import statistics
import time
import unittest
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from polars2svg import Polars2SVG

BASELINE_PATH = Path(__file__).parent / "perf_baseline.json"
WARN_THRESHOLD = 1.5   # emit a warning (and print table) when ratio exceeds this
N_ROWS = 1_000_000
N_RUNS = 3

# Pre-built positions for the 100 nodes used in the linkp workload
_N_NODES_ = 100
_POS_LINK_ = {
    str(i): (float(i % 10) / 9.0, float(i // 10) / 9.0)
    for i in range(_N_NODES_)
}

_N_CHORD_NODES_ = 20


def _make_frames():
    """Build all test DataFrames once using numpy for speed at 1M rows."""
    rng = np.random.default_rng(0)

    df_histo = pl.DataFrame({
        'cat':   pl.Series(rng.choice(['A', 'B', 'C'], N_ROWS)),
        'group': pl.Series(rng.choice(['x', 'y'], N_ROWS)),
        'value': pl.Series(rng.integers(1, 101, N_ROWS).astype(np.int32)),
        'score': pl.Series(rng.uniform(0.0, 10.0, N_ROWS)),
    })

    # timestamps as microseconds since epoch, spanning 2020–2025
    ts_start = 1_577_836_800_000_000   # 2020-01-01 UTC in µs
    ts_end   = 1_767_139_200_000_000   # 2026-01-01 UTC in µs
    df_time = pl.DataFrame({
        'ts':       pl.Series(rng.integers(ts_start, ts_end, N_ROWS)).cast(pl.Datetime('us')),
        'value':    pl.Series(rng.integers(0, 101, N_ROWS).astype(np.int32)),
        'category': pl.Series(rng.choice(['A', 'B', 'C'], N_ROWS)),
        'numeric':  pl.Series(rng.uniform(0.0, 10.0, N_ROWS)),
    })

    df_xy = pl.DataFrame({
        'a': pl.Series(rng.integers(0, 101, N_ROWS).astype(np.int32)),
        'c': pl.Series(rng.uniform(0.0, 1.0, N_ROWS)),
    })

    df_link = pl.DataFrame({
        'fm':    pl.Series(rng.integers(0, _N_NODES_, N_ROWS)).cast(pl.Utf8),
        'to':    pl.Series(rng.integers(0, _N_NODES_, N_ROWS)).cast(pl.Utf8),
        'count': pl.Series(rng.uniform(0.1, 10.0, N_ROWS)),
    })

    df_chord = pl.DataFrame({
        'fm':    pl.Series(rng.choice([str(i) for i in range(_N_CHORD_NODES_)], 10_000)),
        'to':    pl.Series(rng.choice([str(i) for i in range(_N_CHORD_NODES_)], 10_000)),
        'count': pl.Series(rng.uniform(0.1, 10.0, 10_000)),
    }).filter(pl.col('fm') != pl.col('to'))

    _N_SL_NODES_ = 20
    _N_SL_ROWS_  = 5_000
    df_spread = pl.DataFrame({
        'fm':   pl.Series(rng.choice([str(i) for i in range(_N_SL_NODES_)], _N_SL_ROWS_)),
        'to':   pl.Series(rng.choice([str(i) for i in range(_N_SL_NODES_)], _N_SL_ROWS_)),
        'ts':   pl.Series(rng.integers(ts_start, ts_end, _N_SL_ROWS_)).cast(pl.Datetime('us')),
    }).filter(pl.col('fm') != pl.col('to'))

    return df_histo, df_time, df_xy, df_link, df_chord, df_spread


def _fmt_ms(seconds):
    """Format seconds as a human-readable millisecond string."""
    return f"{seconds * 1000:8.1f} ms"


def _print_perf_table(rows):
    """Print a columnar performance comparison table to stdout."""
    w_name = max(len(r[0]) for r in rows)
    w_name = max(w_name, len('component'))
    header = f"  {'component':<{w_name}}   {'baseline':>10}   {'current':>10}   {'ratio':>7}   status"
    sep    = f"  {'-'*w_name}   {'-'*10}   {'-'*10}   {'-'*7}   ------"
    lines = [
        f"\nPerformance vs baseline (warn_threshold={WARN_THRESHOLD}×):",
        header,
        sep,
    ]
    for name, base, current, ratio in rows:
        base_col    = _fmt_ms(base)        if base    is not None else "     (new) "
        current_col = _fmt_ms(current)     if current is not None else "     N/A   "
        ratio_col   = f"{ratio:6.2f}×"    if ratio   is not None else "    N/A "
        flag        = "  *** SLOW"         if ratio   is not None and ratio > WARN_THRESHOLD else ""
        lines.append(f"  {name:<{w_name}}   {base_col}   {current_col}   {ratio_col}{flag}")
    print("\n".join(lines) + "\n")


class TestPerformanceRegression(unittest.TestCase):

    def _make_workloads(self):
        p2s = Polars2SVG()
        df_histo, df_time, df_xy, df_link, df_chord, df_spread = _make_frames()

        # webgpu(): time only the payload build (buffers + base64), not the render
        class _TimedResult_:
            def __init__(self, t): self.t_overall = t
        def _webgpu_payload_(component_fn):
            _component_ = component_fn()
            t0 = time.time()
            _component_.webgpu()
            return _TimedResult_(time.time() - t0)

        return {
            "histop":      lambda: p2s.histop(df_histo, 'cat'),
            "piep":        lambda: p2s.piep(df_histo, 'cat'),
            "timep":       lambda: p2s.timep(df_time, 'ts'),
            "xyp":         lambda: p2s.xyp(df_xy, 'a', 'c'),
            "linkp":       lambda: p2s.linkp(df=df_link, relationships=[('fm', 'to')], pos=_POS_LINK_),
            "chordp":      lambda: p2s.chordp(df=df_chord, relationships=[('fm', 'to')], wxh=(256, 256)),
            "spreadlinesp": lambda: p2s.spreadlinesp(df_spread, [('fm', 'to')], ego='0', time='ts'),
            "xyp_webgpu_payload": lambda: _webgpu_payload_(lambda: p2s.xyp(df_xy, 'a', 'c')),
        }

    def _median_render_time(self, fn):
        return statistics.median(fn().t_overall for _ in range(N_RUNS))

    def test_performance_regression(self):
        """Report render-time regressions (>1.5× baseline) across all five components.

        The test always passes; it emits a UserWarning and prints a columnar table
        when any component exceeds WARN_THRESHOLD.

        First run: UPDATE_PERF_BASELINE=1 python3 tests/test_performance.py  (writes baseline, passes)
        Recalibrate after an intentional change: UPDATE_PERF_BASELINE=1 pytest tests/test_performance.py
        Run with pytest -s to see the timing table in all cases.
        """
        update = os.environ.get("UPDATE_PERF_BASELINE") == "1"

        workloads = self._make_workloads()
        medians = {name: self._median_render_time(fn) for name, fn in workloads.items()}

        if update:
            BASELINE_PATH.write_text(json.dumps(medians, indent=2))
            return

        if not BASELINE_PATH.exists():
            self.skipTest("No perf baseline — run with UPDATE_PERF_BASELINE=1 to generate one")

        baseline = json.loads(BASELINE_PATH.read_text())

        rows = []
        slow = []
        for name, median in medians.items():
            if name not in baseline:
                rows.append((name, None, median, None))
                continue
            ratio = median / baseline[name]
            rows.append((name, baseline[name], median, ratio))
            if ratio > WARN_THRESHOLD:
                slow.append((name, ratio))

        if slow:
            _print_perf_table(rows)
            for name, ratio in slow:
                warnings.warn(
                    f"perf: {name} is {ratio:.2f}× baseline (threshold {WARN_THRESHOLD}×)",
                    UserWarning,
                    stacklevel=2,
                )


if __name__ == '__main__':
    unittest.main()
