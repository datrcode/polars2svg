import json
import os
import platform as _platform_
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
import unittest
import warnings
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as _dist_version
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import polars as pl

import polars2svg
from polars2svg import Polars2SVG

BASELINE_PATH = Path(__file__).parent / "perf_baseline.json"
SCHEMA_VERSION = 2      # 1 == the old flat {component: seconds} file
WARN_THRESHOLD = 1.5    # emit a warning (and print table) when ratio exceeds this
N_ROWS = 1_000_000
N_RUNS = 3

# Pre-built positions for the 100 nodes used in the linkp workload
_N_NODES_ = 100
_POS_LINK_ = {
    str(i): (float(i % 10) / 9.0, float(i // 10) / 9.0)
    for i in range(_N_NODES_)
}

_N_CHORD_NODES_ = 20

# Accelerated-layout workloads. Sized so a single run lands in the tens-to-hundreds
# of milliseconds on the GPU path — big enough that the kernels dominate interpreter
# overhead, small enough that 3 runs stay cheap.
_N_ODFLOW_FLOWS_ = 60
_ODFLOW_ITERS_   = 60
_N_TFDP_NODES_   = 1_500
_TFDP_ITERS_     = 100


# ---------------------------------------------------------------------------
# Platform identity — timings are only comparable within one of these
# ---------------------------------------------------------------------------

def _slug(text):
    """Lowercase, alphanumeric-and-dashes only."""
    out = re.sub(r'[^a-z0-9]+', '-', str(text).lower())
    return out.strip('-')


def _cpu_model():
    """Marketing name of the CPU/SoC ('Apple M5 Max', 'AMD Ryzen 9 7900X ...')."""
    try:
        if sys.platform == 'darwin':
            _out_ = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                   capture_output=True, text=True, timeout=5).stdout.strip()
            if _out_:
                return _out_
        elif sys.platform.startswith('linux'):
            for _line_ in Path('/proc/cpuinfo').read_text().splitlines():
                if _line_.startswith('model name'):
                    return _line_.split(':', 1)[1].strip()
    except Exception:   # noqa: BLE001 - identity is best-effort; never fail a timing run
        pass
    return _platform_.processor() or _platform_.machine() or 'unknown-cpu'


def _cpu_slug(model):
    """'AMD Ryzen 9 7900X 12-Core Processor' -> 'amd-ryzen-9-7900x'."""
    _s_ = _slug(model)
    _s_ = re.sub(r'-\d+-core.*$', '', _s_)          # "-12-core-processor" tail
    _s_ = re.sub(r'-(processor|cpu)$', '', _s_)
    _s_ = re.sub(r'-w-radeon.*$', '', _s_)          # "... w/ Radeon Graphics"
    return _s_ or 'unknown-cpu'


def _gpu_name():
    """Best-effort GPU name; None when nothing reports one."""
    if sys.platform == 'darwin':
        return _cpu_model()     # Apple silicon: the GPU is part of the SoC
    _exe_ = shutil.which('nvidia-smi')
    if _exe_ is not None:
        try:
            _proc_ = subprocess.run([_exe_, '--query-gpu=name', '--format=csv,noheader'],
                                    capture_output=True, text=True, timeout=10)
            # nvidia-smi reports "couldn't communicate with the NVIDIA driver" on
            # stdout and exits non-zero, so without the returncode check that
            # message gets recorded as the GPU name.
            _out_ = _proc_.stdout.strip()
            if _proc_.returncode == 0 and _out_:
                return _out_.splitlines()[0].strip()
        except Exception:   # noqa: BLE001
            pass
    return None


def _accel():
    """What accelerated path this *interpreter* can actually reach.

    Returns (tag, mlx_backend). The tag is the last field of the platform id, and
    is the whole reason the four target platforms are distinguishable:

        nomlx         mlx not installed (the clean-room `.venv`)
        mlx-nobackend mlx installed but no backend library (Linux `mlx` with no extra)
        metal         Apple silicon GPU
        cuda          NVIDIA GPU
        mlx-cpu       mlx present, GPU probe failed -> CPU device
    """
    if find_spec('mlx') is None:
        return 'nomlx', None
    try:
        import mlx.core  # noqa: F401
    except Exception:   # noqa: BLE001 - front-end without a backend library
        return 'mlx-nobackend', None
    try:
        from polars2svg.tfdp_layout import gpu_backend
        _backend_ = gpu_backend()
    except Exception:   # noqa: BLE001
        return 'mlx-nobackend', None
    return ({'cpu': 'mlx-cpu'}.get(_backend_, _backend_)), _backend_


def _dist(name):
    try:
        return _dist_version(name)
    except PackageNotFoundError:
        return None


def _platform_descriptor():
    """(platform_id, metadata) for the machine + interpreter running right now."""
    _accel_tag_, _mlx_backend_ = _accel()
    _cpu_ = _cpu_model()
    _os_  = _platform_.system().lower()
    _arch_ = _platform_.machine().lower()
    _id_ = f"{_os_}-{_arch_}-{_cpu_slug(_cpu_)}-{_accel_tag_}"

    return _id_, {
        'platform_id':   _id_,
        'os':            _os_,
        'os_release':    _platform_.release(),
        'arch':          _arch_,
        'cpu':           _cpu_,
        'cpu_count':     os.cpu_count(),
        'gpu':           _gpu_name(),
        'accel':         _accel_tag_,
        'mlx_backend':   _mlx_backend_,
        'mlx_version':   _dist('mlx'),
        'host':          socket.gethostname(),
        'python':        _platform_.python_version(),
        'interpreter':   sys.prefix,
        'polars2svg':    getattr(polars2svg, '__version__', None),
        'numpy':         np.__version__,
        'polars':        pl.__version__,
        'n_rows':        N_ROWS,
        'n_runs':        N_RUNS,
    }


def _load_baseline():
    """(entries, legacy) — the platform-keyed map, and whether a schema-1 file was found."""
    if not BASELINE_PATH.exists():
        return {}, False
    try:
        _doc_ = json.loads(BASELINE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}, False
    if isinstance(_doc_, dict) and _doc_.get('schema') == SCHEMA_VERSION:
        return _doc_.get('platforms', {}), False
    # Schema 1 was a flat {component: seconds} map with no platform identity, so
    # there is no honest way to say which machine produced it. Report, don't guess.
    return {}, True


def _write_baseline(platform_id, metadata, medians):
    """Merge this platform's timings into the file, leaving other platforms intact.

    The merge is what lets one file accumulate every machine: run the update on
    each, then copy/collect the entries (``tools/perf_report.py`` merges files).
    """
    _entries_, _ = _load_baseline()
    _entries_[platform_id] = {
        'platform':  metadata,
        'recorded':  datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'timings':   medians,
    }
    BASELINE_PATH.write_text(json.dumps(
        {'schema': SCHEMA_VERSION, 'platforms': _entries_}, indent=2, sort_keys=False) + "\n")


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


def _print_perf_table(rows, platform_id):
    """Print a columnar performance comparison table to stdout."""
    w_name = max(len(r[0]) for r in rows)
    w_name = max(w_name, len('component'))
    header = f"  {'component':<{w_name}}   {'baseline':>10}   {'current':>10}   {'ratio':>7}   status"
    sep    = f"  {'-'*w_name}   {'-'*10}   {'-'*10}   {'-'*7}   ------"
    lines = [
        f"\nPerformance vs baseline (platform={platform_id}, warn_threshold={WARN_THRESHOLD}×):",
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
        def _timed_(fn):
            t0 = time.time()
            fn()
            return _TimedResult_(time.time() - t0)

        workloads = {
            "histop":      lambda: p2s.histop(df_histo, 'cat'),
            "piep":        lambda: p2s.piep(df_histo, 'cat'),
            "timep":       lambda: p2s.timep(df_time, 'ts'),
            "xyp":         lambda: p2s.xyp(df_xy, 'a', 'c'),
            "linkp":       lambda: p2s.linkp(df=df_link, relationships=[('fm', 'to')], pos=_POS_LINK_),
            "chordp":      lambda: p2s.chordp(df=df_chord, relationships=[('fm', 'to')], wxh=(256, 256)),
            "spreadlinesp": lambda: p2s.spreadlinesp(df_spread, [('fm', 'to')], ego='0', time='ts'),
            "xyp_webgpu_payload": lambda: _webgpu_payload_(lambda: p2s.xyp(df_xy, 'a', 'c')),
        }

        # --- accelerated layouts -------------------------------------------------
        # The render components above are pure CPU, so they time the same on every
        # interpreter. These two are the only workloads that move when mlx/Metal/CUDA
        # is present, which is what makes the four platform ids worth separating.
        from polars2svg.od_flow_layout import ODFlowLayout
        _rng_ = np.random.default_rng(0)
        _flows_ = [tuple(map(float, r))
                   for r in _rng_.uniform(0.0, 800.0, size=(_N_ODFLOW_FLOWS_, 4))]
        # ODFlowLayout does its work in __init__; results() just returns the points.
        workloads["od_flow_layout"] = lambda: _timed_(
            lambda: ODFlowLayout(_flows_, iterations=_ODFLOW_ITERS_))

        # TFDPLayout is mlx-only: absent from the clean-room platforms by design, so
        # its baseline key simply does not exist there.
        try:
            import networkx as nx
            from polars2svg.tfdp_layout import TFDPLayout
        except ImportError:
            pass
        else:
            _g_ = nx.barabasi_albert_graph(_N_TFDP_NODES_, 3, seed=0)
            workloads["tfdp_layout"] = lambda: _timed_(
                lambda: TFDPLayout(_g_, max_iter=_TFDP_ITERS_, seed=0).results())

        return workloads

    def _median_render_time(self, fn):
        return statistics.median(fn().t_overall for _ in range(N_RUNS))

    def test_performance_regression(self):
        """Report render-time regressions (>1.5× baseline) for this platform.

        Timings are only comparable within one platform id — os/arch/cpu/accelerator —
        so the baseline file holds one entry per platform and this test compares
        against the entry matching the machine and interpreter it is running on.
        The test always passes; it emits a UserWarning and prints a columnar table
        when any component exceeds WARN_THRESHOLD.

        First run on a machine (writes/merges this platform's entry, passes):
            UPDATE_PERF_BASELINE=1 .venv/bin/python -m pytest tests/test_performance.py
        Run with pytest -s to see the timing table in all cases.
        """
        update = os.environ.get("UPDATE_PERF_BASELINE") == "1"
        platform_id, metadata = _platform_descriptor()

        workloads = self._make_workloads()
        medians = {name: self._median_render_time(fn) for name, fn in workloads.items()}

        if update:
            _write_baseline(platform_id, metadata, medians)
            print(f"\nperf baseline written for platform '{platform_id}' "
                  f"({metadata['cpu']}, accel={metadata['accel']}) -> {BASELINE_PATH}\n")
            return

        entries, legacy = _load_baseline()
        if legacy:
            self.skipTest(
                f"perf baseline is the old schema-1 (flat, platform-less) format — "
                f"regenerate with UPDATE_PERF_BASELINE=1")
        if platform_id not in entries:
            self.skipTest(
                f"No perf baseline for platform '{platform_id}' "
                f"(have: {', '.join(sorted(entries)) or 'none'}) — "
                f"run with UPDATE_PERF_BASELINE=1 to generate one")

        baseline = entries[platform_id]['timings']

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
        # Baselined workloads that did not run here (component removed, or an
        # accelerated layout whose backend is missing) — surfaced, not silently dropped.
        for name in baseline:
            if name not in medians:
                rows.append((name, baseline[name], None, None))

        if slow:
            _print_perf_table(rows, platform_id)
            for name, ratio in slow:
                warnings.warn(
                    f"perf: {name} is {ratio:.2f}× baseline (threshold {WARN_THRESHOLD}×) "
                    f"on platform {platform_id}",
                    UserWarning,
                    stacklevel=2,
                )


if __name__ == '__main__':
    unittest.main()
