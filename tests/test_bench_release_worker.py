import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_TOOLS_ = Path(__file__).resolve().parent.parent / 'tools'


def _load_worker_():
    spec = importlib.util.spec_from_file_location('bench_release_worker', _TOOLS_ / 'bench_release_worker.py')
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestBenchReleaseWorker(unittest.TestCase):
    """tools/bench_release_worker.py is run under old releases by the G5 driver, so it must
    keep working against the current tree: every workload renders at a tiny size."""

    def test_netflow_frame_is_deterministic_and_shaped(self):
        w = _load_worker_()
        a, b = w.make_netflow(2000, seed=1), w.make_netflow(2000, seed=1)
        self.assertTrue(a.equals(b))
        self.assertEqual(a.height, 2000)
        self.assertEqual(set(a.columns), {'ts', 'sip', 'dip', 'sip_net', 'dip_net',
                                          'sport', 'dport', 'proto', 'bytes', 'pkts'})

    def test_every_workload_renders_and_reports(self):
        w = _load_worker_()
        with tempfile.TemporaryDirectory() as d:
            w.make_netflow(3000).write_parquet(Path(d) / 'netflow_3000.parquet')
            doc = json.loads(subprocess.run(
                [sys.executable, str(_TOOLS_ / 'bench_release_worker.py'),
                 '--data', d, '--sizes', '3000', '--runs', '1'],
                capture_output=True, text=True, check=True).stdout)
        self.assertEqual(set(doc['results']), {f'{n}@3000' for n in w.WORKLOADS})
        self.assertIn('graph_build', w.WORKLOADS)
        for key, r in doc['results'].items():
            self.assertEqual(r['status'], 'ok', f'{key}: {r}')
            self.assertGreater(r['median_s'], 0.0, key)
            if not key.startswith('graph_build'):             # the one workload with no SVG
                self.assertGreater(r['svg_bytes'], 0, key)


if __name__ == '__main__':
    unittest.main()
