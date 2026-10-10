import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

_TOOLS_ = Path(__file__).resolve().parent.parent / 'tools'


def _load_worker_():
    spec = importlib.util.spec_from_file_location('bench_release_worker', _TOOLS_ / 'bench_release_worker.py')
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestBenchReleaseWorker(unittest.TestCase):
    """tools/bench_release_worker.py is run under old releases by the G5 driver, so it must
    keep working against the current tree: every workload renders at a tiny size."""

    @classmethod
    def setUpClass(cls):
        if not (_TOOLS_ / 'bench_release_worker.py').exists():
            raise unittest.SkipTest('tools/bench_release_worker.py not present (dev-only tooling)')

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
        self.assertLessEqual(set(w.NO_SVG), set(w.WORKLOADS))
        for key, r in doc['results'].items():
            self.assertEqual(r['status'], 'ok', f'{key}: {r}')
            self.assertGreater(r['median_s'], 0.0, key)
            if key.split('@')[0] in w.NO_SVG:                # not renders: no SVG to measure
                self.assertEqual(r['svg_bytes'], 0, key)
            else:
                self.assertGreater(r['svg_bytes'], 0, key)

    def test_synthetic_hosts_become_dotted_quads(self):
        # ipSubnetTreeMapLayout() groups only IPv4-shaped names, so the integer hosts are
        # converted; 256 hosts share a /24.
        w  = _load_worker_()
        df = w._ip_frame_(w.make_netflow(2000))
        self.assertEqual(df.columns, ['sip', 'dip'])
        self.assertTrue(df['sip'].str.contains(r'^10\.\d+\.\d+\.\d+$').all())
        self.assertTrue(df['dip'].str.contains(r'^10\.\d+\.\d+\.\d+$').all())
        self.assertEqual(w._ip_frame_(w.make_netflow(1).with_columns(sip=pl.lit('300')))['sip'][0],
                         '10.0.1.44')

    def test_real_addresses_pass_through(self):
        w   = _load_worker_()
        df  = pl.DataFrame({'sip': ['172.10.0.4', '10.0.0.2'], 'dip': ['10.1.2.3', '172.10.0.4'],
                            'proto': ['tcp', 'udp']})
        self.assertTrue(w._ip_frame_(df).equals(df.select(['sip', 'dip'])))

    def test_treemap_groups_the_synthetic_hosts_by_subnet(self):
        # Without quads every host would fall outside the IPv4 grouping, and the workload
        # would time a layout of nothing in particular.
        from polars2svg import Polars2SVG
        w   = _load_worker_()
        p2s = Polars2SVG()
        g   = p2s.createNetworkXGraph(w._ip_frame_(w.make_netflow(3000)), [('sip', 'dip')])
        pos, cells = p2s.ipSubnetTreeMapLayout(g, return_cells=True)
        self.assertEqual(set(pos), set(g.nodes))
        self.assertEqual(set(cells), {n.rsplit('.', 1)[0] + '.0/24' for n in g.nodes})
        self.assertGreater(len(cells), 1)


if __name__ == '__main__':
    unittest.main()
