import importlib.util
import tempfile
import unittest
from pathlib import Path

import polars as pl

_TOOLS_ = Path(__file__).resolve().parent.parent / 'tools'


def _load_():
    spec = importlib.util.spec_from_file_location('bench_releases', _TOOLS_ / 'bench_releases.py')
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ok_(median, rss=100.0):
    return {'status': 'ok', 'median_s': median, 'cold_s': median * 1.1, 'peak_rss_mb': rss,
            'svg_bytes': 10, 'runs': 3}


class TestBenchReleasesReport(unittest.TestCase):
    """The pure parts of tools/bench_releases.py; the venv / network parts are exercised by
    running the tool, not by the suite."""

    @classmethod
    def setUpClass(cls):
        if not (_TOOLS_ / 'bench_releases.py').exists():
            raise unittest.SkipTest('tools/bench_releases.py not present (dev-only tooling)')

    def test_summarize_takes_median_of_rounds_and_ignores_failed_rounds(self):
        b = _load_()
        s = b.summarize([_ok_(1.0), _ok_(3.0), {'status': 'oom'}, _ok_(2.0)])
        self.assertEqual(s['status'], 'ok')
        self.assertEqual(s['median_s'], 2.0)
        self.assertEqual(s['rounds'], 3)

    def test_summarize_all_failed_reports_the_status(self):
        b = _load_()
        self.assertEqual(b.summarize([{'status': 'oom'}, {'status': 'oom'}])['status'], 'oom')

    def test_table_ratios_are_against_the_first_column(self):
        b    = _load_()
        cols = [b.Column('old', Path('x')), b.Column('new', Path('y'))]
        summ = {('xyp', 10, 'old'): b.summarize([_ok_(1.0)]), ('xyp', 10, 'new'): b.summarize([_ok_(2.0)])}
        t = b.format_table(summ, cols, [10], ['xyp'], 'median_s', lambda v: f'{v:.0f}s', 'median')
        row = [ln for ln in t.splitlines() if ln.startswith('xyp@10')][0]
        self.assertIn('1s', row)
        self.assertIn('2s (2.00x)', row)
        self.assertNotIn('(1.00x)', row)

    def test_table_shows_failures_not_numbers(self):
        b    = _load_()
        cols = [b.Column('old', Path('x')), b.Column('new', Path('y'))]
        summ = {('xyp', 10, 'old'): b.summarize([_ok_(1.0)]), ('xyp', 10, 'new'): b.summarize([{'status': 'oom'}])}
        t = b.format_table(summ, cols, [10], ['xyp'], 'median_s', lambda v: f'{v:.0f}s', 'median')
        self.assertIn('oom', t)

    def test_import_data_maps_columns_and_derives_networks(self):
        b = _load_()
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / 'real.parquet'
            pl.DataFrame({'t': [1, 2], 'a': ['10.1.2.3', '10.1.9.9'], 'z': ['10.2.0.1', '10.2.0.2'],
                          'sp': [1, 2], 'dp': [80, 443], 'p': ['tcp', 'udp'], 'by': [1.0, 2.0],
                          'pk': [1, 2]}).with_columns(pl.col('t').cast(pl.Datetime('us'))).write_parquet(src)
            n = b.import_data(src, dict(ts='t', sip='a', dip='z', sport='sp', dport='dp',
                                        proto='p', bytes='by', pkts='pk'), Path(d) / 'data')
            out = pl.read_parquet(Path(d) / 'data' / f'netflow_{n}.parquet')
        self.assertEqual(n, 2)
        self.assertEqual(out['sip_net'].to_list(), ['10.1', '10.1'])
        self.assertEqual(out['dip_net'].to_list(), ['10.2', '10.2'])

    def test_import_data_concatenates_csv_chunks_parses_dates_and_matches_the_worker_schema(self):
        # The VAST 2013 capture is three CSV chunks with a 'YYYY-MM-DD HH:MM:SS' date column.
        b    = _load_()
        rows = ['T,S,D,SP,DP,P,B,K',
                '2013-04-01 07:50:16,172.20.0.3,172.255.255.255,137,137,UDP,1104,12',
                '2013-04-01 07:50:21,172.10.0.40,10.0.0.1,138,80,TCP,184,2']
        with tempfile.TemporaryDirectory() as d:
            chunks = []
            for i, body in enumerate((rows[:2], [rows[0], rows[2]])):
                f = Path(d) / f'chunk{i}.csv'
                f.write_text('\n'.join(body) + '\n')
                chunks.append(f)
            n = b.import_data(chunks, dict(ts='T', sip='S', dip='D', sport='SP', dport='DP',
                                           proto='P', bytes='B', pkts='K'), Path(d) / 'data')
            out = pl.read_parquet(Path(d) / 'data' / f'netflow_{n}.parquet')
        self.assertEqual(n, 2)
        self.assertEqual(out.schema, b._worker_module_().make_netflow(1).schema)
        self.assertEqual(out['sip'].to_list(), ['172.20.0.3', '172.10.0.40'])     # chunk order kept
        self.assertEqual(out['ts'][1].isoformat(), '2013-04-01T07:50:21')
        self.assertEqual(out['sip_net'].to_list(), ['172.20', '172.10'])
        self.assertEqual(out['bytes'].to_list(), [1104.0, 184.0])


if __name__ == '__main__':
    unittest.main()
