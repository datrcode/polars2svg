import unittest
import polars as pl
import polars2svg.xyp as xypmod
from polars2svg import Polars2SVG


def _x_order(inst):
    '''category value -> resolved __xi__ index, as an ordered dict (by index).'''
    m = {}
    for row in inst.df_flat.iter_rows(named=True):
        k = tuple(row['__x__'].values()) if isinstance(row['__x__'], dict) else row['__x__']
        m[k] = row['__xi__']
    return {k: m[k] for k in sorted(m, key=lambda z: m[z])}


def _contiguous_by(order_keys, pred):
    '''True if the items matching pred form one contiguous block (either orientation).'''
    flags = [bool(pred(k)) for k in order_keys]
    return flags == sorted(flags) or flags == sorted(flags, reverse=True)


def _one_run(order_keys, pred):
    '''True if the items matching pred form a single unbroken run -- unlike
    _contiguous_by, this also holds for a block sitting in the middle.'''
    flags = [bool(pred(k)) for k in order_keys]
    return sum(1 for i, f in enumerate(flags) if f and not (i and flags[i - 1])) == 1


class Testxyp_spectral(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _two_cluster_df(self):
        # a* categories co-occur with y0/y1; b* categories co-occur with y2/y3.
        ycluster = {'a1': ['y0', 'y1'], 'a2': ['y1', 'y0'], 'a3': ['y0', 'y1'],
                    'b1': ['y2', 'y3'], 'b2': ['y3', 'y2'], 'b3': ['y2', 'y3']}
        recs = [(x, y) for x, ys in ycluster.items() for y in ys]
        return pl.DataFrame({'xc': [r[0] for r in recs], 'yc': [r[1] for r in recs]})

    def test_spectral_groups_clusters(self):
        df = self._two_cluster_df()
        for lazy in [True, False]:
            xyp = self.p2s.xyp(df=df, x=('xc', self.p2s.SETp), y=('yc', self.p2s.SETp),
                               x_order='spectral', dot_size=5, use_lazy_execution=lazy)
            order = list(_x_order(xyp).keys())
            self.assertEqual(len(order), 6)
            self.assertTrue(_contiguous_by(order, lambda k: k[0].startswith('a')),
                            f'clusters not contiguous: {order}')

    def test_spectral_single_field_string_axis(self):
        # categorical by dtype (String), no SETp tag
        df = pl.DataFrame({'pet': ['cat', 'dog', 'fish', 'cat', 'dog', 'fish'],
                           'qty': [1, 2, 3, 4, 5, 6]})
        xyp = self.p2s.xyp(df=df, x='pet', y='qty', x_order='spectral', dot_size=5)
        self.assertEqual(len(_x_order(xyp)), 3)

    def test_spectral_by_and_weight(self):
        df = pl.DataFrame({'pet': ['cat', 'cat', 'dog', 'dog', 'fish', 'fish'],
                           'col': ['gray', 'black', 'gray', 'black', 'blue', 'green'],
                           'w':   [3, 1, 2, 2, 5, 5],
                           'qty': [1, 2, 3, 4, 5, 6]})
        # ordering pets by their color distribution, weighted by w -- just needs to run
        self.p2s.xyp(df=df, x='pet', y='qty', x_order='spectral',
                     spectral_by='col', spectral_weight='w', dot_size=5)

    def test_similarity_variants(self):
        df = self._two_cluster_df()
        for sim in ['cosine', 'linear', 'correlation']:
            for norm in [True, False]:
                self.p2s.xyp(df=df, x=('xc', self.p2s.SETp), y=('yc', self.p2s.SETp),
                             x_order='spectral', spectral_similarity=sim,
                             spectral_normalize=norm, dot_size=5)

    def test_both_axes_spectral(self):
        df = self._two_cluster_df()
        xyp = self.p2s.xyp(df=df, x=('xc', self.p2s.SETp), y=('yc', self.p2s.SETp),
                           x_order='spectral', y_order='spectral', dot_size=5)
        self.assertEqual(len(_x_order(xyp)), 6)

    def test_three_disconnected_cliques_stay_three_blocks(self):
        # Three cliques that share no partners at all -> three components, so the
        # Laplacian's zero eigenvalue has multiplicity 3.  A single global Fiedler
        # solve picks an arbitrary vector out of that null space, which used to give
        # two of the cliques the same value and interleave them (~1 run in 4).
        for trial in range(25):
            xs, ys = [], []
            for g in range(3):
                nodes = [f'g{g}n{i}' for i in range(6)]
                for a in nodes:
                    for b in nodes:
                        xs.append(a), ys.append(b)
            df  = pl.DataFrame({'src': xs, 'dst': ys}).sample(fraction=1.0, shuffle=True, seed=trial)
            xyp = self.p2s.xyp(df=df, x='src', y='dst', x_order='spectral',
                               y_order='spectral', dot_size=2)
            order = list(_x_order(xyp).keys())
            self.assertEqual(len(order), 18)
            for g in range(3):
                self.assertTrue(_one_run(order, lambda k, g=g: k.startswith(f'g{g}')),
                                f'clique g{g} not contiguous on trial {trial}: {order}')

    def test_disconnected_order_is_deterministic(self):
        df = pl.DataFrame({'src': [f'g{g}n{i}' for g in range(3) for i in range(6) for _ in range(6)],
                           'dst': [f'g{g}n{j}' for g in range(3) for _ in range(6) for j in range(6)]})
        orders = set()
        for _ in range(5):
            xyp = self.p2s.xyp(df=df, x='src', y='dst', x_order='spectral', dot_size=2)
            orders.add(tuple(_x_order(xyp).keys()))
        self.assertEqual(len(orders), 1, f'spectral order varied between identical runs: {orders}')

    def test_rejects_numeric_axis(self):
        df = pl.DataFrame({'pet': ['cat', 'dog'], 'qty': [1, 2]})
        with self.assertRaises(ValueError):
            self.p2s.xyp(df=df, x='qty', y='pet', x_order='spectral', dot_size=5)

    def test_rejects_bad_string(self):
        df = pl.DataFrame({'pet': ['cat', 'dog'], 'qty': [1, 2]})
        with self.assertRaises(ValueError):
            self.p2s.xyp(df=df, x='pet', y='qty', x_order='fiedler', dot_size=5)

    def test_rejects_missing_spectral_by(self):
        df = pl.DataFrame({'pet': ['cat', 'dog'], 'qty': [1, 2]})
        with self.assertRaises(ValueError):
            self.p2s.xyp(df=df, x='pet', y='qty', x_order='spectral',
                         spectral_by='nope', dot_size=5)

    def test_shared_small_multiples_single_global_solve(self):
        # Facets hold different category subsets; a shared spectral axis must compute
        # ONE global order and apply it identically to every panel.
        facet_map = {'p': ['a1', 'a2', 'b1'], 'q': ['a2', 'b1', 'b2'], 'r': ['a1', 'b1', 'b2']}
        yc = {'a1': ['y0', 'y1'], 'a2': ['y1', 'y0'], 'b1': ['y2', 'y3'], 'b2': ['y3', 'y2']}
        recs = [(f, x, y) for f, xs in facet_map.items() for x in xs for y in yc[x]]
        df = pl.DataFrame({'facet': [r[0] for r in recs],
                           'xc': [r[1] for r in recs], 'yc': [r[2] for r in recs]})
        tmpl = self.p2s.xyp(x=('xc', self.p2s.SETp), y=('yc', self.p2s.SETp),
                            x_order='spectral', dot_size=5, sm_shared={self.p2s.SM_X})

        # count spectral solves during the shared render
        calls = {'n': 0}
        orig = xypmod.XYp.__spectralOrder__
        def spy(self, src):
            calls['n'] += 1
            return orig(self, src)
        xypmod.XYp.__spectralOrder__ = spy
        try:
            lu = {f: df.filter(pl.col('facet') == f) for f in ['p', 'q', 'r']}
            render = tmpl.renderSmallMultiples(df, lu, None)
        finally:
            xypmod.XYp.__spectralOrder__ = orig

        self.assertEqual(calls['n'], 1, 'shared spectral axis should solve exactly once (the global reference)')
        orders = {f: _x_order(render[f]) for f in ['p', 'q', 'r']}
        allcats = set().union(*[set(o) for o in orders.values()])
        for c in allcats:
            idxs = {orders[f][c] for f in orders if c in orders[f]}
            self.assertEqual(len(idxs), 1, f'category {c!r} got inconsistent indices across panels: {idxs}')

    def test_unshared_small_multiples_orders_per_tile(self):
        # Without SM_X in sm_shared, each panel seriates independently (allowed).
        facet_map = {'p': ['a1', 'a2', 'b1'], 'q': ['a2', 'b1', 'b2']}
        yc = {'a1': ['y0', 'y1'], 'a2': ['y1', 'y0'], 'b1': ['y2', 'y3'], 'b2': ['y3', 'y2']}
        recs = [(f, x, y) for f, xs in facet_map.items() for x in xs for y in yc[x]]
        df = pl.DataFrame({'facet': [r[0] for r in recs],
                           'xc': [r[1] for r in recs], 'yc': [r[2] for r in recs]})
        tmpl = self.p2s.xyp(x=('xc', self.p2s.SETp), y=('yc', self.p2s.SETp),
                            x_order='spectral', dot_size=5)  # no sm_shared
        lu = {f: df.filter(pl.col('facet') == f) for f in ['p', 'q']}
        render = tmpl.renderSmallMultiples(df, lu, None)
        for f in ['p', 'q']:
            self.assertGreater(len(_x_order(render[f])), 0)


if __name__ == '__main__':
    unittest.main()
