"""
Tests for rows whose relationship endpoint is null.

Companion to tests/test_histop_null_bin.py and tests/test_piep_null_bin.py, but
chordp's version of the problem is different in kind.  A chord needs two
endpoints, so a row missing one has no arc and no ribbon -- it is simply not in
the diagram, and chordp already said so: __calculateOrder__ has always built its
node set with drop_nulls().  Three defects followed from the rest of the code not
agreeing with that.

1. Construction crashed.  df_edge_weights was built with a plain group_by, which
   keeps a null group, so None reached leafWalkFromEdges' sorted(node_set) and
   raised "'<' not supported between instances of 'str' and 'NoneType'".  Only
   an explicit order= got past it.

2. Selection lost rows.  is_in() yields *null* for a null endpoint, Kleene logic
   carried that through the & / | chain, and filter() drops a null mask row --
   as does the complement, since ~null is null.  So remove_records=True deleted
   every null-endpoint row from BOTH halves: on a 5-row frame, select-all gave 3
   and remove-all gave 0.  Rows vanished from an operation that had nothing to do
   with them, and the interactive stack then held the truncated frame.

3. A relationship may carry a third element (fm, to, weight), which chordp
   accepts at construction, but all three selection methods unpacked it as a pair
   and raised "too many values to unpack".

The invariant that ties 1 and 2 together, and what most of these tests assert:
select and remove must *partition* the frame -- len(select) + len(remove) == len(df)
for any query -- and the diagram must look exactly as it would if the
null-endpoint rows had never been in the frame.
"""
import unittest
from math import cos, sin

import polars as pl
from polars2svg import Polars2SVG


# rows 0-2 are real edges; rows 3-4 each miss one endpoint
_DF_NULL_ = pl.DataFrame({
    'fm':  ['a', 'b', 'c', None, 'a'],
    'to':  ['b', 'c', 'a', 'a',  None],
    'val': [1,   2,   3,   4,    5],
})
# the same frame with the null-endpoint rows already gone
_DF_CLEAN_ = _DF_NULL_.filter(pl.col('fm').is_not_null() & pl.col('to').is_not_null())
_RELS_ = [('fm', 'to')]


def _arc_midpoint(ch, node):
    row   = ch.df_node.filter(pl.col('__nm__').cast(pl.String) == str(node))
    amr   = row['__amr__'][0]
    r_mid = (ch.r + ch.r_inner) / 2.0
    return (ch.cx + r_mid * cos(amr), ch.cy + r_mid * sin(amr))


def _small_bbox(x, y, size=5.0):
    return (x - size, y - size, x + size, y + size)


class TestChordPNullEndpointConstruction(unittest.TestCase):
    """A null endpoint used to make the default order= path raise TypeError."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.ch  = cls.p2s.chordp(df=_DF_NULL_, relationships=_RELS_, wxh=(256, 256))

    def test_constructs_with_the_default_order(self):
        self.assertIsNotNone(self.ch.svg)

    def test_null_is_not_a_node(self):
        self.assertNotIn(None, self.ch.nodes_all)
        self.assertEqual(sorted(self.ch.nodes_all), ['a', 'b', 'c'])

    def test_edge_weights_carry_no_null_endpoint(self):
        """df_edge_weights has to agree with nodes_all, which drops nulls."""
        self.assertEqual(self.ch.df_edge_weights['__fm__'].null_count(), 0)
        self.assertEqual(self.ch.df_edge_weights['__to__'].null_count(), 0)

    def test_edge_weights_hold_only_the_real_edges(self):
        self.assertEqual(len(self.ch.df_edge_weights), 3)

    def test_node_names_carry_no_null(self):
        self.assertEqual(self.ch.df_node['__nm__'].null_count(), 0)

    def test_render_matches_the_frame_without_those_rows(self):
        """A row missing an endpoint contributes nothing, so the SVG is identical."""
        _clean_ = self.p2s.chordp(df=_DF_CLEAN_, relationships=_RELS_, wxh=(256, 256))
        self.assertEqual(self.ch.svg, _clean_.svg)

    def test_rows_are_still_in_the_frame(self):
        """Not rendering them is not the same as discarding them."""
        self.assertEqual(len(self.ch.df), len(_DF_NULL_))


class TestChordPNullEndpointPartition(unittest.TestCase):
    """select and remove must partition the frame for every query."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.ch  = cls.p2s.chordp(df=_DF_NULL_, relationships=_RELS_, wxh=(256, 256))
        cls.w, cls.h = cls.ch.wxh

    def _assert_partitions(self, keep, remove):
        self.assertEqual(len(keep) + len(remove), len(_DF_NULL_),
                         'select and remove must cover the frame exactly once')

    # ── whole-canvas selection ────────────────────────────────────────────────

    def test_rectangle_over_everything_partitions(self):
        _box_ = (0, 0, self.w, self.h)
        self._assert_partitions(self.ch.filterByRectangle(_box_),
                                self.ch.filterByRectangle(_box_, remove_records=True))

    def test_rectangle_over_everything_selects_only_real_edges(self):
        result = self.ch.filterByRectangle((0, 0, self.w, self.h))
        self.assertEqual(sorted(result['val'].to_list()), [1, 2, 3])

    def test_rectangle_over_everything_keeps_null_endpoint_rows(self):
        """The rows have no arc, so 'remove what I selected' must not touch them."""
        result = self.ch.filterByRectangle((0, 0, self.w, self.h), remove_records=True)
        self.assertEqual(sorted(result['val'].to_list()), [4, 5])

    def test_oval_over_everything_partitions(self):
        _oval_ = (self.w / 2, self.h / 2, self.w, self.h)
        self._assert_partitions(self.ch.filterByOval(_oval_),
                                self.ch.filterByOval(_oval_, remove_records=True))

    def test_oval_over_everything_keeps_null_endpoint_rows(self):
        result = self.ch.filterByOval((self.w / 2, self.h / 2, self.w, self.h),
                                      remove_records=True)
        self.assertEqual(sorted(result['val'].to_list()), [4, 5])

    # ── a two-node selection ──────────────────────────────────────────────────

    def test_two_node_rectangle_partitions(self):
        _ax_, _ay_ = _arc_midpoint(self.ch, 'a')
        _bx_, _by_ = _arc_midpoint(self.ch, 'b')
        _box_ = (min(_ax_, _bx_) - 5, min(_ay_, _by_) - 5,
                 max(_ax_, _bx_) + 5, max(_ay_, _by_) + 5)
        self._assert_partitions(self.ch.filterByRectangle(_box_),
                                self.ch.filterByRectangle(_box_, remove_records=True))

    def test_selecting_node_a_never_selects_a_null_endpoint_row(self):
        """Row 3 is (None -> 'a'): 'a' is selected, but the row has no chord."""
        _box_ = _small_bbox(*_arc_midpoint(self.ch, 'a'))
        result = self.ch.filterByRectangle(_box_)
        self.assertNotIn(4, result['val'].to_list())
        self.assertNotIn(5, result['val'].to_list())

    # ── an empty selection ────────────────────────────────────────────────────

    def test_empty_selection_partitions(self):
        _box_ = (0, 0, 1, 1)   # the corner: no arc there
        self._assert_partitions(self.ch.filterByRectangle(_box_),
                                self.ch.filterByRectangle(_box_, remove_records=True))

    def test_empty_selection_removes_nothing(self):
        result = self.ch.filterByRectangle((0, 0, 1, 1), remove_records=True)
        self.assertEqual(len(result), len(_DF_NULL_))

    # ── recordsAt (brush) ─────────────────────────────────────────────────────
    #
    # recordsAt is deliberately the *either*-endpoint contract, where the rectangle
    # and the oval are the *both*-endpoint one -- brushing a node means "rows this
    # node appears in", so a row with one null endpoint is reached through the
    # endpoint it still has.  That is unchanged by the null fix (Kleene OR already
    # gave True for `null | True`) and is asserted here so the difference from the
    # rectangle above reads as designed rather than accidental.

    def test_records_at_reaches_a_null_endpoint_row_by_its_surviving_endpoint(self):
        # rows 4 (None -> a) and 5 (a -> None) both name node a
        result = self.ch.recordsAt(_arc_midpoint(self.ch, 'a'))
        self.assertEqual(sorted(result['val'].to_list()), [1, 3, 4, 5])

    def test_records_at_does_not_reach_a_row_that_never_names_the_node(self):
        for _n_, _expected_ in (('b', [1, 2]), ('c', [2, 3])):
            with self.subTest(node=_n_):
                result = self.ch.recordsAt(_arc_midpoint(self.ch, _n_))
                self.assertEqual(sorted(result['val'].to_list()), _expected_)


class TestChordPNullEndpointDegenerate(unittest.TestCase):
    """Frames where the nulls leave little or nothing to draw."""

    def setUp(self):
        self.p2s = Polars2SVG()

    def _partition_check(self, df, expect_drawn_nodes):
        """expect_drawn_nodes is what ends up on the ring (df_node), which follows the
        surviving edges -- a node whose every edge lost an endpoint has no chord and so
        no arc, even though its name is still in nodes_all."""
        ch = self.p2s.chordp(df=df, relationships=_RELS_, wxh=(256, 256))
        self.assertEqual(sorted(str(n) for n in ch.df_node['__nm__'].to_list()),
                         sorted(str(n) for n in expect_drawn_nodes))
        w, h = ch.wxh
        keep   = ch.filterByRectangle((0, 0, w, h))
        remove = ch.filterByRectangle((0, 0, w, h), remove_records=True)
        self.assertEqual(len(keep) + len(remove), len(df))
        return ch, keep, remove

    def test_every_from_endpoint_null(self):
        """No edge keeps both endpoints, so nothing is drawn -- and nothing is lost."""
        _df_ = pl.DataFrame({'fm': [None, None, None], 'to': ['a', 'b', 'c'], 'val': [1, 2, 3]},
                            schema={'fm': pl.String, 'to': pl.String, 'val': pl.Int64})
        _ch_, _keep_, _remove_ = self._partition_check(_df_, [])
        self.assertEqual(len(_keep_), 0)
        self.assertEqual(len(_remove_), 3)
        self.assertEqual(len(_ch_.df_edge_weights), 0)

    def test_every_row_missing_an_endpoint(self):
        _df_ = pl.DataFrame({'fm': [None, None], 'to': [None, None], 'val': [1, 2]},
                            schema={'fm': pl.String, 'to': pl.String, 'val': pl.Int64})
        self._partition_check(_df_, [])

    def test_one_surviving_edge(self):
        _df_ = pl.DataFrame({'fm': ['a', None], 'to': ['b', None], 'val': [1, 2]})
        _ch_, _keep_, _remove_ = self._partition_check(_df_, ['a', 'b'])
        self.assertEqual(_keep_['val'].to_list(), [1])
        self.assertEqual(_remove_['val'].to_list(), [2])

    def test_numeric_node_ids_with_a_null(self):
        _df_ = pl.DataFrame({'fm': [1, 2, None], 'to': [2, 1, 1], 'val': [1, 2, 3]})
        _ch_, _keep_, _remove_ = self._partition_check(_df_, [1, 2])
        self.assertEqual(sorted(_keep_['val'].to_list()), [1, 2])
        self.assertEqual(_remove_['val'].to_list(), [3])


class TestChordPThreePartRelationship(unittest.TestCase):
    """(fm, to, weight) is accepted at construction, so selection must accept it
    too -- all three methods used to raise 'too many values to unpack'."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.df  = pl.DataFrame({'fm': ['a', 'b', 'c'], 'to': ['b', 'c', 'a'],
                                'w':  [1.0, 2.0, 3.0], 'val': [1, 2, 3]})
        cls.ch  = cls.p2s.chordp(df=cls.df, relationships=[('fm', 'to', 'w')], wxh=(256, 256))
        cls.w, cls.h = cls.ch.wxh

    def test_relationship_kept_its_third_element(self):
        self.assertEqual(len(self.ch.relationships[0]), 3)

    def test_filter_by_rectangle(self):
        self.assertEqual(len(self.ch.filterByRectangle((0, 0, self.w, self.h))), 3)

    def test_filter_by_rectangle_remove(self):
        self.assertEqual(len(self.ch.filterByRectangle((0, 0, self.w, self.h),
                                                       remove_records=True)), 0)

    def test_filter_by_oval(self):
        self.assertEqual(len(self.ch.filterByOval((self.w / 2, self.h / 2, self.w, self.h))), 3)

    def test_records_at(self):
        result = self.ch.recordsAt(_arc_midpoint(self.ch, 'a'))
        self.assertGreater(len(result), 0)

    def test_partitions(self):
        _box_ = (0, 0, self.w, self.h)
        self.assertEqual(len(self.ch.filterByRectangle(_box_)) +
                         len(self.ch.filterByRectangle(_box_, remove_records=True)),
                         len(self.df))


class TestChordPNullEndpointExplicitOrder(unittest.TestCase):
    """An explicit order= used to be the only way past the construction crash --
    and it is the path on which the row loss was measurable."""

    @classmethod
    def setUpClass(cls):
        cls.p2s = Polars2SVG()
        cls.ch  = cls.p2s.chordp(df=_DF_NULL_, relationships=_RELS_,
                                 order=['a', 'b', 'c'], wxh=(256, 256))

    def test_partitions(self):
        w, h = self.ch.wxh
        _box_ = (0, 0, w, h)
        self.assertEqual(len(self.ch.filterByRectangle(_box_)) +
                         len(self.ch.filterByRectangle(_box_, remove_records=True)),
                         len(_DF_NULL_))

    def test_null_endpoint_rows_survive_remove_all(self):
        w, h = self.ch.wxh
        result = self.ch.filterByRectangle((0, 0, w, h), remove_records=True)
        self.assertEqual(sorted(result['val'].to_list()), [4, 5])


if __name__ == '__main__':
    unittest.main()
