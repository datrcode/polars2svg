import unittest
import polars as pl
from polars2svg import Polars2SVG


#
# Unit tests for XYp.recordsAt(xy, shape=, threshold=).
#
# recordsAt() maps a pixel-space query back to the original source records whose
# plotted dots fall inside the query region:
#   SELECT_CIRCLEp     (default) -- within `threshold` px of (x, y)
#   SELECT_HORIZONTALp           -- within `threshold` px of y (a horizontal band)
#   SELECT_VERTICALp             -- within `threshold` px of x (a vertical band)
# It returns self.df's original columns (with the internal __p2s_index__ dropped).
#
# The fixture places records on an L shape so the three shapes select provably
# different subsets: four records share a world-y (one horizontal line) and four
# share a world-x (one vertical line), meeting at the corner record.
#
class TestXypRecordsAt(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        # ids:      0      1      2      3      4      5      6
        # x:        0      1      2      3      0      0      0
        # y:        0      0      0      0      1      2      3
        #   -> ids {0,1,2,3} share world-y 0 (horizontal line)
        #   -> ids {0,4,5,6} share world-x 0 (vertical line), corner = id 0
        self.df = pl.DataFrame({
            'id': [0, 1, 2, 3, 4, 5, 6],
            'x':  [0, 1, 2, 3, 0, 0, 0],
            'y':  [0, 0, 0, 0, 1, 2, 3],
        })
        self.xyp = self.p2s.xyp(self.df, x='x', y='y')
        # pixel coordinates of every record, keyed by id
        _flat_ = self.xyp.df_flat.join(self.xyp.df.select(['__p2s_index__', 'id']), on='__p2s_index__')
        self._px_ = {r['id']: (r['__xpx__'], r['__ypx__']) for r in _flat_.iter_rows(named=True)}

    def test_circle_selects_single_record(self):
        _x_, _y_ = self._px_[0]                       # corner record's pixel
        _res_    = self.xyp.recordsAt((_x_, _y_), threshold=0.5)
        self.assertEqual(set(_res_['id'].to_list()), {0})
        # original columns are returned; the internal index is stripped
        self.assertNotIn('__p2s_index__', _res_.columns)
        self.assertEqual(set(_res_.columns), {'id', 'x', 'y'})

    def test_default_shape_is_circle(self):
        _x_, _y_ = self._px_[2]
        _res_    = self.xyp.recordsAt((_x_, _y_), threshold=0.5)   # shape omitted
        self.assertEqual(set(_res_['id'].to_list()), {2})

    def test_circle_threshold_captures_neighbors(self):
        # a threshold spanning the gap to the adjacent point on the horizontal line
        # picks up both; a tiny threshold picks up only the target.
        _x0_, _y0_ = self._px_[0]
        _x1_, _y1_ = self._px_[1]
        _gap_      = ((_x1_ - _x0_) ** 2 + (_y1_ - _y0_) ** 2) ** 0.5
        _wide_     = self.xyp.recordsAt((_x0_, _y0_), self.p2s.SELECT_CIRCLEp, threshold=_gap_ + 1.0)
        self.assertTrue({0, 1}.issubset(set(_wide_['id'].to_list())))
        _tight_    = self.xyp.recordsAt((_x0_, _y0_), self.p2s.SELECT_CIRCLEp, threshold=_gap_ - 1.0)
        self.assertEqual(set(_tight_['id'].to_list()), {0})

    def test_circle_empty_when_far(self):
        _res_ = self.xyp.recordsAt((5000, 5000), self.p2s.SELECT_CIRCLEp, threshold=1.0)
        self.assertEqual(_res_.height, 0)
        self.assertEqual(set(_res_.columns), {'id', 'x', 'y'})   # schema preserved even when empty

    def test_circle_far_no_int_overflow(self):
        # Regression: __xpx__/__ypx__ are Int32, so a distant query point used to overflow
        # (x-X)^2 into a negative value that passed the `<= threshold**2` test, wrongly
        # matching every record. The squared distance is computed in Float64 now.
        _res_ = self.xyp.recordsAt((100000, 100000), self.p2s.SELECT_CIRCLEp, threshold=1.0)
        self.assertEqual(_res_.height, 0)

    def test_horizontal_selects_shared_y(self):
        # a horizontal band at the corner's y selects every record on that world-y line,
        # regardless of x.
        _x_, _y_ = self._px_[0]
        _res_    = self.xyp.recordsAt((_x_, _y_), self.p2s.SELECT_HORIZONTALp, threshold=0.5)
        self.assertEqual(set(_res_['id'].to_list()), {0, 1, 2, 3})

    def test_vertical_selects_shared_x(self):
        # a vertical band at the corner's x selects every record on that world-x line,
        # regardless of y.
        _x_, _y_ = self._px_[0]
        _res_    = self.xyp.recordsAt((_x_, _y_), self.p2s.SELECT_VERTICALp, threshold=0.5)
        self.assertEqual(set(_res_['id'].to_list()), {0, 4, 5, 6})

    def test_horizontal_and_vertical_differ(self):
        _x_, _y_ = self._px_[0]
        _h_ = set(self.xyp.recordsAt((_x_, _y_), self.p2s.SELECT_HORIZONTALp, threshold=0.5)['id'].to_list())
        _v_ = set(self.xyp.recordsAt((_x_, _y_), self.p2s.SELECT_VERTICALp,   threshold=0.5)['id'].to_list())
        self.assertNotEqual(_h_, _v_)
        self.assertEqual(_h_ & _v_, {0})   # they intersect only at the corner record

    def test_unknown_shape_raises(self):
        _x_, _y_ = self._px_[0]
        with self.assertRaises(ValueError):
            self.xyp.recordsAt((_x_, _y_), shape='not-a-shape')


if __name__ == '__main__':
    unittest.main()
