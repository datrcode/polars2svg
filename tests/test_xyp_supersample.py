#
# test_xyp_supersample.py
#
# Tests for supersampling of integer dot sizes (dot_size_supersample=).
#
# Background: an integer dot_size draws a raster of dot_size x dot_size cells and snaps
# every point to a grid step of dot_size, so positions can only fall on integer
# multiples of dot_size (0, N, 2N, ...).  That coarse quantization merges points that
# are close but land in different sub-cells.
#
# dot_size_supersample=s (int >= 1, default 1) subdivides the grid: points snap to a
# finer step of dot_size/s, so values fall at fractional multiples of the cell (2x ->
# 0, 0.5, 1, 1.5, ... in cell units) for finer positional resolution while the rects
# are still drawn dot_size wide.  s=1 is byte-identical to the original raster path;
# the factor is inert (warns once) for float/field dot sizes.
#
import logging
import re
import unittest

import polars as pl

from polars2svg import Polars2SVG


_INERT_MSG_FRAGMENT_ = 'dot_size_supersample only affects integer dot_size'


def _dot_rects_(svg):
    """Extract (x, y) coordinate strings for the integer-dot rect group only
    (excludes background/context rects)."""
    m = re.search(r'rect-group-\d+"[^>]*>(.*?)</g>', svg, re.S)
    if m is None:
        return []
    return re.findall(r'<rect x="([-\d.]+)" y="([-\d.]+)"', m.group(1))


class _SupersampleBase_(unittest.TestCase):
    def setUp(self):
        self.p2s = Polars2SVG()
        # A regular grid of points, one per (i, j) on a coarse lattice so that at
        # small dot_size the raster keeps them distinct and supersampling refines them.
        xs, ys = [], []
        for i in range(0, 101, 2):
            for j in range(0, 101, 2):
                xs.append(float(i)); ys.append(float(j))
        self.df = pl.DataFrame({'x': xs, 'y': ys})


# ---------------------------------------------------------------------------
# Byte-identity of the default / ss=1 path
# ---------------------------------------------------------------------------

class TestSupersampleDefault(_SupersampleBase_):

    def test_default_is_one(self):
        xyp = self.p2s.xyp(self.df, 'x', 'y', dot_size=4)
        self.assertEqual(xyp.dot_size_supersample, 1)

    def test_omitted_equals_explicit_one(self):
        # Same pixel snapping whether the factor is defaulted or explicitly 1.
        a = self.p2s.xyp(self.df, 'x', 'y', dot_size=4)
        b = self.p2s.xyp(self.df, 'x', 'y', dot_size=4, dot_size_supersample=1)
        self.assertEqual(sorted(_dot_rects_(a.svg)), sorted(_dot_rects_(b.svg)))

    def test_ss1_coords_are_integers(self):
        # The raster invariant: with no supersampling every dot coordinate is a whole
        # pixel (integer multiple of dot_size offset from the plot origin).
        xyp = self.p2s.xyp(self.df, 'x', 'y', dot_size=5)
        for sx, sy in _dot_rects_(xyp.svg):
            self.assertEqual(float(sx), int(float(sx)))
            self.assertEqual(float(sy), int(float(sy)))


# ---------------------------------------------------------------------------
# Supersampling refines the grid
# ---------------------------------------------------------------------------

class TestSupersampleRefines(_SupersampleBase_):

    def test_fractional_positions_with_odd_dot_size(self):
        # Odd dot_size (=3) with 2x supersampling -> step 1.5, so half-cell positions
        # produce fractional pixel coordinates.
        xyp = self.p2s.xyp(self.df, 'x', 'y', dot_size=3, dot_size_supersample=2, wxh=(200, 200))
        coords = [float(v) for pair in _dot_rects_(xyp.svg) for v in pair]
        self.assertTrue(any(abs(v - round(v)) > 1e-9 for v in coords),
                        'expected some fractional coordinates under 2x supersampling')

    def test_snaps_to_finer_step(self):
        # Every dot coordinate (relative to the raster origin) must be a multiple of
        # dot_size/s.  x offset = __xpx__ - plot_origin[0]; y offset (top-left corner)
        # = plot_origin[1] - dot_size - __ypx__.
        dot_size, s = 4, 2
        step = dot_size / s
        xyp = self.p2s.xyp(self.df, 'x', 'y', dot_size=dot_size, dot_size_supersample=s, wxh=(200, 200))
        ox = xyp.plot_origin[0]
        oy = xyp.plot_origin[1] - dot_size
        for sx, sy in _dot_rects_(xyp.svg):
            self.assertAlmostEqual((float(sx) - ox) % step, 0.0, places=6)
            self.assertAlmostEqual((oy - float(sy)) % step, 0.0, places=6)

    def test_more_distinct_cells_than_ss1(self):
        # A finer grid can only resolve at least as many distinct dot positions; on this
        # dense lattice with a coarse dot_size it resolves strictly more.
        base = self.p2s.xyp(self.df, 'x', 'y', dot_size=6, wxh=(180, 180))
        fine = self.p2s.xyp(self.df, 'x', 'y', dot_size=6, dot_size_supersample=3, wxh=(180, 180))
        n_base = len(set(_dot_rects_(base.svg)))
        n_fine = len(set(_dot_rects_(fine.svg)))
        self.assertGreater(n_fine, n_base)

    def test_rects_still_dot_size_wide(self):
        # Supersampling changes position resolution only -- the rects keep their
        # dot_size footprint (the CSS style block is unchanged).
        xyp = self.p2s.xyp(self.df, 'x', 'y', dot_size=7, dot_size_supersample=2)
        self.assertIn('width: 7px', xyp.svg)
        self.assertIn('height: 7px', xyp.svg)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestSupersampleValidation(_SupersampleBase_):

    def test_zero_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.xyp(self.df, 'x', 'y', dot_size=4, dot_size_supersample=0)

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.xyp(self.df, 'x', 'y', dot_size=4, dot_size_supersample=-2)

    def test_float_raises(self):
        with self.assertRaises(ValueError):
            self.p2s.xyp(self.df, 'x', 'y', dot_size=4, dot_size_supersample=2.0)

    def test_bool_raises(self):
        # True is an int subclass but a supersample factor of "True" is a mistake.
        with self.assertRaises(ValueError):
            self.p2s.xyp(self.df, 'x', 'y', dot_size=4, dot_size_supersample=True)

    def test_typo_kwarg_rejected(self):
        # The new param is in the allowlist; a typo of it still raises (drift guard).
        with self.assertRaises(TypeError):
            self.p2s.xyp(self.df, 'x', 'y', dot_size=4, dot_size_supersamp=2)


# ---------------------------------------------------------------------------
# Inert on non-integer dot sizes (warns once, does not change geometry)
# ---------------------------------------------------------------------------

class _CountingHandler_(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
    def emit(self, record):
        self.records.append(record.getMessage())


class TestSupersampleInert(_SupersampleBase_):
    def setUp(self):
        super().setUp()
        self.logger  = logging.getLogger('polars2svg_logger')
        self.handler = _CountingHandler_()
        self.logger.addHandler(self.handler)
        for _f_ in self.logger.filters:
            if type(_f_).__name__ == 'OnceFilter':
                _f_.seen_messages.clear()

    def tearDown(self):
        self.logger.removeHandler(self.handler)

    def _inertWarnings(self):
        return [m for m in self.handler.records if _INERT_MSG_FRAGMENT_ in m]

    def test_float_dot_size_warns_once(self):
        # Two builds -> exactly one warning (once-per-process via OnceFilter).
        self.p2s.xyp(self.df, 'x', 'y', dot_size=3.0, dot_size_supersample=2)
        self.p2s.xyp(self.df, 'x', 'y', dot_size=3.0, dot_size_supersample=2)
        self.assertEqual(len(self._inertWarnings()), 1)

    def test_field_dot_size_warns(self):
        self.p2s.xyp(self.df, 'x', 'y', dot_size='x', dot_size_supersample=2)
        self.assertEqual(len(self._inertWarnings()), 1)

    def test_int_dot_size_does_not_warn(self):
        self.p2s.xyp(self.df, 'x', 'y', dot_size=4, dot_size_supersample=2)
        self.assertEqual(self._inertWarnings(), [])

    def test_ss1_does_not_warn_on_float(self):
        self.p2s.xyp(self.df, 'x', 'y', dot_size=3.0, dot_size_supersample=1)
        self.assertEqual(self._inertWarnings(), [])

    def test_float_geometry_unchanged_by_supersample(self):
        # The float path ignores the factor -- same dot coordinates with and without it.
        a = self.p2s.xyp(self.df, 'x', 'y', dot_size=3.0)
        b = self.p2s.xyp(self.df, 'x', 'y', dot_size=3.0, dot_size_supersample=2)
        self.assertEqual(sorted(a.df_pixels['__xpx__'].to_list()),
                         sorted(b.df_pixels['__xpx__'].to_list()))
        self.assertEqual(sorted(a.df_pixels['__ypx__'].to_list()),
                         sorted(b.df_pixels['__ypx__'].to_list()))


# ---------------------------------------------------------------------------
# Templates, rendering soundness
# ---------------------------------------------------------------------------

class TestSupersampleTemplateAndRender(_SupersampleBase_):

    def test_template_propagates_factor(self):
        tmpl  = self.p2s.xyp(x='x', y='y', dot_size=4, dot_size_supersample=2, wxh=(200, 200))
        clone = self.p2s.xyp(self.df, template=tmpl)
        self.assertEqual(clone.dot_size_supersample, 2)
        # The clone actually renders on the finer grid (fractional or half-step coords).
        step = 4 / 2
        ox   = clone.plot_origin[0]
        for sx, _ in _dot_rects_(clone.svg):
            self.assertAlmostEqual((float(sx) - ox) % step, 0.0, places=6)

    def test_renders_valid_svg_no_control_chars(self):
        xyp = self.p2s.xyp(self.df, 'x', 'y', dot_size=3, dot_size_supersample=2)
        self.assertIn('<svg', xyp.svg)
        self.assertFalse(any(ord(c) < 0x20 and c not in '\t\n\r' for c in xyp.svg),
                         'control character leaked into SVG')

    def test_dots_stay_within_plot_box(self):
        xyp = self.p2s.xyp(self.df, 'x', 'y', dot_size=4, dot_size_supersample=2, wxh=(200, 200))
        f   = xyp.df_flat
        x0  = xyp.plot_origin[0]
        x1  = x0 + xyp.plot_size[0]
        y1  = xyp.plot_origin[1]
        y0  = y1 - xyp.plot_size[1]
        for i in range(f.shape[0]):
            sx, sy = f['__xpx__'][i], f['__ypx__'][i]
            self.assertGreaterEqual(sx, x0)
            self.assertLessEqual(sx, x1)
            self.assertGreaterEqual(sy, y0)
            self.assertLessEqual(sy, y1)


if __name__ == '__main__':
    unittest.main()
