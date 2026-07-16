import unittest

import polars as pl

from polars2svg import Polars2SVG
from label_fidelity_data import (
    FIDELITY_LABELS, KNOWN_ROUNDING_VICTIMS,
    faulty_round_svg_floats, svg_text_contents,
)

# Plots are sized generously so the column-name context label never hits cropText() --
# the longest fidelity string (a full IPv6 address, ~39 chars) fits with room to spare.
_WXH = (900, 600)


class TestContextLabelFidelity(unittest.TestCase):
    '''draw_context axis/field labels name a DataFrame column. Whatever the column is
    called must render true to form -- the same string-fidelity contract as linkp node
    labels, since roundSvgFloats() corrupted any digit-dot-digit run in <text> content
    regardless of which component emitted it.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    # -- renderers keyed by name; each puts the label-under-test in a column name that
    #    surfaces as a draw_context label, and returns the finished SVG. -------------
    def _xyp_x(self, col):
        df = pl.DataFrame({col: [1.0, 2.0, 3.0], 'value': [4.0, 5.0, 6.0]})
        return self.p2s.xyp(df, col, 'value', wxh=_WXH).svg

    def _xyp_y(self, col):
        df = pl.DataFrame({'key': [1.0, 2.0, 3.0], col: [4.0, 5.0, 6.0]})
        return self.p2s.xyp(df, 'key', col, wxh=_WXH).svg

    def _histop_bin(self, col):
        df = pl.DataFrame({col: ['a', 'b', 'a', 'c'], 'value': [1.0, 2.0, 3.0, 4.0]})
        return self.p2s.histop(df, col, wxh=_WXH).svg

    def _histop_count(self, col):
        df = pl.DataFrame({'cat': ['a', 'b', 'a', 'c'], col: [1.0, 2.0, 3.0, 4.0]})
        return self.p2s.histop(df, 'cat', count=col, wxh=_WXH).svg

    def _piep_bin(self, col):
        df = pl.DataFrame({col: ['a', 'b', 'a', 'c'], 'value': [1.0, 2.0, 3.0, 4.0]})
        return self.p2s.piep(df, col, wxh=_WXH).svg

    def _cases(self):
        return {
            'xyp/x-axis':      self._xyp_x,
            'xyp/y-axis':      self._xyp_y,
            'histop/bin':      self._histop_bin,
            'histop/count':    self._histop_count,
            'piep/bin':        self._piep_bin,
        }

    # ------------------------------------------------------------------
    # Positive: every column name appears verbatim in its context label.
    # ------------------------------------------------------------------
    def test_context_labels_render_true_to_form(self):
        for _case_, _render_ in self._cases().items():
            for _col_ in FIDELITY_LABELS:
                with self.subTest(case=_case_, column=_col_):
                    _texts_ = svg_text_contents(_render_(_col_))
                    self.assertIn(_col_, _texts_,
                                  f'{_case_}: column name {_col_!r} missing/corrupted '
                                  f'in draw_context label')

    def test_context_svg_is_well_formed_xml(self):
        # Escape-required column names (&, <, >, quotes) must keep the SVG parseable.
        for _case_, _render_ in self._cases().items():
            for _col_ in ('a & b', 'x < y', 'p > q', '<tag>', 'a"quoted"b'):
                with self.subTest(case=_case_, column=_col_):
                    svg_text_contents(_render_(_col_))  # parses or raises

    # ------------------------------------------------------------------
    # Negative: re-applying the disabled rounding corrupts context labels too.
    # ------------------------------------------------------------------
    def test_faulty_rounding_would_corrupt_context_labels(self):
        for _case_, _render_ in self._cases().items():
            _svg_ = _render_('1.172.32.1')
            self.assertIn('1.172.32.1', svg_text_contents(_svg_))  # correct as rendered
            _after_ = svg_text_contents(faulty_round_svg_floats(_svg_))
            with self.subTest(case=_case_):
                self.assertNotIn('1.172.32.1', _after_)
                self.assertIn('1.17.32.1', _after_)

        # And every known-vulnerable label breaks on the x-axis path.
        for _victim_ in KNOWN_ROUNDING_VICTIMS:
            _after_ = svg_text_contents(faulty_round_svg_floats(self._xyp_x(_victim_)))
            self.assertNotIn(_victim_, _after_,
                             f'faulty rounding should have corrupted axis label {_victim_!r}')

    # ------------------------------------------------------------------
    # Guard: production rounding must stay OFF.
    # ------------------------------------------------------------------
    def test_production_rounding_is_disabled(self):
        probe = '<text x="1.172">1.172.32.1</text> more="123.456789"'
        self.assertEqual(self.p2s.roundSvgFloats(probe), probe,
                         'roundSvgFloats() is enabled again -- it corrupts context labels; '
                         'see the TODO on Polars2SVG.roundSvgFloats')


if __name__ == '__main__':
    unittest.main()
