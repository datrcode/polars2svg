import unittest
import re

import polars as pl

from polars2svg import Polars2SVG
from svg_test_utils import assert_valid_svg


# roundSvgFloats() is disabled (see the TODO on Polars2SVG.roundSvgFloats): its regex
# matched digit-dot-digit runs anywhere in the finished SVG, including inside
# <text>/<tspan> label content, silently corrupting labels that merely looked like a
# float (e.g. node label "1.172.32.1" -> "1.17.32.1"). Re-enable these once trimming is
# restricted to attribute-value floats.
_ROUNDING_DISABLED_REASON_ = 'roundSvgFloats() disabled pending a text-content-safe rewrite'

# a decimal number embedded in an SVG string (matches the production helper's regex)
_NUM_RE_ = re.compile(r'-?\d*\.\d+')


def _max_frac_digits(svg):
    '''Longest fractional-digit run found in svg (0 if there are no decimals).'''
    _tails_ = [len(m.split('.', 1)[1]) for m in _NUM_RE_.findall(svg)]
    return max(_tails_) if _tails_ else 0


class TestRoundSvgFloatsHelper(unittest.TestCase):
    '''Unit tests for Polars2SVG.roundSvgFloats -- the SVG float-precision trimmer.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _r(self, s, **kw):
        # wrap in an attribute so the helper sees a realistic context
        return self.p2s.roundSvgFloats(f'v="{s}"', **kw)

    @unittest.skip(_ROUNDING_DISABLED_REASON_)
    def test_trims_long_tail(self):
        self.assertEqual(self._r('123.456789'), 'v="123.46"')

    @unittest.skip(_ROUNDING_DISABLED_REASON_)
    def test_rounds_up(self):
        self.assertEqual(self._r('0.999'), 'v="1"')
        self.assertEqual(self._r('12.348'), 'v="12.35"')

    def test_short_numbers_untouched(self):
        # already <= 2 fractional digits: byte-for-byte identical (no trailing-zero churn)
        for s in ('1.0', '0.5', '12.34', '3.14', '0.07'):
            self.assertEqual(self._r(s), f'v="{s}"')

    def test_integers_untouched(self):
        for s in ('0', '5', '256', '-40', '1000000'):
            self.assertEqual(self._r(s), f'v="{s}"')

    @unittest.skip(_ROUNDING_DISABLED_REASON_)
    def test_negative_numbers(self):
        self.assertEqual(self._r('-3.14159'), 'v="-3.14"')
        self.assertEqual(self._r('-0.129'), 'v="-0.13"')

    @unittest.skip(_ROUNDING_DISABLED_REASON_)
    def test_collapses_to_zero(self):
        # rounds below the precision threshold -> plain "0" (no "-0" / "0.00")
        self.assertEqual(self._r('0.001'), 'v="0"')
        self.assertEqual(self._r('-0.001'), 'v="0"')
        self.assertEqual(self._r('-0.004'), 'v="0"')

    @unittest.skip(_ROUNDING_DISABLED_REASON_)
    def test_strips_trailing_zeros(self):
        # 1.30something -> 1.3, not 1.30
        self.assertEqual(self._r('1.2999'), 'v="1.3"')
        self.assertEqual(self._r('2.0009'), 'v="2"')

    def test_hex_colors_untouched(self):
        for s in ('#aabbcc', '#ff000080', '#abc', '#123456'):
            self.assertEqual(self._r(s), f'v="{s}"')

    def test_identifiers_untouched(self):
        # ids / class names carry digits but no decimal point
        for s in ('xyp_12345', 'plotClip-999', 'rect-group-42'):
            self.assertEqual(self._r(s), f'v="{s}"')

    @unittest.skip(_ROUNDING_DISABLED_REASON_)
    def test_digits_parameter(self):
        self.assertEqual(self._r('1.23456', digits=3), 'v="1.235"')
        self.assertEqual(self._r('1.23456', digits=0), 'v="1"')

    def test_empty_and_none(self):
        self.assertEqual(self.p2s.roundSvgFloats(''), '')
        self.assertIsNone(self.p2s.roundSvgFloats(None))

    def test_idempotent(self):
        s = '<circle cx="12.34567" cy="0.00019" r="3.14159" />'
        once = self.p2s.roundSvgFloats(s)
        twice = self.p2s.roundSvgFloats(once)
        self.assertEqual(once, twice)

    @unittest.skip(_ROUNDING_DISABLED_REASON_)
    def test_multiple_numbers_in_one_string(self):
        s = '<line x1="0.111" y1="2.999" x2="100.0" y2="3.14" />'
        self.assertEqual(self.p2s.roundSvgFloats(s),
                         '<line x1="0.11" y1="3" x2="100.0" y2="3.14" />')

    @unittest.skip(_ROUNDING_DISABLED_REASON_)
    def test_reduces_length(self):
        s = '<rect x="12.3456789" y="98.7654321" width="10.111111" height="20.222222" />'
        self.assertLess(len(self.p2s.roundSvgFloats(s)), len(s))


class TestComponentFloatPrecision(unittest.TestCase):
    '''Every component's finished SVG must carry no float with more than the
    default 2 fractional digits, must stay valid SVG, and rendering must be
    idempotent under a second rounding pass.'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def _renderers(self):
        p2s = self.p2s
        # xyp / histop / piep / timep
        _df_xy_ = pl.DataFrame({'x': [1.37, 2.91, 3.14, 4.6, 5.5, 6.28],
                                'y': [3.33, 1.1, 4.77, 1.9, 5.2, 9.81]})
        _df_cat_ = pl.DataFrame({'cat': ['a', 'b', 'a', 'c', 'b', 'a', 'c', 'b'],
                                 'val': [1.5, 2.7, 3.9, 4.1, 5.3, 6.6, 7.2, 8.8]})
        _df_ts_ = pl.DataFrame({'ts': ['2021-01-03', '2021-02-11', '2021-02-27',
                                       '2021-05-19', '2021-08-01', '2021-11-30']}
                               ).with_columns(pl.col('ts').str.to_datetime())
        # graph components
        _df_g_ = pl.DataFrame({'fm': ['a', 'b', 'c', 'a', 'd', 'b', 'c', 'a'],
                               'to': ['b', 'a', 'a', 'c', 'a', 'c', 'b', 'd']})
        _df_sl_ = pl.DataFrame({'fm':   ['a', 'b', 'c', 'a', 'd', 'b', 'c', 'a'],
                                'to':   ['b', 'a', 'a', 'c', 'a', 'c', 'b', 'd'],
                                'time': [1, 1, 1, 2, 2, 2, 3, 3]})
        return {
            'xyp':     lambda: p2s.xyp(_df_xy_, 'x', 'y', wxh=(200, 200)),
            'histop':  lambda: p2s.histop(_df_cat_, 'cat', count='val', wxh=(200, 200)),
            'piep':    lambda: p2s.piep(_df_cat_, 'cat', wxh=(200, 200)),
            'timep':   lambda: p2s.timep(_df_ts_, ('ts', p2s.LT_Y_mp), wxh=(256, 128)),
            'linkp':   lambda: p2s.linkp(_df_g_, [('fm', 'to')], wxh=(300, 300)),
            'chordp':  lambda: p2s.chordp(_df_g_, [('fm', 'to')], wxh=(300, 300)),
            'spreadlinesp': lambda: p2s.spreadlinesp(_df_sl_, [('fm', 'to')],
                                                     ego='a', time='time', wxh=(500, 260)),
            'smallp':  lambda: p2s.smallp(_df_cat_, 'cat',
                                          p2s.xyp(_df_cat_, 'val', 'val'), wxh=(300, 300)),
        }

    @unittest.skip(_ROUNDING_DISABLED_REASON_)
    def test_max_two_fractional_digits(self):
        for _name_, _fn_ in self._renderers().items():
            with self.subTest(component=_name_):
                _svg_ = _fn_().svg
                assert_valid_svg(self, _svg_)
                self.assertLessEqual(
                    _max_frac_digits(_svg_), 2,
                    f'{_name_}: SVG still contains a float with >2 fractional digits')

    def test_rounding_is_idempotent_on_output(self):
        for _name_, _fn_ in self._renderers().items():
            with self.subTest(component=_name_):
                _svg_ = _fn_().svg
                self.assertEqual(self.p2s.roundSvgFloats(_svg_), _svg_,
                                 f'{_name_}: output not already at target precision')

    def test_no_control_char_leak(self):
        # the trimmer must not introduce the multi-field separator or other junk
        for _name_, _fn_ in self._renderers().items():
            with self.subTest(component=_name_):
                self.assertNotIn('\x1f', _fn_().svg)

    @unittest.skip(_ROUNDING_DISABLED_REASON_)
    def test_linkp_interactive_rerender_is_rounded(self):
        # linkp rounds inside __renderSVG__ so the invalidate/re-render path is covered
        _df_g_ = pl.DataFrame({'fm': ['a', 'b', 'c', 'a'], 'to': ['b', 'c', 'a', 'd']})
        _ln_ = self.p2s.linkp(_df_g_, [('fm', 'to')], wxh=(300, 300))
        _ln_.invalidateRender()
        _svg_ = _ln_.renderSVG()
        self.assertLessEqual(_max_frac_digits(_svg_), 2)


if __name__ == '__main__':
    unittest.main()
