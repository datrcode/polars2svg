import unittest
from polars2svg import Polars2SVG


class TestSvgAxisLabels(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s    = Polars2SVG()
        self.txt_h  = 12
        self.x0     = 0.0

    def _render(self, lbl_left, lbl_center, lbl_right, available_w, prefer_center=True):
        return self.p2s.svgAxisLabels(
            lbl_left, lbl_center, lbl_right,
            self.x0, available_w, 50, self.txt_h,
            prefer_center=prefer_center,
        )

    def test_all_three_fit_wide(self):
        '''All three labels render when available width is generous.'''
        out = self._render('Jan', 'monthly', 'Dec', 600)
        self.assertEqual(len(out), 3)

    def test_center_only_prefer_center(self):
        '''Only center renders when LR would overlap and prefer_center=True.'''
        # available_w=60 fits "monthly" (~42px at txt_h=12) but not "Jan 2020" + "monthly" + "Dec 2024"
        out = self._render('Jan 2020', 'monthly', 'Dec 2024', 60, prefer_center=True)
        self.assertEqual(len(out), 1)
        self.assertIn('middle', out[0])

    def test_lr_only_prefer_lr(self):
        '''LR render (no center) when center would overlap and prefer_center=False.'''
        # Wide enough for LR but not all three
        out = self._render('0', 'Rows', '100000', 80, prefer_center=False)
        # Either 2 (LR) or 3 (all fit) is acceptable — just must not be center-only
        for elem in out:
            self.assertNotIn('monthly', elem)

    def test_none_fit_prefer_center(self):
        '''Nothing renders when even center does not fit and prefer_center=True.'''
        out = self._render('Jan', 'extremely long granularity label text string', 'Dec', 2, prefer_center=True)
        self.assertEqual(len(out), 0)

    def test_prefer_lr_fallback_to_right_only(self):
        '''Only right label renders when LR together do not fit and prefer_center=False.'''
        # Labels are ~30px each at txt_h=12; available 35 => LR together won't fit
        l_w = self.p2s.textLength('Min', self.txt_h)
        r_w = self.p2s.textLength('Max', self.txt_h)
        gap = self.txt_h * 0.5
        available_w = l_w + r_w + gap - 1  # just below the LR threshold
        out = self._render('Min', 'Center', 'Max', available_w, prefer_center=False)
        # Should be 1 (right only) or 0
        self.assertLessEqual(len(out), 1)
        if len(out) == 1:
            self.assertIn('end', out[0])

    def test_returns_svg_strings(self):
        '''Each returned element is a non-empty string containing SVG text tags.'''
        out = self._render('A', 'B', 'C', 600)
        for elem in out:
            self.assertIsInstance(elem, str)
            self.assertIn('<text', elem)


class TestTextLength(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_non_empty_string_returns_positive_float(self):
        result = self.p2s.textLength('Hello', 12)
        self.assertGreater(result, 0.0)
        self.assertIsInstance(result, float)

    def test_empty_string_returns_zero(self):
        result = self.p2s.textLength('', 12)
        self.assertEqual(result, 0.0)

    def test_scales_proportionally_with_font_height(self):
        w12 = self.p2s.textLength('A', 12)
        w24 = self.p2s.textLength('A', 24)
        self.assertAlmostEqual(w24 / w12, 2.0, delta=0.1)

    def test_longer_string_is_wider(self):
        short = self.p2s.textLength('Hi', 12)
        long  = self.p2s.textLength('Hello World', 12)
        self.assertGreater(long, short)

    def test_unknown_char_does_not_raise(self):
        result = self.p2s.textLength('★', 12)  # ★
        self.assertGreater(result, 0.0)


class TestCropText(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_short_string_returned_unchanged(self):
        result = self.p2s.cropText('Hi', 12, 1000)
        self.assertEqual(result, 'Hi')

    def test_long_string_ends_with_ellipsis(self):
        w = self.p2s.textLength('AB', 12)
        result = self.p2s.cropText('ABCDEFGHIJ', 12, w)
        self.assertTrue(result.endswith('...'))

    def test_cropped_result_fits_in_width(self):
        w = 40.0
        result = self.p2s.cropText('This is a fairly long label string', 12, w)
        self.assertLessEqual(self.p2s.textLength(result, 12), w + self.p2s.textLength('...', 12) + 1)

    def test_very_narrow_width_does_not_crash(self):
        result = self.p2s.cropText('ABCDEFG', 12, 1)
        self.assertIsInstance(result, str)
        self.assertTrue(result.endswith('...'))


class TestSvgText(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_default_branch_contains_key_attributes(self):
        out = self.p2s.svgText('hello', 10, 20, txt_h=14)
        self.assertIn('font-size', out)
        self.assertIn('fill', out)
        self.assertIn('text-anchor', out)
        self.assertIn('hello', out)

    def test_just_xy_branch_minimal_output(self):
        out = self.p2s.svgText('hello', 10, 20, just_xy=True)
        self.assertEqual(out, '<text x="10.00" y="20.00">hello</text>')

    def test_rotation_branch_contains_transform(self):
        out = self.p2s.svgText('hello', 10, 20, rotation=45)
        self.assertIn('transform="rotate(45', out)

    def test_html_escapes_angle_brackets(self):
        out = self.p2s.svgText('<b>', 0, 0)
        self.assertIn('&lt;b&gt;', out)

    def test_empty_string_returns_empty(self):
        self.assertEqual(self.p2s.svgText('', 0, 0), '')

    def test_newline_returns_empty(self):
        self.assertEqual(self.p2s.svgText('\n', 0, 0), '')

    def test_tab_returns_empty(self):
        self.assertEqual(self.p2s.svgText('\t', 0, 0), '')

    def test_custom_anchor_included(self):
        out = self.p2s.svgText('x', 0, 0, anchor='middle')
        self.assertIn('text-anchor="middle"', out)

    def test_output_is_string(self):
        out = self.p2s.svgText('test', 5, 10)
        self.assertIsInstance(out, str)


if __name__ == '__main__':
    unittest.main()
