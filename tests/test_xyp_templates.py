import unittest
from polars2svg import Polars2SVG
from random_dataframe import randomDataFrame
from svg_test_utils import normalize_svg


class Testxyp_templates(unittest.TestCase):
    '''An xyp built without a frame is a template: whatever it is later drawn on, it draws
    what a direct call with the same settings would.  (This test was a placeholder that
    printed "FILL IN" three times and asserted nothing -- PLANNING.md V11.)'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()

    def test_basicTemplate(self):
        df = randomDataFrame(200, seed=31)
        for _kw_ in ({'x': 'a', 'y': 'b'},
                     {'x': 'a', 'y': 'c', 'color': 'j', 'dot_size': 4, 'wxh': (200, 150)},
                     {'x': 'g', 'y': 'a', 'x_distributions': self.p2s.ROW_COUNTp},
                     {'x': 'j', 'y': 'k', 'color': self.p2s.CROW_MAGNITUDEp}):
            with self.subTest(**{k: str(v) for k, v in _kw_.items()}):
                _direct_   = normalize_svg(self.p2s.xyp(df, **_kw_).svg)
                _template_ = self.p2s.xyp(**_kw_)
                self.assertIsNone(_template_.df)
                self.assertEqual(normalize_svg(_template_.render_with(df).svg), _direct_, 'render_with() on a template')
                self.assertEqual(normalize_svg(self.p2s.xyp(df=df, template=_template_).svg), _direct_, 'template=')
                # ... and a view already drawn, redrawn on other rows, is a fresh call on those rows
                self.assertEqual(normalize_svg(self.p2s.xyp(df, **_kw_).render_with(df.head(50)).svg),
                                 normalize_svg(self.p2s.xyp(df.head(50), **_kw_).svg), 'render_with() on a drawn view')

if __name__ == '__main__':
    unittest.main()
