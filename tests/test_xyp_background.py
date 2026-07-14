import unittest
import polars as pl
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
from polars2svg import Polars2SVG

from random_dataframe import randomDataFrame

class Testxyp_background(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p2s = Polars2SVG()
        self.df  = pl.DataFrame({'x': [1,2,3,4,5], 'y': [2,4,1,3,5]})
        self.bg  = {'region_a': Polygon([(1,1),(3,1),(3,4),(1,4)]),
                    'region_b': Polygon([(3,2),(5,2),(5,5),(3,5)])}

    # -------------------------------------------------------------------------
    # Exception: non-scalar axes
    # -------------------------------------------------------------------------

    def test_exception_both_axes_non_numeric(self):
        df = randomDataFrame(50)
        with self.assertRaises(ValueError):
            self.p2s.xyp(df=df, x='j', y='k', background=self.bg)

    def test_exception_x_non_numeric(self):
        df = randomDataFrame(50)
        with self.assertRaises(ValueError):
            self.p2s.xyp(df=df, x='j', y='a', background=self.bg)

    def test_exception_y_non_numeric(self):
        df = randomDataFrame(50)
        with self.assertRaises(ValueError):
            self.p2s.xyp(df=df, x='a', y='k', background=self.bg)

    # -------------------------------------------------------------------------
    # background=None produces empty svg_background
    # -------------------------------------------------------------------------

    def test_no_background(self):
        chart = self.p2s.xyp(df=self.df, x='x', y='y')
        self.assertEqual(chart.svg_background, '')
        self.assertNotIn('fill-opacity', chart.svg)

    # -------------------------------------------------------------------------
    # Shapely Polygon
    # -------------------------------------------------------------------------

    def test_polygon_vary_fill_and_labels(self):
        chart = self.p2s.xyp(df=self.df, x='x', y='y',
                             background=self.bg,
                             background_fill='vary',
                             background_opacity=0.3,
                             background_label_color='vary',
                             background_stroke='default')
        self.assertIn('<path', chart.svg)
        self.assertIn('fill-opacity="0.3"', chart.svg)
        self.assertIn('region_a', chart.svg)
        self.assertIn('region_b', chart.svg)

    def test_polygon_fixed_fill_color(self):
        chart = self.p2s.xyp(df=self.df, x='x', y='y',
                             background={'box': Polygon([(1,1),(4,1),(4,4),(1,4)])},
                             background_fill='#ff0000',
                             background_opacity=0.5)
        self.assertIn('fill="#ff0000"', chart.svg)
        self.assertIn('fill-opacity="0.5"', chart.svg)

    def test_polygon_per_shape_fill_dict(self):
        chart = self.p2s.xyp(df=self.df, x='x', y='y',
                             background=self.bg,
                             background_fill={'region_a': '#aabbcc', 'region_b': '#ddeeff'},
                             background_opacity=1.0)
        self.assertIn('fill="#aabbcc"', chart.svg)
        self.assertIn('fill="#ddeeff"', chart.svg)

    def test_polygon_no_fill(self):
        chart = self.p2s.xyp(df=self.df, x='x', y='y',
                             background=self.bg,
                             background_fill=None)
        self.assertIn('fill-opacity="0.0"', chart.svg)

    # -------------------------------------------------------------------------
    # MultiPolygon
    # -------------------------------------------------------------------------

    def test_multipolygon(self):
        mp = MultiPolygon([Polygon([(0,0),(2,0),(2,2),(0,2)]),
                           Polygon([(3,3),(5,3),(5,5),(3,5)])])
        chart = self.p2s.xyp(df=self.df, x='x', y='y',
                             background={'mp': mp},
                             background_fill='#123456',
                             background_opacity=0.4)
        self.assertIn('<path', chart.svg)
        self.assertIn('fill="#123456"', chart.svg)

    # -------------------------------------------------------------------------
    # LineString / MultiLineString (fill forced to 'none')
    # -------------------------------------------------------------------------

    def test_linestring(self):
        chart = self.p2s.xyp(df=self.df, x='x', y='y',
                             background={'line': LineString([(1,1),(3,3),(5,1)])},
                             background_stroke='default')
        self.assertIn('<path', chart.svg)

    def test_multilinestring(self):
        from shapely.geometry import MultiLineString
        mls = MultiLineString([[(1,1),(3,3)], [(2,4),(4,2)]])
        chart = self.p2s.xyp(df=self.df, x='x', y='y',
                             background={'mls': mls},
                             background_stroke='default')
        self.assertIn('<path', chart.svg)

    # -------------------------------------------------------------------------
    # Points list
    # -------------------------------------------------------------------------

    def test_points_list(self):
        chart = self.p2s.xyp(df=self.df, x='x', y='y',
                             background={'box': [(0,0),(2,0),(2,2),(0,2)]},
                             background_fill='#aabbcc',
                             background_opacity=0.5)
        self.assertIn('fill-opacity="0.5"', chart.svg)
        self.assertIn('fill="#aabbcc"', chart.svg)

    # -------------------------------------------------------------------------
    # SVG circle string → ellipse
    # -------------------------------------------------------------------------

    def test_circle_svg_string(self):
        chart = self.p2s.xyp(df=self.df, x='x', y='y',
                             background={'circ': '<circle cx="3" cy="3" r="1" />'},
                             background_fill='#ff0000',
                             background_opacity=0.4)
        self.assertIn('<ellipse', chart.svg)

    # -------------------------------------------------------------------------
    # SVG path description string
    # -------------------------------------------------------------------------

    def test_svg_path_string(self):
        chart = self.p2s.xyp(df=self.df, x='x', y='y',
                             background={'tri': 'M 1 1 L 3 1 L 2 4 Z'},
                             background_fill='#00ff00',
                             background_opacity=0.6)
        self.assertIn('<path', chart.svg)
        self.assertIn('fill="#00ff00"', chart.svg)

    # -------------------------------------------------------------------------
    # Stroke options
    # -------------------------------------------------------------------------

    def test_stroke_vary(self):
        chart = self.p2s.xyp(df=self.df, x='x', y='y',
                             background=self.bg,
                             background_stroke='vary',
                             background_stroke_w=2.0)
        self.assertIn('stroke-width="2.0"', chart.svg)

    def test_stroke_fixed_color(self):
        chart = self.p2s.xyp(df=self.df, x='x', y='y',
                             background=self.bg,
                             background_stroke='#123456',
                             background_stroke_w=1.5)
        self.assertIn('stroke="#123456"', chart.svg)
        self.assertIn('stroke-width="1.5"', chart.svg)

    def test_no_stroke(self):
        chart = self.p2s.xyp(df=self.df, x='x', y='y',
                             background=self.bg,
                             background_stroke=None)
        self.assertNotIn('stroke-width', chart.svg_background)

    # -------------------------------------------------------------------------
    # Label options
    # -------------------------------------------------------------------------

    def test_label_fixed_color(self):
        chart = self.p2s.xyp(df=self.df, x='x', y='y',
                             background=self.bg,
                             background_label_color='#ff00ff')
        self.assertIn('<text', chart.svg)
        self.assertIn('fill="#ff00ff"', chart.svg)

    def test_no_label(self):
        chart = self.p2s.xyp(df=self.df, x='x', y='y',
                             background=self.bg,
                             background_label_color=None)
        self.assertNotIn('<text', chart.svg_background)

    # -------------------------------------------------------------------------
    # Layering: background behind context and data
    # -------------------------------------------------------------------------

    def test_background_precedes_context_in_svg(self):
        chart = self.p2s.xyp(df=self.df, x='x', y='y',
                             background=self.bg,
                             background_fill='vary',
                             background_opacity=0.3)
        bg_pos   = chart.svg.find(chart.svg_background[:30])
        ctx_pos  = chart.svg.find(chart.svg_context[:30]) if chart.svg_context else len(chart.svg)
        self.assertLess(bg_pos, ctx_pos)

if __name__ == '__main__':
    unittest.main()
