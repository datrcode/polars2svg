import unittest
import polars as pl
from polars2svg import Polars2SVG

class TestXYPMultisetColor(unittest.TestCase):
    def test_multiset_color(self):
        p2s = Polars2SVG()
        # Two rows at identical coordinates with different string colors → triggers CSETp multiset
        df = pl.DataFrame({'x': [0.0, 0.0], 'y': [0.0, 0.0], 'c': ['red', 'blue']})
        xyp = p2s.xyp(df, 'x', 'y', color='c')
        pixels = xyp.df_pixels
        self.assertEqual(len(pixels), 1, f"Expected one merged pixel row, got {len(pixels)}")
        color_set  = pixels['__color_set__'][0].to_list()
        hex_color  = pixels['__hexcolor__'][0]
        set_element = pixels['__set_element__'][0]
        print(f"color_set: {sorted(color_set)}")
        print(f"set_element (multiset sentinel): {set_element}")
        print(f"Multiset color: {hex_color}")
        self.assertEqual(sorted(color_set), ['blue', 'red'])
        self.assertEqual(set_element, '-1')
        self.assertEqual(hex_color, '#7f8367')

if __name__ == '__main__':
    unittest.main()
