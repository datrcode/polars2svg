import contextlib
import io
import unittest

import polars as pl

from polars2svg import Polars2SVG
from polars2svg.p2s_help import documentedParameters, parameterSection
from polars2svg.xyp import XYp
from polars2svg.histop import Histop
from polars2svg.timep import Timep
from polars2svg.piep import Piep
from polars2svg.linkp import LinkP
from polars2svg.smallp import Smallp
from polars2svg.spreadlinesp import SpreadLinesP
from polars2svg.tile import Tile

# PLANNING.md §7 F16 (DT, 2026-10-03): p2s.help('xyp', 'x_distributions') prints just that
# parameter's documentation, sliced out of the component docstring.

_DOC_ = '''
    widget(df, ...)

    An intro.

    size = 1                      # default
         = 'field'

    Prose about size.

    === %< === %< ===

    Render Options

    wxh        = (w, h)
    tall       = True | False     # first line
                                  # a second comment line
    alpha      = 1.0
    beta       = 2.0
    alpha      = 0.5              # alpha again, documented twice
'''


class TestParameterSection(unittest.TestCase):

    def test_a_topic_chunk_comes_back_whole_with_its_prose(self):
        _s_ = parameterSection(_DOC_, 'size')
        self.assertIn("= 'field'", _s_)
        self.assertIn('Prose about size.', _s_)
        self.assertIn('An intro.', _s_)
        self.assertNotIn('wxh', _s_)

    def test_a_list_chunk_gives_only_the_parameters_block(self):
        _s_ = parameterSection(_DOC_, 'tall')
        self.assertEqual(_s_.splitlines()[0].split()[0], 'tall')
        self.assertIn('a second comment line', _s_)
        self.assertNotIn('wxh', _s_)
        self.assertNotIn('alpha', _s_)

    def test_a_parameter_documented_twice_returns_both(self):
        _s_ = parameterSection(_DOC_, 'alpha')
        self.assertIn('= 1.0', _s_)
        self.assertIn('documented twice', _s_)
        self.assertNotIn('beta', _s_)

    def test_an_undocumented_parameter_is_none(self):
        self.assertIsNone(parameterSection(_DOC_, 'gamma'))
        self.assertEqual(documentedParameters(_DOC_), {'size', 'wxh', 'tall', 'alpha', 'beta'})


class TestHelpOnTheRealDocstrings(unittest.TestCase):

    def setUp(self):
        self.p2s = Polars2SVG()

    def test_x_distributions_is_its_section_not_the_docstring(self):
        _t_ = self.p2s.helpText('xyp', 'x_distributions')
        self.assertTrue(_t_.startswith('xyp(x_distributions=...)'))
        self.assertIn('DISTRIBUTION_INSIDEp', _t_)
        self.assertNotIn('dot_size_range', _t_)
        self.assertLess(len(_t_), len(Polars2SVG.xyp.__doc__) / 4)

    def test_a_render_option_is_its_own_block(self):
        _t_ = self.p2s.helpText('xyp', 'aspect')
        self.assertIn("'geo'", _t_)
        self.assertIn('plate carree', _t_)          # the indented explanation under it
        self.assertNotIn('background_fill', _t_)
        self.assertNotIn('draw_border', _t_)

    def test_help_prints_what_helptext_returns(self):
        _out_ = io.StringIO()
        with contextlib.redirect_stdout(_out_):
            self.p2s.help('histop', 'count')
        self.assertEqual(_out_.getvalue(), self.p2s.helpText('histop', 'count') + '\n')

    def test_no_parameter_is_the_whole_docstring(self):
        _t_ = self.p2s.helpText('piep')
        self.assertIn('piep(', _t_)
        self.assertGreater(len(_t_), 1000)

    def test_a_rendered_component_works_as_its_name(self):
        _plot_ = self.p2s.histop(pl.DataFrame({'a': ['x', 'y', 'x']}), 'a')
        self.assertEqual(self.p2s.helpText(_plot_, 'count'), self.p2s.helpText('histop', 'count'))

    def test_a_misspelled_parameter_suggests_the_real_one(self):
        with self.assertRaisesRegex(ValueError, r"did you mean 'x_distributions'"):
            self.p2s.helpText('xyp', 'x_distribution')

    def test_a_misspelled_component_suggests_the_real_one(self):
        with self.assertRaisesRegex(ValueError, r"no component 'xpy' -- did you mean 'xyp'"):
            self.p2s.helpText('xpy', 'wxh')

    def test_something_that_is_not_a_component_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'is not a component'):
            self.p2s.helpText(pl.DataFrame(), 'wxh')


class TestEveryParameterIsDocumented(unittest.TestCase):
    '''A ratchet, not a rule: every public keyword a component accepts should have a
    definition line its help() can find.  These did not on 2026-10-03.  A NEW name
    missing documentation fails; a name documented since must come off the list, so
    the list only shrinks.  `df` (the positional frame) and `_private_` keywords are
    exempt.  chordp is checked only where its optional dependencies import.'''

    _UNDOCUMENTED_ = {
        'xyp':          {'dot_size_supersample', 'x_time_expand_perc'},
        'histop':       set(),
        'timep':        {'x_time_expand_perc'},
        'piep':         {'count_range'},
        'linkp':        {'color_stat_range_shared', 'count_range_shared', 'flowmap_backend',
                         'flowmap_iterations', 'flowmap_max_flows', 'flowmap_samples',
                         'flowmap_time_budget', 'label_ellipsis', 'label_line_width',
                         'label_max_lines'},
        'chordp':       {'bounds_percent', 'color_stat_range_shared', 'count_range_shared',
                         'pos', 'render_skeleton'},
        'smallp':       set(),
        'spreadlinesp': {'alter_inter_d', 'alter_separation_h', 'anno', 'channel_inter_d',
                         'circle_inter_d', 'circle_spacer', 'color_stat_range_shared',
                         'count_range_shared', 'h_collapsed_sections', 'highlight_nodes',
                         'max_bin_h', 'max_bin_w', 'max_channel_w', 'max_rings', 'min_channel_w',
                         'node_labels', 'r_min', 'r_pref', 'sm_shared', 'time_order', 'x_ins',
                         'y_ins'},
        'tile':         set(),
    }

    def _classes_(self):
        _out_ = {'xyp': XYp, 'histop': Histop, 'timep': Timep, 'piep': Piep, 'linkp': LinkP,
                 'smallp': Smallp, 'spreadlinesp': SpreadLinesP, 'tile': Tile}
        try:
            from polars2svg.chordp import ChP
            _out_['chordp'] = ChP
        except ImportError:
            pass
        return _out_

    def test_the_undocumented_list_is_exact(self):
        self.assertEqual(set(self._UNDOCUMENTED_), set(Polars2SVG._HELP_COMPONENTS_))
        for _name_, _cls_ in self._classes_().items():
            with self.subTest(component=_name_):
                _public_ = {_k_ for _k_ in getattr(_cls_, '_VALID_KWARGS', frozenset())
                            if _k_ != 'df' and not (_k_.startswith('_') and _k_.endswith('_'))}
                _missing_ = _public_ - documentedParameters(Polars2SVG._helpDoc_(Polars2SVG(), _name_))
                self.assertEqual(
                    _missing_, self._UNDOCUMENTED_[_name_],
                    f'{_name_}: newly undocumented {sorted(_missing_ - self._UNDOCUMENTED_[_name_])}; '
                    f'documented now, remove from the list: '
                    f'{sorted(self._UNDOCUMENTED_[_name_] - _missing_)}.  Document a parameter '
                    f"with a `name = form   # comment` line in Polars2SVG.{_name_}'s docstring.")


if __name__ == '__main__':
    unittest.main()
