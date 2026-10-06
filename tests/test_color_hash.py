'''The version-stable colour hash: colorHashKey() and colorHashes() (PLANNING.md
C-polars2-color-hash).

pl.Expr.hash() is documented as unstable across polars releases, and 2.0.0rc2 proved it:
every categorical colour moved.  Its replacement hashes a canonical rendering of each
value with blake2b, so the colours depend on neither polars nor the column's dtype.

The reference digests below are the cross-version anchor.  If one changes, every chart
that colours by that kind of value changes colour -- that is a recolour, and it needs
the same deliberate decision as the one that introduced this hash.
'''
import datetime as dt
import decimal
import hashlib
import random
import unittest
import zoneinfo
from unittest import mock

import polars as pl

from polars2svg import Polars2SVG, p2s_colors_mixin
from polars2svg.p2s_colors_mixin import colorHashes, colorHashKey, isHexColor


def _one_(value, dtype=None) -> int:
    return int(colorHashes(pl.Series([value], dtype=dtype))[0])


class TestReferenceDigests(unittest.TestCase):
    # (value, dtype) -> the 32-bit hash it must keep forever
    _REFERENCE_ = [
        ('abc',                             None,                              0xe9dd8e5f),
        ('',                                None,                              0x44bdadb9),
        ('héllo',                           None,                              0xf3e492e6),
        ('10',                              None,                              0xe31940e3),
        (10,                                pl.Int64,                          0xed0aad02),
        (-1,                                pl.Int64,                          0x2ad4a58c),
        (2**64 - 1,                         pl.UInt64,                         0x0ff45d30),
        (1.5,                               pl.Float64,                        0xd3b7334f),
        (-1.0,                              pl.Float64,                        0xd806b373),
        (0.0,                               pl.Float64,                        0x832f0894),
        (float('nan'),                      pl.Float64,                        0xb4f99591),
        (True,                              pl.Boolean,                        0xa8016025),
        (dt.date(2026, 10, 5),              pl.Date,                           0x9ada800d),
        (dt.datetime(2026, 10, 5, 12, 30),  pl.Datetime('us'),                 0x2f8ab8fc),
        (dt.time(12, 30),                   pl.Time,                           0xc91a350c),
        (dt.timedelta(days=1, seconds=2),   pl.Duration('us'),                 0x1e7e8d84),
        (decimal.Decimal('1.50'),           pl.Decimal(10, 2),                 0xa7983db0),
        (b'\x00\xff',                       pl.Binary,                         0x1f2cb921),
        (['tbp'],                           pl.List(pl.String),                0x716a2b94),
        ({'a': 1},                          pl.Struct({'a': pl.Int64}),        0x1fe2eee0),
        (None,                              pl.String,                         0xeb66b2d7),
    ]

    def test_reference_digests_are_unchanged(self):
        for _value_, _dtype_, _expected_ in self._REFERENCE_:
            with self.subTest(value=_value_, dtype=_dtype_):
                self.assertEqual(_one_(_value_, _dtype_), _expected_)

    def test_digest_is_the_first_four_blake2b_bytes_little_endian(self):
        # pins the construction itself, independently of colorHashes()
        for _value_, _dtype_, _expected_ in self._REFERENCE_:
            with self.subTest(value=_value_):
                _digest_ = hashlib.blake2b(colorHashKey(_value_), digest_size=4).digest()
                self.assertEqual(int.from_bytes(_digest_, 'little'), _expected_)


class TestKeyFormat(unittest.TestCase):
    def test_keys_are_a_tag_a_nul_and_a_canonical_rendering(self):
        _cases_ = [
            ('abc',                         b'str\x00abc'),
            (10,                            b'int\x0010'),
            (-0.0,                          b'float\x000.0'),
            (True,                          b'bool\x001'),
            (dt.date(2026, 10, 5),          b'date\x002026-10-05'),
            (dt.timedelta(seconds=90),      b'timedelta\x000,90,0'),
            (['a', 1, 2.5],                 b'json\x00["a",1,2.5]'),
            ({'k': None},                   b'json\x00{"k":null}'),
            (None,                          b'null\x00'),
        ]
        for _value_, _expected_ in _cases_:
            with self.subTest(value=_value_):
                self.assertEqual(colorHashKey(_value_), _expected_)

    def test_nested_non_json_scalars_keep_their_tag(self):
        # a date inside a list must not collide with the same text as a string
        self.assertNotEqual(colorHashKey([dt.date(2026, 10, 5)]), colorHashKey(['2026-10-05']))
        self.assertEqual(colorHashKey([-0.0]), colorHashKey([0.0]))

    def test_aware_datetimes_keep_their_offset(self):
        _berlin_ = dt.datetime(2026, 10, 5, 12, 30, tzinfo=zoneinfo.ZoneInfo('Europe/Berlin'))
        self.assertEqual(colorHashKey(_berlin_), b'datetime\x002026-10-05T12:30:00+02:00')


class TestWhatSharesAColour(unittest.TestCase):
    def test_every_integer_width_hashes_alike(self):
        _hashes_ = {_one_(10, _d_) for _d_ in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                               pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)}
        self.assertEqual(_hashes_, {_one_(10, pl.Int64)})

    def test_string_categorical_and_enum_hash_alike(self):
        self.assertEqual(_one_('abc', pl.Categorical), _one_('abc', pl.String))
        self.assertEqual(_one_('abc', pl.Enum(['abc', 'x'])), _one_('abc', pl.String))

    def test_float_widths_hash_alike_for_exact_values(self):
        self.assertEqual(_one_(1.5, pl.Float32), _one_(1.5, pl.Float64))

    def test_type_tag_keeps_look_alikes_apart(self):
        self.assertNotEqual(_one_('10', pl.String),  _one_(10, pl.Int64))
        self.assertNotEqual(_one_(10, pl.Int64),     _one_(10.0, pl.Float64))
        self.assertNotEqual(_one_(True, pl.Boolean), _one_(1, pl.Int64))
        self.assertNotEqual(_one_(None, pl.String),  _one_('', pl.String))

    def test_negative_zero_is_zero(self):
        # unique() keeps whichever of 0.0 / -0.0 it meets first; they must not differ
        self.assertEqual(_one_(-0.0, pl.Float64), _one_(0.0, pl.Float64))
        _a_ = colorHashes(pl.Series([-0.0, 0.0]).unique())
        _b_ = colorHashes(pl.Series([0.0, -0.0]).unique())
        self.assertEqual(_a_.to_list(), _b_.to_list())


class TestSeriesContract(unittest.TestCase):
    def test_same_length_name_and_dtype_with_no_nulls(self):
        _s_ = pl.Series('v', ['a', None, 'b', 'a'])
        _h_ = colorHashes(_s_)
        self.assertEqual((_h_.name, _h_.dtype, len(_h_), _h_.null_count()), ('v', pl.UInt32, 4, 0))
        self.assertEqual(_h_[0], _h_[3])
        self.assertEqual(_h_[1], _one_(None, pl.String))

    def test_empty_series(self):
        _h_ = colorHashes(pl.Series('v', [], dtype=pl.String))
        self.assertEqual((len(_h_), _h_.dtype), (0, pl.UInt32))

    def test_string_fast_path_matches_the_general_encoder(self):
        _rng_   = random.Random(0)
        _chars_ = 'abcXYZ09 .-é漢\x00'
        _vals_  = [''.join(_rng_.choice(_chars_) for _ in range(_rng_.randint(0, 40))) for _ in range(500)] + [None, '']
        _fast_  = colorHashes(pl.Series(_vals_, dtype=pl.String)).to_list()
        _slow_  = [int.from_bytes(hashlib.blake2b(colorHashKey(_v_), digest_size=4).digest(), 'little') for _v_ in _vals_]
        self.assertEqual(_fast_, _slow_)


class TestColorize(unittest.TestCase):
    '''colorizeColumnPolarsOperations(): the hash runs once, over distinct values only.

    Its colour math used to read pl.Expr.hash() directly, and eager polars has no
    common-subexpression elimination: every with_columns evaluated the hash 94 times.
    '''

    def setUp(self):
        self.p2s = Polars2SVG()

    def _colorize_(self, df: pl.DataFrame, **kwargs) -> list:
        return df.with_columns(self.p2s.colorizeColumnPolarsOperations('v', **kwargs).alias('c'))['c'].to_list()

    def test_reference_colours_are_unchanged(self):
        # hash, HSV band and rounding together; a change here recolours every chart
        for _value_, _dtype_, _hex_ in (('abc', pl.String, '#a64b7a'), (10, pl.Int64, '#802e52'),
                                        (None, pl.String, '#d6498d'), (2.5, pl.Float64, '#77bf16')):
            with self.subTest(value=_value_):
                self.assertEqual(self._colorize_(pl.DataFrame({'v': pl.Series([_value_], dtype=_dtype_)})), [_hex_])

    def test_hashes_once_per_call_and_only_distinct_values(self):
        # two chunks: a map_batches UDF still sees the whole column in one call
        _df_ = pl.concat([pl.DataFrame({'v': ['a', 'b', None] * 400}), pl.DataFrame({'v': ['c', 'a'] * 300})], rechunk=False)
        self.assertEqual(_df_['v'].n_chunks(), 2)
        _real_ = p2s_colors_mixin.colorHashes
        with mock.patch.object(p2s_colors_mixin, 'colorHashes', wraps=_real_) as _spy_:
            self._colorize_(_df_)
        self.assertEqual(_spy_.call_count, 1)
        self.assertEqual(len(_spy_.call_args.args[0]), _df_['v'].n_unique())

    def test_each_row_gets_its_own_values_colour_in_order(self):
        _vals_ = ['b', None, 'a', 'b', 'c', None, 'a']
        _alone_ = {_v_: self._colorize_(pl.DataFrame({'v': pl.Series([_v_], dtype=pl.String)}))[0] for _v_ in set(_vals_)}
        self.assertEqual(self._colorize_(pl.DataFrame({'v': _vals_})), [_alone_[_v_] for _v_ in _vals_])

    def test_null_rows_get_a_colour(self):
        _out_ = self._colorize_(pl.DataFrame({'v': ['a', None]}))
        self.assertTrue(all(isHexColor(_c_) for _c_ in _out_), _out_)

    def test_overrides_win_and_match_on_the_string_cast(self):
        self.p2s.setColorOverrides({'a': '#ff0000', '10': '#00ff00'})
        _plain_ = Polars2SVG()
        _base_b_ = _plain_.colorizeColumnPolarsOperations('v')
        self.assertEqual(self._colorize_(pl.DataFrame({'v': ['a', 'b']})),
                         ['#ff0000', pl.DataFrame({'v': ['b']}).select(_base_b_).item()])
        self.assertEqual(self._colorize_(pl.DataFrame({'v': [10]})), ['#00ff00'])
        self.assertNotEqual(self._colorize_(pl.DataFrame({'v': ['a']}), apply_overrides=False), ['#ff0000'])

    def test_works_under_when_then(self):
        # xyp's CSETp path calls it inside pl.when(...).then(...)
        _df_ = pl.DataFrame({'v': ['a', None, 'b'], 'ok': [True, False, True]})
        _out_ = _df_.with_columns(pl.when(pl.col('ok')).then(self.p2s.colorizeColumnPolarsOperations('v'))
                                    .otherwise(pl.lit('#000000')).alias('c'))['c'].to_list()
        self.assertEqual(_out_, [self._colorize_(pl.DataFrame({'v': ['a']}))[0], '#000000',
                                 self._colorize_(pl.DataFrame({'v': ['b']}))[0]])


class TestColorMemo(unittest.TestCase):
    '''color() / colors() memoize per instance (PLANNING.md C-color-memo-numeric-keys).

    The memo used to be keyed by the value, and Python counts -1 == -1.0, 1 == True and
    0 == False == 0.0 as one dict key, so whichever was asked for first fixed the colour
    of the rest on that instance.  colors() also put every miss into one Series, which
    cannot hold 1 and 'a' (or -1 and -1.0) and raised.
    '''
    # Pairs Python treats as one dict key but the colorizer colours apart
    _LOOK_ALIKES_ = [(-1, -1.0), (1, True), (0, False), (0, 0.0), (False, 0.0), ((-1,), (-1.0,))]

    @staticmethod
    def _fresh_(value) -> str:
        return Polars2SVG().color(value)

    def test_look_alikes_really_differ(self):
        # Without this every test below could pass on a pair the colorizer merges anyway
        for _a_, _b_ in self._LOOK_ALIKES_:
            with self.subTest(a=_a_, b=_b_):
                self.assertNotEqual(self._fresh_(_a_), self._fresh_(_b_))

    def test_color_keeps_look_alikes_apart_in_either_order(self):
        for _a_, _b_ in self._LOOK_ALIKES_:
            for _first_, _second_ in ((_a_, _b_), (_b_, _a_)):
                with self.subTest(first=_first_, second=_second_):
                    _p2s_ = Polars2SVG()
                    self.assertEqual(_p2s_.color(_first_),  self._fresh_(_first_))
                    self.assertEqual(_p2s_.color(_second_), self._fresh_(_second_))

    def test_color_matches_the_colorizer(self):
        _p2s_ = Polars2SVG()
        for _value_ in (-1, -1.0, True, 'x', None, dt.date(2026, 10, 6)):
            with self.subTest(value=_value_):
                _hex_ = pl.DataFrame({'v': [_value_]}).select(_p2s_.colorizeColumnPolarsOperations('v', apply_overrides=False)).item()
                self.assertEqual(_p2s_.color(_value_), _hex_)

    def test_colors_takes_a_list_of_mixed_types(self):
        _values_ = [1, 'a', 2.5, None, dt.date(2026, 10, 6), 'b', 3]
        _got_    = Polars2SVG().colors(_values_)
        self.assertEqual(list(_got_), _values_)                         # input order, not batch order
        for _value_ in _values_:
            with self.subTest(value=_value_):
                self.assertEqual(_got_[_value_], self._fresh_(_value_))

    def test_colors_gives_python_equal_values_one_entry_the_first_ones(self):
        for _a_, _b_ in self._LOOK_ALIKES_:
            with self.subTest(a=_a_, b=_b_):
                _p2s_ = Polars2SVG()
                _got_ = _p2s_.colors([_a_, _b_])
                self.assertEqual(len(_got_), 1)
                self.assertEqual(_got_[_a_], self._fresh_(_a_))
                # ... and the memo it filled does not leak into the other value
                self.assertEqual(_p2s_.color(_b_), self._fresh_(_b_))

    def test_colors_and_color_share_the_memo_without_merging(self):
        # Not a dict: -1 and -1.0 would be one key in it -- the bug under test
        _int_, _float_, _bool_ = self._fresh_(-1), self._fresh_(-1.0), self._fresh_(True)
        _p2s_ = Polars2SVG()
        _p2s_.colors([-1, 1])
        with mock.patch.object(p2s_colors_mixin, 'colorHashes', wraps=p2s_colors_mixin.colorHashes) as _spy_:
            self.assertEqual(_p2s_.color(-1),   _int_)                  # memo hit
            self.assertEqual(_spy_.call_count, 0)
            self.assertEqual(_p2s_.color(-1.0), _float_)                # a miss, not -1's entry
            self.assertEqual(_p2s_.color(True), _bool_)                 # a miss, not 1's entry
            self.assertEqual(_spy_.call_count, 2)


if __name__ == '__main__':
    unittest.main()
