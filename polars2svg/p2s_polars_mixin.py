from typing import Any
import polars as pl
import re
import operator
from functools import reduce

class P2SPolarsMixin:
    # ---------------------------------------------------------------------
    # Host-class attributes this mixin reads off `self`.
    #
    # A mixin is never instantiated on its own -- these are provided by whatever
    # class mixes it in (Polars2SVG).  Declared here so a checker
    # can follow the mixin's own methods; bare annotations, so nothing exists at
    # runtime and nothing is shadowed.
    #
    # Typed `Any` on purpose: the mixin genuinely does not know the concrete type,
    # and several hosts declare the same name with a narrower type of their own.
    # ---------------------------------------------------------------------
    logger: Any

    def __init__(self) -> None:
        pass

    def __p2s_polars_mixin_init__(self) -> None:
        pass

    # -------------------------------------------------------------------------
    # Ported from rtsvg/rtsvg.py (David Trimm, Apache 2.0)
    # -------------------------------------------------------------------------

    def copyDataFrame(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.clone()

    def flattenTuple(self, _tuple_: tuple) -> tuple:
        _ls_: list = []
        for x in _tuple_:
            if isinstance(x, tuple): _ls_.extend(self.flattenTuple(x))
            else:                    _ls_.append(x)
        return tuple(_ls_)

    def createConcatColumn(self, df: pl.DataFrame, columns: tuple, new_column: str) -> pl.DataFrame:
        to_concat_new, str_casts = [], []
        for x in columns:
            if df[x].dtype != pl.String:
                str_casts.append(pl.col(x).cast(str).alias('__' + x + '_as_str__'))
                to_concat_new.append(pl.col('__' + x + '_as_str__'))
            else:
                to_concat_new.append(pl.col(x))
        return df.with_columns(*str_casts).with_columns(pl.concat_str(to_concat_new, separator='|').alias(new_column))

    def polarsFilterColumnsWithNaNs(self, df: pl.DataFrame, cols: tuple) -> pl.DataFrame:
        exprs = [pl.col(col).is_not_null() for col in cols]
        if len(exprs) == 0:
            return df
        return df.filter(reduce(operator.and_, exprs))

    # -------------------------------------------------------------------------

    #
    # polarsConcatString - for a given string, return a list of polars expressions for use with pl.concat_str
    # - handles duplicate column names
    # - performs simple precision operations -- only {_column_:.2f}
    #
    def polarsConcatString(self, s: str) -> list:
        _matches_ = []
        for _match_ in re.findall(r'{[a-zA-Z0-9_\-. :]+}', s): _matches_.append(_match_)
        _parts_ = []
        i = 0
        for _match_ in _matches_:
            j = s.index(_match_, i)
            if i != j: _parts_.append(s[i:j])
            _parts_.append(_match_)
            i = j + len(_match_)
        if i < len(s): _parts_.append(s[i:])
        _expr_ = []
        for _part_ in _parts_:
            if _part_.startswith('{') and _part_.endswith('}'):
                _within_braces_ = _part_[1:-1]
                if ':' in _within_braces_: _col_, _format_ = _within_braces_.split(':')
                else:                      _col_, _format_ = _within_braces_, None

                if _format_ is None: _expr_.append(pl.col(_col_))
                else:
                    if _format_.startswith('.') and _format_.endswith('f'):
                        _precision_ = int(_format_[1:-1])
                        _expr_.append(pl.col(_col_).round(_precision_).cast(pl.String))
                    elif re.fullmatch(r'0\d+d', _format_):
                        _width_ = int(_format_[1:-1])
                        _expr_.append(pl.col(_col_).cast(pl.String).str.zfill(_width_))
                    else:
                        self.logger.warning(f'XYp.polarsConcatString(): Unknown format "{_format_}" in string template "{s}"')
                        _expr_.append(pl.col(_col_))
            else:
                _expr_.append(pl.lit(_part_))
        return _expr_



