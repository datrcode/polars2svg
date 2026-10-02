from typing import Any
import polars as pl

from .p2s_render_mixin import OTHER_LABEL


class P2SBinComponentMixin:
    # ---------------------------------------------------------------------
    # Host-class attributes this mixin reads off `self`.
    #
    # A mixin is never instantiated on its own -- these are provided by whatever
    # class mixes it in (Histop, Timep).  Declared here so a checker
    # can follow the mixin's own methods; bare annotations, so nothing exists at
    # runtime and nothing is shadowed.
    #
    # Typed `Any` on purpose: the mixin genuinely does not know the concrete type,
    # and several hosts declare the same name with a narrower type of their own.
    # ---------------------------------------------------------------------
    _color_field_:       Any
    color:               Any
    count:               Any
    p2s:                 Any
    remainder_threshold: Any

    #
    # P2SBinComponentMixin - shared color-stat / legend / count-formatting logic
    # for the binned distribution components (Histop, Timep). These three methods
    # were byte-identical in both components (no per-component variation), so they
    # move here verbatim. The mixin reads component state it does not own:
    # self.p2s, self.color, self._color_field_, and self.count.
    #
    def __colorStatAggExpr__(self) -> pl.Expr:
        """Aggregation expression for numeric spectrum coloring (default: sum)."""
        _field_ = self._color_field_
        _op_    = pl.col(_field_).sum()
        if isinstance(self.color, tuple):
            for item in self.color:
                if   item in {self.p2s.CMAGNITUDE_MINp,    self.p2s.CSTRETCHED_MINp,    self.p2s.MINp}:    _op_ = pl.col(_field_).min();    break
                elif item in {self.p2s.CMAGNITUDE_MEDIANp, self.p2s.CSTRETCHED_MEDIANp, self.p2s.MEDIANp}: _op_ = pl.col(_field_).median(); break
                elif item in {self.p2s.CMAGNITUDE_MEANp,   self.p2s.CSTRETCHED_MEANp,   self.p2s.MEANp}:   _op_ = pl.col(_field_).mean();   break
                elif item in {self.p2s.CMAGNITUDE_MAXp,    self.p2s.CSTRETCHED_MAXp,    self.p2s.MAXp}:    _op_ = pl.col(_field_).max();    break
                elif item == self.p2s.STDp:                                                                   _op_ = pl.col(_field_).std();    break
        return _op_.alias('__color_stat__')

    def __legendColorFieldName__(self) -> str:
        if isinstance(self.color, str):   return self.color
        if isinstance(self.color, tuple): return '|'.join(_f_ for _f_ in self.color if isinstance(_f_, str))
        return ''

    def __formatCount__(self, count: float) -> str:
        if count is None: return '0'
        _v_ = float(count)
        if   _v_ >= 1_000_000: return f'{_v_/1_000_000:.1f}M'
        elif _v_ >= 1_000:     return f'{_v_/1_000:.1f}K'
        elif _v_ == int(_v_):  return str(int(_v_))
        else:                  return f'{_v_:.2g}'

    #
    # __clampNegativeCounts__() - a bar, or a stacked bar's segment, is drawn from zero, so a
    # count= that sums below zero is drawn as zero (PLANNING.md §5 C-negative-counts).  That
    # used to happen by accident, and silently -- a simple bar came out as nothing, but a
    # negative segment in a stacked bar threw the stack off its scale.  Clamp it here, right
    # after aggregating, so everything downstream (pooling, order=, the count axis) sees what
    # is drawn, and say so once per distinct case.  df_agg has a '__count__' per row.
    #
    def __clampNegativeCounts__(self, df_agg: pl.DataFrame, bin_col: str) -> pl.DataFrame:
        if '__count__' not in df_agg.columns or not df_agg['__count__'].dtype.is_numeric(): return df_agg
        _bins_ = df_agg.filter(pl.col('__count__') < 0)[bin_col].unique().sort().to_list()
        if len(_bins_) == 0: return df_agg
        _shown_ = ', '.join(str(b) for b in _bins_[:5]) + (f' and {len(_bins_) - 5:,} more' if len(_bins_) > 5 else '')
        self.p2s.logger.warning(
            f'{type(self).__name__}: count={self.count!r} sums below zero in {len(_bins_):,} '
            f'bin{"s" if len(_bins_) != 1 else ""} ({_shown_}); a negative count is drawn as zero')
        return df_agg.with_columns(pl.col('__count__').clip(lower_bound=0))

    #
    # __poolThinColors__() - fold every colour value whose largest segment would be under
    # remainder_threshold px into '(other)'.  px_per_count is the scale a count is drawn
    # at.  It is only an estimate while the aggregates are built, before the geometry
    # exists, so each component pools again once it knows its plot (PLANNING.md §5
    # C-histop-two-remainders: the first pass alone measured against the whole canvas, and
    # kept values the plot could not draw).  df_agg has one row per (bin, colour) with a
    # '__count__'; the result is sorted as the input was built.
    #
    def __poolThinColors__(self, df_agg: pl.DataFrame, bin_col: str, px_per_count: float) -> pl.DataFrame:
        _cf_      = self._color_field_
        _stats_   = df_agg.group_by(_cf_).agg(pl.col('__count__').max().alias('__max_in_bin__'))
        _visible_ = set(_stats_.filter(pl.col('__max_in_bin__') * px_per_count >= self.remainder_threshold)[_cf_].to_list())
        if len(_visible_) == len(_stats_): return df_agg
        _visible_str_ = {str(v) for v in _visible_}
        return (df_agg
                .with_columns(pl.col(_cf_).cast(pl.String))
                .with_columns(pl.when(pl.col(_cf_).is_in(_visible_str_))
                                .then(pl.col(_cf_))
                                .otherwise(pl.lit(OTHER_LABEL))
                                .alias(_cf_))
                .group_by([bin_col, _cf_])
                .agg(pl.col('__count__').sum())
                .sort([bin_col, _cf_]))

