from typing import Any
import polars as pl


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
    _color_field_: Any
    color:         Any
    p2s:           Any

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
