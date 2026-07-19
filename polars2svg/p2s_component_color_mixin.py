import polars as pl

__name__ = 'p2s_component_color_mixin'


class P2SComponentColorMixin:
    #
    # P2SComponentColorMixin - shared edge/node color-resolution logic for the
    # graph components (LinkP, ChP). These five methods were byte-identical in
    # both components except for the component's own display name; that name is
    # now supplied by the _COMPONENT_NAME_ class attribute (used for dtype-keyed
    # color logging and validation error messages), which each component sets.
    #
    # The mixin reads component state it does not own: self.p2s, self.df,
    # self.color / self.node_color, the _color_stat_min_/_max_ and
    # _legend_stat_min_/_max_ accumulators, and color_stat_range_shared. Every
    # component mixing this in provides them.
    #
    _COMPONENT_NAME_ = 'Component'

    #
    # __effectiveColorSpec__() - resolve the color spec for links or nodes
    #
    def __effectiveColorSpec__(self, target):
        if target == 'links': return self.color
        return self.node_color

    #
    # __colorModeInfo__() - classify a color spec into a mode dict
    # Returns: {'kind': str, 'field': str|None, 'stat': str, 'hex': str|None}
    # Kinds: 'default' | 'fixed_hex' | 'categorical' | 'crow_magnitude' | 'crow_stretched' |
    #        'cset_magnitude' | 'cset_stretched' | 'stat_magnitude' | 'stat_stretched'
    #
    def __colorModeInfo__(self, spec):
        _p2s_ = self.p2s
        _cmag_  = {_p2s_.CMAGNITUDE_SUMp, _p2s_.CMAGNITUDE_MINp, _p2s_.CMAGNITUDE_MEDIANp,
                   _p2s_.CMAGNITUDE_MEANp, _p2s_.CMAGNITUDE_MAXp}
        _cstr_  = {_p2s_.CSTRETCHED_SUMp, _p2s_.CSTRETCHED_MINp, _p2s_.CSTRETCHED_MEDIANp,
                   _p2s_.CSTRETCHED_MEANp, _p2s_.CSTRETCHED_MAXp}
        _smap_  = {
            _p2s_.CMAGNITUDE_SUMp: 'sum',    _p2s_.CSTRETCHED_SUMp: 'sum',
            _p2s_.CMAGNITUDE_MINp: 'min',    _p2s_.CSTRETCHED_MINp: 'min',
            _p2s_.CMAGNITUDE_MEDIANp:'median',_p2s_.CSTRETCHED_MEDIANp:'median',
            _p2s_.CMAGNITUDE_MEANp: 'mean',  _p2s_.CSTRETCHED_MEANp: 'mean',
            _p2s_.CMAGNITUDE_MAXp: 'max',    _p2s_.CSTRETCHED_MAXp: 'max',
            _p2s_.SUMp: 'sum',   _p2s_.MINp: 'min',
            _p2s_.MEDIANp: 'median', _p2s_.MEANp: 'mean',
            _p2s_.MAXp: 'max',   _p2s_.STDp: 'std',
        }
        _info_ = {'kind': 'default', 'field': None, 'stat': 'sum', 'hex': None}
        if spec is None:
            pass
        elif isinstance(spec, self.p2s.HexColorString):
            _info_['kind'] = 'fixed_hex';  _info_['hex'] = spec
        elif spec == _p2s_.CROW_MAGNITUDEp:
            _info_['kind'] = 'crow_magnitude'
        elif spec == _p2s_.CROW_STRETCHEDp:
            _info_['kind'] = 'crow_stretched'
        elif spec == _p2s_.COLOR_BY_NODE_NAME:
            _info_['kind'] = 'categorical'   # field=None → colorize by node name
        elif isinstance(spec, str) and self.df is not None and spec in self.df.columns:
            _is_num_ = self.p2s.numericColumn(self.df, spec)
            self.p2s.logDtypeKeyedColor(self._COMPONENT_NAME_, spec, _is_num_)
            if _is_num_:
                _info_['kind'] = 'stat_magnitude'; _info_['field'] = spec; _info_['stat'] = 'sum'
            else:
                _info_['kind'] = 'cset'; _info_['field'] = spec
        elif isinstance(spec, tuple):
            _strs_  = [f for f in spec if isinstance(f, str)]
            _enums_ = [e for e in spec if not isinstance(e, str)]
            _field_ = _strs_[0] if _strs_ else None
            _enum_  = _enums_[0] if _enums_ else None
            if   _enum_ == _p2s_.CSETp:
                _info_['kind'] = 'cset';           _info_['field'] = _field_
            elif _enum_ == _p2s_.CSET_MAGNITUDEp:
                _info_['kind'] = 'cset_magnitude'; _info_['field'] = _field_
            elif _enum_ == _p2s_.CSET_STRETCHEDp:
                _info_['kind'] = 'cset_stretched'; _info_['field'] = _field_
            elif _enum_ in _cmag_:
                _info_['kind'] = 'stat_magnitude'; _info_['field'] = _field_; _info_['stat'] = _smap_.get(_enum_, 'sum')
            elif _enum_ in _cstr_:
                _info_['kind'] = 'stat_stretched'; _info_['field'] = _field_; _info_['stat'] = _smap_.get(_enum_, 'sum')
            elif _enum_ in _smap_:
                _info_['kind'] = 'stat_magnitude'; _info_['field'] = _field_; _info_['stat'] = _smap_[_enum_]
            elif _field_:
                _info_['kind'] = 'categorical';    _info_['field'] = _field_
        return _info_

    #
    # __colorAggExprs__() - return agg expressions needed by a color mode (added into group_by().agg())
    #
    def __colorAggExprs__(self, mode_info, prefix):
        kind = mode_info['kind']
        if kind in ('categorical', 'cset'):
            return [
                pl.col(f'__{prefix}_cat__').n_unique().alias(f'__{prefix}_nuniq__'),
                pl.col(f'__{prefix}_cat__').first().alias(f'__{prefix}_first__'),
            ]
        elif kind in ('cset_magnitude', 'cset_stretched') and mode_info['field']:
            return [pl.col(mode_info['field']).n_unique().alias(f'__{prefix}_stat__')]
        elif kind in ('stat_magnitude', 'stat_stretched') and mode_info['field']:
            _field_ = mode_info['field']
            _op_ = {
                'sum':    pl.col(_field_).sum(),
                'min':    pl.col(_field_).min(),
                'median': pl.col(_field_).median(),
                'mean':   pl.col(_field_).mean(),
                'max':    pl.col(_field_).max(),
                'std':    pl.col(_field_).std(),
            }.get(mode_info['stat'], pl.col(_field_).sum())
            return [_op_.alias(f'__{prefix}_stat__')]
        elif kind in ('crow_magnitude', 'crow_stretched'):
            return [pl.len().alias(f'__{prefix}_row_count__')]
        return []

    #
    # __applyColorToDF__() - add f'__{prefix}_hex__' column to an aggregated DataFrame
    #
    def __applyColorToDF__(self, df, mode_info, prefix, default_hex):
        kind    = mode_info['kind']
        col_hex = f'__{prefix}_hex__'
        if kind == 'fixed_hex':
            return df.with_columns(pl.lit(mode_info['hex']).alias(col_hex))
        elif kind == 'categorical':
            return df.with_columns(
                pl.when(pl.col(f'__{prefix}_nuniq__') == 1)
                  .then(pl.col(f'__{prefix}_first__'))
                  .otherwise(pl.lit(default_hex))
                  .alias(col_hex)
            )
        elif kind == 'cset':
            df = df.with_columns(
                pl.when(pl.col(f'__{prefix}_nuniq__') == 1)
                  .then(pl.col(f'__{prefix}_first__'))
                  .otherwise(pl.lit(-1))
                  .alias(f'__{prefix}_set_elem__')
            )
            return df.with_columns(
                self.p2s.colorizeColumnPolarsOperations(f'__{prefix}_set_elem__').alias(col_hex)
            )
        elif kind in ('crow_magnitude', 'crow_stretched', 'cset_magnitude', 'cset_stretched',
                      'stat_magnitude', 'stat_stretched'):
            _sc_     = f'__{prefix}_row_count__' if kind in ('crow_magnitude', 'crow_stretched') else f'__{prefix}_stat__'
            _norm_   = f'__{prefix}_norm__'
            _r_, _g_, _b_ = f'__{prefix}_r__', f'__{prefix}_g__', f'__{prefix}_b__'
            # legend-only stat accumulator (kept separate from _color_stat_min_/_max_,
            # which smallp SM_COLOR sharing reads and which stretched modes never touch)
            _lg_min_ = df[_sc_].cast(pl.Float64).min()
            _lg_max_ = df[_sc_].cast(pl.Float64).max()
            if _lg_min_ is not None and (getattr(self, '_legend_stat_min_', None) is None or _lg_min_ < self._legend_stat_min_):
                self._legend_stat_min_ = float(_lg_min_)
            if _lg_max_ is not None and (getattr(self, '_legend_stat_max_', None) is None or _lg_max_ > self._legend_stat_max_):
                self._legend_stat_max_ = float(_lg_max_)
            if kind in ('crow_stretched', 'cset_stretched', 'stat_stretched'):
                _n_unique_ = df[_sc_].n_unique()
                df = df.with_columns(
                    ((pl.col(_sc_).rank('dense') - 1).cast(pl.Float64) / max(_n_unique_ - 1, 1)).alias(_norm_)
                )
            else:
                if self.color_stat_range_shared is not None:
                    _cs_min_ = float(self.color_stat_range_shared[0])
                    _cs_max_ = float(self.color_stat_range_shared[1])
                else:
                    _min_v_ = df[_sc_].cast(pl.Float64).min()
                    _max_v_ = df[_sc_].cast(pl.Float64).max()
                    _cs_min_ = float(_min_v_) if _min_v_ is not None else 0.0
                    _cs_max_ = float(_max_v_) if _max_v_ is not None else 1.0
                if self._color_stat_min_ is None or _cs_min_ < self._color_stat_min_:
                    self._color_stat_min_ = _cs_min_
                if self._color_stat_max_ is None or _cs_max_ > self._color_stat_max_:
                    self._color_stat_max_ = _cs_max_
                df = df.with_columns(
                    ((pl.col(_sc_).cast(pl.Float64) - _cs_min_) /
                     (0.001 + _cs_max_ - _cs_min_))
                    .clip(0.0, 1.0).alias(_norm_)
                )
            df = df.with_columns(
                self.p2s.colorSpectrumPolarsOperations(_norm_, _r_, _g_, _b_)
            ).with_columns(
                self.p2s.hexColorFromRGBTriplesPolarsOperations(_r_, _g_, _b_).alias(col_hex)
            )
            return df
        else:
            return df.with_columns(pl.lit(default_hex).alias(col_hex))

    #
    # __validateColorSpec__() - raise ValueError if a node_color value is not a recognized form
    #
    def __validateColorSpec__(self, spec, param_name, allow_dict=False):
        if spec is None: return
        if isinstance(spec, dict):
            if not allow_dict:
                raise ValueError(f'{self._COMPONENT_NAME_}.__validateInput__(): {param_name} does not support dict values')
            return
        if isinstance(spec, tuple): return
        _p2s_ = self.p2s
        if spec in (_p2s_.CROW_MAGNITUDEp, _p2s_.CROW_STRETCHEDp, _p2s_.COLOR_BY_NODE_NAME): return
        if isinstance(spec, self.p2s.HexColorString): return
        if isinstance(spec, str):
            if self.df is not None and spec in self.df.columns: return
            raise ValueError(
                f'{self._COMPONENT_NAME_}.__validateInput__(): {param_name}={spec!r} is not a hex color, '
                f'a recognized constant, or a DataFrame column name'
            )
        raise ValueError(
            f'{self._COMPONENT_NAME_}.__validateInput__(): {param_name}={spec!r} has unsupported type {type(spec).__name__}'
        )
