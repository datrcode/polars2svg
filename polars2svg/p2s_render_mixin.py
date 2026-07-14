import polars as pl

from .exceptions import InvalidSpecError

__name__ = 'p2s_render_mixin'

class P2SRenderMixin:
    def __init__(self):
        pass

    def __p2s_render_mixin_init__(self):
        pass


    #
    # colorizeOrder() - determine the global ordering of colors across all bars
    # Returns a list of color values sorted by their global total (descending),
    # to be passed as color_order to colorizeBar() for consistent cross-bar ordering.
    #
    def colorizeOrder(self, df, count, color):
        color_enums = set()
        if isinstance(color, tuple):
            _strs_ = []
            for i in range(len(color)):
                if isinstance(color[i], str): _strs_.append(color[i])
                else:                         color_enums.add(color[i])
            df = df.with_columns(pl.concat_str(_strs_, separator=self.MULTI_FIELD_SEP).alias('__color__'))
            color = '__color__'
        #
        # ROW COUNTING
        #
        if   count == self.ROW_COUNTp:
            df_gb = df.group_by(color).len().rename({'len':'__sum__'})
        #
        # VARIATIONS OF SCALAR COUNTING
        #
        elif isinstance(count, str) and self.numericColumn(df, count):
            df_gb = df.group_by(color).agg(pl.col(count).sum().alias('__sum__'))
        elif isinstance(count, tuple) and len(count) == 1 and self.numericColumn(df, count[0]):
            df_gb = df.group_by(color).agg(pl.col(count[0]).sum().alias('__sum__'))
        elif isinstance(count, tuple) and len(count) == 2 and self.numericColumn(df, count[0]) and count[1] == self.SCALARp:
            df_gb = df.group_by(color).agg(pl.col(count[0]).sum().alias('__sum__'))
        #
        # VARIATIONS OF SET COUNTING
        #
        elif isinstance(count, str) and count == color:
            df_gb = df.select([color]).unique().with_columns(pl.lit(1.0).alias('__sum__'))
        elif isinstance(count, str) and self.ROW_COUNTp in color_enums:
            df_gb = df.group_by([count, color]).len().group_by(color).len().rename({'len':'__sum__'})
        elif isinstance(count, tuple) and self.ROW_COUNTp in color_enums:
            _strs_ = [s for s in count if isinstance(s, str)]
            if color not in _strs_: _strs_.append(color)
            df_gb = df.group_by(_strs_).len().group_by(color).len().rename({'len':'__sum__'})
        elif isinstance(count, str):
            df_gb = (
                df.select([count, color])
                .unique()
                .with_columns( (1.0 / pl.col(color).count().over(count)).alias('__weight__') )
                .group_by(color)
                .agg(pl.col('__weight__').sum().alias('__sum__'))
            )
        elif isinstance(count, tuple):
            _strs_, _enums_ = [], set()
            for i in range(len(count)):
                if isinstance(count[i], str): _strs_.append(count[i])
                else:                         _enums_.add(count[i])
            if color not in _strs_: _strs_.append(color)
            df_gb = (
                df.select(_strs_)
                .unique()
                .with_columns( (1.0 / pl.col(color).count().over(_strs_)).alias('__weight__') )
                .group_by(color)
                .agg(pl.col('__weight__').sum().alias('__sum__'))
            )
        else: raise InvalidSpecError(f'colorizeOrder(): unknown count type {count=}')

        return df_gb.sort(['__sum__', color], descending=True)[color].to_list()

    #
    # colorizeBar() - colorize a bar for a barchart using polars operations
    # xywh convention: (x, y_bottom, bar_width, bar_height) - bars extend upward from y_bottom
    #
    def colorizeBar(self, df, xywh, count, color, color_order=None, orientation='horizontal', remainder_threshold=3.0, dl=None):
        if orientation not in ('horizontal', 'vertical'):
            raise ValueError('colorizeBar(): orientation must be "horizontal" or "vertical"')
        bar_h, bar_w = xywh[3], xywh[2]
        _x_, _y_ = xywh[0], xywh[1]
        _color_data_ = self.colorTyped('data', 'default')
        # Easiest case first - bar extends upward from _y_ (bottom baseline)
        if   color is None and orientation == 'horizontal':
            _s_ = f'<rect x="{xywh[0]}" y="{xywh[1]}" width="{bar_h}" height="{bar_w}" fill="{_color_data_}" stroke="none" />'
            if dl is not None: return dl.rect(xywh[0], xywh[1], bar_h, bar_w, _color_data_, svg=_s_)
            return _s_
        elif color is None and orientation == 'vertical':
            _s_ = f'<rect x="{xywh[0]}" y="{xywh[1] - bar_h}" width="{bar_w}" height="{bar_h}" fill="{_color_data_}" stroke="none" />'
            if dl is not None: return dl.rect(xywh[0], xywh[1] - bar_h, bar_w, bar_h, _color_data_, svg=_s_)
            return _s_
        # Concatenate the color fields if necessary & set the color field to the concatenation
        color_enums = set()
        if isinstance(color, tuple):
            _strs_ = []
            for i in range(len(color)):
                if isinstance(color[i], str): _strs_.append(color[i])
                else:                         color_enums.add(color[i])
            df = df.with_columns(pl.concat_str(_strs_, separator=self.MULTI_FIELD_SEP).alias('__color__'))
            color = '__color__'
        #
        # Use the correct method for counting
        #
        # ROW COUNTING
        #
        if   count == self.ROW_COUNTp:                                 
            df_gb = df.group_by(color).len().rename({'len':'__sum__'})
        #
        # VARIATIONS OF SCALAR COUNTING
        #
        elif isinstance(count, str) and self.numericColumn(df, count): 
            df_gb = df.group_by(color).agg(pl.col(count).sum().alias('__sum__'))
        elif isinstance(count, tuple) and len(count) == 1 and self.numericColumn(df, count[0]):
            df_gb = df.group_by(color).agg(pl.col(count[0]).sum().alias('__sum__'))
        elif isinstance(count, tuple) and len(count) == 2 and self.numericColumn(df, count[0]) and count[1] == self.SCALARp:
            df_gb = df.group_by(color).agg(pl.col(count[0]).sum().alias('__sum__'))
        #
        # VARIATIONS OF SET COUNTING
        #
        elif isinstance(count, str) and count == color: 
            df_gb = df.select([color]).unique().with_columns(pl.lit(1.0).alias('__sum__'))
        elif isinstance(count, str) and self.ROW_COUNTp in color_enums:
            df_gb = df.group_by([count, color]).len().group_by(color).len().rename({'len':'__sum__'})
        elif isinstance(count, tuple) and self.ROW_COUNTp in color_enums:
            _strs_ = [s for s in count if isinstance(s, str)]
            if color not in _strs_: _strs_.append(color)
            df_gb = df.group_by(_strs_).len().group_by(color).len().rename({'len':'__sum__'})
        elif isinstance(count, str):
            df_gb = (
                df.select([count, color])
                .unique()
                .with_columns( (1.0 / pl.col(color).count().over(count)).alias('__weight__') )
                .group_by(color)
                .agg(pl.col('__weight__').sum().alias('__sum__'))
            )
        elif isinstance(count, tuple):
            _strs_, _enums_ = [], set()
            for i in range(len(count)):
                if isinstance(count[i], str): _strs_.append(count[i])
                else:                         _enums_.add(count[i])
            if color not in _strs_: _strs_.append(color)
            df_gb = (
                df.select(_strs_)
                .unique()
                .with_columns( (1.0 / pl.col(color).count().over(_strs_)).alias('__weight__') )
                .group_by(color)
                .agg(pl.col('__weight__').sum().alias('__sum__'))
            )
        #
        # Unrecognized count type -- raise
        #
        else: raise InvalidSpecError(f'colorizeBar(): unknown count type {count=}')

        # Order segments: global order if color_order provided, else local sort by count
        if color_order is not None:
            _rank_df_ = pl.DataFrame({color: color_order, '__rank__': list(range(len(color_order)))})
            df_gb = df_gb.join(_rank_df_, on=color, how='left') \
                        .with_columns(pl.col('__rank__').fill_null(len(color_order))) \
                        .sort(['__rank__', '__sum__', color], descending=[False, True, True]) \
                        .drop('__rank__')
        else:
            df_gb = df_gb.sort(['__sum__', color], descending=True)

        # Determine the percentage covered & the bar height
        df_gb = df_gb.with_columns((pl.col('__sum__')/pl.col('__sum__').sum()).alias('__perc__')) \
                    .with_columns((pl.col('__perc__') * bar_h).alias('__h_in_px__'))
        # Split into what needs to be aggregated based on bar height
        df_above = df_gb.filter(pl.col('__h_in_px__') >= remainder_threshold).with_columns(self.colorizeColumnPolarsOperations(color).alias('__hexcolor__'))
        df_below = df_gb.filter(pl.col('__h_in_px__') <  remainder_threshold)
        if len(df_below) > 0: df_below = df_below.select(pl.selectors.numeric()).sum().with_columns(pl.lit(_color_data_).alias('__hexcolor__'))
        # Put it back together
        df       = pl.concat([df_above, df_below], how='diagonal') if len(df_below) > 0 else df_above
        # Stack segments from bottom (_y_) upward: y = _y_ - cumsum_inclusive
        if orientation == 'horizontal': df = df.with_columns(pl.col('__h_in_px__').cum_sum().shift(1).fill_null(0.0).alias('__px__')).with_columns(pl.col('__px__') + _x_)
        else:                           df = df.with_columns((_y_ - pl.col('__h_in_px__').cum_sum()).alias('__px__'))
        # Render
        if orientation == 'horizontal': _op_     = self.polarsConcatString(f'<rect x="{{__px__}}" y="{_y_}" width="{{__h_in_px__}}" height="{bar_w}" fill="{{__hexcolor__}}" stroke="none" />')
        else:                           _op_     = self.polarsConcatString(f'<rect x="{_x_}" y="{{__px__}}" width="{bar_w}" height="{{__h_in_px__}}" fill="{{__hexcolor__}}" stroke="none" />')
        df       = df.with_columns(pl.concat_str(_op_).alias('__svg__'))
        if dl is not None:
            df = df.with_columns(self.rgbFromHexPolarsOperations('__hexcolor__', '__r_f__', '__g_f__', '__b_f__'))
            _rgba_ = ('__r_f__', '__g_f__', '__b_f__')
            if orientation == 'horizontal': return dl.rects_table(df, '__px__', _y_, '__h_in_px__', bar_w, _rgba_)
            else:                           return dl.rects_table(df, _x_, '__px__', bar_w, '__h_in_px__', _rgba_)
        return ''.join(df['__svg__'])

    #
    # colorizeAllBarsVertical() - render all vertical stacked bars in one Polars pipeline
    #
    # Equivalent to calling colorizeBar(..., orientation='vertical') once per bin, but
    # processes the entire aggregated DataFrame in a single pass — O(1) Polars operations
    # regardless of the number of bins.
    #
    # df must already be aggregated: one row per (bin_col, color) with a '__count__' column.
    # x_lookup is a small DataFrame with columns (bin_col, '__x__') giving the x position per bin.
    # count_min / count_max define the y-axis scale (bars are proportionally scaled to plot_h).
    #
    def colorizeAllBarsVertical(self, df, bin_col, x_lookup, y_bottom, bar_w,
                                plot_h, count_min, count_max, color,
                                color_order=None, remainder_threshold=3.0,
                                hexcolor_col=None, dl=None):
        _color_data_ = self.colorTyped('data', 'default')
        _span_        = max(float(count_max) - float(count_min), 1e-9)
        _count_min_f_ = float(count_min)

        # 1. Attach a sort rank to every row (one join over the full df, not per-bin)
        if color_order is not None:
            _n_colors_ = len(color_order)
            _rank_df_  = pl.DataFrame({color: color_order, '__rank__': list(range(_n_colors_))},
                                       schema_overrides={'__rank__': pl.Int64})
            df = df.join(_rank_df_, on=color, how='left') \
                   .with_columns(pl.col('__rank__').fill_null(_n_colors_))
        else:
            _totals_   = df.group_by(color).agg(pl.col('__count__').sum().alias('__ct__')) \
                           .sort(['__ct__', color], descending=True) \
                           .with_columns(pl.int_range(pl.len()).alias('__rank__')) \
                           .drop('__ct__')
            _n_colors_ = len(_totals_)
            df = df.join(_totals_, on=color, how='left') \
                   .with_columns(pl.col('__rank__').fill_null(_n_colors_))

        # 2. Per-bin total → per-bin bar height → per-segment pixel height  (all window ops)
        df = df.with_columns(pl.col('__count__').sum().over(bin_col).cast(pl.Float64).alias('__bin_total__')) \
               .with_columns(
                   ((pl.col('__bin_total__') - _count_min_f_).clip(lower_bound=0.0) / _span_ * plot_h)
                   .alias('__bar_h__')
               ) \
               .with_columns(
                   (pl.col('__count__').cast(pl.Float64) / pl.col('__bin_total__') * pl.col('__bar_h__'))
                   .alias('__h_in_px__')
               )

        # 3. Split above/below remainder threshold; assign segment colors
        if hexcolor_col is not None:
            df_above = df.filter(pl.col('__h_in_px__') >= remainder_threshold) \
                         .with_columns(pl.col(hexcolor_col).alias('__hexcolor__'))
        else:
            df_above = df.filter(pl.col('__h_in_px__') >= remainder_threshold) \
                         .with_columns(pl.col(color).cast(pl.String).alias(color)) \
                         .with_columns(self.colorizeColumnPolarsOperations(color).alias('__hexcolor__'))
        df_below = df.filter(pl.col('__h_in_px__') <  remainder_threshold)
        if len(df_below) > 0:
            df_below = df_below.group_by(bin_col).agg([
                pl.col('__h_in_px__').sum(),
                pl.lit(_n_colors_ + 1).cast(pl.Int64).alias('__rank__'),
                pl.lit(_color_data_).alias('__hexcolor__'),
            ])
            df = pl.concat([df_above, df_below], how='diagonal')
        else:
            df = df_above

        # 4. Sort so cum_sum within each bin follows rank order, then join x positions
        df = df.sort([bin_col, '__rank__']) \
               .join(x_lookup, on=bin_col, how='left')

        # 5. Cumulative y position within each bin (over() respects current sort order)
        df = df.with_columns(
            (y_bottom - pl.col('__h_in_px__').cum_sum().over(bin_col)).alias('__py__')
        )

        # 6. Render all segments
        _op_ = self.polarsConcatString(
            f'<rect x="{{__x__}}" y="{{__py__}}" width="{bar_w}" height="{{__h_in_px__}}" fill="{{__hexcolor__}}" stroke="none" />'
        )
        df = df.with_columns(pl.concat_str(_op_).alias('__svg__'))
        if dl is not None:
            df = df.with_columns(self.rgbFromHexPolarsOperations('__hexcolor__', '__r_f__', '__g_f__', '__b_f__'))
            return dl.rects_table(df, '__x__', '__py__', bar_w, '__h_in_px__', ('__r_f__', '__g_f__', '__b_f__'))
        return ''.join(df['__svg__'])

    #
    # colorizeAllBarsHorizontal() - render all horizontal stacked bars in one Polars pipeline
    #
    # Equivalent to calling colorizeBar(..., orientation='horizontal') once per bin, but
    # processes the entire aggregated DataFrame in a single pass — O(1) Polars operations
    # regardless of the number of bins.
    #
    # df must already be aggregated: one row per (bin_col, color) with a '__count__' column.
    # y_lookup is a small DataFrame with columns (bin_col, '__y__') giving the top y position per bin.
    # x_left is the fixed left x-coordinate of all bars (= plot_x0).
    # bar_h is the fixed height of each bar.
    # count_min / count_max define the x-axis scale (bars are proportionally scaled to plot_w).
    #
    def colorizeAllBarsHorizontal(self, df, bin_col, y_lookup, x_left, bar_h,
                                  plot_w, count_min, count_max, color,
                                  color_order=None, remainder_threshold=3.0,
                                  hexcolor_col=None, dl=None):
        _color_data_ = self.colorTyped('data', 'default')
        _span_        = max(float(count_max) - float(count_min), 1e-9)
        _count_min_f_ = float(count_min)

        # 1. Attach a sort rank to every row (one join over the full df, not per-bin)
        if color_order is not None:
            _n_colors_ = len(color_order)
            _rank_df_  = pl.DataFrame({color: color_order, '__rank__': list(range(_n_colors_))},
                                       schema_overrides={'__rank__': pl.Int64})
            df = df.join(_rank_df_, on=color, how='left') \
                   .with_columns(pl.col('__rank__').fill_null(_n_colors_))
        else:
            _totals_   = df.group_by(color).agg(pl.col('__count__').sum().alias('__ct__')) \
                           .sort(['__ct__', color], descending=True) \
                           .with_columns(pl.int_range(pl.len()).alias('__rank__')) \
                           .drop('__ct__')
            _n_colors_ = len(_totals_)
            df = df.join(_totals_, on=color, how='left') \
                   .with_columns(pl.col('__rank__').fill_null(_n_colors_))

        # 2. Per-bin total → per-bin bar width → per-segment pixel width  (all window ops)
        df = df.with_columns(pl.col('__count__').sum().over(bin_col).cast(pl.Float64).alias('__bin_total__')) \
               .with_columns(
                   ((pl.col('__bin_total__') - _count_min_f_).clip(lower_bound=0.0) / _span_ * plot_w)
                   .alias('__bar_w__')
               ) \
               .with_columns(
                   (pl.col('__count__').cast(pl.Float64) / pl.col('__bin_total__') * pl.col('__bar_w__'))
                   .alias('__w_in_px__')
               )

        # 3. Split above/below remainder threshold; assign segment colors
        if hexcolor_col is not None:
            df_above = df.filter(pl.col('__w_in_px__') >= remainder_threshold) \
                         .with_columns(pl.col(hexcolor_col).alias('__hexcolor__'))
        else:
            df_above = df.filter(pl.col('__w_in_px__') >= remainder_threshold) \
                         .with_columns(pl.col(color).cast(pl.String).alias(color)) \
                         .with_columns(self.colorizeColumnPolarsOperations(color).alias('__hexcolor__'))
        df_below = df.filter(pl.col('__w_in_px__') <  remainder_threshold)
        if len(df_below) > 0:
            df_below = df_below.group_by(bin_col).agg([
                pl.col('__w_in_px__').sum(),
                pl.lit(_n_colors_ + 1).cast(pl.Int64).alias('__rank__'),
                pl.lit(_color_data_).alias('__hexcolor__'),
            ])
            df = pl.concat([df_above, df_below], how='diagonal')
        else:
            df = df_above

        # 4. Sort so cum_sum within each bin follows rank order, then join y positions
        df = df.sort([bin_col, '__rank__']) \
               .join(y_lookup, on=bin_col, how='left')

        # 5. Cumulative x position within each bin (stacks left-to-right)
        df = df.with_columns(
            (x_left + pl.col('__w_in_px__').cum_sum().over(bin_col) - pl.col('__w_in_px__')).alias('__px__')
        )

        # 6. Render all segments
        _op_ = self.polarsConcatString(
            f'<rect x="{{__px__}}" y="{{__y__}}" width="{{__w_in_px__}}" height="{bar_h}" fill="{{__hexcolor__}}" stroke="none" />'
        )
        df = df.with_columns(pl.concat_str(_op_).alias('__svg__'))
        if dl is not None:
            df = df.with_columns(self.rgbFromHexPolarsOperations('__hexcolor__', '__r_f__', '__g_f__', '__b_f__'))
            return dl.rects_table(df, '__px__', '__y__', '__w_in_px__', bar_h, ('__r_f__', '__g_f__', '__b_f__'))
        return ''.join(df['__svg__'])

