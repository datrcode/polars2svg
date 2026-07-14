#
# Polars implementation of the following:
#
# H. Rave, V. Molchanov and L. Linsen, "Uniform Sample Distribution in Scatterplots via Sector-based Transformation," 
# 2024 IEEE Visualization and Visual Analytics (VIS), St. Pete Beach, FL, USA, 2024, pp. 156-160, 
# doi: 10.1109/VIS55277.2024.00039. 
# keywords: {Data analysis;Visual analytics;Clutter;Scatterplot de-cluttering;spatial transformation},
#
# This version includes the tile optimization...
#
import polars  as     pl
import numpy   as     np
from   math    import pi, sin, cos, atan2
import os
from   os.path import exists
import time
import platformdirs
from   importlib.resources import files as _pkg_files

from .exceptions import DataError, Polars2SVGError

__name__ = 'udist_scatterplots_via_sectors_tile_opt'

class UDistScatterPlotsViaSectorsTileOpt(object):
    #
    # __init__()
    #
    # Parameters
    # ----------
    # x_vals : list[float]
    #     List of x values
    # y_vals : list[float]
    #     List of y values
    # weights : list[float] | None
    #     List of weights
    # colors : list[str] | None
    #     List of colors
    # static_points : list[0,1] | None
    #     List of static points -- these points will not move but will be considered in the sector density calculations
    # vector_scalar : float, optional
    #     Vector scalar, by default 0.05 -- how much to adjust each point by per iteration
    # iterations : int, optional
    #     Number of iterations, by default 4 -- number of iterations of the algorithm to run
    # num_of_tiles : int, optional
    #     Number of tiles, by default 128 -- number of tiles to use for the tile optimization
    # decay_rate : float, optional
    #     Decay rate, by default None -- how the vector scalar decays with each iteration
    #
    def __init__(self, x_vals=[], y_vals=[], weights=None, colors=None, static_points=None, vector_scalar=0.05, iterations=4, num_of_tiles=128, decay_rate=None):
        self.vector_scalar = vector_scalar
        self.iterations    = iterations
        self.num_of_tiles  = num_of_tiles
        self.time_lu       = {'prepare_df':0.0, 'normalize':0.0, 'all_sectors':0.0, 'explode_points':0.0, 'arctangents':0.0, 'sector_sums':0.0, 
                              'add_missing_sectors':0.0, 'prepare_sector_angles':0.0, 'join_sector_angles':0.0, 'ray_segment_intersections':0.0,
                              'area_calc':0.0, 'sector_uv_summation':0.0, 'point_update':0.0, 'determine_tile':0.0, 'tile_sums':0.0,
                              'cross_join_tile_offsets':0.0, 'join_sector_info':0.0, 'hard_way_arctangents':0.0,
                              'medium_way_crossproducts':0.0, 'separate_easy_hard_way (easy way)':0.0,
                              'separate_easy_hard_way (non-easy way)':0.0, 'separate_easy_hard_way (medium way)':0.0,
                              'separate_easy_hard_way (hard way)':0.0,}


        # Create weights and static points if none were set
        if weights       is None: weights       = np.ones(len(x_vals))
        if static_points is None: static_points_exist, static_points = False, np.zeros(len(x_vals)) # by default, all points will move
        else:                     static_points_exist                = True

        def rayIntersectsSegment(xy_ray, uv_ray, xy0_segment, xy1_segment, include_xy1_endpoint=False, epsilon=1e-9):
            x_r,  y_r  = xy_ray
            dx_r, dy_r = uv_ray
            x0,   y0   = xy0_segment
            x1,   y1   = xy1_segment    
            # Segment direction vector
            dx_s, dy_s = x1 - x0, y1 - y0
            # Compute determinant
            det = -dx_r * dy_s + dy_r * dx_s
            if abs(det) < 1e-10: return None # Lines are parallel or collinear
            # Compute parameters t and u
            t = ((x_r - x0) * dy_s - (y_r - y0) * dx_s) / det
            u = ((x_r - x0) * dy_r - (y_r - y0) * dx_r) / det
            # Check if intersection is valid (t >= 0 for ray, 0 <= u <= 1 for segment)
            if t >= 0.0:
                if include_xy1_endpoint:
                    if 0.0 <= u <= 1.0+epsilon: return (x_r + t * dx_r, y_r + t * dy_r)
                else:
                    if 0.0 <= u <  1.0-epsilon: return (x_r + t * dx_r, y_r + t * dy_r)
            return None

        #
        # Prepare the xo/yo dataframe
        #
        t = time.time()
        # Prefer the copy shipped inside the installed package (located via
        # importlib.resources so it works for zip-safe installs, not just __file__
        # math). If it is missing, regenerate into a user cache dir rather than the
        # package directory -- site-packages is read-only in many environments
        # (system installs, Nix, some conda), so writing there would crash.
        _shipped = _pkg_files('polars2svg') / 'config' / 'udist_scatterplots_via_sectors_tile_opt.parquet'
        if _shipped.is_file():
            self.xoyo_filename = str(_shipped)
        else:
            _cache_dir         = platformdirs.user_cache_dir('polars2svg')
            os.makedirs(_cache_dir, exist_ok=True)
            self.xoyo_filename = os.path.join(_cache_dir, 'udist_scatterplots_via_sectors_tile_opt.parquet')
            if exists(self.xoyo_filename) == False: self.createXoYoDataframeFile()
        self.df_xoyo_sector = pl.read_parquet(self.xoyo_filename).filter(pl.col('num_of_tiles') == num_of_tiles).drop('num_of_tiles').unique()
        if len(self.df_xoyo_sector) == 0: raise DataError('No xo/yo sector data found for num_of_tiles = %d' % num_of_tiles)
        self.time_lu['xoyo_sector_creation'] = time.time() - t

        #
        # Prepare the initial dataframe
        #
        t = time.time()
        if colors is None: df = pl.DataFrame({'x':x_vals, 'y':y_vals, 'w':weights, 's':static_points})            .with_row_index('__index__').with_columns(pl.lit('#000000').alias('c'))
        else:              df = pl.DataFrame({'x':x_vals, 'y':y_vals, 'w':weights, 's':static_points, 'c':colors}).with_row_index('__index__')
        df_orig = df.clone()
        self.time_lu['prepare_df'] += (time.time() - t)

        self.df_anim = []

        #
        # Perform each iteration
        #
        for _iteration_ in range(iterations):
            # Determine the overall weight sums
            df_weight_sum = df['w'].sum()

            #
            # Normalize the points to 0.02 to 0.98 (want to give it a little space around the edges to that there are sectors to move into)
            #
            t = time.time()
            if set(['_u_', '_v_', 'x_right', 'y_right']) & set(df.columns) == set(['_u_', '_v_', 'x_right', 'y_right']): df = df.drop(['_u_', '_v_', 'x_right', 'y_right'])
            if 's_right' in df.columns: df = df.drop(['s_right'])
            # When all x (or y) values are identical (including the single-point case), max - min == 0, which would
            # otherwise produce a 0/0 NaN; fall back to the midpoint of the normalized range in that situation.
            x_range, y_range = pl.col('x').max() - pl.col('x').min(), pl.col('y').max() - pl.col('y').min()
            df = df.with_columns(pl.when(x_range == 0.0).then(0.5).otherwise(0.02 + 0.96 * (pl.col('x') - pl.col('x').min())/x_range).alias('x'),
                                 pl.when(y_range == 0.0).then(0.5).otherwise(0.02 + 0.96 * (pl.col('y') - pl.col('y').min())/y_range).alias('y'))
            self.time_lu['normalize'] += (time.time() - t)

            self.df_anim.append(df)

            #
            # All Sectors DataFrame
            #
            t = time.time()
            df_all_sectors = df.join(pl.DataFrame({'sector': [i for i in range(16)]}), how='cross').drop(['w','c','s'])
            self.time_lu['all_sectors'] += (time.time() - t)
            
            #
            # vvv -- New Performant Version -- vvv
            #

            # Determine the x/y tile of each point (xi,yi)
            t = time.time()
            df_w_tile = df.with_columns((pl.col('x') * num_of_tiles).cast(pl.Int16).alias('xi'), 
                                        (pl.col('y') * num_of_tiles).cast(pl.Int16).alias('yi'))
            self.time_lu['determine_tile'] += (time.time() - t)

            # Determine the sum of the weights in each tile (tile_sum)
            t = time.time()
            df_tile_sums = df_w_tile.group_by(['xi','yi']).agg(pl.col('w').sum().alias('tile_sum'))
            self.time_lu['tile_sums'] += (time.time() - t)

            # Cross join all filled in tiles with the original points & then compute the x/y offset (xo,yo)
            t = time.time()
            df           = df_w_tile.join(df_tile_sums, how='cross').rename({'xi_right':'xi_tile_sums', 'yi_right':'yi_tile_sums'}). \
                                                                     with_columns((pl.col('xi_tile_sums') - pl.col('xi')).alias('xo'),
                                                                                  (pl.col('yi_tile_sums') - pl.col('yi')).alias('yo'))
            self.time_lu['cross_join_tile_offsets'] += (time.time() - t)

            # Pull the sector information from the df_xoyo_sector dataframe ... any sector == -1 needs to be done the hard way
            # ... hard way == atan2 on the individual point level
            t = time.time()
            df          = df.join(self.df_xoyo_sector, on=['xo','yo'])
            self.time_lu['join_sector_info'] += (time.time() - t)

            # Separate into "easy way" and "hard way"
            t = time.time()
            df_easy_way   = df.filter(pl.col('sector') != -1)
            self.time_lu['separate_easy_hard_way (easy way)'] += (time.time() - t)

            t = time.time()
            df_hard_way   = df.filter(pl.col('sector') == -1) \
                              .join(df_w_tile.drop(['c']), left_on=['xi_tile_sums','yi_tile_sums'], right_on=['xi','yi']) \
                              .filter(pl.col('__index__') != pl.col('__index___right'))
            self.time_lu['separate_easy_hard_way (non-easy way)'] += (time.time() - t)
            
            t = time.time()
            df_medium_way = df_hard_way.filter(pl.col('u').is_not_null())
            self.time_lu['separate_easy_hard_way (medium way)'] += (time.time() - t)

            t = time.time()
            df_hard_way   = df_hard_way.filter(pl.col('u').is_null())
            self.time_lu['separate_easy_hard_way (hard way)'] += (time.time() - t)

            # Medium way calculation ... determine the sector based on crossproduct
            t = time.time()
            _x1_, _y1_ = pl.col('x') + pl.col('u'), pl.col('y') + pl.col('v')
            df_medium_way = df_medium_way.with_columns(pl.when((pl.col('x') - pl.col('x_right')) * (_y1_ - pl.col('y_right')) - 
                                                               (pl.col('y') - pl.col('y_right')) * (_x1_ - pl.col('x_right')) > 0.0)
                                                         .then     (pl.col('lsector'))
                                                         .otherwise(pl.col('rsector'))
                                                         .alias    ('sector'))
            self.time_lu['medium_way_crossproducts'] += (time.time() - t)

            # Hard way calculation ... determine the sector on a per point basis
            t = time.time()
            _dx_, _dy_ = pl.col('x_right') - pl.col('x'), pl.col('y_right') - pl.col('y')
            df_hard_way = df_hard_way.with_columns(((16*(pl.arctan2(_dy_, _dx_) + pl.lit(pi))/(pl.lit(2*pi))).cast(pl.Int64)).alias('sector'))
            self.time_lu['hard_way_arctangents'] += (time.time() - t)

            # Sector Summation
            t = time.time()
            df_easy_way   = df_easy_way  .drop(set(df_easy_way  .columns) - set(['__index__','sector','tile_sum'])).rename({'tile_sum':'_w_sum_'})
            df_medium_way = df_medium_way.drop(set(df_medium_way.columns) - set(['__index__','sector','w'       ])).rename({'w':       '_w_sum_'})
            df_hard_way   = df_hard_way  .drop(set(df_hard_way  .columns) - set(['__index__','sector','w_right' ])).rename({'w_right': '_w_sum_'})
            df_hard_way = pl.DataFrame({'__index__': df_hard_way['__index__'], '_w_sum_': df_hard_way['_w_sum_'], 'sector': df_hard_way['sector']}) # Re-align the columns
            df            = pl.concat([df_easy_way, df_medium_way, df_hard_way])
            df            = df.group_by(['__index__','sector']).agg(pl.col('_w_sum_').sum()).with_columns((pl.col('_w_sum_') / df_weight_sum).alias('_w_ratio_'))
            self.time_lu['sector_sums'] += (time.time() - t)

            #
            # ^^^ -- New Performant Version -- ^^^
            #

            #
            # Add the missing sectors back in...
            #
            t = time.time()
            df = df_all_sectors.join(df, on=['__index__','sector'], how='left').with_columns(pl.col('_w_sum_').fill_null(0), pl.col('_w_ratio_').fill_null(0))
            self.time_lu['add_missing_sectors'] += (time.time() - t)

            #
            # Create the sector angle dataframe
            # ... this is a small dataframe that covers just 16 sectors ...
            # ... it will be joined with the points dataframe to calculate the area of each sector for each point
            #
            t = time.time()
            _lu_ = {'sector':[], 
                    'a0':[],       'a0u':[],       'a0v':[],                 # Ray 0 angle & uv components
                    'a1':[],       'a1u':[],       'a1v':[],                 # Ray 1 angle & uv components
                    'corner_x':[], 'corner_y':[],                            # Corner between segment0 and segment 1
                    'anchor_a':[], 'anchor_u':[],  'anchor_v':[],            # Anchor angle & uv components
                    's0x0':[],     's0x1':[],      's0y0':[],     's0y1':[], # Segment 0
                    's1x0':[],     's1x1':[],      's1y0':[],     's1y1':[]} # Segment 1
            for _sector_ in range(16):
                _sector_align_ = (_sector_ + 8)%16
                _lu_['sector'].append(_sector_align_)
                _a0_ = _sector_*pi/8.0
                _lu_['a0'].append(_a0_), _lu_['a0u'].append(cos(_a0_)), _lu_['a0v'].append(sin(_a0_))
                _a1_ = (_sector_+1)*pi/8.0
                _lu_['a1'].append(_a1_), _lu_['a1u'].append(cos(_a1_)), _lu_['a1v'].append(sin(_a1_))
                _anchor_ = _a0_ + pi + pi / 16.0 # half angle on top of that
                _lu_['anchor_a'].append(_anchor_), _lu_['anchor_u'].append(cos(_anchor_)), _lu_['anchor_v'].append(sin(_anchor_))
                if   _sector_ >= 0 and _sector_ <  4:
                    _lu_['s0x0']    .append(1.0), _lu_['s0x1']    .append(1.0), _lu_['s0y0'].append(0.0), _lu_['s0y1'].append(1.0) # segment 0 (x0,y0) -> (x1,y1) (1,0) -> (1,1)
                    _lu_['s1x0']    .append(0.0), _lu_['s1x1']    .append(1.0), _lu_['s1y0'].append(1.0), _lu_['s1y1'].append(1.0) # segment 1 (x0,y0) -> (x1,y1) (0,1) -> (1,1)
                    _lu_['corner_x'].append(1.0), _lu_['corner_y'].append(1.0)
                elif _sector_ >= 4 and _sector_ <  8:
                    _lu_['s0x0']    .append(0.0), _lu_['s0x1']    .append(1.0), _lu_['s0y0'].append(1.0), _lu_['s0y1'].append(1.0) # (0,1) -> (1,1)
                    _lu_['s1x0']    .append(0.0), _lu_['s1x1']    .append(0.0), _lu_['s1y0'].append(0.0), _lu_['s1y1'].append(1.0) # (0,0) -> (0,1)
                    _lu_['corner_x'].append(0.0), _lu_['corner_y'].append(1.0)
                elif _sector_ >= 8 and _sector_ < 12:
                    _lu_['s0x0']    .append(0.0), _lu_['s0x1']    .append(0.0), _lu_['s0y0'].append(0.0), _lu_['s0y1'].append(1.0) # (0,0) -> (0,1)
                    _lu_['s1x0']    .append(0.0), _lu_['s1x1']    .append(1.0), _lu_['s1y0'].append(0.0), _lu_['s1y1'].append(0.0) # (0,0) -> (1,0)
                    _lu_['corner_x'].append(0.0), _lu_['corner_y'].append(0.0)
                else:
                    _lu_['s0x0']    .append(0.0), _lu_['s0x1']    .append(1.0), _lu_['s0y0'].append(0.0), _lu_['s0y1'].append(0.0) # (0,0) -> (1,0)
                    _lu_['s1x0']    .append(1.0), _lu_['s1x1']    .append(1.0), _lu_['s1y0'].append(0.0), _lu_['s1y1'].append(1.0) # (1,0) -> (1,1)
                    _lu_['corner_x'].append(1.0), _lu_['corner_y'].append(0.0)
            df_sector_angles = pl.DataFrame(_lu_)
            df_sector_angles = df_sector_angles.with_columns((pl.col('s0x1') - pl.col('s0x0')).alias('s0u'), (pl.col('s0y1') - pl.col('s0y0')).alias('s0v'),
                                                             (pl.col('s1x1') - pl.col('s1x0')).alias('s1u'), (pl.col('s1y1') - pl.col('s1y0')).alias('s1v'))
            self.time_lu['prepare_sector_angles'] += (time.time() - t)

            # Join w/ sector information
            t  = time.time()
            df = df.join(df_sector_angles, on='sector', how='left')
            self.time_lu['join_sector_angles'] += (time.time() - t)

            # Create rays for each sector angles
            t  = time.time()
            df = df.with_columns((pl.col('a0').cos()).alias('xa0'), (pl.col('a0').sin()).alias('ya0'), # uv for angle 0
                                 (pl.col('a1').cos()).alias('xa1'), (pl.col('a1').sin()).alias('ya1')) # uv for angle 1

            # Intersect each ray with each segment (uses the multistep version of ray-segment intersection)
            # ... determinate "r0s0_det" (ray 0, segment 0) .... done four ways (ray 0 & 1 ... segment 0 & 1)
            df = df.with_columns((-pl.col('a0u') * pl.col('s0v') + pl.col('a0v') * pl.col('s0u')).alias('r0s0_det'),
                                 (-pl.col('a0u') * pl.col('s1v') + pl.col('a0v') * pl.col('s1u')).alias('r0s1_det'),
                                 (-pl.col('a1u') * pl.col('s0v') + pl.col('a1v') * pl.col('s0u')).alias('r1s0_det'),
                                 (-pl.col('a1u') * pl.col('s1v') + pl.col('a1v') * pl.col('s1u')).alias('r1s1_det'))
            # ... "t" ("r0s0_t") and "u" values ("r0s0_u") (ray 0, segment 0) for all four ways (ray 0 & 1 ... segment 0 & 1)
            df = df.with_columns((((pl.col('x') - pl.col('s0x0')) * pl.col('s0v') - (pl.col('y') - pl.col('s0y0')) * pl.col('s0u')) / pl.col('r0s0_det')).alias('r0s0_t'),
                                 (((pl.col('x') - pl.col('s0x0')) * pl.col('a0v') - (pl.col('y') - pl.col('s0y0')) * pl.col('a0u')) / pl.col('r0s0_det')).alias('r0s0_u'),

                                 (((pl.col('x') - pl.col('s1x0')) * pl.col('s1v') - (pl.col('y') - pl.col('s1y0')) * pl.col('s1u')) / pl.col('r0s1_det')).alias('r0s1_t'),
                                 (((pl.col('x') - pl.col('s1x0')) * pl.col('a0v') - (pl.col('y') - pl.col('s1y0')) * pl.col('a0u')) / pl.col('r0s1_det')).alias('r0s1_u'),

                                 (((pl.col('x') - pl.col('s0x0')) * pl.col('s0v') - (pl.col('y') - pl.col('s0y0')) * pl.col('s0u')) / pl.col('r1s0_det')).alias('r1s0_t'),
                                 (((pl.col('x') - pl.col('s0x0')) * pl.col('a1v') - (pl.col('y') - pl.col('s0y0')) * pl.col('a1u')) / pl.col('r1s0_det')).alias('r1s0_u'),

                                 (((pl.col('x') - pl.col('s1x0')) * pl.col('s1v') - (pl.col('y') - pl.col('s1y0')) * pl.col('s1u')) / pl.col('r1s1_det')).alias('r1s1_t'),
                                 (((pl.col('x') - pl.col('s1x0')) * pl.col('a1v') - (pl.col('y') - pl.col('s1y0')) * pl.col('a1u')) / pl.col('r1s1_det')).alias('r1s1_u'),)
            # ... the x and y intersects (r0s0_xi, r0s0_yi) (ray 0, segment 0) for all four ways (ray 0 & 1) and segment (0 & 1)
            df = df.with_columns(pl.when((pl.col('r0s0_t') >= 0.0) & (pl.col('r0s0_u') >= 0.0) & (pl.col('r0s0_u') <= 1.0)).then(pl.col('x') + pl.col('r0s0_t') * pl.col('a0u')).otherwise(None).alias('r0s0_xi'),
                                 pl.when((pl.col('r0s0_t') >= 0.0) & (pl.col('r0s0_u') >= 0.0) & (pl.col('r0s0_u') <= 1.0)).then(pl.col('y') + pl.col('r0s0_t') * pl.col('a0v')).otherwise(None).alias('r0s0_yi'),

                                 pl.when((pl.col('r0s1_t') >= 0.0) & (pl.col('r0s1_u') >= 0.0) & (pl.col('r0s1_u') <= 1.0)).then(pl.col('x') + pl.col('r0s1_t') * pl.col('a0u')).otherwise(None).alias('r0s1_xi'),
                                 pl.when((pl.col('r0s1_t') >= 0.0) & (pl.col('r0s1_u') >= 0.0) & (pl.col('r0s1_u') <= 1.0)).then(pl.col('y') + pl.col('r0s1_t') * pl.col('a0v')).otherwise(None).alias('r0s1_yi'),

                                 pl.when((pl.col('r1s0_t') >= 0.0) & (pl.col('r1s0_u') >= 0.0) & (pl.col('r1s0_u') <= 1.0)).then(pl.col('x') + pl.col('r1s0_t') * pl.col('a1u')).otherwise(None).alias('r1s0_xi'),
                                 pl.when((pl.col('r1s0_t') >= 0.0) & (pl.col('r1s0_u') >= 0.0) & (pl.col('r1s0_u') <= 1.0)).then(pl.col('y') + pl.col('r1s0_t') * pl.col('a1v')).otherwise(None).alias('r1s0_yi'),

                                 pl.when((pl.col('r1s1_t') >= 0.0) & (pl.col('r1s1_u') >= 0.0) & (pl.col('r1s1_u') <= 1.0)).then(pl.col('x') + pl.col('r1s1_t') * pl.col('a1u')).otherwise(None).alias('r1s1_xi'),
                                 pl.when((pl.col('r1s1_t') >= 0.0) & (pl.col('r1s1_u') >= 0.0) & (pl.col('r1s1_u') <= 1.0)).then(pl.col('y') + pl.col('r1s1_t') * pl.col('a1v')).otherwise(None).alias('r1s1_yi'),)
            self.time_lu['ray_segment_intersections'] += (time.time() - t)

            #
            # Area Calculation using Shoelace Formula
            #
            # Case 0 ... which is X_X_ ... which is the first and second ray both hit the first segment
            t = time.time()
            _c0_0p_x_, _c0_0p_y_, _c0_0q_x_, _c0_0q_y_ = pl.col('r0s0_xi'), pl.col('r0s0_yi'), pl.col('r1s0_xi'), pl.col('r1s0_yi')
            _c0_1p_x_, _c0_1p_y_, _c0_1q_x_, _c0_1q_y_ = pl.col('r1s0_xi'), pl.col('r1s0_yi'), pl.col('x'),       pl.col('y')
            _c0_2p_x_, _c0_2p_y_, _c0_2q_x_, _c0_2q_y_ = pl.col('x'),       pl.col('y'),       pl.col('r0s0_xi'), pl.col('r0s0_yi')
            _c0_op_ = (((_c0_0p_x_*_c0_0q_y_ - _c0_0q_x_*_c0_0p_y_) + (_c0_1p_x_*_c0_1q_y_ - _c0_1q_x_*_c0_1p_y_) + (_c0_2p_x_*_c0_2q_y_ - _c0_2q_x_*_c0_2p_y_))/2.0).abs().alias('area')

            # Case 1 ... which is _X_X ... which is the first and second ray both hit the second segment
            _c1_0p_x_, _c1_0p_y_, _c1_0q_x_, _c1_0q_y_ = pl.col('r0s1_xi'), pl.col('r0s1_yi'), pl.col('r1s1_xi'), pl.col('r1s1_yi')
            _c1_1p_x_, _c1_1p_y_, _c1_1q_x_, _c1_1q_y_ = pl.col('r1s1_xi'), pl.col('r1s1_yi'), pl.col('x'),       pl.col('y')
            _c1_2p_x_, _c1_2p_y_, _c1_2q_x_, _c1_2q_y_ = pl.col('x'),       pl.col('y'),       pl.col('r0s1_xi'), pl.col('r0s1_yi')
            _c1_op_ = (((_c1_0p_x_*_c1_0q_y_ - _c1_0q_x_*_c1_0p_y_) + (_c1_1p_x_*_c1_1q_y_ - _c1_1q_x_*_c1_1p_y_) + (_c1_2p_x_*_c1_2q_y_ - _c1_2q_x_*_c1_2p_y_))/2.0).abs().alias('area')

            # Case 2 ... which is X__X ... which is the first and second ray both hit different segments... so needs the corner position
            _c2_0p_x_, _c2_0p_y_, _c2_0q_x_, _c2_0q_y_ = pl.col('r0s0_xi'),  pl.col('r0s0_yi'),  pl.col('corner_x'), pl.col('corner_y')
            _c2_1p_x_, _c2_1p_y_, _c2_1q_x_, _c2_1q_y_ = pl.col('corner_x'), pl.col('corner_y'), pl.col('r1s1_xi'),  pl.col('r1s1_yi')
            _c2_2p_x_, _c2_2p_y_, _c2_2q_x_, _c2_2q_y_ = pl.col('r1s1_xi'), pl.col('r1s1_yi'),   pl.col('x'),       pl.col('y')
            _c2_3p_x_, _c2_3p_y_, _c2_3q_x_, _c2_3q_y_ = pl.col('x'),       pl.col('y'),         pl.col('r0s0_xi'), pl.col('r0s0_yi')
            _c2_op_ = (((_c2_0p_x_*_c2_0q_y_ - _c2_0q_x_*_c2_0p_y_) + 
                        (_c2_1p_x_*_c2_1q_y_ - _c2_1q_x_*_c2_1p_y_) + 
                        (_c2_2p_x_*_c2_2q_y_ - _c2_2q_x_*_c2_2p_y_) +
                        (_c2_3p_x_*_c2_3q_y_ - _c2_3q_x_*_c2_3p_y_))/2.0).abs().alias('area')
            df = df.with_columns(pl.when(pl.col('r0s0_xi').is_not_null() & pl.col('r1s0_xi').is_not_null()).then(_c0_op_)
                                   .when(pl.col('r0s1_xi').is_not_null() & pl.col('r1s1_xi').is_not_null()).then(_c1_op_)
                                   .when(pl.col('r0s0_xi').is_not_null() & pl.col('r1s1_xi').is_not_null()).then(_c2_op_)
                                   .otherwise(pl.lit(None).alias('area')))
            self.time_lu['area_calc'] += (time.time() - t)

            #
            # With the sector sums, adjust the point based on the ratio of the sector area / sector density...
            # ... results of this iteration will be stored in the _xnext_ and _ynext_ fields of the dataframe
            # ... really important:  empty sectors need to be included in the calculation...
            #
            t = time.time()
            _diff_op_ = (pl.col('_w_ratio_') - pl.col('area')) # diff = sector_sum/weight_sum - sector_area/area_total # area_total == 1.0 since that was the normalization
            df_uv     = df.group_by(['__index__','x','y']).agg( (vector_scalar * _diff_op_ * pl.col('anchor_u')).sum().alias('_u_'),
                                                                (vector_scalar * _diff_op_ * pl.col('anchor_v')).sum().alias('_v_'))
            if decay_rate is not None and decay_rate < 1.0: vector_scalar *= decay_rate
            self.time_lu['sector_uv_summation'] += (time.time() - t)

            t = time.time()
            if static_points_exist:
                df_uv     = df_uv.join(df_orig.drop(['x', 'y', 'w', 'c']), on=['__index__'], how='left') # add the static info back in
                df_uv     = df_uv.with_columns(
                    pl.when(pl.col('s') == 0).then(pl.col('x') + pl.col('_u_')).otherwise(pl.col('x')).alias('x'), 
                    pl.when(pl.col('s') == 0).then(pl.col('y') + pl.col('_v_')).otherwise(pl.col('y')).alias('y')
                )
            else:
                df_uv     = df_uv.with_columns((pl.col('x') + pl.col('_u_')).alias('x'), 
                                               (pl.col('y') + pl.col('_v_')).alias('y'))
            df        = df_uv.join(df_orig, on=['__index__'], how='left') # add the weight back in
            self.time_lu['point_update'] += (time.time() - t)

        # Normalize the frame so that they are all of the same for the animation sequence
        df = df.drop(['_u_', '_v_', 'x_right', 'y_right'])
        x_range, y_range = pl.col('x').max() - pl.col('x').min(), pl.col('y').max() - pl.col('y').min()
        df = df.with_columns(pl.when(x_range == 0.0).then(0.5).otherwise(0.02 + 0.96 * (pl.col('x') - pl.col('x').min())/x_range).alias('x'),
                             pl.when(y_range == 0.0).then(0.5).otherwise(0.02 + 0.96 * (pl.col('y') - pl.col('y').min())/y_range).alias('y'))
        self.df_anim.append(df)

        # Save the final frame as the results
        self.df_results = df.sort(['__index__'])

    #
    # results() - return the results as xvals, yvals
    #
    def results(self): return self.df_results['x'], self.df_results['y']

    #
    # _repr_svg_() -- simple SVG representation of the results
    #
    def _repr_svg_(self):
        x0, y0, x1, y1 = self.df_results['x'].min(), self.df_results['y'].min(), self.df_results['x'].max(), self.df_results['y'].max()
        xperc, yperc   = (x1-x0)*0.01, (y1-y0)*0.01
        x0, y0, x1, y1 = x0-xperc, y0-yperc, x1+xperc, y1+yperc
        svg = []
        svg.append(f'<svg x="0" y="0" width="256" height="256" viewBox="{x0} {y0} {x1-x0} {y1-y0}" xmlns="http://www.w3.org/2000/svg">')
        svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" x="0" y="0" fill="#ffffff" />')
        if 'c' in self.df_results.columns:
            for i in range(len(self.df_results)): svg.append(f'<circle cx="{self.df_results["x"][i]}" cy="{self.df_results["y"][i]}" r="{0.005}" fill="{self.df_results["c"][i]}" />')
        else:
            for i in range(len(self.df_results)): svg.append(f'<circle cx="{self.df_results["x"][i]}" cy="{self.df_results["y"][i]}" r="{0.005}" fill="#404040" />')
        svg.append('</svg>')
        return ''.join(svg)


    #
    # svgAnimation() - produce an SVG animation of the iterations
    #
    def svgAnimation(self, duration='10s', w=256, h=256, r=0.005):
        df = self.df_anim[0]
        x_cols = [f'x{i}' for i in range(0, len(self.df_anim))]
        y_cols = [f'y{i}' for i in range(0, len(self.df_anim))]
        x_cols.extend(x_cols[::-1]), y_cols.extend(y_cols[::-1])
        for i in range(1, len(self.df_anim)):
            df = df.join(self.df_anim[i].drop(['s','w','c']), on=['__index__']) \
                   .rename({'x_right':f'x{i}', 'y_right':f'y{i}'})
        df = df.rename({'x':'x0', 'y':'y0'})

        df = df.with_columns(pl.concat_str(x_cols, separator=';').alias('x_values_str'), 
                             pl.concat_str(y_cols, separator=';').alias('y_values_str'))

        _str_ops_ = [pl.lit(f'<circle r="{r}" fill="'), 
                     pl.col('c'), 
                     pl.lit('">'),
                     pl.lit('<animate attributeName="cx" values="'),
                     pl.col('x_values_str'),
                     pl.lit(f'" begin="0s" dur="{duration}" repeatCount="indefinite" />'),
                     pl.lit('<animate attributeName="cy" values="'),
                     pl.col('y_values_str'),
                     pl.lit(f'" begin="0s" dur="{duration}" repeatCount="indefinite" />'),
                     pl.lit('</circle>')]

        svg = []
        svg.append(f'<svg x="0" y="0" width="{w}" height="{h}" viewBox="0 0 1 1" xmlns="http://www.w3.org/2000/svg">')
        svg.append('<rect x="0.0" y="0.0" width="1.0" height="1.0" x="0" y="0" fill="#ffffff" />')

        df = df.with_columns(pl.concat_str(*_str_ops_, separator='').alias('svg'))
        svg.extend(df['svg'])

        svg.append('</svg>')
        return ''.join(svg)

    #
    # tileBounds()
    #
    def tileBounds(self, xi, yi):
        x0, x1 = xi/float(self.num_of_tiles), (xi+1)/float(self.num_of_tiles)
        y0, y1 = yi/float(self.num_of_tiles), (yi+1)/float(self.num_of_tiles)
        return (x0, y0, x1, y1)

    #
    # createXoYoDataframeFile() -- create the offset tiles dataframe
    # ... this is created ahead of time & saved off
    # ... it can be used for any values of the data
    # ... generation time is over 24m on 7900x 96gb
    #
    def createXoYoDataframeFile(self):

        def rayIntersectsSegment(xy_ray, uv_ray, xy0_segment, xy1_segment, include_xy1_endpoint=False, epsilon=1e-9):
            x_r,  y_r  = xy_ray
            dx_r, dy_r = uv_ray
            x0,   y0   = xy0_segment
            x1,   y1   = xy1_segment    
            # Segment direction vector
            dx_s, dy_s = x1 - x0, y1 - y0
            # Compute determinant
            det = -dx_r * dy_s + dy_r * dx_s
            if abs(det) < 1e-10: return None # Lines are parallel or collinear
            # Compute parameters t and u
            t = ((x_r - x0) * dy_s - (y_r - y0) * dx_s) / det
            u = ((x_r - x0) * dy_r - (y_r - y0) * dx_r) / det
            # Check if intersection is valid (t >= 0 for ray, 0 <= u <= 1 for segment)
            if t >= 0.0:
                if include_xy1_endpoint:
                    if 0.0 <= u <= 1.0+epsilon: return (x_r + t * dx_r, y_r + t * dy_r)
                else:
                    if 0.0 <= u <  1.0-epsilon: return (x_r + t * dx_r, y_r + t * dy_r)
            return None

        angles = []
        for i in range(16): angles.append(i*2*pi/16)

        dfs = []
        for num_of_tiles in [16, 32, 64, 128, 256, 512, 1024]:
            #
            # Prepare the xo/yo dataframe
            #
            _iota_ = 10e-6
            # Tile (xi,yi) --> (x0, y0, x1, y1) where the bounds are [x0,x1) and [y0,y1)
            tile_to_rect = {}
            for xi in range(num_of_tiles):
                x0, x1 = xi/float(num_of_tiles), (xi+1)/float(num_of_tiles)
                for yi in range(num_of_tiles):
                    y0, y1 = yi/float(num_of_tiles), (yi+1)/float(num_of_tiles)
                    tile_to_rect[(xi,yi)] = (x0, y0, x1, y1)
            # Determine which tiles (xo,yo) need to be checked for sector comparisons the hard way
            offtiles_intersected_by_rays = set()
            offtiles_to_sectors          = {}
            offtiles_to_uvs              = {}
            for _tile_ in [(0,0), (0, num_of_tiles-1), (num_of_tiles-1, num_of_tiles-1), (num_of_tiles-1, 0)]:
                xi,  yi  = _tile_
                x0,  y0, x1, y1 = tile_to_rect[_tile_]
                xm,  ym         = (x0+x1)/2.0, (y0+y1)/2.0
                _positions_ = [(x0+_iota_, y0+_iota_),(xm, y0+_iota_),(x1-_iota_, y0+_iota_),
                               (x0+_iota_, ym),       (xm, ym),       (x1-_iota_, ym),
                               (x0+_iota_, y1-_iota_),(xm, y1-_iota_),(x1-_iota_, y1-_iota_)]
                for _position_ in _positions_:
                    xpt, ypt = _position_
                    for _sector_ in range(16):
                        a0, a1         = angles[_sector_], angles[(_sector_+1)%16]
                        u0, v0, u1, v1 = cos(a0), sin(a0), cos(a1), sin(a1)
                        for _tile_ in tile_to_rect:
                            x0, y0, x1, y1 = tile_to_rect[_tile_]
                            _xo_, _yo_ = _tile_[0]-xi, _tile_[1]-yi
                            k          = (_xo_, _yo_)

                            if rayIntersectsSegment((xpt, ypt), (u0, v0), (x0, y0), (x0, y1)) or rayIntersectsSegment((xpt, ypt), (u0, v0), (x1, y0), (x1, y1)) or \
                               rayIntersectsSegment((xpt, ypt), (u0, v0), (x0, y0), (x1, y0)) or rayIntersectsSegment((xpt, ypt), (u0, v0), (x0, y1), (x1, y1)):
                                offtiles_intersected_by_rays.add(k)
                                if k not in offtiles_to_sectors: offtiles_to_sectors[k], offtiles_to_uvs[k] = set(), set()
                                offtiles_to_uvs[k].add((u0, v0))
                                offtiles_to_sectors[k].add((int(16*(atan2(y0-ypt, x0-xpt)+pi)/(2*pi)) + 16)%16)
                                offtiles_to_sectors[k].add((int(16*(atan2(y0-ypt, x1-xpt)+pi)/(2*pi)) + 16)%16)
                                offtiles_to_sectors[k].add((int(16*(atan2(y1-ypt, x1-xpt)+pi)/(2*pi)) + 16)%16)
                                offtiles_to_sectors[k].add((int(16*(atan2(y1-ypt, x0-xpt)+pi)/(2*pi)) + 16)%16)

                            if rayIntersectsSegment((xpt, ypt), (u1, v1), (x0, y0), (x0, y1)) or rayIntersectsSegment((xpt, ypt), (u1, v1), (x1, y0), (x1, y1)) or \
                               rayIntersectsSegment((xpt, ypt), (u1, v1), (x0, y0), (x1, y0)) or rayIntersectsSegment((xpt, ypt), (u1, v1), (x0, y1), (x1, y1)):
                                offtiles_intersected_by_rays.add(k)
                                if k not in offtiles_to_sectors: offtiles_to_sectors[k], offtiles_to_uvs[k] = set(), set()
                                offtiles_to_uvs[k].add((u1, v1))
                                offtiles_to_sectors[k].add((int(16*(atan2(y0-ypt, x0-xpt)+pi)/(2*pi)) + 16)%16)
                                offtiles_to_sectors[k].add((int(16*(atan2(y0-ypt, x1-xpt)+pi)/(2*pi)) + 16)%16)
                                offtiles_to_sectors[k].add((int(16*(atan2(y1-ypt, x1-xpt)+pi)/(2*pi)) + 16)%16)
                                offtiles_to_sectors[k].add((int(16*(atan2(y1-ypt, x0-xpt)+pi)/(2*pi)) + 16)%16)

            # Determine the min and max x/y offsets
            xo_min, xo_max, yo_min, yo_max = 0, 0, 0, 0 # xi_min=-63 yi_min=-63 xi_max=63 yi_max=63 // for num_of_tiles = 64
            for _xyo_ in offtiles_intersected_by_rays:
                xo, yo = _xyo_
                xo_min, xo_max = min(xo, xo_min), max(xo, xo_max)
                yo_min, yo_max = min(yo, yo_min), max(yo, yo_max)
            xoyo_to_sector, sectors_seen = {}, set()
            # These tiles that are completely in a sector and do not touch the rays -- this determines the sector lookup
            for xo in range(xo_min, xo_max+1):
                for yo in range(yo_min, yo_max+1):
                    if (xo,yo) in offtiles_intersected_by_rays: continue
                    xm, ym   = xo, yo
                    _sector_ = int(16*(atan2(ym, xm)+pi)/(2*pi))
                    if _sector_ < 0: raise Polars2SVGError('This should not happen // _sector_ < 0')
                    xoyo_to_sector[(xo,yo)] = _sector_
                    sectors_seen.add(_sector_)
            # Transform into a dataframe
            _lu_ = {'xo':[], 'yo':[], 'sector':[], 'u':[], 'v':[], 'rsector':[], 'lsector':[]}
            # These are the tiles that intersect the rays ... and so have to be figured out on a per point basis
            for _xyo_ in offtiles_intersected_by_rays:
                xo, yo = _xyo_
                _lu_['xo'].append(xo), _lu_['yo'].append(yo), _lu_['sector'].append(-1)
                # This shouldn't happen -- if there's an intersection, it should be at least two sectors (if not more)
                if   _xyo_ in offtiles_to_sectors and len(offtiles_to_sectors[_xyo_]) == 1: 
                    raise Polars2SVGError('This should not happen // offtiles_to_sectors[_xyo_] == len(1)')
                # Two sectors intersected -- this can be used to determine the sector via the cross product
                elif _xyo_ in offtiles_to_sectors and len(offtiles_to_sectors[_xyo_]) == 2:
                    if len(offtiles_to_uvs[_xyo_]) != 1: # this shouldn't happen -- only one ray should have intersected if there are two sectors
                        raise Polars2SVGError(f'This should not happen // len(offtiles_to_uvs[_xyo_]) != 1 // {_xyo_} {offtiles_to_uvs[_xyo_]}')
                    _sectors_ = sorted(list(offtiles_to_sectors[_xyo_]))
                    if _sectors_[0] == 0 and _sectors_[1] == 15: _sector0_, _sector1_ = 15,           0
                    else:                                        _sector0_, _sector1_ = _sectors_[0], _sectors_[1]
                    _lu_['u'].append(list(offtiles_to_uvs[_xyo_])[0][0]), _lu_['v'].append(list(offtiles_to_uvs[_xyo_])[0][1]), _lu_['rsector'].append(_sector0_), _lu_['lsector'].append(_sector1_)
                # Otherwise, we'll need to use the arctangent
                else:
                    _lu_['u'].append(None), _lu_['v'].append(None), _lu_['rsector'].append(None), _lu_['lsector'].append(None)
            # These are the tiles that don't intersect the rays ... and therefore can be optimized to a single sector
            for _xyo_ in xoyo_to_sector:
                xo, yo   = _xyo_
                _sector_ = xoyo_to_sector[_xyo_]
                _lu_['xo'].append(xo), _lu_['yo'].append(yo), _lu_['sector'].append(_sector_)
                _lu_['u'].append(None), _lu_['v'].append(None), _lu_['rsector'].append(None), _lu_['lsector'].append(None)
            # Create the dataframe and add it to the list of dataframes (each added one cover a specific num_of_tiles)
            df_xoyo_sector = pl.DataFrame(_lu_).with_columns(pl.lit(num_of_tiles).alias('num_of_tiles'))
            dfs.append(df_xoyo_sector)
        # Concatenate the dataframes together & write out to a file for use when the class starts up
        df = pl.concat(dfs)
        df.write_parquet(self.xoyo_filename)
