import param
from panel.reactive import ReactiveHTML

from . import od_flow_layout as _ofl_


# _mlxCudaStatus_() - (mlx_available, cuda_available) for the header indicators below.
# Reuses od_flow_layout's cached GPU probe (mx is None when mlx isn't installed at all;
# _default_device() falls back to mx.cpu when the GPU backend probe fails) so this never
# runs a second, redundant device-resolution kernel.
def _mlxCudaStatus_():
    mx = _ofl_.mx
    if mx is None:
        return False, False
    if _ofl_._default_device() == mx.cpu:
        return True, False
    _metal_ = getattr(mx, 'metal', None)
    _is_metal_ = _metal_ is not None and _metal_.is_available()
    return True, not _is_metal_


# Indicator rows in the widget header, rendered top to bottom.  Adding a row here is
# the only edit needed:  the frame stack below reserves its space via headerHeight(),
# and the layout tests derive their component geometry from that same helper.
_INDICATOR_LABELS_ = ('MLX', 'CUDA')


def _statusTxtH_(txt_h):
    return max(6, txt_h - 2)


def headerHeight(txt_h=10):
    """Vertical space the indicator header claims from the frame stack's budget."""
    return len(_INDICATOR_LABELS_) * (_statusTxtH_(txt_h) + 4)


# _placement() - decide which stack indices get an icon and how many frames are
# hidden on each side, per the "Stack Control Requirements" worked out in
# polars2svg_prototyping/stack_control_shrinkage.ipynb:
#
#   1) the current index always renders (it is the mandatory baseline -- the
#      caller reserves one `hc` for it before computing `avail`)
#   2) base (index 0), then top (index n-1), are tried next -- but rule 5 only
#      fires "after rendering the current index and the base dataframe", so a
#      failed base attempt also skips top (and, transitively, outward growth):
#      the acceptable-rendering examples in the notebook never show top without
#      base, or an outward frame without both
#   3) with base and top settled, fill outward from the current index
#      (index-1, index+1, index-2, index+2, ...), each direction stopping
#      independently once it no longer fits
#   4) at most two "... N stack frame(s) ..." labels are ever needed -- one
#      below the contiguous current-index cluster, one above it -- because the
#      cluster only ever grows outward from a single seed
#
# `avail` is the vertical budget left after the mandatory current-index icon;
# `slot` is the marginal cost of one more icon (hc + hgap) and `ell_h` the
# fixed height of one skip label. Costs are tracked as budget deltas (icon
# cost minus any skip-label saved by newly closing a gap) so a merge that
# collapses a gap can be cheaper than its raw icon cost.
def _placement(n, index, avail, slot, ell_h):
    if n <= 1:
        return 0, 0, True, True, 0, 0

    def _gap_below_(lo, base_on):
        if lo == 0:
            return 0
        return (lo - 1) if base_on else lo

    def _gap_above_(hi, top_on):
        if hi == n - 1:
            return 0
        return (n - 2 - hi) if top_on else (n - 1 - hi)

    def _label_cost_(count):
        return ell_h if count > 0 else 0

    lo = hi = index
    base_on  = (index == 0)
    top_on   = (index == n - 1)
    budget   = avail - _label_cost_(_gap_below_(lo, base_on)) - _label_cost_(_gap_above_(hi, top_on))
    chain_ok = True

    if not base_on:
        before = _label_cost_(_gap_below_(lo, base_on))
        merges = (lo == 1)
        new_lo = 0 if merges else lo
        after  = _label_cost_(_gap_below_(new_lo, True))
        cost   = slot + after - before
        if cost <= budget:
            budget -= cost
            base_on, lo = True, new_lo
        else:
            chain_ok = False

    if chain_ok and not top_on:
        before = _label_cost_(_gap_above_(hi, top_on))
        merges = (hi == n - 2)
        new_hi = n - 1 if merges else hi
        after  = _label_cost_(_gap_above_(new_hi, True))
        cost   = slot + after - before
        if cost <= budget:
            budget -= cost
            top_on, hi = True, new_hi
        else:
            chain_ok = False

    if chain_ok:
        active_lo, active_hi = lo > 0, hi < n - 1
        go_low = True
        while active_lo or active_hi:
            if go_low and not active_lo:
                go_low = False
            elif not go_low and not active_hi:
                go_low = True
            if go_low:
                before  = _label_cost_(_gap_below_(lo, base_on))
                new_lo  = lo - 1
                merges  = (new_lo == 0)
                icon_co = 0 if (merges and base_on) else slot
                new_bon = base_on or merges
                after   = _label_cost_(_gap_below_(new_lo, new_bon))
                cost    = icon_co + after - before
                if cost <= budget:
                    budget -= cost
                    lo, base_on = new_lo, new_bon
                    active_lo = lo > 0
                else:
                    active_lo = False
            else:
                before  = _label_cost_(_gap_above_(hi, top_on))
                new_hi  = hi + 1
                merges  = (new_hi == n - 1)
                icon_co = 0 if (merges and top_on) else slot
                new_ton = top_on or merges
                after   = _label_cost_(_gap_above_(new_hi, new_ton))
                cost    = icon_co + after - before
                if cost <= budget:
                    budget -= cost
                    hi, top_on = new_hi, new_ton
                    active_hi = hi < n - 1
                else:
                    active_hi = False
            if not active_lo and not active_hi:
                break
            go_low = not go_low

    return lo, hi, base_on, top_on, _gap_below_(lo, base_on), _gap_above_(hi, top_on)


def stack_controli(component, stack_name='default', insets=(2, 2), hgap=4,
                   wxh=(160, 256), txt_h=10, **kwargs):
    w, h   = wxh
    wc, hc = component.wxh
    x0_val     = insets[0]
    inset_y_val = insets[1]
    _cls_ref_ = [None]
    _mlx_avail_, _cuda_avail_ = _mlxCudaStatus_()
    _row_txt_h_ = int(min(10, max(1, (hc - 4) / 2)))

    # Requirement (1): the widget needs room for the MLX/CUDA header, one icon
    # (the current index always renders), and two skip labels -- anything less
    # can never satisfy "always render the current index" alongside rule (7)'s
    # up-to-two labels, so refuse to build it rather than silently clipping.
    _ell_h_min_ = 2 * hgap + txt_h
    _min_h_ = 2 * inset_y_val + headerHeight(txt_h) + hc + 2 * _ell_h_min_
    if h < _min_h_:
        raise ValueError(
            f'stack_controli: wxh height {h} is too small (need >= {_min_h_}) for the '
            f'MLX/CUDA header, one icon, and two "... N stack frame(s) ..." labels')

    def _render_svg_content(dfs, index, cache):
        p2s_ref = component.p2s
        x0      = x0_val
        inset_y = inset_y_val
        gap     = hgap
        slot    = hc + gap
        n       = len(dfs)
        frame_map = []  # list of (y_top, y_bot, stack_idx)

        _bg_     = p2s_ref.colorTyped('background', 'default')
        _border_ = p2s_ref.colorTyped('axis', 'inner')
        _svg_ = [f'<svg width="{w}" height="{h}">',
                 f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_bg_}"/>']

        # MLX / CUDA availability header — one small row each, faded green when the
        # feature is usable, else a gray close to the background (visible but muted).
        _status_txt_h_ = _statusTxtH_(txt_h)
        _status_row_h_ = _status_txt_h_ + 4
        _header_h_     = headerHeight(txt_h)
        _avail_co_     = p2s_ref.colorTyped('indicator', 'available')
        _unavail_co_   = p2s_ref.colorTyped('indicator', 'unavailable')
        for _i_, (_label_, _ok_) in enumerate(zip(_INDICATOR_LABELS_, (_mlx_avail_, _cuda_avail_))):
            _svg_.append(p2s_ref.svgText(f'{_label_}: {"available" if _ok_ else "unavailable"}',
                                         x0, inset_y + _i_ * _status_row_h_ + _status_txt_h_,
                                         txt_h=_status_txt_h_,
                                         color=_avail_co_ if _ok_ else _unavail_co_))

        ELL_H  = 2 * gap + txt_h
        y_base = h - inset_y - hc
        y_top  = inset_y + _header_h_
        avail  = y_base - y_top

        def _add_frame(df, y, stack_idx):
            df_id = id(df)
            if df_id not in cache:
                cache[df_id] = component.render_with(df)._repr_svg_()
            tile_svg    = cache[df_id]
            is_selected = (stack_idx == index)
            label_color = p2s_ref.colorTyped('label', 'defaultfg') if is_selected else p2s_ref.colorTyped('label', 'inner')
            x_lbl = x0 + wc + 4
            mid   = y + hc // 2
            half  = (_row_txt_h_ + 2) // 2
            if n == 1 or stack_idx == 0:
                pos_str = 'base'
            elif stack_idx == n - 1:
                pos_str = 'top'
            else:
                pos_str = f'{stack_idx + 1} of {n}'
            _svg_.append(p2s_ref.svgText(f'{p2s_ref.unitizeInt(len(df))} Rows',
                                         x_lbl, mid - half, txt_h=_row_txt_h_, color=label_color))
            _svg_.append(p2s_ref.svgText(pos_str,
                                         x_lbl, mid + half, txt_h=_row_txt_h_, color=label_color))
            _svg_.append(f'<g transform="translate({x0}, {y})">')
            _svg_.append(tile_svg)
            _svg_.append('</g>')
            _svg_.append(f'<rect x="{x0}" y="{y}" width="{wc}" height="{hc}"'
                         f' fill="none" stroke="{_border_}" stroke-width="1"/>')
            frame_map.append((y, y + hc, stack_idx))

        def _add_skip_label(y, count):
            noun     = 'frame' if count == 1 else 'frames'
            label    = f'... {count} stack {noun} ...'
            cx       = w / 2   # centered on the full widget width so it can't clip off the left edge
            _skip_co_ = p2s_ref.colorTyped('label', 'inner')
            text_y   = y + ELL_H / 2 + txt_h / 3
            _svg_.append(
                f'<text x="{cx}" y="{text_y}" text-anchor="middle"'
                f' font-family="Helvetica,Arial,sans-serif" font-size="{txt_h}px"'
                f' font-style="italic" fill="{_skip_co_}">{label}</text>')

        lo, hi, base_on, top_on, gap_below, gap_above = _placement(n, index, avail, slot, ELL_H)

        if lo == 0 and hi == n - 1:
            # Fully contiguous: no skip labels needed.
            for i in range(lo, hi + 1):
                _add_frame(dfs[i], y_base - (i - lo) * slot, i)

        elif lo == 0:
            # Cluster anchored at the base, growing upward; top is either an
            # isolated icon above a gap, or simply out of reach.
            for i in range(lo, hi + 1):
                _add_frame(dfs[i], y_base - (i - lo) * slot, i)
            cluster_top = y_base - (hi - lo) * slot
            region_top  = y_top
            if top_on:
                _add_frame(dfs[n - 1], y_top, n - 1)
                region_top = y_top + hc
            if gap_above > 0:
                _add_skip_label((region_top + cluster_top) / 2 - ELL_H / 2, gap_above)

        elif hi == n - 1:
            # Cluster anchored at the top, growing downward; base is either an
            # isolated icon below a gap, or simply out of reach.
            for i in range(hi, lo - 1, -1):
                _add_frame(dfs[i], y_top + (hi - i) * slot, i)
            cluster_bot   = y_top + (hi - lo) * slot + hc
            region_bottom = y_base + hc
            if base_on:
                _add_frame(dfs[0], y_base, 0)
                region_bottom = y_base
            if gap_below > 0:
                _add_skip_label((cluster_bot + region_bottom) / 2 - ELL_H / 2, gap_below)

        else:
            # Cluster floats free of both ends: draw any isolated base/top,
            # then center the [skip?][cluster][skip?] block in the space
            # between them (or between the header and the floor, if neither
            # base nor top made it in).
            if base_on:
                _add_frame(dfs[0], y_base, 0)
            if top_on:
                _add_frame(dfs[n - 1], y_top, n - 1)

            inner_top = (y_top + hc + gap) if top_on else y_top
            inner_bot = (y_base - gap) if base_on else (y_base + hc)
            cluster_n = hi - lo + 1
            block_items = cluster_n + (1 if gap_below > 0 else 0) + (1 if gap_above > 0 else 0)
            block_h = (cluster_n * hc + (ELL_H if gap_below > 0 else 0) + (ELL_H if gap_above > 0 else 0)
                      + gap * max(0, block_items - 1))
            block_bottom = (inner_top + inner_bot) / 2 + block_h / 2

            y = block_bottom
            if gap_below > 0:
                y -= ELL_H
                _add_skip_label(y, gap_below)
                y -= gap
            for i in range(lo, hi + 1):
                y -= hc
                _add_frame(dfs[i], y, i)
                y -= gap
            if gap_above > 0:
                y -= ELL_H
                _add_skip_label(y, gap_above)

        _svg_.append('</svg>')
        return ''.join(_svg_), frame_map

    # Pre-compute the initial SVG so it becomes the class-level param default.
    # Panel seeds the JS data object from class defaults, not __init__ instance
    # assignments, so the render script sees the real content on first paint —
    # if mvc already carries a populated stack, the default must reflect it too,
    # or the browser paints just the base frame no matter what __init__ assigns.
    # Also pre-populate the tile cache for the base df so __init__ starts warm.
    _initial_mvc_ = kwargs.get('mvc')
    if _initial_mvc_ is not None and stack_name in _initial_mvc_.stacks:
        _initial_stack_ = _initial_mvc_.stacks[stack_name]
        _initial_dfs_, _initial_index_ = _initial_stack_['dfs'], _initial_stack_['index']
    else:
        _initial_dfs_, _initial_index_ = [component.df_orig], 0
    _initial_cache_ = {}
    _initial_svg_, _initial_frame_map_ = _render_svg_content(_initial_dfs_, _initial_index_, _initial_cache_)

    def __init__(self, **kwargs):
        mvc = kwargs.pop('mvc', None)
        super(_cls_ref_[0], self).__init__(**kwargs)
        self.mvc         = mvc
        self._frame_map_ = list(_initial_frame_map_)
        self._svg_cache_ = dict(_initial_cache_)
        if mvc is not None:
            mvc.view_stack[id(self)] = stack_name
            mvc.view_refs[id(self)]  = self
            if stack_name in mvc.stacks:
                s = mvc.stacks[stack_name]
                svg, fm          = _render_svg_content(s['dfs'], s['index'], self._svg_cache_)
                self.mod_inner   = svg
                self._frame_map_ = fm
        self.param.watch(self.applyClickOp, 'click_op_finished')

    async def applyClickOp(self, event):
        cy  = self.click_y
        idx = None
        for (y_top, y_bot, stack_idx) in self._frame_map_:
            if y_top <= cy < y_bot:
                idx = stack_idx
                break
        if idx is None or self.mvc is None:
            return
        sn = self.mvc.view_stack.get(id(self))
        if sn is None:
            return
        s = self.mvc.stacks.get(sn)
        if s is None or idx < 0 or idx >= len(s['dfs']):
            return
        s['index'] = idx
        df = s['dfs'][idx]
        for vid, vsn in self.mvc.view_stack.items():
            if vsn == sn:
                view = self.mvc.view_refs.get(vid)
                if view is not None:
                    await view.display(df, s['dfs'], idx)

    async def display(self, df, dfs, dfs_index):
        if df is not dfs[dfs_index]:
            return
        svg, fm          = _render_svg_content(dfs, dfs_index, self._svg_cache_)
        self.mod_inner   = svg
        self._frame_map_ = fm
        _ids_ = {id(d) for d in dfs}
        for _id_ in list(self._svg_cache_.keys()):
            if _id_ not in _ids_:
                del self._svg_cache_[_id_]

    def sketchHtml(self, use_webgpu=False):
        # Static snapshot for panelizeSketch(): the current stack frame already
        # lives in mod_inner as a complete <svg>...</svg>, so reuse it verbatim.
        return self.mod_inner or None

    cls = type('STACKCONTROLI', (ReactiveHTML,), {
        'mod_inner':         param.String(default=_initial_svg_),
        'click_y':           param.Integer(default=0),
        'click_op_finished': param.Boolean(default=False),
        'wxh':               wxh,
        '_template': (
            f'<svg id="svgstackcontrol" width="{w}" height="{h}">'
            f'<svg id="mod" width="{w}" height="{h}">${{mod_inner}}</svg>'
            f'<rect id="screen" x="0" y="0" width="{w}" height="{h}" opacity="0"'
            f' style="cursor:pointer;" onclick="${{script(\'myOnClick\')}}" />'
            f'</svg>'
        ),
        '_scripts': {
            'render':    'mod.innerHTML = data.mod_inner;',
            'mod_inner': 'mod.innerHTML = data.mod_inner;',
            'myOnClick': 'data.click_y = Math.round(event.offsetY); data.click_op_finished = !data.click_op_finished;',
        },
        '__init__':     __init__,
        'display':      display,
        'applyClickOp': applyClickOp,
        'sketchHtml':   sketchHtml,
    })
    _cls_ref_[0] = cls
    return cls(**kwargs)
