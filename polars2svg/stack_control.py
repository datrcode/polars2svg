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


def stack_controli(component, stack_name='default', insets=(2, 2), hgap=4,
                   wxh=(160, 256), txt_h=10, **kwargs):
    w, h   = wxh
    wc, hc = component.wxh
    x0_val     = insets[0]
    inset_y_val = insets[1]
    _cls_ref_ = [None]
    _mlx_avail_, _cuda_avail_ = _mlxCudaStatus_()

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

        def _add_frame(df, y, is_selected, stack_idx):
            df_id = id(df)
            if df_id not in cache:
                cache[df_id] = component.render_with(df)._repr_svg_()
            tile_svg    = cache[df_id]
            label_color = p2s_ref.colorTyped('label', 'defaultfg') if is_selected else p2s_ref.colorTyped('label', 'inner')
            x_lbl = x0 + wc + 4
            mid   = y + hc // 2 + txt_h // 3
            if stack_idx == 0:
                sub = '(Base)'
            elif stack_idx == n - 1:
                sub = 'Top'
            else:
                sub = None
            if sub is not None:
                half = (txt_h + 2) // 2
                _svg_.append(p2s_ref.svgText(f'{p2s_ref.unitizeInt(len(df))} Rows',
                                             x_lbl, mid - half, txt_h=txt_h, color=label_color))
                _svg_.append(p2s_ref.svgText(sub,
                                             x_lbl, mid + half, txt_h=txt_h, color=label_color))
            else:
                _svg_.append(p2s_ref.svgText(f'{p2s_ref.unitizeInt(len(df))} Rows',
                                             x_lbl, mid, txt_h=txt_h, color=label_color))
            _svg_.append(f'<g transform="translate({x0}, {y})">')
            _svg_.append(tile_svg)
            _svg_.append('</g>')
            _svg_.append(f'<rect x="{x0}" y="{y}" width="{wc}" height="{hc}"'
                         f' fill="none" stroke="{_border_}" stroke-width="1"/>')
            frame_map.append((y, y + hc, stack_idx))

        def _add_skip_label(y, count):
            noun     = 'frame' if count == 1 else 'frames'
            label    = f'... {count} stack {noun} ...'
            cx       = x0 + wc // 2
            _skip_co_ = p2s_ref.colorTyped('label', 'inner')
            text_y   = y + ELL_H // 2 + txt_h // 3
            _svg_.append(
                f'<text x="{cx}" y="{text_y}" text-anchor="middle"'
                f' font-family="Helvetica,Arial,sans-serif" font-size="{txt_h}px"'
                f' font-style="italic" fill="{_skip_co_}">{label}</text>')

        _add_frame(dfs[0], y_base, index == 0, 0)

        if n == 1:
            pass

        elif (n - 1) * slot <= avail:
            for i in range(1, n):
                _add_frame(dfs[i], y_base - i * slot, i == index, i)

        else:
            _add_frame(dfs[n - 1], y_top, index == n - 1, n - 1)

            inner_space = y_base - (y_top + slot)

            def spiral_fill(start_sign, space_budget):
                tr = [False] * n
                tr[0] = tr[n - 1] = True
                sign, offset = start_sign, 0
                while space_budget >= slot:
                    idx = index + sign * offset
                    if sign == -start_sign:
                        sign = start_sign
                    else:
                        sign, offset = -start_sign, offset + 1
                    if 0 <= idx < n and not tr[idx]:
                        tr[idx] = True
                        space_budget -= slot
                return tr

            one_label_budget = inner_space - ELL_H
            two_label_budget = inner_space - 2 * ELL_H

            for render_case in range(3):
                if render_case == 0:
                    tr = spiral_fill(1, one_label_budget)
                    fits = tr[1]
                elif render_case == 1:
                    tr = spiral_fill(-1, one_label_budget)
                    fits = tr[n - 2]
                else:
                    tr = spiral_fill(-1, two_label_budget)
                    fits = True
                if fits:
                    to_render = tr
                    break

            if render_case == 0:
                # Cluster packed from base upward; single skip label near top
                y, i = y_base - slot, 1
                while to_render[i]:
                    _add_frame(dfs[i], y, i == index, i)
                    y -= slot
                    i += 1
                missing = sum(1 for j in range(i, n - 1) if not to_render[j])
                if missing > 0:
                    top_lbl_y = (y_top + hc + y + slot) // 2
                    _add_skip_label(top_lbl_y - ELL_H // 2, missing)

            elif render_case == 1:
                # Cluster packed from top downward; single skip label near base
                y, i = y_top + slot, n - 2
                while to_render[i]:
                    _add_frame(dfs[i], y, i == index, i)
                    y += slot
                    i -= 1
                missing = sum(1 for j in range(1, i + 1) if not to_render[j])
                if missing > 0:
                    bot_lbl_y = (y - gap + y_base) // 2
                    _add_skip_label(bot_lbl_y - ELL_H // 2, missing)

            else:
                # Cluster centered in middle space; skip labels above and below
                inner = [i for i in range(1, n - 1) if to_render[i]]
                if not inner:
                    # Budget too tight for even one middle frame: base and top with a
                    # single skip label standing in for everything between them.
                    _add_skip_label((y_top + hc + y_base) // 2 - ELL_H // 2, n - 2)
                else:
                    mid = (y_top + hc + gap + y_base) // 2
                    cluster_h = len(inner) * slot - gap
                    start_y = mid + cluster_h // 2 - hc
                    y = start_y
                    for i in inner:
                        _add_frame(dfs[i], y, i == index, i)
                        y -= slot
                    missing_top = (n - 2) - inner[-1]
                    if missing_top > 0:
                        top_lbl_y = (y_top + hc + y + slot) // 2
                        _add_skip_label(top_lbl_y - ELL_H // 2, missing_top)
                    missing_bot = inner[0] - 1
                    if missing_bot > 0:
                        bot_lbl_y = (start_y + hc + y_base) // 2
                        _add_skip_label(bot_lbl_y - ELL_H // 2, missing_bot)

        _svg_.append('</svg>')
        return ''.join(_svg_), frame_map

    # Pre-compute the initial SVG so it becomes the class-level param default.
    # Panel seeds the JS data object from class defaults, not __init__ instance
    # assignments, so the render script sees the real content on first paint.
    # Also pre-populate the tile cache for the base df so __init__ starts warm.
    _initial_cache_ = {}
    _initial_svg_, _initial_frame_map_ = _render_svg_content([component.df_orig], 0, _initial_cache_)

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
