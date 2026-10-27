import asyncio
import json
import time
from typing import Any
import polars as pl
import param
from panel.custom import JSComponent

from .p2s_esm import esm

from . import od_flow_layout as _ofl_
from .interactive_controller import SelectionAPIMixin, _helpSubRow_, _keyboardHelpSvg_
from .stack_shelf import _MAX_SHELF_


# _mlxCudaStatus_() - (mlx_available, cuda_available) for the header indicators below.
# Reuses od_flow_layout's cached GPU probe (mx is None when mlx isn't installed at all;
# _default_device() falls back to mx.cpu when the GPU backend probe fails) so this never
# runs a second, redundant device-resolution kernel.
def _mlxCudaStatus_() -> tuple:
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


def _statusTxtH_(txt_h: int) -> int:
    return max(6, txt_h - 2)


def headerHeight(txt_h: int = 10) -> int:
    """Vertical space the header claims from the frame stack's budget: the indicator
    rows, then the status row that says why an operation was refused (F18, Q6)."""
    return (len(_INDICATOR_LABELS_) + 1) * (_statusTxtH_(txt_h) + 4)


# The shelf (PLANNING.md §7 F18) sits between the header and the frame stack: one
# text-only row per slot -- name, row count, trashcan -- and a rule under the last.
def _shelfRowH_(txt_h: int) -> int:
    return _statusTxtH_(txt_h) + 6


def shelfHeight(txt_h: int = 10, rows: int = _MAX_SHELF_, gap: int = 4) -> int:
    """Vertical space a shelf of ``rows`` slots claims from the frame stack's budget."""
    return rows * _shelfRowH_(txt_h) + gap if rows > 0 else 0


# Every row's trashcan sits in a strip this wide at the right edge, so a click is on a
# trashcan by x alone.
_TC_W_ = 12

# A shelf trashcan deletes on a second click within this many seconds (Q14).
_CONFIRM_S_ = 2.0

# The click's modifier keys (sent as 'c' / 's' / 'cs') -> the set operation, following
# linkpi's [[p(cs)^2]] convention: ctrl adds, shift removes, ctrl-shift intersects.
_MODIFIER_OPS_ = {'c': 'union', 's': 'subtract', 'cs': 'intersect'}


def _trashcanSvg_(x: float, y: float, color: str) -> str:
    # 9 x 10 px: lid, handle, tapered bin.
    return (f'<g transform="translate({x:.1f},{y:.1f})" fill="none" stroke="{color}" stroke-width="1">'
            f'<path d="M0.5 2 H8.5 M3 2 V0.5 H6 V2 M1.5 2 L2.2 9.5 H6.8 L7.5 2"/></g>')


def _defaultShelfName_(taken: list[str]) -> str:
    # What the naming dialog pre-fills: the first 'frame N' not on the shelf.
    _n_ = 1
    while f'frame {_n_}' in taken:
        _n_ += 1
    return f'frame {_n_}'


def _fitText_(p2s_ref: Any, txt: str, txt_h: float, w: float) -> str:
    # The longest prefix that fits with an ellipsis; cropText() can overshoot its width.
    if p2s_ref.textLength(txt, txt_h) <= w:
        return txt
    while txt and p2s_ref.textLength(txt + '\u2026', txt_h) > w:
        txt = txt[:-1]
    return txt + '\u2026' if txt else ''


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
def _placement(n: int, index: int, avail: int, slot: int, ell_h: int) -> tuple:
    if n <= 1:
        return 0, 0, True, True, 0, 0

    def _gap_below_(lo: int, base_on: bool) -> int:
        if lo == 0:
            return 0
        return (lo - 1) if base_on else lo

    def _gap_above_(hi: int, top_on: bool) -> int:
        if hi == n - 1:
            return 0
        return (n - 2 - hi) if top_on else (n - 1 - hi)

    def _label_cost_(count: int) -> int:
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


# Keyboard help overlay — same look & feel as the other interactive components'
# help menus (translucent gray rounded box, Courier text, toggled by 'h'); see
# interactive_controller._keyboard_help_svg_ for the sibling.  Kept concise so
# the box stays legible on the narrow stack-control widget; the root <svg> is
# also given overflow:visible so a wider box is never clipped.
# The six other views' help builder and sub-row rule, so this table lines up the way theirs
# do: ctrl-shift-c is a sub-row of c rather than a wider key cell, and every space survives
# SVG's whitespace collapsing (PLANNING.md §5 C-stack-help-key-column).
# The mouse rows (F18) take a 'clk' key cell, so every row's first '|' still lines up.
_STACK_KEYBOARD_COMMANDS_ = '\n'.join([
    'h . | toggle help',
    'c . | keep base + current',
    _helpSubRow_('ctrl-shift-c', 'current -> base'),
    's . | shelve current: type a name, enter',
    'clk | a frame: show it; a shelf entry: push it',
    _helpSubRow_('ctrl-click',  'current + it (union)'),
    _helpSubRow_('shift-click', 'current - it (subtract)'),
    _helpSubRow_('ctrl-shift',  'current & it (intersect)'),
    _helpSubRow_('trashcan',    'delete it (on the shelf: click twice)'),
])

_STACK_HELP_SVG_ = _keyboardHelpSvg_(_STACK_KEYBOARD_COMMANDS_)


# ─────────────────────────────────────────────────────────────────────────────
# The ESM module
#
# The help box is wider than the (narrow) widget, so the root <svg> needs
# overflow:visible for it to show in full when open.  That means the help must
# be *hidden by not rendering it* (display:none) rather than parked off-screen —
# an off-screen copy would spill past the widget's left edge into the
# neighbouring component instead of being clipped away.
#
# Sizes come from `model.svg_w` / `model.svg_h` at render time, where the template used
# jinja.  _STACK_HELP_SVG_ is no longer concatenated into markup here -- there is no
# markup to concatenate it into -- so it travels as the `kbd_help_svg` param instead.
# ─────────────────────────────────────────────────────────────────────────────

# The browser half lives in polars2svg/js/p2s_stack_control.js (PLANNING.md W1).  No GPU
# variant: this widget has no plot to rasterise.
_STACK_CONTROL_ESM_ = esm('fragments/p2s_dom.js',
                          'p2s_stack_control.js')


class STACKCONTROLI(SelectionAPIMixin, JSComponent):
    """Stack navigator widget.

    One static class for every widget size: ``svg_w`` / ``svg_h`` are params the ESM
    module reads when it builds the DOM, so this no longer needs to be built per call
    with ``type()``.  The frame renderer stays a closure built by
    :func:`stack_controli` (it carries the geometry settled there) and is handed
    to the instance rather than captured by a generated class.
    """

    svg_w:             Any = param.Integer(default=0)
    svg_h:             Any = param.Integer(default=0)
    mod_inner:         Any = param.String(default='')
    click_x:           Any = param.Integer(default=0)
    click_y:           Any = param.Integer(default=0)
    # The click's modifiers: '' / 'c' / 's' / 'cs'.  Read through _MODIFIER_OPS_, so a
    # value it does not know is a plain click.
    click_mods:        Any = param.String(default='')
    click_op_finished: Any = param.Boolean(default=False)
    # [[x0, y0, x1, y1, text], ...] as JSON: the full text of every name the render cut
    # off, which the browser shows on hover.  A <title> cannot do it -- the click screen
    # covers the whole widget.
    hover_regions:     Any = param.String(default='[]')
    key_op_finished:   Any = param.String(default='')
    # The naming dialog (`s`, F18 / Q11).  shelf_name is what was typed, written by the
    # browser just before key_op_finished = 'shelve', and only ever read through
    # cleanShelfName().  shelve_blocked is why the displayed frame cannot be shelved
    # whatever it is called ('' when it can), so the dialog does not open only to be
    # refused; shelf_default_name pre-fills it; shelf_dialog_box is [x, y, w, h] as
    # JSON, the first free shelf row -- where the new entry will be drawn, so the
    # dialog covers no existing name ('' when there is no shelf).
    shelf_name:         Any = param.String(default='')
    shelve_blocked:     Any = param.String(default='not on a stack yet')
    shelf_default_name: Any = param.String(default='frame 1')
    shelf_dialog_box:   Any = param.String(default='')
    help_display:      Any = param.String(default='none')
    # The keyboard-help overlay markup.  It was concatenated straight into the template
    # before; it is a param now because the ESM module has no markup of its own to
    # concatenate it into.  Same bytes on the wire either way -- the template shipped per
    # view too -- and it keeps the SVG out of the JavaScript.
    kbd_help_svg:      Any = param.String(default=_STACK_HELP_SVG_)

    _esm = _STACK_CONTROL_ESM_

    def __init__(self, wxh: tuple, stack_name: str, render_svg_content: Any,
                 initial_svg: str, initial_frame_map: list, initial_cache: dict,
                 max_shelf: int = _MAX_SHELF_, tc_x0: int = 0,
                 initial_shelf_map: list | None = None,
                 shelf_row0: tuple | None = None, **kwargs: Any) -> None:
        mvc = kwargs.pop('mvc', None)
        # mod_inner goes through the constructor like any other param.  Under
        # ReactiveHTML it could not: bound as a content ${mod_inner} it was a *child*,
        # and a _init_params() override existed to seed it for first paint.
        super().__init__(svg_w=wxh[0], svg_h=wxh[1], mod_inner=initial_svg, **kwargs)
        self.wxh                  = wxh       # read by _sketch_placeholder_html()
        self._stack_name_         = stack_name
        self._render_svg_content_ = render_svg_content
        # The shelf slots this widget draws; panelize() copies it onto the controller's
        # max_shelf, because the widget is built before the controller exists.
        self.max_shelf            = max_shelf
        self._tc_x0_              = tc_x0     # a click at x >= this is on a trashcan
        # [x, y, w, h] of the first shelf row; row i is h * i further down.
        self._shelf_row0_         = shelf_row0
        self._placeDialog_(0)
        self.mvc                  = mvc
        self._frame_map_          = list(initial_frame_map)
        self._shelf_map_: list    = list(initial_shelf_map or [])   # [(y_top, y_bot, name)]
        self._svg_cache_          = dict(initial_cache)
        self._status_             = ''        # why the last operation was refused
        self._armed_: tuple[str, float] | None = None   # a shelf trashcan clicked once
        if mvc is not None:
            mvc.view_stack[id(self)] = stack_name
            mvc.view_refs[id(self)]  = self
            mvc.max_shelf            = max_shelf
            self.refreshShelf()
        self.param.watch(self.applyClickOp, 'click_op_finished')
        self.param.watch(self.applyKeyOp,   'key_op_finished')

    # _stackRef_() -- (stack name, stack) this widget draws, or None when it has no
    # controller yet (panelize() attaches one after construction).
    def _stackRef_(self) -> tuple[str, dict] | None:
        if self.mvc is None:
            return None
        sn = self.mvc.view_stack.get(id(self), self._stack_name_)
        s  = self.mvc.stacks.get(sn)
        return None if s is None or not s['dfs'] else (sn, s)

    def _render_(self, dfs: list, index: int) -> None:
        _ref_ = self._stackRef_()
        if _ref_ is not None:
            sn     = _ref_[0]
            shelf  = [(_nm_, len(_d_), self.mvc.entryRefusal(sn, _d_) is None)
                      for _nm_, _d_ in self.mvc.shelfEntries(sn)]
            labels = [self.mvc.frameLabel(sn, _d_) for _d_ in dfs]
            # A name no entry can have, so only the name-independent refusals remain.
            self.shelve_blocked     = self.mvc.shelveRefusal(sn, '\x00', dfs[index]) or ''
            self.shelf_default_name = _defaultShelfName_([_e_[0] for _e_ in shelf])
            self._placeDialog_(len(shelf))
        else:
            shelf, labels = [], [None] * len(dfs)
        _armed_ = self._armed_[0] if self._armed_ is not None else None
        svg, fm, shm, hov  = self._render_svg_content_(dfs, index, self._svg_cache_, shelf=shelf,
                                                        labels=labels, status=self._status_,
                                                        armed=_armed_)
        self.mod_inner     = svg
        self._frame_map_   = fm
        self._shelf_map_   = shm
        self.hover_regions = json.dumps(hov)

    # The dialog goes on the row the new entry will take: the first free one.  With more
    # entries than this widget draws, that is the last row, the "... N more" one.
    def _placeDialog_(self, n_entries: int) -> None:
        if self._shelf_row0_ is None or self.max_shelf <= 0:
            self.shelf_dialog_box = ''
            return
        _x_, _y_, _w_, _h_ = self._shelf_row0_
        self.shelf_dialog_box = json.dumps([_x_, _y_ + min(n_entries, self.max_shelf - 1) * _h_, _w_, _h_])

    def refreshShelf(self) -> None:
        """Re-render from the controller's current stack and shelf -- after a shelf
        change, a refusal, or an armed trashcan timing out."""
        _ref_ = self._stackRef_()
        if _ref_ is not None:
            self._render_(_ref_[1]['dfs'], _ref_[1]['index'])

    # _broadcastStackChange() — push the (already mutated) stack out to every view
    # registered on the same stack name, so peers re-render in lock-step.  Shared
    # by the click handler and the keyboard ops below.
    async def _broadcastStackChange(self, sn: str, df: pl.DataFrame, dfs: list, index: int) -> None:
        for vid, vsn in self.mvc.view_stack.items():
            if vsn == sn:
                view = self.mvc.view_refs.get(vid)
                if view is not None:
                    await view.display(df, dfs, index)

    # applyClickOp() -- a click, hit-tested here: a shelf row or a stack frame by y, its
    # trashcan by x, and the modifiers pick the set operation (PLANNING.md §7 F18).
    # Modifiers are ignored on a trashcan.  A refusal's reason goes on the status row.
    async def applyClickOp(self, event: Any) -> None:
        _ref_ = self._stackRef_()
        if _ref_ is None:
            return
        sn, s  = _ref_
        cx, cy = self.click_x, self.click_y
        on_tc  = cx >= self._tc_x0_
        op     = _MODIFIER_OPS_.get(self.click_mods)
        name   = next((_nm_ for (_y0_, _y1_, _nm_) in self._shelf_map_ if _y0_ <= cy < _y1_), None)
        idx    = next((_i_ for (_y0_, _y1_, _i_) in self._frame_map_ if _y0_ <= cy < _y1_), None)
        if name is None and idx is None:
            return
        if not (on_tc and name is not None):
            self._armed_ = None
        self._status_ = ''
        if name is not None:
            why = (await self._shelfTrashcan_(sn, name)) if on_tc else \
                  (await self.mvc.useShelfEntry(sn, name, op))
        elif on_tc and idx != s['index']:
            why = await self.mvc.deleteFrame(sn, idx)
        else:
            why = await self.mvc.useStackFrame(sn, idx, op)
        self._status_ = why or ''
        self.refreshShelf()

    # A shelf entry is the frame the user chose to keep, so its trashcan asks twice
    # (Q14): the first click arms it (drawn red) and the second, within _CONFIRM_S_,
    # deletes.  Stack trashcans delete at once.
    async def _shelfTrashcan_(self, sn: str, name: str) -> str | None:
        _now_ = time.monotonic()
        if self._armed_ is not None and self._armed_[0] == name and _now_ - self._armed_[1] <= _CONFIRM_S_:
            self._armed_ = None
            return await self.mvc.deleteShelfEntry(sn, name) or None
        self._armed_ = (name, _now_)
        try:
            asyncio.get_running_loop().call_later(_CONFIRM_S_, self._disarm_, self._armed_)
        except RuntimeError:          # no loop: nothing will redraw it, but the time check holds
            pass
        return 'click again to delete'

    def _disarm_(self, armed: tuple[str, float]) -> None:
        if self._armed_ == armed:
            self._armed_ = None
            if self._status_ == 'click again to delete':
                self._status_ = ''
            self.refreshShelf()

    # applyKeyOp() — keyboard shortcuts routed from myOnKeyDown.  The 'h' help
    # toggle lives entirely in JS (it just flips help_display); only the two
    # stack-collapsing ops need Python:
    #   'collapse' (c)            -> drop every frame except the base and the
    #                                currently visible one, leaving [base, current]
    #                                (or just [base] when the base itself is visible)
    #   'rebase'   (ctrl+shift+c) -> discard the whole stack and make the visible
    #                                dataframe the new base, leaving [current]
    #   'shelve'   (s, then Enter) -> shelve the visible dataframe as shelf_name
    #   'shelve_blocked' (s)       -> it cannot be shelved; show why
    async def applyKeyOp(self, event: Any) -> None:
        _op_ = self.key_op_finished
        if _op_:
            self.key_op_finished = ''      # reset so an identical next press re-fires
        if not _op_ or self.mvc is None:
            return
        sn = self.mvc.view_stack.get(id(self))
        if sn is None:
            return
        s = self.mvc.stacks.get(sn)
        if s is None or not s['dfs']:
            return
        idx = s['index']
        if idx < 0 or idx >= len(s['dfs']):
            return
        base_df, cur_df = s['dfs'][0], s['dfs'][idx]
        # A stale tile can no longer be served for the kept frame: the tile cache
        # is identity-guarded (see _add_frame), so a reused id re-renders rather
        # than reusing a since-freed frame's render.
        if _op_ == 'collapse':
            new_dfs = [base_df] if cur_df is base_df else [base_df, cur_df]
            new_idx = len(new_dfs) - 1
            s['dfs'], s['index'] = new_dfs, new_idx
            await self._broadcastStackChange(sn, new_dfs[new_idx], new_dfs, new_idx)
        elif _op_ == 'rebase':
            await self.mvc.replaceStack(self, cur_df)
        elif _op_ == 'shelve':
            # Diagram 5: the displayed frame goes on the shelf under the typed name.
            self._status_ = await self.mvc.shelveDisplayed(sn, self.shelf_name) or ''
            self.refreshShelf()
        elif _op_ == 'shelve_blocked':
            # `s` pressed when the frame cannot be shelved at all: say why, no dialog.
            self._status_ = self.shelve_blocked
            self.refreshShelf()

    async def display(self, df: pl.DataFrame, dfs: list, dfs_index: int) -> None:
        if df is not dfs[dfs_index]:
            return
        self._render_(dfs, dfs_index)
        _ids_ = {id(d) for d in dfs}
        for _id_ in list(self._svg_cache_.keys()):
            if _id_ not in _ids_:
                del self._svg_cache_[_id_]

    def sketchHtml(self, use_webgpu: bool = False) -> str | None:
        # Static snapshot for panelizeSketch(): the current stack frame already
        # lives in mod_inner as a complete <svg>...</svg>, so reuse it verbatim.
        return self.mod_inner or None



def stack_controli(component: Any, stack_name: str = 'default', insets: tuple = (2, 2), hgap: int = 4,
                   wxh: tuple = (160, 256), txt_h: int = 10, max_shelf: int = _MAX_SHELF_,
                   **kwargs: Any) -> Any:
    """The stack navigator.  ``max_shelf`` is the number of shelf slots it draws above
    the frame stack (PLANNING.md §7 F18); 0 draws no shelf."""
    if isinstance(max_shelf, bool) or not isinstance(max_shelf, int) or max_shelf < 0:
        raise ValueError(f'stack_controli: max_shelf must be an int >= 0, got {max_shelf!r}')
    w, h   = wxh
    wc, hc = component.wxh
    x0_val     = insets[0]
    inset_y_val = insets[1]
    _mlx_avail_, _cuda_avail_ = _mlxCudaStatus_()
    _row_txt_h_ = int(min(10, max(1, (hc - 4) / 2)))
    _shelf_h_   = shelfHeight(txt_h, max_shelf, hgap)
    _tc_x0_     = w - x0_val - _TC_W_

    # Requirement (1): the widget needs room for the header, the shelf, one icon
    # (the current index always renders), and two skip labels -- anything less
    # can never satisfy "always render the current index" alongside rule (7)'s
    # up-to-two labels, so refuse to build it rather than silently clipping.
    _ell_h_min_ = 2 * hgap + txt_h
    _min_h_ = 2 * inset_y_val + headerHeight(txt_h) + _shelf_h_ + hc + 2 * _ell_h_min_
    if h < _min_h_:
        raise ValueError(
            f'stack_controli: wxh height {h} is too small (need >= {_min_h_}) for the '
            f'header, {max_shelf} shelf slot(s), one icon, and two "... N stack frame(s) ..." '
            f'labels -- make it taller, or pass a smaller max_shelf')

    def _render_svg_content(dfs: list, index: int, cache: dict, shelf: list | None = None,
                            labels: list | None = None, status: str = '',
                            armed: str | None = None) -> tuple:
        # shelf  -- [(name, n_rows, usable)], in the order shelved
        # labels -- one per frame in dfs, None where the frame has none
        # Returns (svg, frame_map, shelf_map, hover_regions).
        p2s_ref = component.p2s
        x0      = x0_val
        inset_y = inset_y_val
        gap     = hgap
        slot    = hc + gap
        n       = len(dfs)
        shelf   = shelf or []
        labels  = labels if labels is not None else [None] * n
        frame_map: list = []   # [(y_top, y_bot, stack_idx)]
        shelf_map: list = []   # [(y_top, y_bot, name)]
        hover:     list = []   # [[x0, y0, x1, y1, full text]] for text the render cut off

        _bg_     = p2s_ref.colorTyped('background', 'default')
        _border_ = p2s_ref.colorTyped('axis', 'inner')
        _fg_     = p2s_ref.colorTyped('label', 'defaultfg')
        _dim_    = p2s_ref.colorTyped('label', 'inner')
        _alert_  = p2s_ref.colorTyped('indicator', 'more_rows')
        _off_    = p2s_ref.colorTyped('indicator', 'unavailable')
        _svg_ = [f'<svg width="{w}" height="{h}">',
                 f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_bg_}"/>']

        # MLX / CUDA availability header — one small row each, faded green when the
        # feature is usable, else a gray close to the background (visible but muted).
        # The status row under them says why the last operation was refused (Q6).
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
        if status:
            _y_ = inset_y + len(_INDICATOR_LABELS_) * _status_row_h_
            _svg_.append(p2s_ref.svgText(_fitText_(p2s_ref, status, _status_txt_h_, w - 2 * x0),
                                         x0, _y_ + _status_txt_h_, txt_h=_status_txt_h_, color=_alert_))
            if p2s_ref.textLength(status, _status_txt_h_) > w - 2 * x0:
                hover.append([0, _y_, w, _y_ + _status_row_h_, status])

        # The shelf: text-only rows in the order shelved.  More entries than slots (a controller
        # whose max_shelf is above this widget's) leaves the last slot saying how many
        # more there are.
        if max_shelf > 0:
            _row_h_   = _shelfRowH_(txt_h)
            _y_shelf_ = inset_y + _header_h_
            _shown_   = shelf if len(shelf) <= max_shelf else shelf[:max_shelf - 1]
            for _i_, (_nm_, _rows_, _ok_) in enumerate(_shown_):
                _y0_   = _y_shelf_ + _i_ * _row_h_
                _ty_   = _y0_ + (_row_h_ + _status_txt_h_) / 2 - 1
                _co_   = _fg_ if _ok_ else _off_
                _rtxt_ = f'{p2s_ref.unitizeInt(_rows_)} Rows'
                _rw_   = p2s_ref.textLength(_rtxt_, _status_txt_h_)
                _svg_.append(p2s_ref.svgText(_rtxt_, _tc_x0_ - 3, _ty_, txt_h=_status_txt_h_,
                                             color=_co_ if _ok_ else _off_, anchor='end'))
                _nw_   = _tc_x0_ - 3 - _rw_ - 6 - x0
                _shown_nm_ = _fitText_(p2s_ref, _nm_, _status_txt_h_, _nw_)
                _svg_.append(p2s_ref.svgText(_shown_nm_, x0, _ty_, txt_h=_status_txt_h_, color=_co_))
                if _shown_nm_ != _nm_:
                    hover.append([x0, _y0_, x0 + _nw_, _y0_ + _row_h_, _nm_])
                _svg_.append(_trashcanSvg_(_tc_x0_ + 1.5, _y0_ + (_row_h_ - 10) / 2,
                                           _alert_ if _nm_ == armed else _dim_))
                shelf_map.append((_y0_, _y0_ + _row_h_, _nm_))
            if len(_shown_) < len(shelf):
                _y0_ = _y_shelf_ + len(_shown_) * _row_h_
                _svg_.append(p2s_ref.svgText(f'... {len(shelf) - len(_shown_)} more on the shelf ...',
                                             w / 2, _y0_ + (_row_h_ + _status_txt_h_) / 2 - 1,
                                             txt_h=_status_txt_h_, color=_dim_, anchor='middle',
                                             font_style='italic'))
            _y_rule_ = _y_shelf_ + max_shelf * _row_h_ + gap / 2
            _svg_.append(f'<line class="p2s-shelf-rule" x1="{x0}" y1="{_y_rule_}" x2="{w - x0}" y2="{_y_rule_}"'
                         f' stroke="{_border_}" stroke-width="1"/>')

        ELL_H  = 2 * gap + txt_h
        y_base = h - inset_y - hc
        y_top  = inset_y + _header_h_ + _shelf_h_
        avail  = y_base - y_top

        def _add_frame(df: pl.DataFrame, y: float, stack_idx: int) -> None:
            # Cache entries are (df, tile_svg): id() alone is an unsafe key because a
            # dropped frame's id can be reused by a later dataframe, so guard every
            # hit with an identity check and re-render on a mismatch.
            df_id = id(df)
            entry = cache.get(df_id)
            if entry is None or entry[0] is not df:
                entry = (df, component.render_with(df)._repr_svg_())
                cache[df_id] = entry
            tile_svg    = entry[1]
            is_selected = (stack_idx == index)
            label_color = _fg_ if is_selected else _dim_
            x_lbl = x0 + wc + 4
            mid   = y + hc // 2
            half  = (_row_txt_h_ + 2) // 2
            if n == 1 or stack_idx == 0:
                pos_str = 'base'
            elif stack_idx == n - 1:
                pos_str = 'top'
            else:
                pos_str = f'{stack_idx + 1} of {n}'
            # Line 2 is the frame's label when it has one (a shelf name, or a set
            # operation and its operand -- Q2), else its position.
            _lbl_   = labels[stack_idx]
            _line2_ = pos_str if _lbl_ is None else _fitText_(p2s_ref, _lbl_, _row_txt_h_, _tc_x0_ - 2 - x_lbl)
            _svg_.append(p2s_ref.svgText(f'{p2s_ref.unitizeInt(len(df))} Rows',
                                         x_lbl, mid - half, txt_h=_row_txt_h_, color=label_color))
            _svg_.append(p2s_ref.svgText(_line2_,
                                         x_lbl, mid + half, txt_h=_row_txt_h_, color=label_color))
            if _lbl_ is not None and _line2_ != _lbl_:
                hover.append([x_lbl, y, _tc_x0_, y + hc, _lbl_])
            _svg_.append(f'<g transform="translate({x0}, {y})">')
            _svg_.append(tile_svg)
            _svg_.append('</g>')
            _svg_.append(f'<rect x="{x0}" y="{y}" width="{wc}" height="{hc}"'
                         f' fill="none" stroke="{_border_}" stroke-width="1"/>')
            # Every frame but the displayed one can be deleted, so only it has no trashcan.
            if not is_selected:
                _svg_.append(_trashcanSvg_(_tc_x0_ + 1.5, mid - 5, _dim_))
            frame_map.append((y, y + hc, stack_idx))

        def _add_skip_label(y: float, count: int) -> None:
            noun     = 'frame' if count == 1 else 'frames'
            label    = f'... {count} stack {noun} ...'
            cx       = w / 2   # centered on the full widget width so it can't clip off the left edge
            _skip_co_ = _dim_
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
        return ''.join(_svg_), frame_map, shelf_map, hover

    # Pre-compute the initial SVG so the widget paints its real content on the
    # first frame.  It reaches the browser as an ordinary constructor param now; under
    # ReactiveHTML it needed an _init_params() override to seed the data model,
    # because mod_inner was a *child* and ReactiveHTML left data.mod_inner at the class
    # default.  If mvc already carries a populated stack the
    # initial render must reflect it too, or the browser paints just the base
    # frame.  Also pre-populates the tile cache so __init__ starts warm.
    _initial_mvc_ = kwargs.get('mvc')
    if _initial_mvc_ is not None and stack_name in _initial_mvc_.stacks:
        _initial_stack_ = _initial_mvc_.stacks[stack_name]
        _initial_dfs_, _initial_index_ = _initial_stack_['dfs'], _initial_stack_['index']
    else:
        _initial_dfs_, _initial_index_ = [component.df_orig], 0
    _initial_cache_: dict = {}
    _initial_svg_, _initial_frame_map_, _initial_shelf_map_, _ = \
        _render_svg_content(_initial_dfs_, _initial_index_, _initial_cache_)


    return STACKCONTROLI(wxh=wxh, stack_name=stack_name,
                         render_svg_content=_render_svg_content,
                         initial_svg=_initial_svg_,
                         initial_frame_map=_initial_frame_map_,
                         initial_cache=_initial_cache_,
                         max_shelf=max_shelf, tc_x0=_tc_x0_,
                         initial_shelf_map=_initial_shelf_map_,
                         shelf_row0=(x0_val, inset_y_val + headerHeight(txt_h),
                                     w - 2 * x0_val, _shelfRowH_(txt_h)),
                         **kwargs)
