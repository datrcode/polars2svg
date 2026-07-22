import asyncio
import copy
import json
import re
import time
from math import sqrt
from typing import Any
import networkx as nx
import polars as pl
import panel as pn
import param
import pyperclip
from panel.reactive import ReactiveHTML
from shapely.geometry import Polygon

from .polars_force_directed_layout import PolarsForceDirectedLayout
from .convey_proximity_layout       import ConveyProximityLayout
from .mds_at_scale                  import LandmarkMDSLayout, PivotMDSLayout
from .p2s_webgpu_runtime            import P2S_GPU_JS

#
# Known limitation: in a browser, ctrl-click / shift-ctrl-click can open the
# browser popup and interrupt a drag selection.  Workaround: press the modifier
# keys after the drag selection has started (and before it is released).
#

def _gpu_error_overlay(msg, w, h):
    # SVG content for the `mod` overlay when WebGPU rendering fails. We deliberately do
    # NOT fall back to SVG automatically: the failure is surfaced to the user, who must
    # re-create the view with use_webgpu=False to switch to SVG rendering.
    import html as _html
    reason = _html.escape(str(msg)) if msg else 'unknown error'
    return (
        f'<rect x="0" y="0" width="{w}" height="{h}" fill="rgba(255,255,255,0.94)"/>'
        f'<foreignObject x="10" y="10" width="{max(w - 20, 40)}" height="{max(h - 20, 40)}">'
        f'<div xmlns="http://www.w3.org/1999/xhtml" '
        f'style="font-family:monospace;font-size:12px;color:#a00;line-height:1.45;">'
        f'<b>WebGPU rendering failed.</b><br/>{reason}<br/><br/>'
        f'Re-create this view with <code>use_webgpu=False</code> to render with SVG instead.'
        f'</div></foreignObject>'
    )

try:
    from .tfdp_layout import TFDPLayout as _TFDPLayout
    _TFDP_AVAILABLE = True
except ImportError:
    _TFDPLayout = None
    _TFDP_AVAILABLE = False

# (mnemonic, label) — single source of truth for the shift-G / shift-W picker
# menus, the Python layout_modes/layout_operations lists, and the JS menu
# arrays. Mnemonics are case-sensitive, must be unique within a menu, and must
# avoid the picker navigation keys (j, k, W, G, Enter, Escape, arrows).
_LAYOUT_MODE_MENU_ = [
    ('g', 'grid'),
    ('c', 'circle'),
    ('s', 'sunflower'),
    ('o', 'grid (color)'),
    ('d', 'grid (color, clouds)'),
    ('r', 'rescale'),
]
_LAYOUT_OP_MENU_ = [
    ('s', 'spring nx'),
    ('f', 'force directed'),
    ('h', 'hyper tree'),
    ('d', 'hyper tree donut'),
    ('v', 'convey proximity'),
    ('l', 'landmark mds'),
    ('L', 'landmark mds pos'),
    ('p', 'pivot mds'),
    ('c', 'connected components'),
    ('C', 'circle pack'),
    ('n', 'neighborhood (spatial)'),
    ('N', 'neighborhood (graph)'),
]
if _TFDP_AVAILABLE:
    _LAYOUT_OP_MENU_.append(('t', 't-fdp'))


class _ContractedLayoutView_:
    """Lightweight stand-in for a LinkNode that overrides ``.pos`` with the
    contracted (one-representative-node-per-location) position map while
    proxying every other attribute to the real node. Layout handlers only read
    ``.pos`` (plus the graph and the selection passed alongside), so this is
    enough to run any of them on the contracted graph."""
    def __init__(self, base, pos):
        self._base = base
        self.pos   = pos

    def __getattr__(self, name):
        return getattr(self._base, name)

BRUSH_STATES = [
    None,                          # 0 = off
    ('SELECT_CIRCLEp',      5),    # 1 circle, small radius
    ('SELECT_CIRCLEp',     15),    # 2 circle, larger radius
    ('SELECT_VERTICALp',    1),    # 3 vertical band, radius=1
    ('SELECT_VERTICALp',    3),    # 4 vertical band, radius=3
    ('SELECT_HORIZONTALp',  1),    # 5 horizontal band, radius=1
    ('SELECT_HORIZONTALp',  3),    # 6 horizontal band, radius=3
]

class InteractionController(object):
    # Constructor
    def __init__(self):
        self.stacks     = {}   # {'name': {'dfs': [df], 'index': 0}}
        self.view_stack = {}   # {id(view): 'stack_name'}
        self.links      = {}   # {(id(source), event_type): [target_views]}
        self.view_refs  = {}   # {id(view): view} — reverse lookup for brush propagation
    # addStack()
    def addStack(self, name, df):
        self.stacks[name] = {'dfs': [df], 'index': 0}
    # link()
    def link(self, source, targets, on='stack', stack=None):
        if stack is not None: self.view_stack[id(source)] = stack
        self.view_refs[id(source)] = source
        if isinstance(on, str): on_as_set = set([on])
        else:                   on_as_set = on
        for _on_ in on_as_set:
            key = (id(source), _on_)
            if key not in self.links: self.links[key] = []
            for t in (targets if isinstance(targets, list) else [targets]):
                self.view_refs[id(t)] = t
                if t not in self.links[key]: self.links[key].append(t)
    # Stack insight
    def stackTopDataFrame(self, caller):
        s = self.stacks[self.view_stack[id(caller)]]
        return s['dfs'][0]
    def stackCurrentDataFrame(self, caller):
        s = self.stacks[self.view_stack[id(caller)]]
        return s['dfs'][s['index']]
    # Stack control
    # subtractCurrentStackFromTop()
    async def subtractCurrentStackFromTop(self, caller):
        _df_top_     = self.stackTopDataFrame(caller)
        _df_current_ = self.stackCurrentDataFrame(caller)
        if len(_df_top_) != len(_df_current_):
            _df_subtracted_ = _df_top_.join(_df_current_, on=_df_top_.columns, how="anti", nulls_equal=True)
            if len(_df_subtracted_) > 0: await self.pushStack(caller, _df_subtracted_)
    # popStack()
    async def popStack(self, caller):
        s = self.stacks[self.view_stack[id(caller)]]
        if s['index'] > 0:
            s['index'] -= 1
            _targets_ = self.links.get((id(caller), 'stack'), [])
            for view in ([caller] + [v for v in _targets_ if v is not caller]):
                await view.display(s['dfs'][s['index']], s['dfs'], s['index'])
    # pushStack()
    async def pushStack(self, caller, df):
        s = self.stacks[self.view_stack[id(caller)]]
        if s['index'] != len(s['dfs']) - 1: s['dfs'] = s['dfs'][:s['index']+1]
        s['dfs'].append(df)
        s['index'] += 1
        _targets_ = self.links.get((id(caller), 'stack'), [])
        for view in ([caller] + [v for v in _targets_ if v is not caller]):
            await view.display(s['dfs'][s['index']], s['dfs'], s['index'])
    # setStackIndex()
    async def setStackIndex(self, caller, index):
        s = self.stacks[self.view_stack[id(caller)]]
        if index >= 0 and index < len(s['dfs']):
            s['index'] = index
            _targets_ = self.links.get((id(caller), 'stack'), [])
            for view in ([caller] + [v for v in _targets_ if v is not caller]):
                await view.display(s['dfs'][s['index']], s['dfs'], s['index'])
    # replaceStack() — replace the entire stack with a new base dataframe and re-render all views
    # caller may be a registered view object or a stack name string (e.g. 'default')
    async def replaceStack(self, caller, df):
        sn = caller if isinstance(caller, str) else self.view_stack.get(id(caller))
        if sn is None or sn not in self.stacks: return
        self.stacks[sn] = {'dfs': [df], 'index': 0}
        for vid, vsn in self.view_stack.items():
            if vsn == sn:
                view = self.view_refs.get(vid)
                if view is not None:
                    if hasattr(view, 'replaceBaseDataframe'):
                        await view.replaceBaseDataframe(df)
                    else:
                        await view.display(df, [df], 0)
    # Brush control — auto-discovers all stack peers, no explicit 'brush' links needed
    # brushUpdate() — broadcast brushed df to all peers on the same stack
    async def brushUpdate(self, caller, df):
        sn = self.view_stack.get(id(caller))
        if sn is None: return
        s = self.stacks[sn]
        for vid, vsn in self.view_stack.items():
            if vsn == sn and vid != id(caller):
                view = self.view_refs.get(vid)
                if view is not None:
                    await view.display(df, s['dfs'], s['index'])
    # brushClear() — revert all peers to current stack df
    async def brushClear(self, caller):
        sn = self.view_stack.get(id(caller))
        if sn is None: return
        s   = self.stacks[sn]
        df  = s['dfs'][s['index']]
        for vid, vsn in self.view_stack.items():
            if vsn == sn and vid != id(caller):
                view = self.view_refs.get(vid)
                if view is not None:
                    await view.display(df, s['dfs'], s['index'])
    # selectionUpdate() — propagate entity selection to 'selection'-linked views
    async def selectionUpdate(self, caller, entities):
        _targets_ = self.links.get((id(caller), 'selection'), [])
        for view in _targets_:
            if hasattr(view, 'receiveSelection'):
                await view.receiveSelection(entities)
    # selectionClear()
    async def selectionClear(self, caller):
        await self.selectionUpdate(caller, set())
    # positionsUpdate() — propagate a node-layout change to 'positions'-linked views.
    # Callers broadcast liberally (any refresh where positions MAY have moved);
    # receivers are expected to diff against their last snapshot and skip no-op
    # re-renders. Safe to call from sync code: the fan-out is scheduled on the
    # running event loop (and silently skipped when no loop is running, e.g.
    # during view construction).
    def positionsUpdate(self, caller, pos):
        _targets_ = self.links.get((id(caller), 'positions'), [])
        if not _targets_: return
        _snapshot_ = {k: (float(v[0]), float(v[1])) for k, v in pos.items()}
        async def _fan_out_():
            for view in _targets_:
                if hasattr(view, 'receivePositions'):
                    await view.receivePositions(_snapshot_)
        try:                  _loop_ = asyncio.get_running_loop()
        except RuntimeError:  return
        _loop_.create_task(_fan_out_())

_BRUSH_MODE_NAMES_ = ['', 'circ r=5', 'circ r=15', 'vert r=1', 'vert r=3', 'horiz r=1', 'horiz r=3']

def _resolve_set_op(shiftkey, ctrlkey):
    if shiftkey and ctrlkey: return 'intersect'
    if shiftkey:             return 'subtract'
    if ctrlkey:              return 'add'
    return 'replace'

_INTERACTIVEP_CONFIG_ = {
    'timepi': {
        'class_name':    'TIMEPI',
        'svg_parent_id': 'svgparenttimepi',
        'render_fn':     'timep',
        'fallback_shape':'SELECT_VERTICALp',
        'brush_seq':     '[0,1,2,3,4]',
        'has_z_key':     False,
        'kbd_s_desc':    'toggle brush on/off',
    },
    'histopi': {
        'class_name':    'HISTOPI',
        'svg_parent_id': 'svgparenthistopi',
        'render_fn':     'histop',
        'fallback_shape':'SELECT_HORIZONTALp',
        'brush_seq':     '[0,1,2,5,6]',
        'has_z_key':     False,
        'has_search':    True,
        'kbd_s_desc':    'toggle brush on/off',
    },
    'xypi': {
        'class_name':    'XYPI',
        'svg_parent_id': 'svgparentxypi',
        'render_fn':     'xyp',
        'fallback_shape':'SELECT_HORIZONTALp',
        'brush_seq':     '[0,1,2,3,4,5,6]',
        'has_z_key':     True,
        'kbd_s_desc':    'toggle brush on/off',
    },
    'chordpi': {
        'class_name':    'CHORDPI',
        'svg_parent_id': 'svgparentchordpi',
        'render_fn':     'chordp',
        'fallback_shape':'SELECT_CIRCLEp',
        'brush_seq':     '[0,1,2]',
        'has_z_key':     False,
        'kbd_s_desc':    'toggle brush on/off',
    },
    'piepi': {
        'class_name':    'PIEPI',
        'svg_parent_id': 'svgparentpiepi',
        'render_fn':     'piep',
        'fallback_shape':'SELECT_CIRCLEp',
        'brush_seq':     '[0,1,2]',
        'has_z_key':     False,
        'has_search':    True,
        'kbd_s_desc':    'toggle brush on/off',
    },
}

def _interactivep(_plot_, kind, **kwargs):
    cfg            = _INTERACTIVEP_CONFIG_[kind]
    use_webgpu     = kwargs.pop('use_webgpu', False)
    if use_webgpu and getattr(_plot_, 'webgpu', None) is None:
        raise ValueError(f'_interactivep(): use_webgpu=True is not (yet) supported for "{kind}"')
    _gpu_payload_default_ = _plot_.webgpu() if use_webgpu else None
    _svg_          = _plot_._repr_svg_() if not use_webgpu else ''
    _w_, _h_       = _plot_.wxh[0], _plot_.wxh[1]
    class_name     = cfg['class_name']
    svg_parent     = cfg['svg_parent_id']
    render_fn      = cfg['render_fn']
    fallback_shape = cfg['fallback_shape']
    brush_seq      = cfg['brush_seq']
    has_z_key      = cfg['has_z_key']
    has_search     = cfg.get('has_search', False)
    _cls_ref_      = [None]
    # Constructor
    def __init__(self, **kwargs):
        mvc = kwargs.pop('mvc', None) # don't pass to the super
        super(_cls_ref_[0], self).__init__(**kwargs)
        # Locking variable
        self.lock = asyncio.Lock()
        # Model/View/Controller — single-df fallback: build a default stack and register self
        if mvc is None:
            self.mvc = InteractionController()
            self.mvc.addStack('default', _plot_.df_orig)
            self.mvc.view_stack[id(self)] = 'default'
        else:
            self.mvc = mvc
        # Render state
        self._plot_  = _plot_
        self._cache_ = {id(_plot_.df_orig): _plot_}
        self.template = _plot_
        # Watch for callbacks
        self.param.watch(self.applyDragOp,     'drag_op_finished')
        self.param.watch(self.applyKeyOp,      'key_op_finished')
        self.param.watch(self.applyBrushOp,    'brush_changed')
        self.param.watch(self.applyBrushLeave, 'brush_leave_done')
        if use_webgpu:
            self.param.watch(self.applyGpuError, 'gpu_error')
        if has_search:
            self.param.watch(self.applySearchOp, 'search_op_finished')
    # Refresh the view
    def __refreshView__(self):
        if use_webgpu:
            if self.gpu_error: self.mod_inner   = _gpu_error_overlay(self.gpu_error, _w_, _h_)
            else:              self.gpu_payload = self._plot_.webgpu()
        else:
            self.mod_inner = self._plot_._repr_svg_()
    # WebGPU rendering failed in the browser -> surface the error in the overlay.
    # No automatic SVG fallback: the user must re-create the view with use_webgpu=False.
    async def applyGpuError(self, event):
        if self.gpu_error:
            self.mod_inner = _gpu_error_overlay(self.gpu_error, _w_, _h_)
    # Render the view
    def __renderView__(self, df):
        return getattr(self.template.p2s, render_fn)(df=df, template=self.template)
    # Core brush logic: call recordsAt and broadcast to peers
    async def _doBrushAt(self, xy, state_idx):
        state_def = BRUSH_STATES[state_idx]
        shape     = getattr(self._plot_.p2s, state_def[0])
        threshold = state_def[1]
        try:
            filtered = self._plot_.recordsAt(xy, shape=shape, threshold=threshold)
        except ValueError:
            try:
                filtered = self._plot_.recordsAt(xy, shape=getattr(self._plot_.p2s, fallback_shape), threshold=threshold)
            except Exception:
                return
        if len(filtered) == 0:
            await self.mvc.brushClear(self)
        else:
            await self.mvc.brushUpdate(self, filtered)
    # Callbacks - applyDragOp()
    async def applyDragOp(self, event):
        async with self.lock:
            if not self.drag_op_finished: return
            _coords_ = (self.drag_x0, self.drag_y0, self.drag_x1, self.drag_y1)
            _shift_  = self.shiftkey
            _shape_  = self.select_shape
            self.drag_op_finished = False
        if _shape_ == 'oval':
            # press point (drag_x0/y0) is the oval center; drag edge sets the radii
            _cx_, _cy_ = self.drag_x0, self.drag_y0
            _rx_ = abs(self.drag_x1 - self.drag_x0)
            _ry_ = abs(self.drag_y1 - self.drag_y0)
            _df_ = self._plot_.filterByOval((_cx_, _cy_, _rx_, _ry_), _shift_)
        else:
            _df_ = self._plot_.filterByRectangle(_coords_, _shift_)
        if len(_df_) > 0: await self.mvc.pushStack(self, _df_)
        else:             await self.mvc.popStack(self)
    # Callbacks - applyKeyOp()
    async def applyKeyOp(self, event):
        async with self.lock:
            _key_   = self.key_op_finished
            _xy_    = (self.x_mouse, self.y_mouse)
            _shift_ = self.shiftkey
            self.key_op_finished = ''
        if has_z_key and (_key_ == 'z' or _key_ == 'Z'):
            _df_ = self._plot_.filterByColorAtXY(_xy_, _shift_)
            if _df_ is not None and len(_df_) > 0: await self.mvc.pushStack(self, _df_)
            else:                                  await self.mvc.popStack(self)
        elif _key_ == 'q':
            await self.mvc.subtractCurrentStackFromTop(self)
    # Callbacks - applyBrushOp() — fires on mouse move (when brush active) or on brush state change
    async def applyBrushOp(self, event):
        async with self.lock:
            _state_ = self.brush_state
            _xy_    = (self.x_mouse, self.y_mouse)
        if _state_ == 0:
            await self.mvc.brushClear(self)
        else:
            await self._doBrushAt(_xy_, _state_)
    # Callbacks - applyBrushLeave() — fires when mouse leaves the component while brush is active
    async def applyBrushLeave(self, event):
        async with self.lock:
            if not self.brush_leave_done: return
            if self.brush_state == 0:
                self.brush_leave_done = False
                return
            self.brush_leave_done = False
        await self.mvc.brushClear(self)
    # Callbacks - applySearchOp() — fires when user commits a '/' search string
    async def applySearchOp(self, event):
        async with self.lock:
            _s_ = self.search_str
        if not _s_:
            return
        if _s_.startswith('-'):
            _sub_, _remove_bins_ = _s_[1:], True
        else:
            _sub_, _remove_bins_ = _s_, False
        if _sub_:
            _df_ = self._plot_.filterBySubstring(_sub_, remove_bins=_remove_bins_)
            if len(_df_) > 0: await self.mvc.pushStack(self, _df_)
            else:             await self.mvc.popStack(self)
    # MVC
    async def display(self, df, dfs, dfs_index):
        async with self.lock:
            # render if not already rendered
            if id(df) not in self._cache_: self._cache_[id(df)] = self.__renderView__(df)
            # set the current & refresh
            self._plot_ = self._cache_[id(df)]
            self.__refreshView__()
            # clean up the cache
            _ids_ = set([id(df) for df in dfs])
            for _id_ in list(self._cache_.keys()):
                if _id_ not in _ids_:
                    del self._cache_[_id_]

    # Build keyboard commands string
    _z_key_cmd_ = '\nz . | filter to color nearest to mouse (shift filters those records out)' if has_z_key else ''
    _search_cmd_ = '\n/ . | filter bins: type substring + Enter (prefix -remove); Escape to cancel' if has_search else ''
    _keyboard_commands_ = f"""
h . | toggle help display
q . | subtract the current from the top
F . | pick selection shape (rectangle | oval)
s . | {cfg['kbd_s_desc']}
S . | cycle brush shape{_z_key_cmd_}{_search_cmd_}
        """

    # Build static SVG for keyboard help overlay
    _help_lines_  = _keyboard_commands_.strip().split('\n')
    _help_w_      = max(len(l) for l in _help_lines_) * 7 + 20
    _help_h_      = len(_help_lines_) * 14 + 12
    _help_font_style_ = "font-family: 'Courier New', monospace; font-size: 11px; fill: #222;"
    _help_text_lines_ = ''.join(
        f'<text x="10" y="{12 + i*14}" style="{_help_font_style_}">{l}</text>'
        for i, l in enumerate(_help_lines_)
    )
    _keyboard_help_svg_ = (
        f'<rect x="0" y="0" width="{_help_w_}" height="{_help_h_}" '
        f'fill="rgba(240,240,240,0.95)" stroke="#888" stroke-width="1" rx="3"/>'
        f'{_help_text_lines_}'
    )

    # Build JS z-key block (prepended to 's' check in myOnKeyDown when has_z_key)
    _z_block_ = (
        """if      (event.key == 'z' || event.key == 'Z') { data.key_op_finished = "z"; }\n                else """
        if has_z_key else ""
    )

    # Build JS search blocks for myOnKeyDown (histopi only)
    _search_block_top_ = ("""
                if (state.search_mode) {
                    if (event.key === 'Enter') {
                        if (state.search_buffer) {
                            data.search_str = state.search_buffer;
                            data.search_op_finished = !data.search_op_finished;
                        }
                        state.search_mode   = false;
                        state.search_buffer = '';
                        searchtext.textContent = '';
                    } else if (event.key === 'Escape') {
                        state.search_mode   = false;
                        state.search_buffer = '';
                        searchtext.textContent = '';
                    } else if (event.key === 'Backspace') {
                        state.search_buffer = state.search_buffer.slice(0, -1);
                        searchtext.textContent = '/ ' + state.search_buffer + '▋';
                    } else if (event.key.length === 1) {
                        state.search_buffer += event.key;
                        searchtext.textContent = '/ ' + state.search_buffer + '▋';
                    }
                    return;
                }
""") if has_search else ''

    _search_block_bottom_ = ("""
                else if (event.key == '/') {
                    state.search_mode   = true;
                    state.search_buffer = '';
                    searchtext.textContent = '/ ▋';
                }
""") if has_search else ''

    _search_init_ = """
                state.search_mode   = false;
                state.search_buffer = '';
""" if has_search else ''

    _search_text_elem_ = (
        f'<text id="searchtext" x="{_w_//2}" y="{_h_-2}" text-anchor="middle" '
        f'fill="#0000cc" font-size="11px" font-family="monospace"></text>'
    ) if has_search else ''

    # Template: in GPU mode the plot renders on a canvas underneath the (transparent)
    # interaction SVG; the SVG keeps all controller chrome + mouse/key handling
    _svg_root_ = f"""
<svg id="{svg_parent}" width="{_w_}" height="{_h_}" tabindex="0" onkeydown="${{script('myOnKeyDown')}}" onkeyup="${{script('myOnKeyUp')}}"{' style="position:absolute;left:0;top:0;"' if use_webgpu else ''}>
    <svg id="mod" width="{_w_}" height="{_h_}"> ${{mod_inner}} </svg>
    <g   id="brushindicator"></g>
    <g   id="brushmodelabel"></g>
    <g   id="keyboardhelp" transform="translate(${{keyboardhelp_x}} 0)">{_keyboard_help_svg_}</g>
    <rect id="drag"   x="-10" y="-10" width="5"     height="5" stroke="#000000" stroke-width="2" fill="none" />
    <ellipse id="dragoval" cx="-10" cy="-10" rx="0" ry="0" stroke="#000000" stroke-width="2" fill="none" display="none" />
    <rect id="screen" x="0"   y="0"   width="{_w_}" height="{_h_}" opacity="0.05"
          onmouseover="${{script('myOnMouseOver')}}"  onmouseout="${{script('myOnMouseOut')}}"
          onmousedown="${{script('downSelect')}}"     onmousemove="${{script('myOnMouseMove')}}"
          onmouseup="${{script('myOnMouseUp')}}" />
    <text id="infostr" x="5" y="{_h_-3}" fill="#000000" font-size="10px"> ${{info_str}} </text>
    {_search_text_elem_}
    <g id="pickermenu" pointer-events="none"></g>
</svg>
"""
    if use_webgpu:
        _template_ = f"""
<div id="gpuwrap" style="position:relative;width:{_w_}px;height:{_h_}px;">
    <canvas id="gpucanvas" width="{_w_}" height="{_h_}" style="position:absolute;left:0;top:0;"></canvas>
    {_svg_root_}
</div>
        """
    else:
        _template_ = _svg_root_

    # GPU additions to the JS table: install the runtime + first paint on mount,
    # and re-render whenever gpu_payload changes; any GPU failure sets gpu_error, and
    # the Python watcher shows an error overlay (no automatic SVG fallback)
    _gpu_render_block_ = (f"""
{P2S_GPU_JS}
                if (!window.__P2S_GPU__.supported()) {{ data.gpu_error = 'WebGPU is not available in this browser.'; }}
                else {{
                    window.__P2S_GPU__.render(gpucanvas, data.gpu_payload)
                        .catch(function(e) {{ console.warn('p2s webgpu:', e); data.gpu_error = (e && e.message) ? e.message : String(e); }});
                }}
""") if use_webgpu else ''
    _gpu_payload_script_ = ("""
                if (window.__P2S_GPU__ && window.__P2S_GPU__.supported() && !data.gpu_error) {
                    window.__P2S_GPU__.render(gpucanvas, data.gpu_payload)
                        .catch(function(e) { console.warn('p2s webgpu:', e); data.gpu_error = (e && e.message) ? e.message : String(e); });
                }
""") if use_webgpu else ''

    # Dynamic Class
    cls = type(class_name, (ReactiveHTML,), {
        #
        # Keyboard Commands
        #
        '_keyboard_commands_': _keyboard_commands_,
        #
        # Panel Params
        #
        'mod_inner':         param.String(default=_svg_),
        'info_str':          param.String(default=''),
        'keyboardhelp_x':    param.Integer(default=-1000),
        'x0_middle':         param.Integer(default=0),
        'y0_middle':         param.Integer(default=0),
        'x1_middle':         param.Integer(default=0),
        'y1_middle':         param.Integer(default=0),
        'middle_op_finished':param.Boolean(default=False),
        'wheel_x':           param.Integer(default=0),
        'wheel_y':           param.Integer(default=0),
        'wheel_rots':        param.Integer(default=0),
        'wheel_op_finished': param.Boolean(default=False),
        'drag_op_finished':  param.Boolean(default=False),
        'drag_x0':           param.Integer(default=0),
        'drag_y0':           param.Integer(default=0),
        'drag_x1':           param.Integer(default=10),
        'drag_y1':           param.Integer(default=10),
        'select_shape':      param.String(default='rectangle'),
        'shiftkey':          param.Boolean(default=False),
        'ctrlkey':           param.Boolean(default=False),
        'last_key':          param.String(default=''),
        'key_op_finished':   param.String(default=''),
        'x_mouse':           param.Integer(default=0),
        'y_mouse':           param.Integer(default=0),
        'has_focus':         param.Boolean(default=False),
        'brushing_mode':     param.Boolean(default=False),
        'brush_state':       param.Integer(default=0),
        'brush_changed':     param.Integer(default=0),
        'brush_leave_done':  param.Boolean(default=False),
        **({'search_str': param.String(default=''), 'search_op_finished': param.Boolean(default=False)} if has_search else {}),
        **({'gpu_payload': param.Dict(default=_gpu_payload_default_), 'gpu_error': param.String(default='')} if use_webgpu else {}),
        #
        # Template / Required by ReactiveHTML @ Initialization
        #
        '_template': _template_,
        #
        # Functions
        #
        '__init__':              __init__,
        **({'applyGpuError': applyGpuError} if use_webgpu else {}),
        'applyDragOp':           applyDragOp,
        'applyKeyOp':            applyKeyOp,
        'applyBrushOp':          applyBrushOp,
        'applyBrushLeave':       applyBrushLeave,
        **({'applySearchOp': applySearchOp} if has_search else {}),
        '_doBrushAt':            _doBrushAt,
        '__renderView__':        __renderView__,
        '__refreshView__':       __refreshView__,
        'display':               display,
        #
        # JavaScript Table
        #
        '_scripts': {
            'render': f"""
                mod.innerHTML      = data.mod_inner;
                infostr.innerHTML  = data.info_str;
                state.x0_drag      = state.y0_drag = -10;
                state.x1_drag      = state.y1_drag =  -5;
                data.has_focus     = false;
                data.shiftkey      = false;
                data.ctrlkey       = false;
                state.drag_op      = false;
                data.brush_state   = 0;
                data.brushing_mode = false;
                data.brush_changed = 0;
                state.last_brush_x = -999;
                state.last_brush_y = -999;
                state.cur_mouse_x  = -999;
                state.cur_mouse_y  = -999;
                state.brush_defs   = [null,['circle',5],['circle',15],['vertical',1],['vertical',3],['horizontal',1],['horizontal',3]];
                state.brush_names  = ['','circ r=5','circ r=15','vert r=1','vert r=3','horiz r=1','horiz r=3'];
                state.select_shape = 'rectangle';
                state.menu_items   = {{'select_shape': [['r','rectangle'],['o','oval']]}};
                state.menu_open    = false;
                state.menu_kind    = '';
                state.menu_index   = 0;
                state.menu_timer   = null;
{_search_init_}
                screen.addEventListener('wheel', function(event) {{
                    event.preventDefault();
                    data.wheel_x = event.offsetX; data.wheel_y = event.offsetY;
                    data.wheel_rots = Math.round(10*event.deltaY);
                    data.wheel_op_finished = true;
                }}, {{passive: false}});
                // On macOS ctrl+click is a secondary click -> the browser raises a
                // contextmenu (popup) during ctrl / shift-ctrl rectangular drags.
                // Swallow it on the panel so the intersection/add selection survives.
                {svg_parent}.addEventListener('contextmenu', function(event) {{
                    event.preventDefault();
                }});
{_gpu_render_block_}
            """,
            **({'gpu_payload': _gpu_payload_script_} if use_webgpu else {}),
            'updateBrushCursor': f"""
                var _bs_ = data.brush_state;
                if (_bs_ == 0) {{
                    brushindicator.innerHTML = '';
                    brushmodelabel.innerHTML = '';
                    return;
                }}
                var _x_ = state.cur_mouse_x, _y_ = state.cur_mouse_y;
                var _d_ = state.brush_defs[_bs_], _r_ = _d_[1];
                var _s_ = 'stroke="rgba(100,150,255,0.8)" fill="none" pointer-events="none"';
                if      (_d_[0] == 'circle')     {{ brushindicator.innerHTML = '<circle cx="'+_x_+'" cy="'+_y_+'" r="'+_r_+'" '+_s_+' stroke-width="1.5"/>'; }}
                else if (_d_[0] == 'vertical')   {{ brushindicator.innerHTML = '<line x1="'+_x_+'" y1="0" x2="'+_x_+'" y2="{_h_}" '+_s_+' stroke-width="'+Math.max(1,_r_)+'"/>'; }}
                else if (_d_[0] == 'horizontal') {{ brushindicator.innerHTML = '<line x1="0" y1="'+_y_+'" x2="{_w_}" y2="'+_y_+'" '+_s_+' stroke-width="'+Math.max(1,_r_)+'"/>'; }}
                var _nm_ = state.brush_names[_bs_];
                var _tw_ = _nm_.length * 6 + 10, _rx_ = {_w_} - _tw_ - 3;
                brushmodelabel.innerHTML = '<rect x="'+_rx_+'" y="3" width="'+_tw_+'" height="15" rx="3" fill="rgba(100,150,255,0.3)" stroke="rgba(100,150,255,0.7)" stroke-width="0.5" pointer-events="none"/>'
                    + '<text x="'+(_rx_+5)+'" y="14" font-size="10px" fill="rgba(40,60,200,1.0)" font-family="monospace" pointer-events="none">'+_nm_+'</text>';
            """,
            'myOnMouseOver': f"""
                data.has_focus = true;
                {svg_parent}.focus();
            """,
            'myOnMouseOut':"""
                data.has_focus           = false;
                brushindicator.innerHTML = '';
                if (data.brush_state > 0) { data.brush_leave_done = true; }
            """,
            # key events don't have access to event.offsetX/Y
            'myOnKeyDown': f"""
                event.stopPropagation();
{_search_block_top_}                if (state.menu_open) {{
                    event.preventDefault();
                    var _items_ = state.menu_items[state.menu_kind];
                    if      (event.key === 'Escape') {{ self.menuClose();  }}
                    else if (event.key === 'Enter')  {{ self.menuCommit(); }}
                    else if (event.key === 'ArrowDown' || event.key === 'j' || event.key === 'F') {{
                        state.menu_index = (state.menu_index + 1) % _items_.length;
                        self.menuRender(); self.menuArmTimer();
                    }}
                    else if (event.key === 'ArrowUp' || event.key === 'k') {{
                        state.menu_index = (state.menu_index - 1 + _items_.length) % _items_.length;
                        self.menuRender(); self.menuArmTimer();
                    }}
                    else if (event.key.length === 1) {{
                        for (var _i_ = 0; _i_ < _items_.length; _i_++) {{
                            if (_items_[_i_][0] === event.key) {{ state.menu_index = _i_; self.menuCommit(); break; }}
                        }}
                    }}
                    return;
                }}
                data.shiftkey = event.shiftKey;
                data.ctrlkey  = event.ctrlKey;
                data.x_mouse  = state.cur_mouse_x;
                data.y_mouse  = state.cur_mouse_y;
                {_z_block_}if (event.key == 's') {{
                    if (data.brush_state == 0) {{
                        var _seq_ = {brush_seq};
                        data.brush_state = _seq_[1];
                    }} else {{
                        data.brush_state = 0;
                    }}
                    data.brushing_mode = data.brush_state > 0;
                    data.brush_changed += 1;
                    self.updateBrushCursor();
                }}
                else if (event.key == 'S') {{
                    var _seq_ = {brush_seq};
                    var _non0_ = _seq_.filter(function(x) {{ return x > 0; }});
                    if (data.brush_state == 0) {{
                        data.brush_state = _non0_[0];
                    }} else {{
                        var _i_ = _non0_.indexOf(data.brush_state);
                        data.brush_state = _non0_[(_i_ + 1) % _non0_.length];
                    }}
                    data.brushing_mode = true;
                    data.brush_changed += 1;
                    self.updateBrushCursor();
                }}
                else if (event.key == 'q') {{ data.key_op_finished = "q"; }}
                else if (event.key == 'F') {{ state.menu_kind = 'select_shape'; self.menuOpen(); }}
                else if (event.key == 'h') {{
                    if (data.keyboardhelp_x == -1000) {{ data.keyboardhelp_x =     5; }}
                    else                               {{ data.keyboardhelp_x = -1000; }}
                }}{_search_block_bottom_}
            """,
            # key events don't have access to event.offsetX/Y
            'myOnKeyUp':"""
                data.shiftkey = event.shiftKey;
                data.ctrlkey  = event.ctrlKey;
            """,
            'myOnMouseMove':"""
                state.cur_mouse_x = event.offsetX;
                state.cur_mouse_y = event.offsetY;
                state.x1_drag     = event.offsetX;
                state.y1_drag     = event.offsetY;
                if (state.drag_op) { self.myUpdateDragRect(); }
                if (data.brush_state > 0) {
                    self.updateBrushCursor();
                    var _dx_ = event.offsetX - state.last_brush_x;
                    var _dy_ = event.offsetY - state.last_brush_y;
                    if (_dx_*_dx_ + _dy_*_dy_ >= 9) {
                        state.last_brush_x = event.offsetX;
                        state.last_brush_y = event.offsetY;
                        data.x_mouse       = event.offsetX;
                        data.y_mouse       = event.offsetY;
                        data.brush_changed += 1;
                    }
                }
            """,
            'downSelect':"""
                if (event.button == 0) {
                    state.x0_drag  = event.offsetX;
                    state.y0_drag  = event.offsetY;
                    state.x1_drag  = event.offsetX;
                    state.y1_drag  = event.offsetY;
                    state.drag_op  = true;
                    self.myUpdateDragRect();
                } else if (event.button == 1) {
                    data.x0_middle = data.x1_middle = event.offsetX;
                    data.y0_middle = data.y1_middle = event.offsetY;
                }
            """,
            'myOnMouseUp':"""
                if (event.button == 0) {
                    state.x1_drag         = event.offsetX;
                    state.y1_drag         = event.offsetY;
                    if (state.drag_op) {
                        state.shiftkey        = event.shiftKey;
                        state.ctrlkey         = event.ctrlKey;
                        state.drag_op         = false;
                        self.myUpdateDragRect();
                        data.drag_x0          = state.x0_drag;
                        data.drag_y0          = state.y0_drag;
                        data.drag_x1          = state.x1_drag;
                        data.drag_y1          = state.y1_drag;
                        data.drag_op_finished = true;
                    }
                }
            """,
            'myOnMouseWheel':"""
                event.preventDefault();
                data.wheel_x = event.offsetX; data.wheel_y = event.offsetY; data.wheel_rots = Math.round(10*event.deltaY);
                data.wheel_op_finished = true;
            """,
            'mod_inner':"""
                mod.innerHTML     = data.mod_inner;
                infostr.innerHTML = data.info_str;
            """,
            'info_str': """
                infostr.innerHTML = data.info_str;
            """,
            'myUpdateDragRect':"""
                var _stroke_ = (data.shiftkey && data.ctrlkey) ? '#0000ff'
                             : (data.shiftkey)                  ? '#ff0000'
                             : (data.ctrlkey)                   ? '#00ff00'
                             :                                    '#000000';
                if (state.drag_op && state.select_shape == 'oval') {
                    var cx = state.x0_drag, cy = state.y0_drag;
                    var rx = Math.abs(state.x1_drag - state.x0_drag);
                    var ry = Math.abs(state.y1_drag - state.y0_drag);
                    dragoval.setAttribute('cx',cx); dragoval.setAttribute('cy',cy);
                    dragoval.setAttribute('rx',rx); dragoval.setAttribute('ry',ry);
                    dragoval.setAttribute('stroke',_stroke_);
                    dragoval.setAttribute('display','inline');
                    drag.setAttribute('x',-10);   drag.setAttribute('y',-10);
                    drag.setAttribute('width',5); drag.setAttribute('height',5);
                } else if (state.drag_op) {
                    x = Math.min(state.x0_drag, state.x1_drag);
                    y = Math.min(state.y0_drag, state.y1_drag);
                    w = Math.abs(state.x1_drag - state.x0_drag)
                    h = Math.abs(state.y1_drag - state.y0_drag)
                    drag.setAttribute('x',x);     drag.setAttribute('y',y);
                    drag.setAttribute('width',w); drag.setAttribute('height',h);
                    drag.setAttribute('stroke',_stroke_);
                    dragoval.setAttribute('display','none');
                } else {
                    drag.setAttribute('x',-10);   drag.setAttribute('y',-10);
                    drag.setAttribute('width',5); drag.setAttribute('height',5);
                    dragoval.setAttribute('display','none');
                }
        """,
            # ── selection-shape picker menu (Shift+F) ──
            # Modal JS state machine mirrored from the linkp layout picker; nothing reaches
            # Python until menuCommit writes data.select_shape (which applyDragOp reads).
            'menuOpen':"""
                var _items_ = state.menu_items[state.menu_kind];
                state.menu_index = 0;
                for (var _i_ = 0; _i_ < _items_.length; _i_++) {
                    if (_items_[_i_][1] == state.select_shape) { state.menu_index = _i_; break; }
                }
                state.menu_open = true;
                self.menuRender();
                self.menuArmTimer();
            """,
            'menuRender':"""
                if (!state.menu_open) { return; }
                var _items_  = state.menu_items[state.menu_kind];
                var _header_ = 'selection shape:';
                var _maxlen_ = _header_.length;
                for (var _i_ = 0; _i_ < _items_.length; _i_++) {
                    _maxlen_ = Math.max(_maxlen_, _items_[_i_][1].length + 4);
                }
                var _w_menu_ = _maxlen_ * 7 + 20,
                    _h_menu_ = (_items_.length + 1) * 14 + 12,
                    _style_  = 'font-family: \\'Courier New\\', monospace; font-size: 11px; fill: #222;';
                var _html_ = '<rect x="8" y="8" width="' + _w_menu_ + '" height="' + _h_menu_ + '"'
                           + ' fill="rgba(240,240,240,0.95)" stroke="#888" stroke-width="1" rx="3"/>'
                           + '<rect x="10" y="' + (8 + 1 + (state.menu_index + 1) * 14) + '" width="' + (_w_menu_ - 4) + '" height="13"'
                           + ' fill="rgba(100,150,255,0.3)"/>'
                           + '<text x="18" y="' + (8 + 12) + '" style="' + _style_ + ' font-weight: bold;">' + _header_ + '</text>';
                for (var _i_ = 0; _i_ < _items_.length; _i_++) {
                    _html_ += '<text x="18" y="' + (8 + 12 + (_i_ + 1) * 14) + '" style="' + _style_ + '">'
                            + '[' + _items_[_i_][0] + '] ' + _items_[_i_][1] + '</text>';
                }
                pickermenu.innerHTML = _html_;
            """,
            'menuCommit':"""
                state.select_shape = state.menu_items[state.menu_kind][state.menu_index][1];
                data.select_shape  = state.select_shape;
                self.menuClose();
            """,
            'menuClose':"""
                if (state.menu_timer != null) { clearTimeout(state.menu_timer); }
                state.menu_timer     = null;
                state.menu_open      = false;
                state.menu_kind      = '';
                pickermenu.innerHTML = '';
            """,
            'menuArmTimer':"""
                if (state.menu_timer != null) { clearTimeout(state.menu_timer); }
                var _self_ = self;
                state.menu_timer = setTimeout(function() { if (state.menu_open) { _self_.menuCommit(); } }, 2500);
            """
        }
    })
    _cls_ref_[0] = cls
    return cls(**kwargs)

def timepi(_timep_, **kwargs):     return _interactivep(_timep_,   'timepi',   **kwargs)
def histopi(_histop_, **kwargs):   return _interactivep(_histop_,  'histopi',  **kwargs)
def xypi(_xyp_, **kwargs):         return _interactivep(_xyp_,     'xypi',     **kwargs)
def chordpi(_chordp_, **kwargs):   return _interactivep(_chordp_,  'chordpi',  **kwargs)
def piepi(_piep_, **kwargs):       return _interactivep(_piep_,    'piepi',    **kwargs)

# ---------------------------------------------------------------------------
# Smallp hit-test helpers (used by SMALLPI callbacks)
# ---------------------------------------------------------------------------

def _tile_at_(plot, cx, cy):
    tw = plot.sm_template.wxh[0]
    th = plot.sm_template.wxh[1] + (plot.txt_h + 3 if plot.draw_labels else 0)
    for key, (tx, ty) in plot.category_to_xy.items():
        if tx <= cx < tx + tw and ty <= cy < ty + th:
            return key
    return None

def _tiles_overlapping_(plot, rx0, ry0, rx1, ry1):
    tw = plot.sm_template.wxh[0]
    th = plot.sm_template.wxh[1] + (plot.txt_h + 3 if plot.draw_labels else 0)
    return [key for key, (tx, ty) in plot.category_to_xy.items()
            if tx < rx1 and tx + tw > rx0 and ty < ry1 and ty + th > ry0]

# ---------------------------------------------------------------------------
# smallpi — interactive wrapper for Smallp
# ---------------------------------------------------------------------------

def smallpi(_smallp_, **kwargs):
    use_webgpu = kwargs.pop('use_webgpu', False)
    _w_, _h_  = _smallp_.wxh_actual
    _gpu_payload_default_ = _smallp_.webgpu() if use_webgpu else None
    _svg_     = _smallp_._repr_svg_() if not use_webgpu else ''
    _cls_ref_ = [None]

    def __init__(self, **kwargs):
        _mvc_ = kwargs.pop('mvc', None)
        super(_cls_ref_[0], self).__init__(**kwargs)
        self.lock = asyncio.Lock()
        if _mvc_ is None:
            self.mvc = InteractionController()
            self.mvc.addStack('default', _smallp_.df_orig)
            self.mvc.view_stack[id(self)] = 'default'
            self.mvc.view_refs[id(self)]  = self
        else:
            self.mvc = _mvc_
            self.mvc.view_stack[id(self)] = 'default'
            self.mvc.view_refs[id(self)]  = self
        self._plot_  = _smallp_
        self._cache_ = {id(_smallp_.df_orig): _smallp_}
        self.param.watch(self.applyDragOp,     'drag_op_finished')
        self.param.watch(self.applyBrushOp,    'brush_changed')
        self.param.watch(self.applyBrushLeave, 'brush_leave_done')
        self.param.watch(self.applyKeyOp,      'key_op_finished')
        if use_webgpu:
            self.param.watch(self.applyGpuError, 'gpu_error')

    def __renderView__(self, df):
        return _smallp_.render_with_df(df)

    def __refreshView__(self):
        if use_webgpu:
            if self.gpu_error: self.mod_inner   = _gpu_error_overlay(self.gpu_error, _w_, _h_)
            else:              self.gpu_payload = self._plot_.webgpu()
        else:
            self.mod_inner = self._plot_._repr_svg_()

    # WebGPU rendering failed in the browser -> surface the error in the overlay.
    # No automatic SVG fallback: the user must re-create the view with use_webgpu=False.
    async def applyGpuError(self, event):
        if self.gpu_error:
            self.mod_inner = _gpu_error_overlay(self.gpu_error, _w_, _h_)

    async def display(self, df, dfs, dfs_index):
        async with self.lock:
            if id(df) not in self._cache_:
                self._cache_[id(df)] = self.__renderView__(df)
            self._plot_ = self._cache_[id(df)]
            self.__refreshView__()
            _ids_ = {id(d) for d in dfs}
            for _id_ in list(self._cache_.keys()):
                if _id_ not in _ids_:
                    del self._cache_[_id_]

    async def applyBrushOp(self, event):
        if not self.brush_on:
            return
        async with self.lock:
            cx, cy = self.x_mouse, self.y_mouse
        key = _tile_at_(self._plot_, cx, cy)
        if key is not None:
            tile_df = self._plot_.category_to_df.get(key)
            if tile_df is not None and len(tile_df) > 0:
                await self.mvc.brushUpdate(self, tile_df)
                return
        await self.mvc.brushClear(self)

    async def applyBrushLeave(self, event):
        await self.mvc.brushClear(self)

    async def applyDragOp(self, event):
        async with self.lock:
            x0, y0 = self.drag_x0, self.drag_y0
            x1, y1 = self.drag_x1, self.drag_y1
            shift  = self.shiftkey

        is_click = abs(x1 - x0) < 5 and abs(y1 - y0) < 5

        if is_click:
            key = _tile_at_(self._plot_, x0, y0)
            selected_keys = [key] if key is not None else []
        else:
            rx0, ry0 = min(x0, x1), min(y0, y1)
            rx1, ry1 = max(x0, x1), max(y0, y1)
            selected_keys = _tiles_overlapping_(self._plot_, rx0, ry0, rx1, ry1)

        if not selected_keys:
            if is_click:
                await self.mvc.popStack(self)
            return

        selected_dfs = []
        selected_idx = set()
        for key in selected_keys:
            tile_df = self._plot_.category_to_df.get(key)
            if tile_df is not None and len(tile_df) > 0:
                selected_dfs.append(tile_df)
                selected_idx.update(tile_df['__p2s_index__'].to_list())

        if not selected_dfs:
            return

        if shift:
            # Subtract selected rows from the current rendered df (always has __p2s_index__)
            new_df = self._plot_.df.filter(~pl.col('__p2s_index__').is_in(selected_idx))
        else:
            # Push union of selected tile dfs directly (they already have __p2s_index__)
            new_df = pl.concat(selected_dfs) if len(selected_dfs) > 1 else selected_dfs[0]

        if len(new_df) > 0:
            await self.mvc.pushStack(self, new_df)

    async def applyKeyOp(self, event):
        async with self.lock:
            op = self.key_op_finished
        if   op == 'q':         await self.mvc.popStack(self)
        elif op == 'Q':         await self.mvc.setStackIndex(self, 0)
        elif op == 'brush_off': await self.mvc.brushClear(self)

    _svg_root_ = f"""
<svg id="svgparentsmallpi" width="{_w_}" height="{_h_}" tabindex="0"
     onkeydown="${{script('myOnKeyDown')}}"{' style="position:absolute;left:0;top:0;"' if use_webgpu else ''}>
  <svg id="mod" x="0" y="0" width="{_w_}" height="{_h_}">${{mod_inner}}</svg>
  <rect id="selbox" x="0" y="0" width="0" height="0" display="none"
        fill="rgba(100,100,255,0.08)" stroke="#4488ff" stroke-width="1"
        stroke-dasharray="4,2" pointer-events="none"/>
  <rect id="screen" x="0" y="0" width="{_w_}" height="{_h_}"
        style="fill:none;pointer-events:all;"
        onmouseover="${{script('myOnMouseOver')}}"
        onmousedown="${{script('myOnMouseDown')}}"
        onmousemove="${{script('myOnMouseMove')}}"
        onmouseup="${{script('myOnMouseUp')}}"
        onmouseleave="${{script('myOnMouseLeave')}}"/>
</svg>"""
    if use_webgpu:
        _template = f"""
<div id="gpuwrap" style="position:relative;width:{_w_}px;height:{_h_}px;">
    <canvas id="gpucanvas" width="{_w_}" height="{_h_}" style="position:absolute;left:0;top:0;"></canvas>
    {_svg_root_}
</div>"""
    else:
        _template = _svg_root_

    _gpu_render_block_ = (f"""
{P2S_GPU_JS}
            if (!window.__P2S_GPU__.supported()) {{ data.gpu_error = 'WebGPU is not available in this browser.'; }}
            else {{
                window.__P2S_GPU__.render(gpucanvas, data.gpu_payload)
                    .catch(function(e) {{ console.warn('p2s webgpu:', e); data.gpu_error = (e && e.message) ? e.message : String(e); }});
            }}
""") if use_webgpu else ''
    _gpu_payload_script_ = ("""
            if (window.__P2S_GPU__ && window.__P2S_GPU__.supported() && !data.gpu_error) {
                window.__P2S_GPU__.render(gpucanvas, data.gpu_payload)
                    .catch(function(e) { console.warn('p2s webgpu:', e); data.gpu_error = (e && e.message) ? e.message : String(e); });
            }
""") if use_webgpu else ''

    _scripts = {
        'render': f"""
            state.dragging = false;
            state.sx = 0; state.sy = 0;
            mod.innerHTML = data.mod_inner;
{_gpu_render_block_}
        """,
        **({'gpu_payload': _gpu_payload_script_} if use_webgpu else {}),
        'myOnMouseOver': """
            svgparentsmallpi.focus();
        """,
        'myOnMouseDown': """
            state.sx = event.offsetX; state.sy = event.offsetY;
            state.dragging = true;
            data.drag_x0 = Math.round(event.offsetX);
            data.drag_y0 = Math.round(event.offsetY);
        """,
        'myOnMouseMove': """
            data.x_mouse = event.offsetX; data.y_mouse = event.offsetY;
            if (data.brush_on) { data.brush_changed += 1; }
            if (state.dragging) {
                selbox.setAttribute('x',       Math.min(state.sx, event.offsetX));
                selbox.setAttribute('y',       Math.min(state.sy, event.offsetY));
                selbox.setAttribute('width',   Math.abs(event.offsetX - state.sx));
                selbox.setAttribute('height',  Math.abs(event.offsetY - state.sy));
                selbox.setAttribute('display', 'block');
            }
        """,
        'myOnMouseUp': """
            if (!state.dragging) return;
            state.dragging = false;
            data.drag_x1 = Math.round(event.offsetX);
            data.drag_y1 = Math.round(event.offsetY);
            data.shiftkey = event.shiftKey;
            selbox.setAttribute('display', 'none');
            data.drag_op_finished = !data.drag_op_finished;
        """,
        'myOnMouseLeave': """
            state.dragging = false;
            selbox.setAttribute('display', 'none');
            data.brush_leave_done = !data.brush_leave_done;
        """,
        'myOnKeyDown': """
            var k = event.key;
            if (k === 's') {
                data.brush_on = !data.brush_on;
                if (!data.brush_on) { data.key_op_finished = 'brush_off'; }
            } else if (k === 'q' && !event.shiftKey) {
                data.key_op_finished = 'q';
            } else if (k === 'Q' || (event.shiftKey && k === 'q')) {
                data.key_op_finished = 'Q';
            }
        """,
        'mod_inner': """
            mod.innerHTML = data.mod_inner;
        """,
    }

    cls = type('SMALLPI', (ReactiveHTML,), {
        'mod_inner':         param.String(default=_svg_),
        'x_mouse':           param.Integer(default=0),
        'y_mouse':           param.Integer(default=0),
        'brush_on':          param.Boolean(default=False),
        'brush_changed':     param.Integer(default=0),
        'brush_leave_done':  param.Boolean(default=False),
        'drag_x0':           param.Integer(default=0),
        'drag_y0':           param.Integer(default=0),
        'drag_x1':           param.Integer(default=0),
        'drag_y1':           param.Integer(default=0),
        'drag_op_finished':  param.Boolean(default=False),
        'shiftkey':          param.Boolean(default=False),
        'key_op_finished':   param.String(default=''),
        **({'gpu_payload': param.Dict(default=_gpu_payload_default_), 'gpu_error': param.String(default='')} if use_webgpu else {}),
        '__init__':          __init__,
        '__renderView__':    __renderView__,
        '__refreshView__':   __refreshView__,
        **({'applyGpuError': applyGpuError} if use_webgpu else {}),
        'display':           display,
        'applyBrushOp':      applyBrushOp,
        'applyBrushLeave':   applyBrushLeave,
        'applyDragOp':       applyDragOp,
        'applyKeyOp':        applyKeyOp,
        '_template':         _template,
        '_scripts':          _scripts,
    })
    _cls_ref_[0] = cls
    return cls(**kwargs)

_PLOT_TYPE_TO_WRAPPER_ = {
    'Timep':  timepi,
    'Histop': histopi,
    'XYp':    xypi,
    'ChP':    chordpi,
    'Piep':   piepi,
    'Smallp': smallpi,
}

def _collect_leaves(layout):
    leaves = []
    for item in layout:
        if isinstance(item, list): leaves.extend(_collect_leaves(item))
        else:                      leaves.append(item)
    return leaves

def _sketch_placeholder_html(item):
    # Generic static stand-in for an interactive-only leaf that exposes no
    # snapshot (no sketchHtml() result and no _repr_svg_()). Sized to the
    # widget's wxh when known so the sketch keeps the layout's proportions.
    _wxh_ = getattr(item, 'wxh', None) or (160, 120)
    _w_, _h_ = _wxh_
    _label_  = getattr(item, 'sketch_label', None) or type(item).__name__
    _cx_, _cy_ = _w_ // 2, _h_ // 2
    return (
        f'<svg width="{_w_}" height="{_h_}">'
        f'<rect x="0.5" y="0.5" width="{_w_ - 1}" height="{_h_ - 1}" fill="#f4f4f4"'
        f' stroke="#bdbdbd" stroke-width="1" stroke-dasharray="4 3"/>'
        f'<text x="{_cx_}" y="{_cy_ - 4}" text-anchor="middle"'
        f' font-family="Helvetica,Arial,sans-serif" font-size="11px"'
        f' fill="#8a8a8a">{_label_}</text>'
        f'<text x="{_cx_}" y="{_cy_ + 10}" text-anchor="middle"'
        f' font-family="Helvetica,Arial,sans-serif" font-size="9px"'
        f' font-style="italic" fill="#b0b0b0">interactive</text>'
        f'</svg>'
    )


def _sketch_leaf_html(item, use_webgpu):
    # Resolution order, uniform across static and interactive-only leaves:
    #   1. webgpu() render when requested and available (mirrors panelize());
    #      the GPU tier works the same for any leaf that exposes webgpu().
    #   2. sketchHtml() static snapshot for interactive-only widgets
    #      (e.g. stack_controli returns its current frame).
    #   3. _repr_svg_() for static plot components.
    #   4. a generic labeled placeholder, so a sketch never raises.
    if use_webgpu and getattr(item, 'webgpu', None) is not None:
        from .p2s_webgpu_runtime import standalone_html
        _payload_ = item.webgpu()
        if _payload_ is not None:
            return standalone_html(_payload_, border='none')
    _sketch_fn_ = getattr(item, 'sketchHtml', None)
    if callable(_sketch_fn_):
        _html_ = _sketch_fn_(use_webgpu)
        if _html_ is not None:
            return _html_
    _repr_svg_ = getattr(item, '_repr_svg_', None)
    if callable(_repr_svg_):
        return _repr_svg_()
    return _sketch_placeholder_html(item)

def _build_sketch_html(layout, orientation='column', use_webgpu=False):
    flex_dir         = 'column' if orientation == 'column' else 'row'
    next_orientation = 'row'    if orientation == 'column' else 'column'
    inner = ''
    for item in layout:
        if isinstance(item, list): inner += _build_sketch_html(item, next_orientation, use_webgpu)
        else:                      inner += _sketch_leaf_html(item, use_webgpu)
    return f'<div style="display:flex;flex-direction:{flex_dir};gap:0;margin:0;padding:0;">{inner}</div>'

def _build_interactive(layout, view_map, orientation='column'):
    container        = pn.Column if orientation == 'column' else pn.Row
    next_orientation = 'row'     if orientation == 'column' else 'column'
    children = []
    for item in layout:
        if isinstance(item, list): children.append(_build_interactive(item, view_map, next_orientation))
        else:                      children.append(view_map[id(item)])
    return container(*children)

def panelizeSketch(layout, use_webgpu=False):
    pn.extension()
    return pn.pane.HTML(_build_sketch_html(layout, use_webgpu=use_webgpu))

def panelize(layout: Any, stack: str = 'default', use_webgpu: bool = False) -> Any:
    pn.extension()
    plots = _collect_leaves(layout)
    mvc   = InteractionController()
    _init_df_ = next((p.df_orig for p in plots if hasattr(p, 'df_orig')), None)
    mvc.addStack(stack, _init_df_)
    views = []
    for p in plots:
        if isinstance(p, ReactiveHTML) or (isinstance(p, pn.viewable.Viewable) and hasattr(p, 'display')):
            if hasattr(p, 'mvc'): p.mvc = mvc
            mvc.view_stack[id(p)] = stack
            mvc.view_refs[id(p)]  = p
            views.append(p)
        else:
            # use_webgpu applies only to components with a webgpu() representation
            # (currently xyp, histop); the rest render through their SVG wrappers
            _kwargs_ = {'use_webgpu': True} if use_webgpu and getattr(p, 'webgpu', None) is not None else {}
            _wrapper_ = _PLOT_TYPE_TO_WRAPPER_.get(type(p).__name__)
            if   _wrapper_ is not None:        views.append(_wrapper_(p, mvc=mvc, **_kwargs_))
            elif hasattr(p, 'panelWrapper'):
                # Extension point for components outside this module (e.g. dev-tree
                # prototypes): the component supplies its own Panel view, which may
                # implement display() / receiveSelection() / receivePositions() to
                # take part in the corresponding MVC event fan-outs.
                views.append(p.panelWrapper(mvc=mvc))
            else:
                raise TypeError(f'panelize(): no wrapper for component type '
                                f'"{type(p).__name__}" (register it in '
                                f'_PLOT_TYPE_TO_WRAPPER_ or give it a panelWrapper() method)')
    for p, v in zip(plots, views):
        v.sizing_mode = 'fixed'
        _wxh_ = getattr(p, 'wxh_actual', None) or getattr(p, 'wxh', None)
        if _wxh_ is not None:
            v.width  = _wxh_[0]
            v.height = _wxh_[1]
    for view in views:
        others = [v for v in views if v is not view]
        mvc.link(view, others, on='stack', stack=stack)
        if type(view).__name__ in ('LINKPI', 'SLPI'):
            sel_targets = [v for v in others if hasattr(v, 'receiveSelection')]
            if sel_targets:
                mvc.link(view, sel_targets, on='selection', stack=stack)
        if type(view).__name__ == 'LINKPI':
            pos_targets = [v for v in others if hasattr(v, 'receivePositions')]
            if pos_targets:
                mvc.link(view, pos_targets, on='positions', stack=stack)
    _container_ = _build_interactive(layout, {id(p): v for p, v in zip(plots, views)})
    _container_.mvc = mvc
    return _container_

def linkpi(_linkp_, mvc=None, use_webgpu=False, **kwargs):
    _w_, _h_  = _linkp_.wxh
    _gpu_payload_default_ = _linkp_.webgpu() if use_webgpu else None
    _svg_     = '' if use_webgpu else _linkp_._repr_svg_()
    _cls_ref_ = [None]

    # ── link-size / node-size / link-opacity cycle menus (shift-L/O/P + ctrl) ──
    # These mirror the shift-G / shift-W layout pickers: a modal list overlay that
    # cycles forward on the shift key and backward on ctrl, committing on Enter /
    # timeout / mouse-out. The named sizes are always offered; a user-supplied
    # hardcoded number is the only float/int added to the size cycle (and becomes
    # the current selection). Opacity cycles the fixed 10..100 grid. Each item's
    # label doubles as the value committed back to Python; the 'none' label maps
    # back to a real None (links / nodes not drawn) via __sizeLabelToValue__.
    _NAMED_SIZES_    = ['none', 'nil', 'small', 'medium', 'large', 'vary']
    _SIZE_MNEMONICS_ = {'none': 'o', 'nil': 'n', 'small': 's', 'medium': 'm', 'large': 'g', 'vary': 'v'}

    def _num_size_label(v):
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)

    def _build_size_menu(current):
        items = [[_SIZE_MNEMONICS_[nm], nm] for nm in _NAMED_SIZES_]
        if isinstance(current, bool):
            cur = 'small'
        elif isinstance(current, (int, float)):
            cur = _num_size_label(current)
            items.append(['#', cur])          # only a user-supplied number is added
        elif current is None:
            cur = 'none'
        else:
            cur = str(current)
            if cur not in _NAMED_SIZES_:
                items.append(['#', cur])
        return items, cur

    _link_size_items_, _link_size_cur_ = _build_size_menu(_linkp_.link_size)
    _node_size_items_, _node_size_cur_ = _build_size_menu(_linkp_.node_size)

    _link_opacity_items_ = [[str((p // 10) % 10), str(p)] for p in range(10, 101, 10)]
    _link_opacity_cur_   = (str(int(round(float(_linkp_.link_opacity) * 100)))
                            if _linkp_.link_opacity is not None else '100')

    _LINK_SHAPES_      = ['line', 'curve', 'flowmap']
    _link_shape_items_ = [[str(_i_ + 1), _nm_] for _i_, _nm_ in enumerate(_LINK_SHAPES_)]
    _link_shape_cur_   = str(getattr(_linkp_, 'link_shape', 'line') or 'line')
    if _link_shape_cur_ not in _LINK_SHAPES_:
        _link_shape_items_.append(['#', _link_shape_cur_])

    #
    # Constructor
    #
    def __init__(self, **kwargs):
        _mvc_ = kwargs.pop('mvc', mvc)   # allow override via kwargs, fall back to closure mvc
        super(_cls_ref_[0], self).__init__(**kwargs)

        from .polars2svg import Polars2SVG
        self.rt_self   = Polars2SVG()
        self.w, self.h = _w_, _h_

        self.dfs        = [_linkp_.df_orig]
        self.dfs_layout = [_linkp_]
        self.df_level   = 0
        self.graphs     = [self.rt_self.createNetworkXGraph(_linkp_.df_orig, _linkp_.relationships)]

        self.selected_entities = set()
        self.label_mode        = 'all labels' if _linkp_.draw_labels else 'no labels'
        self.sticky_labels     = set(_linkp_.label_only) if _linkp_.label_only else set()

        self.ln_params = {
            'relationships': _linkp_.relationships,
            'pos':           _linkp_.pos,
            'draw_labels':   _linkp_.draw_labels,
            'label_only':    _linkp_.label_only,
        }
        if _linkp_.node_labels is not None:
            self.ln_params['node_labels'] = _linkp_.node_labels

        self.GRID                 = 'grid'
        self.CIRCLE               = 'circle'
        self.SUNFLOWER            = 'sunflower'
        self.GRID_BY_COLOR        = 'grid (color)'
        self.GRID_BY_COLOR_CLOUDS = 'grid (color, clouds)'
        self.RESCALE              = 'rescale'
        self.layout_modes         = [label for _, label in _LAYOUT_MODE_MENU_]

        self.SPRING_NX            = 'spring nx'
        self.FORCE_DIRECTED       = 'force directed'
        self.HYPERTREE            = 'hyper tree'
        self.HYPERTREE_DONUT      = 'hyper tree donut'
        self.CONVEY_PROXIMITY     = 'convey proximity'
        self.LANDMARK_MDS         = 'landmark mds'
        self.LANDMARK_MDS_POS     = 'landmark mds pos'
        self.PIVOT_MDS            = 'pivot mds'
        self.CONNECTED_COMPONENTS = 'connected components'
        self.CIRCLE_PACK          = 'circle pack'
        self.NEIGHBORHOOD_SPATIAL = 'neighborhood (spatial)'
        self.NEIGHBORHOOD_GRAPH   = 'neighborhood (graph)'
        self.TFDP_LAYOUT          = 't-fdp'
        self.layout_operations    = [label for _, label in _LAYOUT_OP_MENU_]
        self.layout_mode          = self.GRID
        self.layout_operation     = self.SPRING_NX

        # Current selections for the link-size / node-size / link-opacity cycle
        # pickers (committed labels round-trip through these params -> applySizeChoice).
        self.link_size_choice     = _link_size_cur_
        self.node_size_choice     = _node_size_cur_
        self.link_opacity_choice  = _link_opacity_cur_
        self.link_shape_choice    = _link_shape_cur_

        self.previous_layouts = []
        self.max_undo_levels  = 20
        self.lock             = asyncio.Lock()

        # Background cycling (the 'b' key): 0 = none, 1 = background, 2 = background + labels.
        # layout_background holds the {name: shape} dict returned by the last layout
        # operation that supports a background (e.g. the donut / circle-pack / subnet
        # layouts via their return_cells parameter); it is None when the current layout
        # does not provide one, in which case cycling has nothing to draw.
        self.background_state  = 0
        self.layout_background = None
        self._bg_label_color_  = '#000000'

        # Community detection (the 'd' key): the LinkP's node_color spec as authored,
        # so that shift-d can restore it after community colors have been pushed over
        # every level of the stack. community_colors holds the {node: '#rrggbb'} map
        # while communities are being shown (None otherwise).
        self._orig_node_color_ = _linkp_.node_color
        self.community_colors  = None

        if _mvc_ is None:
            self.mvc = InteractionController()
            self.mvc.addStack('default', _linkp_.df_orig)
            self.mvc.view_stack[id(self)] = 'default'
            self.mvc.view_refs[id(self)]  = self
        else:
            self.mvc = _mvc_
            self.mvc.view_refs[id(self)] = self

        # GPU mode: the plot renders on the canvas (gpu_payload default); the
        # mod SVG layer stays empty. SVG mode: mod carries the rendered plot.
        if use_webgpu: self.mod_inner = ''
        else:          self.mod_inner = _linkp_._repr_svg_()
        self.allentitiespath = _linkp_.__createPathDescriptionForAllEntities__()
        self.selectionpath   = 'M -100 -100 l 10 0 l 0 10 l -10 0 l 0 -10 Z'
        self.info_str        = f'0 Selected | {self.label_mode} | {self.layout_mode} | {self.layout_operation} | {self.__backgroundStateLabel__()}'

        self._layout_registry = self.__buildLayoutRegistry__()

        self.param.watch(self.applyDragOp,            'drag_op_finished')
        self.param.watch(self.applyMoveOp,            'move_op_finished')
        self.param.watch(self.applyWheelOp,           'wheel_op_finished')
        self.param.watch(self.applyMiddleOp,          'middle_op_finished')
        self.param.watch(self.applyKeyOp,             'key_op_finished')
        self.param.watch(self.applyLayoutInteraction, 'layout_shape')
        self.param.watch(self.unselectedMoveOp,       'unselected_move_op_finished')
        self.param.watch(self.applySearchOp,          'search_op_finished')
        self.param.watch(self.applyLayoutChoice,      ['layout_mode', 'layout_operation'])
        self.param.watch(self.applySizeChoice,        ['link_size_choice', 'node_size_choice', 'link_opacity_choice', 'link_shape_choice'])
        if use_webgpu:
            self.param.watch(self.applyGpuError,      'gpu_error')

    # WebGPU rendering failed in the browser -> surface the error in the overlay.
    # No automatic SVG fallback: the user must re-create the view with use_webgpu=False.
    async def applyGpuError(self, event):
        if self.gpu_error:
            self.mod_inner = _gpu_error_overlay(self.gpu_error, _w_, _h_)

    #
    # saveLayout() - save the current layout
    #
    def saveLayout(self, filename):
        _pos_ = self.dfs_layout[self.df_level].pos
        _lu_ = {'node': list(_pos_.keys()),
                'x':    [v[0] for v in _pos_.values()],
                'y':    [v[1] for v in _pos_.values()]}
        pl.DataFrame(_lu_).write_parquet(filename)

    #
    # loadLayout() - load a layout
    #
    def loadLayout(self, filename):
        if filename.lower().endswith('.csv'): _df_ = pl.read_csv(filename)
        else:                                 _df_ = pl.read_parquet(filename)
        _pos_ = self.dfs_layout[self.df_level].pos
        for row in _df_.iter_rows(named=True): _pos_[row['node']] = (float(row['x']), float(row['y']))
        self.__refreshView__(info=False)

    #
    # display() - MVC callback: navigate the stack or show a brush selection
    #
    async def display(_self_, df, dfs, dfs_index):
        _is_brush_ = df is not dfs[dfs_index]
        async with _self_.lock:
            if _is_brush_:
                _self_.setSelectedEntitiesAndNotifyOthers(_self_._extractNodes_(df))
                _self_.__refreshView__(comp=False, all_ents=False)
            else:
                while _self_.df_level < dfs_index:
                    _self_.pushStack(dfs[_self_.df_level + 1])
                while _self_.df_level > dfs_index:
                    _self_.popStack()

    #
    # replaceBaseDataframe() - MVC callback: swap in a new base dataframe, reset stack, preserve node positions
    #
    async def replaceBaseDataframe(self, df):
        _pos_ = dict(self.dfs_layout[self.df_level].pos)
        _vw_  = self.dfs_layout[self.df_level].view_window
        _new_ln_ = _linkp_.render_with(df, pos=_pos_, view_window=_vw_)
        _g_      = self.rt_self.createNetworkXGraph(df, self.ln_params['relationships'])
        self.dfs               = [df]
        self.dfs_layout        = [_new_ln_]
        self.graphs            = [_g_]
        self.df_level          = 0
        self.previous_layouts  = []
        self.selected_entities = set()
        self.__refreshView__()

    #
    # receiveSelection() - MVC callback: apply an incoming node selection
    #
    async def receiveSelection(_self_, entities):
        _self_.selected_entities = set(entities) & set(_self_.graphs[_self_.df_level].nodes())
        _self_.__refreshView__(comp=False)

    #
    # _extractNodes_() - collect all node names visible in a filtered DataFrame
    #
    def _extractNodes_(_self_, df):
        nodes = set()
        for rel in _linkp_.relationships:
            fm, to = rel[0], rel[1]
            if fm in df.columns: nodes.update(df[fm].drop_nulls().unique().to_list())
            if to in df.columns: nodes.update(df[to].drop_nulls().unique().to_list())
        return nodes

    #
    # _matchNodesByRegex_() - regex-match nodes/labels against one or more patterns
    # - mirrors the substring-matching branch of selectEntities(), but with re.search()
    # - invalid patterns (e.g. an unbalanced group typed into a live search box) are
    #   skipped rather than raised, so they just contribute no matches
    #
    def _matchNodesByRegex_(_self_, patterns, all_nodes, ignore_case=True):
        if isinstance(patterns, str): _patterns_ = set([patterns])
        else:                         _patterns_ = set(patterns)
        _flags_    = re.IGNORECASE if ignore_case else 0
        _compiled_ = []
        for _pattern_ in _patterns_:
            try:             _compiled_.append(re.compile(_pattern_, _flags_))
            except re.error: pass # invalid regex -- contributes no matches

        _set_ = set()
        _node_labels_ = _linkp_.node_labels or {}
        str_to_node   = {str(n): n for n in all_nodes} if _node_labels_ else {}
        for _regex_ in _compiled_:
            if _node_labels_:
                for _label_key_ in _node_labels_.keys():
                    _actual_node_ = str_to_node.get(str(_label_key_))
                    if _actual_node_ is not None and _regex_.search(str(_node_labels_[_label_key_])):
                        _set_.add(_actual_node_)
            for _node_ in all_nodes:
                if _regex_.search(str(_node_)): _set_.add(_node_)
        return _set_

    #
    # selectEntities() - set the selected entities
    #
    def selectEntities(self, 
                       selection,                # string or set
                       set_op       = 'replace', # "replace", "add", "subtract", "intersect"
                       method       = 'exact',   # "exact", "substring", "regex"
                       ignore_case  = True):     # ignore the case when performing the match
        # Get all nodes in the current graph // these are the non-labeled variants
        all_nodes = set(self.graphs[self.df_level].nodes())

        # Perform either substring or regex matching if selected
        if   method == 'substring': # SUBSTRING MATCHES
            if isinstance(selection, str): _substrings_ = set([selection])
            else:                          _substrings_ = set(selection)
            _set_ = set()
            _node_labels_ = _linkp_.node_labels or {}
            str_to_node   = {str(n): n for n in all_nodes} if _node_labels_ else {}
            for _substring_ in _substrings_:
                if ignore_case: _substring_ = _substring_.lower()
                if _node_labels_:
                    for _label_key_ in _node_labels_.keys():
                        _actual_node_ = str_to_node.get(str(_label_key_))
                        if _actual_node_ is not None: # only match nodes in the graph
                            if   ignore_case:
                                if _substring_ in str(_node_labels_[_label_key_]).lower(): _set_.add(_actual_node_)
                            elif _substring_ in str(_node_labels_[_label_key_]): _set_.add(_actual_node_)
                for _node_ in all_nodes:
                    if   ignore_case:
                        if _substring_ in str(_node_).lower(): _set_.add(_node_)
                    elif _substring_ in str(_node_): _set_.add(_node_)
        elif method == 'regex':     # REGEX MATCHES
            _set_ = self._matchNodesByRegex_(selection, all_nodes, ignore_case)
        else:                       # EXACT MATCHES
            # Fix up the selection so that it's definitely a set...
            if    selection is None:                                         selection_as_set = set()
            elif  isinstance(selection, list) or isinstance(selection, set): selection_as_set = set(selection)
            elif  isinstance(selection, dict):                               selection_as_set = set(selection.keys())
            else:                                                            selection_as_set = set([selection])

            # Fix the case...
            if ignore_case: selection_as_set = {x.lower() for x in selection_as_set}

            # Iterate through the nodes...
            _node_labels_ = _linkp_.node_labels or {}
            if _node_labels_: # node labels handled a little differently
                _set_ = set()
                str_to_node = {str(n): n for n in all_nodes}
                for _label_key_ in _node_labels_.keys():
                    _label_       = _node_labels_[_label_key_]
                    _actual_node_ = str_to_node.get(str(_label_key_))
                    if _actual_node_ is None: continue

                    if ignore_case: _label_, _node_cased_ = _label_.lower(), str(_label_key_).lower()
                    else:           _label_, _node_cased_ = _label_, str(_label_key_)

                    if _node_cased_ in selection_as_set or _label_ in selection_as_set: _set_.add(_actual_node_)
                for _node_ in all_nodes:
                    _node_cased_ = str(_node_).lower() if ignore_case else _node_
                    if _node_cased_ in selection_as_set: _set_.add(_node_)
                self.setSelectedEntitiesAndNotifyOthers(_set_)
            else: # just use the selection
                if ignore_case:
                    _set_ = set()
                    for _node_ in all_nodes:
                        _node_cased_ = str(_node_).lower()
                        if _node_cased_ in selection_as_set: _set_.add(_node_)
                else:
                    _set_ = selection_as_set & all_nodes

        if   set_op == 'replace':   self.setSelectedEntitiesAndNotifyOthers(_set_)
        elif set_op == 'add':       self.setSelectedEntitiesAndNotifyOthers(self.selected_entities | _set_)
        elif set_op == 'subtract':  self.setSelectedEntitiesAndNotifyOthers(self.selected_entities - _set_)
        elif set_op == 'intersect': self.setSelectedEntitiesAndNotifyOthers(self.selected_entities & _set_)

        self.__refreshView__(comp=False)

    #
    # selectedEntities() - return the selected entities
    #
    def selectedEntities(self):
        _node_labels_ = _linkp_.node_labels or {}
        _set_ = set()
        if _node_labels_:
            for _node_ in self.selected_entities:
                if _node_ in _node_labels_: _set_.add(_node_labels_[_node_])
                else:                       _set_.add(_node_)
        else:
            _set_ = self.selected_entities
        return _set_

    #
    # selectedNodes() - return the selected nodes
    # - distinction is that the node is the representation within the dataframe
    # - versus the entity may be the lookup label if the node_labels is set
    # - if there are no node_labels, this should return the same as selectedEntities()
    #
    def selectedNodes(self):
        _node_labels_ = _linkp_.node_labels or {}
        if _node_labels_:
            _set_, covered = set(), set()
            for _node_ in _node_labels_.keys():
                if _node_                  in self.selectedEntities() or \
                   _node_labels_[_node_]   in self.selectedEntities(): _set_.add(_node_)
                covered.add(_node_), covered.add(_node_labels_[_node_])
            for _node_ in self.selectedEntities():
                if _node_ not in covered: _set_.add(_node_)
            return _set_
        else:
            return set(self.selected_entities)

    #
    # updateLinkNodeParam() - update a param & refresh the views
    # - performs at all levels of the stack
    #
    def updateLinkNodeParam(self, name, value):
        for i in range(len(self.dfs_layout)):
            setattr(self.dfs_layout[i], name, value)
            self.dfs_layout[i].invalidateRender()
        self.__refreshView__(comp=True)

    #
    # ^^^ -- These methods are for external callers
    #

    #
    # __renderView__() - create a new LinkP for the given DataFrame using current pos/view
    #
    def __renderView__(self, df):
        _pos_ = dict(self.dfs_layout[self.df_level].pos)
        _vw_  = self.dfs_layout[self.df_level].view_window
        return _linkp_.render_with(df, pos=_pos_, view_window=_vw_)

    #
    # __cacheNodePositions__() - cache the node positions for undo operations
    #
    def __cacheNodePositions__(self):
        _copy_ = copy.deepcopy(self.dfs_layout[self.df_level].pos)
        # Appended unconditionally: a != dedupe against the previous entry fails on
        # nx.spring_layout() output (numpy array values), so no skip is attempted.
        self.previous_layouts.append(_copy_)
        while len(self.previous_layouts) > self.max_undo_levels: self.previous_layouts.pop(0)

    #
    # setSelectedEntitiesAndNotifyOthers() - set the selected entities & notify via mvc
    #
    def setSelectedEntitiesAndNotifyOthers(self, _set_):
        self.selected_entities = set(_set_)

        if self.mvc is not None:
            try:
                asyncio.get_event_loop().call_soon(
                    lambda s=set(_set_): asyncio.ensure_future(self.mvc.selectionUpdate(self, s))
                )
            except RuntimeError:
                pass  # no event loop in test/non-async context — skip mvc notification

    #
    # __buildLayoutRegistry__() - build the layout dispatch registry once at init time.
    # Each entry maps an operation name to a callable(ln, g, sel) -> Optional[dict].
    #
    def __buildLayoutRegistry__(self):
        rt = self.rt_self

        def _landmark_mds_pos_(ln, g, sel):
            if len(sel) == 0:
                return LandmarkMDSLayout(g, rt_self=rt).results()
            lm_pos = {n: ln.pos[n] for n in sel}
            return LandmarkMDSLayout(g, landmark_pos=lm_pos, rt_self=rt).results()

        def _circle_pack_(ln, g, sel):
            if len(sel) > 0:
                return None
            pos, shapes = rt.circlePackLayout(g, ln.pos)
            return pos, shapes   # shapes -> background (the enclosing circles)

        registry = {
            self.SPRING_NX:            lambda ln, g, sel: nx.spring_layout(g) if len(sel) == 0 else None,
            self.FORCE_DIRECTED:       lambda ln, g, sel: (
                PolarsForceDirectedLayout(g).results() if len(sel) == 0
                else PolarsForceDirectedLayout(g, pos=ln.pos, static_nodes=set(g.nodes()) - set(sel)).results()
            ),
            self.HYPERTREE:            lambda ln, g, sel: rt.hyperTreeLayout(g, roots=sel),
            self.HYPERTREE_DONUT:      lambda ln, g, sel: rt.hyperTreeDonutLayout(g, roots=sel, return_cells=True),
            self.CONNECTED_COMPONENTS: lambda ln, g, sel: rt.treeMapLayout(g, ln.pos) if len(sel) == 0 else None,
            self.CIRCLE_PACK:          _circle_pack_,
            # Neighborhood layouts (return (pos, cells) -> cells become the background).
            # 'spatial' clusters the current layout (positions unchanged); 'graph'
            # detects weighted communities and repositions them apart.
            self.NEIGHBORHOOD_SPATIAL: lambda ln, g, sel: (
                rt.neighborhoodLayout(g, pos=ln.pos, mode='spatial', return_cells=True) if len(sel) == 0 else None
            ),
            self.NEIGHBORHOOD_GRAPH:   lambda ln, g, sel: (
                rt.neighborhoodLayout(g, mode='graph', return_cells=True) if len(sel) == 0 else None
            ),
            self.CONVEY_PROXIMITY:     lambda ln, g, sel: ConveyProximityLayout(g, use_resistive_distances=True).results() if len(sel) == 0 else None,
            self.LANDMARK_MDS:         lambda ln, g, sel: (
                LandmarkMDSLayout(g, rt_self=rt).results() if len(sel) == 0
                else LandmarkMDSLayout(g, landmarks=sel, rt_self=rt).results()
            ),
            self.LANDMARK_MDS_POS:     _landmark_mds_pos_,
            self.PIVOT_MDS:            lambda ln, g, sel: PivotMDSLayout(g, rt_self=rt).results() if len(sel) == 0 else None,
        }
        if _TFDP_AVAILABLE:
            registry[self.TFDP_LAYOUT] = lambda ln, g, sel: (
                _TFDPLayout(g).results() if len(sel) == 0
                else _TFDPLayout(g, pos=ln.pos, selection=set(sel), pin_background=True).results()
            )
        return registry

    #
    # __backgroundStateLabel__() - human-readable label for the current background state
    #
    def __backgroundStateLabel__(self):
        return ('no background', 'background', 'background + labels')[self.background_state]

    #
    # __applyBackgroundState__() - push the current background-cycle state onto the
    # active LinkP. State 0 hides the background; state 1 shows the layout-provided
    # shapes; state 2 shows the shapes plus their labels. With no layout background
    # available, nothing is drawn regardless of state.
    #
    def __applyBackgroundState__(self, refresh=True):
        _ln_ = self.dfs_layout[self.df_level]
        if self.background_state == 0 or self.layout_background is None:
            _ln_.background             = None
            _ln_.background_label_color = None
        else:
            _ln_.background             = self.layout_background
            _ln_.background_label_color = self._bg_label_color_ if self.background_state == 2 else None
        _ln_.invalidateRender()
        if refresh: self.__refreshView__()

    #
    # __contractCollapsedGraph__() - collapse nodes that share an exact (x,y)
    # location into a single representative node so that the layout algorithm
    # sees the group as one node. Every edge incident on a member is rewired to
    # the representative (parallel edges are merged, their weights summed) and
    # intra-group edges are dropped. Returns (g_c, pos_c, sel_c, members) where
    # `members` maps each representative to the list of underlying nodes, or
    # None when nothing collapses (no two nodes share a location).
    #
    def __contractCollapsedGraph__(self, _ln_, _g_, _sel_):
        _pos_ = _ln_.pos

        # Group graph nodes by exact location. Nodes without a position get a
        # unique key so they never merge with anything.
        _loc_to_nodes_ = {}
        for _node_ in _g_.nodes():
            if _node_ in _pos_:
                _xy_  = _pos_[_node_]
                _key_ = (float(_xy_[0]), float(_xy_[1]))
            else:
                _key_ = ('__nopos__', _node_)
            _loc_to_nodes_.setdefault(_key_, []).append(_node_)

        # Representative (first node encountered) -> its members.
        _members_, _rep_of_ = {}, {}
        for _nodes_ in _loc_to_nodes_.values():
            _rep_ = _nodes_[0]
            _members_[_rep_] = _nodes_
            for _node_ in _nodes_: _rep_of_[_node_] = _rep_

        if all(len(_ns_) == 1 for _ns_ in _members_.values()):
            return None   # nothing collapses -> run the algorithm unchanged

        _g_c_ = type(_g_)()
        _g_c_.add_nodes_from(_members_.keys())
        for _u_, _v_, _data_ in _g_.edges(data=True):
            _ru_, _rv_ = _rep_of_[_u_], _rep_of_[_v_]
            if _ru_ == _rv_:
                continue   # intra-group edge -> represented by the node itself
            if _g_c_.has_edge(_ru_, _rv_):
                _g_c_[_ru_][_rv_]['weight'] = _g_c_[_ru_][_rv_].get('weight', 1) + _data_.get('weight', 1)
            else:
                _g_c_.add_edge(_ru_, _rv_, **_data_)

        _pos_c_ = {_rep_: _pos_[_rep_] for _rep_ in _members_ if _rep_ in _pos_}

        # A representative is selected if any of its members is selected.
        _sel_set_ = set(_sel_)
        _sel_c_   = [_rep_ for _rep_ in _members_ if any(_m_ in _sel_set_ for _m_ in _members_[_rep_])]

        return _g_c_, _pos_c_, _sel_c_, _members_

    #
    # __expandContractedResult__() - map a layout result computed on the
    # contracted graph back onto every underlying node. Each representative's
    # computed position is assigned to all of its members (so a collapsed group
    # stays together at its newly-placed location). Background cells are passed
    # through unchanged.
    #
    def __expandContractedResult__(self, _result_, _members_):
        if isinstance(_result_, tuple): _pos_, _cells_ = _result_
        else:                           _pos_, _cells_ = _result_, None
        if _pos_ is None:
            return _result_
        _pos_full_ = {}
        for _rep_, _xy_ in _pos_.items():
            for _node_ in _members_.get(_rep_, (_rep_,)):
                _pos_full_[_node_] = _xy_
        if isinstance(_result_, tuple): return _pos_full_, _cells_
        return _pos_full_

    #
    # __layoutOperation__() - apply a layout operation via the registry.
    #
    def __layoutOperation__(self, _layout_op_, _ln_, _g_, _sel_):
        handler = self._layout_registry.get(_layout_op_)
        if handler is None:
            return False
        # Collapse exactly-coincident nodes into representatives so the
        # algorithm treats each stacked group as a single node (with its edges).
        _contracted_ = self.__contractCollapsedGraph__(_ln_, _g_, _sel_)
        if _contracted_ is None:
            _result_ = handler(_ln_, _g_, _sel_)
        else:
            _g_c_, _pos_c_, _sel_c_, _members_ = _contracted_
            _result_ = handler(_ContractedLayoutView_(_ln_, _pos_c_), _g_c_, _sel_c_)
            _result_ = self.__expandContractedResult__(_result_, _members_)
        # A layout that supports a background returns (pos, cells); others return pos (or None).
        if isinstance(_result_, tuple): _pos_, _cells_ = _result_
        else:                           _pos_, _cells_ = _result_, None
        if _pos_ is not None:
            for _node_ in _pos_: _ln_.pos[_node_] = (float(_pos_[_node_][0]), float(_pos_[_node_][1]))
            # Capture the layout's background (or clear a stale one from a prior layout).
            self.layout_background = _cells_ if _cells_ else None
            return True
        return False

    #
    # applyLayoutInteraction() - apply layout interaction to the selected entities.
    #
    def applyLayoutInteraction(self, event):
        try:
            x0, y0, x1, y1   = self.drag_x0, self.drag_y0, self.drag_x1, self.drag_y1
            _updated_pos_     = self.apply_layout_interaction(x0, y0, x1, y1, self.layout_shape)
            if _updated_pos_:
                self.__refreshView__(info=False)
        finally:
            self.layout_shape = ""

    #
    # applyMiddleOp() - apply middle operation -- either pan view or reset view
    #
    async def applyMiddleOp(self,event):
        async with self.lock:
            try:
                if self.middle_op_finished:
                    x0, y0, x1, y1 = self.x0_middle, self.y0_middle, self.x1_middle, self.y1_middle
                    dx, dy         = x1 - x0, y1 - y0
                    _comp_ , _adj_coordinate_ = self.dfs_layout[self.df_level], (x0,y0)
                    if _comp_ is not None:
                        if (abs(self.x0_middle - self.x1_middle) <= 1) and (abs(self.y0_middle - self.y1_middle) <= 1):
                            if _comp_.applyMiddleClick(_adj_coordinate_):
                                self._propagate_view_changes_()
                        else:
                            if _comp_.applyMiddleDrag(_adj_coordinate_, (dx,dy)):
                                self._propagate_view_changes_()
            finally:
                self.middle_op_finished = False

    #
    # applyWheelOp() - apply mouse wheel operation (zoom in & out)
    #
    async def applyWheelOp(self,event):
        async with self.lock:
            try:
                if self.wheel_op_finished:
                    x, y = self.wheel_x, self.wheel_y
                    rots = max(-800, min(800, self.wheel_rots))  # clamp: factor stays in [0.2, 1.8]
                    if rots != 0:
                        _comp_ , _adj_coordinate_ = self.dfs_layout[self.df_level], (x,y)
                        if _comp_ is not None:
                            if _comp_.applyScrollEvent(rots, _adj_coordinate_):
                                self._propagate_view_changes_()
            finally:
                self.wheel_op_finished = False
                self.wheel_rots        = 0

    #
    # setAnimation() - set the animation string (and thus the SVG view)
    #
    def setAnimation(self, animation):
        time.sleep(0.001) 
        self.animation_inner = ''
        time.sleep(0.001) 
        self.animation_inner = animation

    #
    # __refreshView__() - refresh the view
    #
    def __refreshView__(self, comp=True, info=True, all_ents=True, sel_ents=True):
        if (comp):
            _ln_ = self.dfs_layout[self.df_level]
            if   not use_webgpu: self.mod_inner   = _ln_.renderSVG()
            elif self.gpu_error: self.mod_inner   = _gpu_error_overlay(self.gpu_error, _w_, _h_)
            else:                self.gpu_payload = _ln_.webgpu()
            # A comp refresh is the single chokepoint after every operation that
            # may have moved nodes (drag-moves, layout ops, undo, load); notify
            # 'positions'-linked views (they diff & skip when nothing moved).
            self.mvc.positionsUpdate(self, _ln_.pos)
        if (info):     self.info_str         = f'{len(self.selected_entities)} Selected | {self.label_mode} | {self.layout_mode} | {self.layout_operation} | {self.__backgroundStateLabel__()}'
        if (all_ents): self.allentitiespath  = self.dfs_layout[self.df_level].__createPathDescriptionForAllEntities__()
        if (sel_ents): self.selectionpath    = self.dfs_layout[self.df_level].__createPathDescriptionOfSelectedEntities__(my_selection=self.selected_entities)

    #
    # popStack() - as long as there are items on the stack, go up the stack
    #
    def popStack(self):
        if self.df_level == 0:
            at_top = 'TOP' if self.df_level == 0 else ''
            self.setAnimation(f'<text x="5" y="15" fill="black"> popStack [{len(self.dfs)} @ {self.df_level}] {at_top} </text>')
            return

        self.df_level -= 1

        self.__refreshView__()

        at_top = 'TOP' if self.df_level == 0 else ''
        self.setAnimation(f'<text x="5" y="15" fill="black"> popStack [{len(self.dfs)} @ {self.df_level}] {at_top} </text>')

    #
    # setStackPosition() - set to a specific position
    #
    def setStackPostion(self, i_found):
        if i_found < 0 or i_found >= len(self.dfs_layout): return

        self.df_level = i_found

        self.__refreshView__()

        self.setAnimation(f'<text x="5" y="15" fill="black"> setStackPosition [{len(self.dfs)} @ {self.df_level}] </text>')

    #
    # pushStack() - push a dataframe onto the stack
    #
    def pushStack(self, df, g=None):
        if g is None: g = self.rt_self.createNetworkXGraph(df, self.ln_params['relationships'])

        _ln_ = self.__renderView__(df)
        _ln_.applyViewConfiguration(self.dfs_layout[self.df_level])

        # This is necessary to shrink the stack
        if len(self.dfs_layout) > (self.df_level+1):
            new_dfs, new_dfs_layout, new_graphs = [], [], []
            for i in range(self.df_level+1):
                new_dfs.append(self.dfs[i]), new_dfs_layout.append(self.dfs_layout[i]), new_graphs.append(self.graphs[i])
            self.dfs, self.dfs_layout, self.graphs = new_dfs, new_dfs_layout, new_graphs

        # Render the new view and update all of the stack variables
        self.dfs        .append(df)
        self.dfs_layout .append(_ln_)
        self.graphs     .append(g)
        self.df_level += 1

        # Update selected entities based on what's available
        self.setSelectedEntitiesAndNotifyOthers(self.selected_entities & g.nodes())
        self.__refreshView__()

        self.setAnimation(f'<text x="5" y="15" fill="black"> pushStack [{len(self.dfs)}]</text>')

    #
    # applyKeyOp() - apply specified key operation
    #
    async def applyKeyOp(self, event):
        _mvc_stack_action_ = None
        async with self.lock:
            _ln_ = self.dfs_layout[self.df_level]
            #
            # "E" - Expand / Expand w/ Directed
            #
            if self.key_op_finished == 'e' or self.key_op_finished == 'E':
                if   self.ctrlkey and len(self.selected_entities) > 0:
                    _entities_, _xs_, _ys_, _weights_ = [], [], [], []
                    for _entity_ in self.selected_entities:
                        _xy_ = _ln_.pos[_entity_]
                        _entities_.append(_entity_), _xs_.append(_xy_[0]), _ys_.append(_xy_[1]), _weights_.append(self.graphs[self.df_level].degree(_entity_))
                    _df_      = pl.DataFrame({'e':_entities_, 'x':_xs_, 'y':_ys_, 'w':_weights_})
                    _results_ = self.rt_self.uniformSampleDistributionInScatterplotsViaSectorBasedTransformation(_df_, 'x', 'y', 'w')
                    for i in range(len(_results_)):
                        _entity_, _x_, _y_ = _results_['e'][i], _results_['x'][i], _results_['y'][i]
                        _ln_.pos[_entity_] = (_x_, _y_)
                    _ln_.invalidateRender()
                    self.__refreshView__()
                elif self.key_op_finished == 'E':
                    _digraph_ = self.rt_self.createNetworkXGraph(self.dfs[self.df_level], self.ln_params['relationships'], use_digraph=True)
                    _new_set_ = set(self.selected_entities)
                    for _node_ in self.selected_entities:
                        for _nbor_ in _digraph_.neighbors(_node_):
                            _new_set_.add(_nbor_)
                    self.setSelectedEntitiesAndNotifyOthers(_new_set_)
                    self.__refreshView__(comp=False, all_ents=False)
                else:
                    _new_set_ = set(self.selected_entities)
                    for _node_ in self.selected_entities:
                        for _nbor_ in self.graphs[self.df_level].neighbors(_node_):
                            _new_set_.add(_nbor_)
                    self.setSelectedEntitiesAndNotifyOthers(_new_set_)
                    self.__refreshView__(comp=False, all_ents=False)

            #
            # "Q" - Invert Selection / Common Neighbors
            #            
            elif self.key_op_finished == 'q' or self.key_op_finished == 'Q':
                if   self.key_op_finished == 'Q': # common neighbors
                    inter_set = None
                    for _node_ in self.selected_entities:
                        nbor_set = set()
                        for _nbor_ in self.graphs[self.df_level].neighbors(_node_):
                            nbor_set.add(_nbor_)
                        if inter_set is None: inter_set = nbor_set             # first time, it gets the nbors
                        else:                 inter_set = inter_set & nbor_set # all other times it's and'ed
                    self.setSelectedEntitiesAndNotifyOthers(inter_set if inter_set is not None else set())
                else:                   # invert selection
                    _new_set_ = set()
                    for _node_ in self.graphs[self.df_level]:
                        if _node_ not in self.selected_entities:
                            _new_set_.add(_node_)
                    self.setSelectedEntitiesAndNotifyOthers(_new_set_)

                self.__refreshView__(comp=False, all_ents=False)

            #
            # "S" - Set Sticky Labels & Remove Sticky Labels
            #
            elif self.key_op_finished == 's' or self.key_op_finished == 'S':
                _label_set_changed_ = True
                if   self.shiftkey and self.ctrlkey:
                    if   self.label_mode == 'all labels':    
                        self.label_mode = 'sticky labels'
                        _ln_.labelOnly(self.sticky_labels)
                        _ln_.drawLabels(True)
                        self.ln_params['draw_labels'] = True
                    elif self.label_mode == 'sticky labels':
                        self.label_mode = 'no labels'
                        _ln_.drawLabels(False)
                        self.ln_params['draw_labels'] = False                        
                    else:                                    
                        self.label_mode = 'all labels'
                        _ln_.drawLabels(True)
                        self.ln_params['draw_labels'] = True
                        _ln_.labelOnly(set())
                    _label_set_changed_ = False
                    self.__refreshView__(all_ents=False, sel_ents=False)
                elif self.shiftkey:
                    self.sticky_labels  = self.sticky_labels - self.selected_entities # subtract from the current set
                elif                   self.ctrlkey:
                    self.sticky_labels = self.sticky_labels | self.selected_entities  # add to the current set
                else:
                    self.sticky_labels  = set(self.selected_entities)                 # make a new set object with the selected

                # if the set of sticky labels has changed, update the label set & refresh
                if _label_set_changed_:
                    if self.label_mode == 'sticky labels': _ln_.labelOnly(self.sticky_labels)
                    self.ln_params['label_only'] = self.sticky_labels
                    self.__refreshView__(info=False, all_ents=False, sel_ents=False)

            #
            # "T" - Collapse (to a point, horizontal line, or vertical line)
            #
            elif len(self.selected_entities) > 0 and (self.key_op_finished == 't' or self.key_op_finished == 'T'):
                self.__cacheNodePositions__()
                self.apply_collapse_to(self.x_mouse, self.y_mouse, self.shiftkey, self.ctrlkey)
                self.__refreshView__(info=False)

            elif self.key_op_finished == 'u' and len(self.previous_layouts) > 0:
                self.apply_undo()
                self.__refreshView__(info=False)

            #
            # "B" - Cycle the background display (none -> background -> background + labels)
            #
            elif self.key_op_finished == 'b' or self.key_op_finished == 'B':
                self.background_state = (self.background_state + 1) % 3
                self.__applyBackgroundState__()

            #
            # "D" - Detect graph communities (louvain) & color the nodes by community;
            #       shift-d restores the node coloring that the LinkP was created with.
            #
            elif self.key_op_finished == 'd' or self.key_op_finished == 'D':
                if self.key_op_finished == 'D':
                    self.community_colors = None
                    self.updateLinkNodeParam('node_color', self._orig_node_color_)
                    self.setAnimation('<text x="5" y="15" fill="black"> communities: cleared </text>')
                else:
                    # Mirrors the 'w' layout branch: an algorithm failure leaves the
                    # view untouched rather than killing the callback.
                    try:              _node_color_ = self.apply_community_detection()
                    except Exception: _node_color_ = None
                    if _node_color_ is not None:
                        self.updateLinkNodeParam('node_color', _node_color_)
                        _communities_found_ = len(set(_node_color_.values()))
                        self.setAnimation(f'<text x="5" y="15" fill="black"> {_communities_found_} communities </text>')

            #
            # "A" - Toggle link arrows
            #
            elif self.key_op_finished == 'a':
                _new_arrows_ = not getattr(_ln_, 'link_arrows', False)
                self.updateLinkNodeParam('link_arrows', _new_arrows_)
                self.setAnimation(f'<text x="5" y="15" fill="black"> link arrows: {"on" if _new_arrows_ else "off"} </text>')

            #
            # "Z" - Select nodes with the same color as the one that the mouse is over
            #
            elif self.key_op_finished == 'z' or self.key_op_finished == 'Z':
                self._select_by_attribute_at_mouse_(lambda ln, e: ln.nodeColor(e), lambda ln, a: ln.nodesWithColor(a))

            #
            # "N" - Select nodes with the same shape as the one that the mouse is over
            #
            elif self.key_op_finished == 'n' or self.key_op_finished == 'N':
                self._select_by_attribute_at_mouse_(lambda ln, e: ln.nodeShape(e), lambda ln, a: ln.nodesWithShape(a))

            #
            # 'C' - Center on Selected (if selected) or Reset View (if not selected) / Selected + Neighbors
            #
            elif self.key_op_finished == 'c' or self.key_op_finished == 'C':
                _rerender_ = False
                def _viewForEntities_(lp, entities):
                    _xs_ = [lp.pos[e][0] for e in entities if e in lp.pos]
                    _ys_ = [lp.pos[e][1] for e in entities if e in lp.pos]
                    if not _xs_: return lp.view_window
                    _x0_, _x1_ = min(_xs_), max(_xs_)
                    _y0_, _y1_ = min(_ys_), max(_ys_)
                    _px_ = max((_x1_ - _x0_) * 0.1, 0.05)
                    _py_ = max((_y1_ - _y0_) * 0.1, 0.05)
                    return (_x0_ - _px_, _y0_ - _py_, _x1_ + _px_, _y1_ + _py_)
                if   self.ctrlkey: # copy to the clipboard
                    if len(self.selected_entities) > 0:
                        if self.shiftkey: # copy the label lookups (if they exist)
                            _list_ = []
                            for x in self.selected_entities:
                                if 'node_labels' in self.ln_params and x in self.ln_params['node_labels']: _list_.append(self.ln_params['node_labels'][x])
                                else:                                                                      _list_.append(x)
                            pyperclip.copy('\n'.join(list(_list_)))
                        else: # copy the nodes as they are named within the dataframe
                            pyperclip.copy('\n'.join(list(self.selected_entities)))
                elif self.key_op_finished == 'C': # recenter on the selected entities & neighbors
                    if len(self.selected_entities) > 0:
                        _new_set_ = set(self.selected_entities)
                        for _node_ in self.selected_entities:
                            for _nbor_ in self.graphs[self.df_level].neighbors(_node_):
                                _new_set_.add(_nbor_)
                        _ln_.setViewWindow(_viewForEntities_(_ln_, _new_set_))
                        _rerender_ = True
                else:
                    if len(self.selected_entities) > 0: # Zoom to selected entities
                        _ln_.setViewWindow(_viewForEntities_(_ln_, self.selected_entities))
                        _rerender_ = True
                    else:                               # Recenter complete view
                        _ln_.view_window = None
                        _ln_.__calculateGeometry__()
                        _rerender_ = True
                
                if _rerender_:
                    self._propagate_view_changes_(invalidate_all=True)
            #
            # 'x' - remove selected nodes from the dataset (push the stack)
            # ... 'X' restore removed nodes (pop the stack)
            #
            elif self.key_op_finished == 'x' or self.key_op_finished == 'X':
                _level_before_ = self.df_level
                if   self.key_op_finished == 'X': self.apply_pop()
                elif self.key_op_finished == 'x': self.apply_push_selected()
                if   self.df_level > _level_before_: _mvc_stack_action_ = ('push', self.dfs[self.df_level])
                elif self.df_level < _level_before_: _mvc_stack_action_ = ('pop',  None)

            #
            # 'ctrl-shift-x' - collapse visualized edges to one row each (push the stack)
            #
            elif self.key_op_finished == 'ctrl_shift_x':
                _level_before_ = self.df_level
                self.apply_collapse_edges()
                if self.df_level > _level_before_: _mvc_stack_action_ = ('push', self.dfs[self.df_level])

            #
            # Degree Related Operations
            #
            elif len(self.key_op_finished) == 1 and self.key_op_finished in '0123456789':
                _match_ = set()
                c       = self.key_op_finished
                min_degree = 7  if c == '7' else 20 if c == '8' else 50  if c == '9' else 100    if c == '0' else None
                max_degree = 20 if c == '7' else 50 if c == '8' else 100 if c == '9' else 10_000 if c == '0' else None

                if min_degree is not None:
                    for _node_ in self.graphs[self.df_level]:
                        if self.graphs[self.df_level].degree(_node_) >= min_degree and self.graphs[self.df_level].degree(_node_) < max_degree: _match_.add(_node_)
                else:
                    _degree_ = int(self.key_op_finished)
                    for _node_ in self.graphs[self.df_level]:
                        if self.graphs[self.df_level].degree(_node_) == _degree_: _match_.add(_node_)

                if   self.shiftkey and self.ctrlkey: self.setSelectedEntitiesAndNotifyOthers(self.selected_entities & _match_)
                elif self.shiftkey:                  self.setSelectedEntitiesAndNotifyOthers(self.selected_entities - _match_)
                elif self.ctrlkey:                   self.setSelectedEntitiesAndNotifyOthers(self.selected_entities | _match_)  
                else:                                self.setSelectedEntitiesAndNotifyOthers(_match_)

                self.__refreshView__(comp=False, all_ents=False)

            #
            # (Layout mode / operation selection is handled entirely in JS by
            #  the shift-G / shift-W picker menus; commits arrive through the
            #  layout_mode / layout_operation params and applyLayoutChoice.)
            #

            #
            # Apply a layout operation to the selected nodes (or all nodes if no selection in place)
            #
            elif self.key_op_finished == 'w':
                self.__cacheNodePositions__()

                # Write new positions to _ln_.pos[_node_] = (x, y)
                try:
                    _pos_modified_ = self.__layoutOperation__(self.layout_operation, _ln_, self.graphs[self.df_level], self.selected_entities)
                except Exception:
                    _pos_modified_ = False
                if _pos_modified_: # If positions were modified, recenter and re-render
                    _ln_.view_window = None  # clear so __calculateGeometry__ recomputes from new positions
                    _ln_.__calculateGeometry__()

                    # Sync the (possibly new) layout background onto the active view
                    # without an extra refresh; the invalidate+refresh below repaints it.
                    self.__applyBackgroundState__(refresh=False)

                    # Invalidate the stack of views & re-render
                    for i in range(len(self.dfs_layout)): self.dfs_layout[i].invalidateRender()
                    self.__refreshView__(info=True)

            self.key_op_finished = ''
        if _mvc_stack_action_ is not None:
            if _mvc_stack_action_[0] == 'push': await self.mvc.pushStack(self, _mvc_stack_action_[1])
            else:                               await self.mvc.popStack(self)

    # -------------------------------------------------------------------------
    # Synchronous interaction helpers — each mirrors one user action exactly.
    # The async Panel callbacks (applyDragOp, applyMoveOp, applyKeyOp) delegate
    # to these so the same logic is reachable from unit tests without a browser.
    # -------------------------------------------------------------------------

    def _propagate_view_changes_(self, invalidate_all=False):
        if invalidate_all:
            for i in range(len(self.dfs_layout)):
                self.dfs_layout[i].invalidateRender()
        self.__refreshView__(info=False)
        _ln_ = self.dfs_layout[self.df_level]
        for i in range(len(self.dfs_layout)):
            if i != self.df_level:
                self.dfs_layout[i].applyViewConfiguration(_ln_)

    def _select_by_attribute_at_mouse_(self, get_attr_fn, nodes_with_attr_fn):
        _ln_ = self.dfs_layout[self.df_level]
        _entities_ = _ln_.entitiesAtPoint((self.x_mouse, self.y_mouse)) or set()
        _attrs_ = {get_attr_fn(_ln_, e) for e in _entities_}
        _result_ = set()
        for _a_ in _attrs_: _result_ |= set(nodes_with_attr_fn(_ln_, _a_))
        self.selectEntities(_result_, _resolve_set_op(self.shiftkey, self.ctrlkey), 'exact')

    def _propagate_positions(self, updated_pos):
        """Push updated world positions to every stack level except the current one."""
        for i in range(len(self.dfs_layout)):
            if i != self.df_level:
                for _key_, _new_pos_ in updated_pos.items():
                    if _key_ in self.dfs_layout[i].pos:
                        self.dfs_layout[i].pos[_key_] = _new_pos_
            self.dfs_layout[i].invalidateRender()

    def apply_drag_select(self, x0, y0, x1, y1, shiftkey=False, ctrlkey=False):
        """Simulate a left-button drag selection over the rectangle (x0,y0)→(x1,y1).
        Updates self.selected_entities; does not trigger mvc notifications."""
        _x0, _y0 = min(x0, x1), min(y0, y1)
        _x1, _y1 = max(x0, x1), max(y0, y1)
        if _x0 == _x1: _x1 += 1
        if _y0 == _y1: _y1 += 1
        _rect_ = Polygon([(_x0,_y0), (_x0,_y1), (_x1,_y1), (_x1,_y0)])
        _overlapping_ = set(self.dfs_layout[self.df_level].overlappingEntities(_rect_))
        if   shiftkey and ctrlkey: self.selected_entities = set(self.selected_entities) & _overlapping_
        elif shiftkey:             self.selected_entities = set(self.selected_entities) - _overlapping_
        elif ctrlkey:              self.selected_entities = set(self.selected_entities) | _overlapping_
        else:                      self.selected_entities = _overlapping_

    def apply_move_selected(self, dx, dy):
        """Simulate dragging the selected nodes by (dx, dy) screen pixels.
        Caches positions for undo and propagates new world positions to all stack levels."""
        self.__cacheNodePositions__()
        _updated_pos_ = self.dfs_layout[self.df_level].__moveSelectedEntities__(
            (dx, dy), my_selection=self.selected_entities)
        self._propagate_positions(_updated_pos_)
        return _updated_pos_

    def apply_push_selected(self):
        """Simulate the 'x' key: remove selected nodes and push a filtered df."""
        if not self.selected_entities:
            return False
        _g_ = copy.deepcopy(self.graphs[self.df_level])
        for _entity_ in self.selected_entities:
            _g_.remove_node(_entity_)
        _df_ = self.rt_self.filterDataFrameByGraph(
            self.dfs[self.df_level], self.ln_params['relationships'], _g_)
        if len(_df_) > 0:
            self.pushStack(_df_, g=_g_)
            return True
        return False

    def apply_pop(self):
        """Simulate the 'X' key: pop the top stack frame."""
        self.popStack()

    def apply_collapse_edges(self):
        """Simulate ctrl-shift-x: keep one row per visualized edge, push the result.
        No selection -> collapse every edge. With selection -> collapse only edges
        adjacent to a selected node; other edges keep all of their rows."""
        _df_orig_ = self.dfs[self.df_level]
        _df_ = self.rt_self.collapseDataFrameEdgesToOneRow(
            _df_orig_, self.ln_params['relationships'], self.selected_entities)
        if _df_ is not None and len(_df_) > 0 and len(_df_) < len(_df_orig_):
            self.pushStack(_df_)
            return True
        return False

    def apply_community_detection(self):
        """Simulate the 'd' key: louvain communities over the graph at this stack level,
        one color per community. Nodes that share an exact position are merged first (the
        same treatment the layout algorithms give them) so a stacked group counts as a
        single community member. Node positions are not touched. Returns the
        {node: '#rrggbb'} map, or None when there is nothing to color."""
        _ln_, _g_ = self.dfs_layout[self.df_level], self.graphs[self.df_level]
        if _g_ is None or _g_.number_of_nodes() == 0: return None

        _contracted_ = self.__contractCollapsedGraph__(_ln_, _g_, set())
        if _contracted_ is None: _g_c_, _members_ = _g_, {_n_: [_n_] for _n_ in _g_.nodes()}
        else:                    _g_c_, _, _, _members_ = _contracted_

        _communities_ = nx.community.louvain_communities(nx.to_undirected(_g_c_),
                                                         weight='weight', resolution=1.0, seed=42)
        if not _communities_: return None

        # One color per community, hashed off the community's canonical (lexicographically
        # smallest) member so that re-running 'd' keeps colors stable rather than
        # reshuffling them with louvain's community ordering.
        _keys_    = [min(str(_m_) for _m_ in _comm_) for _comm_ in _communities_]
        _key_hex_ = self.rt_self.colors(_keys_)

        # Expand each representative's color back onto every node it stands for
        _node_color_ = {}
        for _comm_, _key_ in zip(_communities_, _keys_):
            for _rep_ in _comm_:
                for _node_ in _members_.get(_rep_, (_rep_,)):
                    _node_color_[_node_] = _key_hex_[_key_]
        self.community_colors = _node_color_
        return _node_color_

    def apply_collapse_to(self, sx, sy, shiftkey=False, ctrlkey=False):
        """Simulate the 't' key: collapse selected nodes to screen position (sx, sy).
        Propagates new world positions to all stack levels."""
        _ln_ = self.dfs_layout[self.df_level]
        _target_wx_ = _ln_.xT_inv(sx)
        _target_wy_ = _ln_.yT_inv(sy)
        _updated_pos_ = {}
        if shiftkey:
            for _entity_ in self.selected_entities:
                if _entity_ in _ln_.pos:
                    _ln_.pos[_entity_] = (_ln_.pos[_entity_][0], _target_wy_)
                    _updated_pos_[_entity_] = _ln_.pos[_entity_]
        elif ctrlkey:
            for _entity_ in self.selected_entities:
                if _entity_ in _ln_.pos:
                    _ln_.pos[_entity_] = (_target_wx_, _ln_.pos[_entity_][1])
                    _updated_pos_[_entity_] = _ln_.pos[_entity_]
        else:
            for _entity_ in self.selected_entities:
                _ln_.pos[_entity_] = (_target_wx_, _target_wy_)
                _updated_pos_[_entity_] = (_target_wx_, _target_wy_)
        self._propagate_positions(_updated_pos_)
        return _updated_pos_

    def apply_layout_interaction(self, x0, y0, x1, y1, layout_shape):
        """Simulate 'y' (line) or 'g' (grid/pattern) layout on selected nodes.
        Propagates new world positions to all stack levels."""
        as_list       = list(self.selected_entities)
        _ln_          = self.dfs_layout[self.df_level]
        _updated_pos_ = {}
        if len(as_list) > 1:
            if layout_shape == self.GRID:
                pos_adj = self.rt_self.rectangularLayout(self.graphs[self.df_level], as_list, bounds=(x0,y0,x1,y1))
                self.__cacheNodePositions__()
                for _node_ in pos_adj:
                    _ln_.pos[_node_] = (float(_ln_.xT_inv(pos_adj[_node_][0])), float(_ln_.yT_inv(pos_adj[_node_][1])))
                    _updated_pos_[_node_] = _ln_.pos[_node_]
            elif layout_shape == self.GRID_BY_COLOR or layout_shape == self.GRID_BY_COLOR_CLOUDS:
                _node_to_color_ = {}
                for _node_ in as_list: _node_to_color_[_node_] = _ln_.nodeColor(_node_)
                pos_adj = self.rt_self.treeMapNodeColorLayout(self.graphs[self.df_level], as_list, _node_to_color_,
                                                               collapse=(layout_shape == self.GRID_BY_COLOR_CLOUDS),
                                                               bounds=(x0,y0,x1,y1))
                for _node_ in pos_adj:
                    _ln_.pos[_node_] = (float(_ln_.xT_inv(pos_adj[_node_][0])), float(_ln_.yT_inv(pos_adj[_node_][1])))
                    _updated_pos_[_node_] = _ln_.pos[_node_]
            elif layout_shape == self.RESCALE:
                x0_orig = _ln_.pos[as_list[0]][0]; y0_orig = _ln_.pos[as_list[0]][1]
                x1_orig = _ln_.pos[as_list[0]][0]; y1_orig = _ln_.pos[as_list[0]][1]
                for _node_ in as_list:
                    x0_orig, y0_orig = min(x0_orig, _ln_.pos[_node_][0]), min(y0_orig, _ln_.pos[_node_][1])
                    x1_orig, y1_orig = max(x1_orig, _ln_.pos[_node_][0]), max(y1_orig, _ln_.pos[_node_][1])
                for _node_ in as_list:
                    x,     y     = _ln_.pos[_node_]
                    xperc, yperc = (x - x0_orig)/(x1_orig - x0_orig), (y - y0_orig)/(y1_orig - y0_orig)
                    x_new, y_new = x0 + xperc*(x1 - x0), y0 + yperc*(y1 - y0)
                    _ln_.pos[_node_] = (float(_ln_.xT_inv(x_new)), float(_ln_.yT_inv(y_new)))
                    _updated_pos_[_node_] = _ln_.pos[_node_]
            elif layout_shape == self.CIRCLE:
                wx0, wy0 = _ln_.xT_inv(x0), _ln_.yT_inv(y0)
                wx1, wy1 = _ln_.xT_inv(x1), _ln_.yT_inv(y1)
                r = sqrt((wx0 - wx1)**2 + (wy0 - wy1)**2)
                if r < 0.001: r = 0.001
                pos_adj = self.rt_self.circularOptimizedLayout(self.graphs[self.df_level], as_list, _ln_.pos, xy=(wx0,wy0), r=r)
                self.__cacheNodePositions__()
                for _node_ in pos_adj:
                    _ln_.pos[_node_] = (pos_adj[_node_][0], pos_adj[_node_][1])
                    _updated_pos_[_node_] = _ln_.pos[_node_]
            elif layout_shape == self.SUNFLOWER:
                r = sqrt((x0 - x1)**2 + (y0 - y1)**2)
                pos_adj = self.rt_self.sunflowerSeedLayout(self.graphs[self.df_level], as_list, xy=(x0,y0), r_max=r)
                self.__cacheNodePositions__()
                for _node_ in pos_adj:
                    _ln_.pos[_node_] = (float(_ln_.xT_inv(pos_adj[_node_][0])), float(_ln_.yT_inv(pos_adj[_node_][1])))
                    _updated_pos_[_node_] = _ln_.pos[_node_]
            elif layout_shape in ("line", "v-line", "h-line"):
                if   layout_shape == "v-line": x0, x1 = x1, x1
                elif layout_shape == "h-line": y0, y1 = y1, y1
                wx0, wy0 = _ln_.xT_inv(x0), _ln_.yT_inv(y0)
                wx1, wy1 = _ln_.xT_inv(x1), _ln_.yT_inv(y1)
                pos_adj = self.rt_self.linearOptimizedLayout(self.graphs[self.df_level], as_list, _ln_.pos, ((wx0,wy0),(wx1,wy1)))
                self.__cacheNodePositions__()
                for _node_ in pos_adj:
                    _ln_.pos[_node_] = (pos_adj[_node_][0], pos_adj[_node_][1])
                    _updated_pos_[_node_] = _ln_.pos[_node_]
        elif len(as_list) == 1:
            self.__cacheNodePositions__()
            _ln_.pos[as_list[0]] = (float(_ln_.xT_inv((x0+x1)/2)), float(_ln_.yT_inv((y0+y1)/2)))
            _updated_pos_[as_list[0]] = _ln_.pos[as_list[0]]
        if _updated_pos_:
            self._propagate_positions(_updated_pos_)
        return _updated_pos_

    def apply_undo(self):
        """Simulate the 'u' key: restore the last cached layout and propagate to all stack levels."""
        if not self.previous_layouts:
            return {}
        _previous_pos_ = self.previous_layouts[-1]
        _ln_          = self.dfs_layout[self.df_level]
        _updated_pos_ = {}
        for _entity_ in _previous_pos_:
            _ln_.pos[_entity_] = _previous_pos_[_entity_]
            _updated_pos_[_entity_] = _previous_pos_[_entity_]
        self.previous_layouts = self.previous_layouts[:-1]
        self._propagate_positions(_updated_pos_)
        return _updated_pos_

    #
    # applyDragOp() - select the nodes within the drag operations bounding box.
    #
    async def applyDragOp(self, event):
        async with self.lock:
            if self.drag_op_finished:
                self.apply_drag_select(
                    self.drag_x0, self.drag_y0, self.drag_x1, self.drag_y1,
                    self.shiftkey, self.ctrlkey)
                self.setSelectedEntitiesAndNotifyOthers(self.selected_entities)
                self.__refreshView__(comp=False, all_ents=False)
            self.drag_op_finished = False

    #
    # applyMoveOp() - apply a move operation to the selected node(s)
    # - may also be used to de-select a selected node when the op string is "Subtract" and no drag occurs
    #
    async def applyMoveOp(self, event):
        async with self.lock:
            try:
                if self.move_op_finished:
                    if self.drag_x0 == self.drag_x1 and self.drag_y0 == self.drag_y1 and self.shiftkey:
                        _overlapping_ = self.dfs_layout[self.df_level].entitiesAtPoint((self.drag_x0, self.drag_y0))
                        if _overlapping_:
                            self.setSelectedEntitiesAndNotifyOthers(set(self.selected_entities) - set(_overlapping_))
                        self.__refreshView__(comp=False, all_ents=False)
                    else:
                        self.apply_move_selected(self.drag_x1 - self.drag_x0, self.drag_y1 - self.drag_y0)
                        self.__refreshView__()
            finally:
                self.move_op_finished = False

    #
    # unselectedMoveOp() - occurs when user clicks directly on an unselected node.
    #
    async def unselectedMoveOp(self, event):
        async with self.lock:
            if self.unselected_move_op_finished:
                _x_, _y_ = self.allentities_x0, self.allentities_y0
                _overlapping_entities_ = self.dfs_layout[self.df_level].entitiesAtPoint((_x_, _y_))
                if _overlapping_entities_ is None: _overlapping_entities_ = set()

                if   self.ctrlkey:  self.setSelectedEntitiesAndNotifyOthers(set(self.selected_entities) | set(_overlapping_entities_))
                elif self.shiftkey: self.setSelectedEntitiesAndNotifyOthers(set(self.selected_entities) - set(_overlapping_entities_))
                else:               self.setSelectedEntitiesAndNotifyOthers(set(_overlapping_entities_))

                if self.drag_x0 == self.drag_x1 and self.drag_y0 == self.drag_y1:
                    self.__refreshView__(comp=False, all_ents=False)
                elif self.shiftkey:
                    self.__refreshView__(comp=False, all_ents=False)
                else:
                    self.apply_move_selected(self.drag_x1 - self.drag_x0, self.drag_y1 - self.drag_y0)
                    self.__refreshView__()

            self.unselected_move_op_finished = False

    async def applySearchOp(self, event):
        async with self.lock:
            if self.search_str:
                _s_ = self.search_str
                if   _s_.startswith('+'): set_op, _s_ = 'add',       _s_[1:]
                elif _s_.startswith('-'): set_op, _s_ = 'subtract',  _s_[1:]
                elif _s_.startswith('&'): set_op, _s_ = 'intersect', _s_[1:]
                else:                     set_op       = 'replace'
                if len(_s_) >= 2 and _s_.startswith('/') and _s_.endswith('/'):
                    method, _s_ = 'regex', _s_[1:-1]
                else:
                    method      = 'substring'
                if _s_:
                    self.selectEntities(_s_, set_op=set_op, method=method, ignore_case=True)

    #
    # applyLayoutChoice() - a picker-menu commit in JS set layout_mode /
    # layout_operation; only the info line needs to reflect the new choice.
    #
    def applyLayoutChoice(self, *events):
        self.__refreshView__(comp=False, all_ents=False, sel_ents=False)

    #
    # __sizeLabelToValue__() - convert a size picker label back to the value the
    # LinkP expects: 'none' becomes None (not drawn); a numeric string
    # (user-supplied hardcoded size) becomes a float; a named size ('small',
    # 'vary', ...) stays a string.
    #
    def __sizeLabelToValue__(self, label):
        if label == 'none':             return None
        try:                            return float(label)
        except (TypeError, ValueError): return label

    #
    # applySizeChoice() - a size/opacity/shape picker-menu commit in JS set one
    # of the *_choice params; push the new value onto the LinkP(s) and re-render.
    #
    def applySizeChoice(self, event):
        if   event.name == 'link_opacity_choice':
            try:                            _op_ = int(event.new) / 100.0
            except (TypeError, ValueError): return
            self.updateLinkNodeParam('link_opacity', _op_)
        elif event.name == 'link_size_choice':
            self.updateLinkNodeParam('link_size', self.__sizeLabelToValue__(event.new))
        elif event.name == 'node_size_choice':
            self.updateLinkNodeParam('node_size', self.__sizeLabelToValue__(event.new))
        elif event.name == 'link_shape_choice':
            self.updateLinkNodeParam('link_shape', event.new)

    _keyboard_commands_ = """
/ . | search: type substring + Enter (prefix +add -remove &intersect); Escape to cancel
a . | toggle link arrows (on | off)
b . | cycle background (none | background | background + labels)
c . | reset view or focus view on selected
 .. | shift-c ........ | focus view on selected + neighbors
 .. | ctrl-c ......... | copy selected nodes to clipboard (ctrl-shift-c uses node labels)
d . | detect communities (louvain) & color nodes by community
 .. | shift-d ........ | clear community colors
e . | expand selection | shift-e uses directed graph
 .. | ctrl-e ......... | even out distribution of selected nodes
g . | layout upon next mouse drag
 .. | shift-g ........ | open layout-mode picker: mnemonic key selects; ctrl-g reverses
h . | toggle help display
l . | open link shape picker (line | curve | flowmap); l cycles
 .. | shift-l ........ | open link size picker (ctrl-l reverses)
n . | select node under mouse by shape (shift, ctrl, and ctrl-shift apply)
 .. | shift-o ........ | open link opacity picker (ctrl-o reverses)
 .. | shift-p ........ | open node size picker (ctrl-p reverses)
q . | invert selection
 .. | shift-q ........ | common neighbors
s . | set sticky labels
 .. | shift-s ........ | remove sticky labels from selected
 .. | ctrl-s ......... | add selected to sticky labels
 .. | ctrl-shift-s ... | cycle label visibility (all | sticky | none) 
t . | consolidate .... | shift-t (horizontal) | ctrl-t (vertical)
u . | undo last layout action (limited undo's)
w . | apply layout operation to [selected] nodes
 .. | shift-w ........ | open layout-operation picker; ctrl-w reverses
x   | remove selected nodes (push stack)
 .. | shift-x ........ | pop stack
 .. | ctrl-shift-x ... | collapse edges to one row (selected-adjacent, or all)
y . | line layout ...  | shift-y (horizontal) | ctrl-y (vertical)
z . | select node under mouse by color (shift, ctrl, and ctrl-shift apply)
1-6 | select numbered degree
7 . | select degree 7 -> 20
8 . | select degree 20 -> 50
9 . | select degree 50 -> 100
0 . | select degree 100 -> 10_000
"""

    # Build static SVG for keyboard help overlay
    _help_lines_  = _keyboard_commands_.strip().split('\n')
    _help_w_      = max(len(l) for l in _help_lines_) * 7 + 20
    _help_h_      = len(_help_lines_) * 14 + 12
    _font_style_  = "font-family: 'Courier New', monospace; font-size: 11px; fill: #222;"
    _text_lines_  = ''.join(
        f'<text x="10" y="{12 + i*14}" style="{_font_style_}">{l.replace(" ", " ")}</text>'
        for i, l in enumerate(_help_lines_)
    )
    _keyboard_help_svg_ = (
        f'<rect x="0" y="0" width="{_help_w_}" height="{_help_h_}" '
        f'fill="rgba(240,240,240,0.95)" stroke="#888" stroke-width="1" rx="3"/>'
        f'{_text_lines_}'
    )

    # Root interaction SVG (the plot lives in #mod for SVG mode, on #gpucanvas for GPU mode;
    # all other children are interaction chrome that stays SVG in both modes)
    _svg_root_ = f"""
<svg id="svgparent" width="{_w_}" height="{_h_}" tabindex="0" style="user-select:none;{' position:absolute;left:0;top:0;' if use_webgpu else ''}" onkeydown="${{script('myOnKeyDown')}}" onkeyup="${{script('myOnKeyUp')}}">
    <svg id="mod" width="{_w_}" height="{_h_}"> ${{mod_inner}} </svg>
    <g id="keyboardhelp" transform="translate(${{keyboardhelp_x}} 0)">{_keyboard_help_svg_}</g>
    <g fill-opacity="0.0">
      <g id="opanimation"> ${{animation_inner}} </g>
      <animate id="myanimate" attributeName="fill-opacity" values="0.0;1.0;1.0;0.0" dur="2s" repeatCount="1" />
    </g>
    <rect id="drag" x="-10" y="-10" width="5" height="5" stroke="#000000" stroke-width="2" fill="none" />
    <line   id="layoutline"      x1="-10" y1="-10" x2="-10"    y2="-10"    stroke="#000000" stroke-width="2" />
    <rect   id="layoutrect"      x="-10"  y="-10"  width="10"  height="10" stroke="#000000" stroke-width="2" />
    <circle id="layoutcircle"    cx="-10" cy="-10" r="5"       fill="none" stroke="#000000" stroke-width="6" />
    <circle id="layoutsunflower" cx="-10" cy="-10" r="5"                   stroke="#000000" stroke-width="2" />
    <rect id="screen" x="0" y="0" width="{_w_}" height="{_h_}" opacity="0.05"
          onmouseover="${{script('myOnMouseOver')}}"      onmouseout="${{script('myOnMouseOut')}}"
          onmousedown="${{script('downSelect')}}"         onmousemove="${{script('myOnMouseMove')}}"
          onmouseup="${{script('myOnMouseUp')}}" />
    <text id="infostr" x="5"   y="{_h_-2}" fill="#000000" font-size="10px"> ${{info_str}} </text>
    <path id="allentitieslayer" d="" fill="#000000" fill-opacity="0.01" stroke="none"
          onmouseover="${{script('myOnMouseOver')}}"      onmouseout="${{script('myOnMouseOut')}}"
          onmousedown="${{script('downAllEntities')}}"    onmousemove="${{script('myOnMouseMove')}}"
          onmouseup="${{script('myOnMouseUp')}}" />
    <path id="selectionlayer" d="" fill="#ff0000" transform="" stroke="none"
          onmouseover="${{script('myOnMouseOver')}}"      onmouseout="${{script('myOnMouseOut')}}"
          onmousedown="${{script('downMove')}}"           onmousemove="${{script('myOnMouseMove')}}"
          onmouseup="${{script('myOnMouseUp')}}" />
    <text id="searchtext" x="{_w_//2}" y="{_h_-2}" text-anchor="middle" fill="#0000cc" font-size="11px" font-family="monospace"></text>
    <g id="pickermenu" pointer-events="none"></g>
</svg>
"""
    if use_webgpu:
        _template_ = f"""
<div id="gpuwrap" style="position:relative;width:{_w_}px;height:{_h_}px;">
    <canvas id="gpucanvas" width="{_w_}" height="{_h_}" style="position:absolute;left:0;top:0;"></canvas>
    {_svg_root_}
</div>
"""
    else:
        _template_ = _svg_root_

    # GPU JS: install runtime + first paint on mount; re-render on gpu_payload change;
    # any failure sets gpu_error so the Python watcher shows an error overlay (no SVG fallback)
    _gpu_render_block_ = (f"""
{P2S_GPU_JS}
            if (!window.__P2S_GPU__.supported()) {{ data.gpu_error = 'WebGPU is not available in this browser.'; }}
            else {{
                window.__P2S_GPU__.render(gpucanvas, data.gpu_payload)
                    .catch(function(e) {{ console.warn('p2s webgpu:', e); data.gpu_error = (e && e.message) ? e.message : String(e); }});
            }}
""") if use_webgpu else ''
    _gpu_payload_script_ = ("""
            if (window.__P2S_GPU__ && window.__P2S_GPU__.supported() && !data.gpu_error) {
                window.__P2S_GPU__.render(gpucanvas, data.gpu_payload)
                    .catch(function(e) { console.warn('p2s webgpu:', e); data.gpu_error = (e && e.message) ? e.message : String(e); });
            }
""") if use_webgpu else ''

    # Picker-menu data + state, prepended to the render script. Built separately
    # because the render script is a plain (non-f) string with JS brace literals.
    _menu_init_js_ = (
        "            state.menu_items = " + json.dumps({
            'operation':    [[m, l] for m, l in _LAYOUT_OP_MENU_],
            'mode':         [[m, l] for m, l in _LAYOUT_MODE_MENU_],
            'link_size':    _link_size_items_,
            'link_opacity': _link_opacity_items_,
            'node_size':    _node_size_items_,
            'link_shape':   _link_shape_items_,
        }) + ";\n"
        "            state.menu_open  = false; state.menu_kind = ''; state.menu_index = 0; state.menu_timer = null;\n"
    )

    # Dynamic Class
    cls = type('LINKPI', (ReactiveHTML,), {
        #
        # Keyboard Commands
        #
        '_keyboard_commands_': _keyboard_commands_,
        #
        # Python Methods
        #
        '__init__':                           __init__,
        **({'applyGpuError': applyGpuError} if use_webgpu else {}),
        'saveLayout':                         saveLayout,
        'loadLayout':                         loadLayout,
        '__renderView__':                     __renderView__,
        '__refreshView__':                    __refreshView__,
        'display':                            display,
        'replaceBaseDataframe':               replaceBaseDataframe,
        'receiveSelection':                   receiveSelection,
        '_extractNodes_':                     _extractNodes_,
        '_matchNodesByRegex_':                _matchNodesByRegex_,
        'selectEntities':                     selectEntities,
        'selectedEntities':                   selectedEntities,
        'selectedNodes':                      selectedNodes,
        'updateLinkNodeParam':                updateLinkNodeParam,
        '__cacheNodePositions__':             __cacheNodePositions__,
        'setSelectedEntitiesAndNotifyOthers': setSelectedEntitiesAndNotifyOthers,
        '__buildLayoutRegistry__':            __buildLayoutRegistry__,
        '__backgroundStateLabel__':           __backgroundStateLabel__,
        '__applyBackgroundState__':           __applyBackgroundState__,
        '__contractCollapsedGraph__':         __contractCollapsedGraph__,
        '__expandContractedResult__':         __expandContractedResult__,
        '__layoutOperation__':                __layoutOperation__,
        'applyLayoutInteraction':             applyLayoutInteraction,
        'applyMiddleOp':                      applyMiddleOp,
        'applyWheelOp':                       applyWheelOp,
        'setAnimation':                       setAnimation,
        'popStack':                           popStack,
        'setStackPostion':                    setStackPostion,
        'pushStack':                          pushStack,
        '_propagate_view_changes_':            _propagate_view_changes_,
        '_select_by_attribute_at_mouse_':      _select_by_attribute_at_mouse_,
        '_propagate_positions':               _propagate_positions,
        'apply_drag_select':                  apply_drag_select,
        'apply_move_selected':                apply_move_selected,
        'apply_push_selected':                apply_push_selected,
        'apply_pop':                          apply_pop,
        'apply_collapse_edges':               apply_collapse_edges,
        'apply_community_detection':          apply_community_detection,
        'apply_collapse_to':                  apply_collapse_to,
        'apply_layout_interaction':           apply_layout_interaction,
        'apply_undo':                         apply_undo,
        'applyKeyOp':                         applyKeyOp,
        'applyDragOp':                        applyDragOp,
        'applyMoveOp':                        applyMoveOp,
        'unselectedMoveOp':                   unselectedMoveOp,
        'applySearchOp':                      applySearchOp,
        'applyLayoutChoice':                  applyLayoutChoice,
        'applySizeChoice':                    applySizeChoice,
        '__sizeLabelToValue__':               __sizeLabelToValue__,
        #
        # Panel Params
        #
    'mod_inner'                   : param.String(default=_svg_),
    **({'gpu_payload': param.Dict(default=_gpu_payload_default_), 'gpu_error': param.String(default='')} if use_webgpu else {}),
    'animation_inner'             : param.String(default='<rect x="0" y="0" width="10" height="10" fill="none" stroke="none"/>'),
    'allentitiespath'             : param.String(default="M -100 -100 l 10 0 l 0 10 l -10 0 l 0 -10 Z"),
    'selectionpath'               : param.String(default="M -100 -100 l 10 0 l 0 10 l -10 0 l 0 -10 Z"),
    'info_str'                    : param.String(default=" | | grid"),
    'layout_mode'                 : param.String(default="grid"),
    'layout_operation'            : param.String(default="spring nx"),
    'link_size_choice'            : param.String(default=_link_size_cur_),
    'node_size_choice'            : param.String(default=_node_size_cur_),
    'link_opacity_choice'         : param.String(default=_link_opacity_cur_),
    'link_shape_choice'           : param.String(default=_link_shape_cur_),
    'keyboardhelp_x'              : param.Integer(default=-1000),
    'x0_middle'                   : param.Integer(default=0),
    'y0_middle'                   : param.Integer(default=0),
    'x1_middle'                   : param.Integer(default=0),
    'y1_middle'                   : param.Integer(default=0),
    'middle_op_finished'          : param.Boolean(default=False),
    'wheel_x'                     : param.Integer(default=0),
    'wheel_y'                     : param.Integer(default=0),
    'wheel_rots'                  : param.Integer(default=0), # Mult by 10 and rounded...
    'wheel_op_finished'           : param.Boolean(default=False),
    'drag_op_finished'            : param.Boolean(default=False),
    'drag_x0'                     : param.Integer(default=0),
    'drag_y0'                     : param.Integer(default=0),
    'drag_x1'                     : param.Integer(default=10),
    'drag_y1'                     : param.Integer(default=10),
    'allentities_x0'              : param.Integer(default=10),
    'allentities_y0'              : param.Integer(default=10),
    'unselected_move_op_finished' : param.Boolean(default=False),
    'move_op_finished'            : param.Boolean(default=False),
    'layout_shape'                : param.String(default=""),
    'shiftkey'                    : param.Boolean(default=False),
    'ctrlkey'                     : param.Boolean(default=False),
    'last_key'                    : param.String(default=''),
    'key_op_finished'             : param.String(default=''),
    'x_mouse'                     : param.Integer(default=0),
    'y_mouse'                     : param.Integer(default=0),
    'has_focus'                   : param.Boolean(default=False),
    'search_str'                  : param.String(default=''),
    'search_op_finished'          : param.Boolean(default=False),

    #
    # Panel Template
    #
    '_template': _template_,

    #
    # Panel Javascript Definitions
    #
    '_scripts': {
        'render': _menu_init_js_ + """
            mod.innerHTML            = data.mod_inner;
            infostr.innerHTML        = data.info_str;
            opanimation.innerHTML    = data.animation_inner;
            allentitieslayer.setAttribute("d", data.allentitiespath);
            selectionlayer.setAttribute("d", data.selectionpath);
            state.x0_drag            = state.y0_drag = -10;
            state.x1_drag            = state.y1_drag =  -5;
            // Only seed the mouse position on first render; this script re-runs
            // on every refresh (e.g. after a 't' collapse pushes a new mod_inner),
            // and clobbering cur_mouse back to 0 would make the next key op that
            // reads it (e.g. a repeated 't') collapse to the upper-left corner
            // until the mouse moves again and myOnMouseMove repopulates it.
            if (state.cur_mouse_x === undefined) { state.cur_mouse_x = 0; }
            if (state.cur_mouse_y === undefined) { state.cur_mouse_y = 0; }
            data.has_focus           = false;
            data.shiftkey            = false;
            data.ctrlkey             = false;
            state.drag_op            = false;
            state.move_op            = false;
            state.unselected_move_op = false;
            state.layout_op          = false; // true if next mouse button 1 press is the begin of a layout
            state.layout_line_flag   = false; // true if the shape will be overrode by the line version
            state.layout_op_shape    = "";    // trigger field for python to peform the layout operation
            data.middle_op_finished  = false;
            data.move_op_finished    = false;
            state.search_mode        = false;
            state.search_buffer      = '';

            myanimate.addEventListener("endEvent", () => { data.animation_inner = ""; opanimation.innerHTML = data.animation_inner; });

            var _wheelFn_ = function(event) {
                event.preventDefault();
                data.wheel_x = event.offsetX; data.wheel_y = event.offsetY;
                data.wheel_rots = Math.round(10*event.deltaY);
                data.wheel_op_finished = true;
            };
            screen.addEventListener('wheel', _wheelFn_, {passive: false});
            allentitieslayer.addEventListener('wheel', _wheelFn_, {passive: false});
            selectionlayer.addEventListener('wheel', _wheelFn_, {passive: false});
""" + _gpu_render_block_ + """
        """,

        **({'gpu_payload': _gpu_payload_script_} if use_webgpu else {}),

        'myOnMouseOver':"""
                data.has_focus = true;
                svgparent.focus();
        """,

        'myOnMouseOut':"""
                data.has_focus = false;
                if (state.menu_open) { self.menuCommit(); }
        """,

        # ── picker menu (shift-W: layout operations, shift-G: layout modes) ──
        # Modal JS state machine; nothing reaches Python until menuCommit writes
        # data.layout_operation / data.layout_mode (select-only — 'w' / 'g'-drag
        # still apply the choice).
        'menuOpen':"""
            var _items_   = state.menu_items[state.menu_kind];
            var _current_ = (state.menu_kind == 'operation')    ? data.layout_operation
                          : (state.menu_kind == 'mode')         ? data.layout_mode
                          : (state.menu_kind == 'link_size')    ? data.link_size_choice
                          : (state.menu_kind == 'link_opacity') ? data.link_opacity_choice
                          : (state.menu_kind == 'link_shape')   ? data.link_shape_choice
                          :                                       data.node_size_choice;
            state.menu_index = 0;
            for (var _i_ = 0; _i_ < _items_.length; _i_++) {
                if (_items_[_i_][1] == _current_) { state.menu_index = _i_; break; }
            }
            state.menu_open = true;
            self.menuRender();
            self.menuArmTimer();
        """,
        'menuRender':"""
            if (!state.menu_open) { return; }
            var _items_  = state.menu_items[state.menu_kind];
            var _header_ = (state.menu_kind == 'operation')    ? 'layout operation:'
                         : (state.menu_kind == 'mode')         ? 'layout mode:'
                         : (state.menu_kind == 'link_size')    ? 'link size:'
                         : (state.menu_kind == 'link_opacity') ? 'link opacity:'
                         : (state.menu_kind == 'link_shape')   ? 'link shape:'
                         :                                       'node size:';
            var _maxlen_ = _header_.length;
            for (var _i_ = 0; _i_ < _items_.length; _i_++) {
                _maxlen_ = Math.max(_maxlen_, _items_[_i_][1].length + 4);
            }
            var _w_menu_ = _maxlen_ * 7 + 20,
                _h_menu_ = (_items_.length + 1) * 14 + 12,
                _style_  = 'font-family: \\'Courier New\\', monospace; font-size: 11px; fill: #222;';
            var _html_ = '<rect x="8" y="8" width="' + _w_menu_ + '" height="' + _h_menu_ + '"'
                       + ' fill="rgba(240,240,240,0.95)" stroke="#888" stroke-width="1" rx="3"/>'
                       + '<rect x="10" y="' + (8 + 1 + (state.menu_index + 1) * 14) + '" width="' + (_w_menu_ - 4) + '" height="13"'
                       + ' fill="rgba(100,150,255,0.3)"/>'
                       + '<text x="18" y="' + (8 + 12) + '" style="' + _style_ + ' font-weight: bold;">' + _header_ + '</text>';
            for (var _i_ = 0; _i_ < _items_.length; _i_++) {
                _html_ += '<text x="18" y="' + (8 + 12 + (_i_ + 1) * 14) + '" style="' + _style_ + '">'
                        + '[' + _items_[_i_][0] + '] ' + _items_[_i_][1] + '</text>';
            }
            pickermenu.innerHTML = _html_;
        """,
        'menuCommit':"""
            var _lbl_ = state.menu_items[state.menu_kind][state.menu_index][1];
            if      (state.menu_kind == 'operation')    { data.layout_operation   = _lbl_; }
            else if (state.menu_kind == 'mode')         { data.layout_mode        = _lbl_; }
            else if (state.menu_kind == 'link_size')    { data.link_size_choice    = _lbl_; }
            else if (state.menu_kind == 'link_opacity') { data.link_opacity_choice = _lbl_; }
            else if (state.menu_kind == 'link_shape')   { data.link_shape_choice   = _lbl_; }
            else                                        { data.node_size_choice    = _lbl_; }
            self.menuClose();
        """,
        'menuClose':"""
            if (state.menu_timer != null) { clearTimeout(state.menu_timer); }
            state.menu_timer     = null;
            state.menu_open      = false;
            state.menu_kind      = '';
            pickermenu.innerHTML = '';
        """,
        'menuArmTimer':"""
            if (state.menu_timer != null) { clearTimeout(state.menu_timer); }
            var _self_ = self;
            state.menu_timer = setTimeout(function() { if (state.menu_open) { _self_.menuCommit(); } }, 2500);
        """,

        'myOnKeyDown':"""
            event.stopPropagation();
            if (state.search_mode) {
                if (event.key === 'Enter') {
                    if (state.search_buffer) {
                        data.search_str = state.search_buffer;
                        data.search_op_finished = !data.search_op_finished;
                    }
                    state.search_mode   = false;
                    state.search_buffer = '';
                    searchtext.textContent = '';
                } else if (event.key === 'Escape') {
                    state.search_mode   = false;
                    state.search_buffer = '';
                    searchtext.textContent = '';
                } else if (event.key === 'Backspace') {
                    state.search_buffer = state.search_buffer.slice(0, -1);
                    searchtext.textContent = '/ ' + state.search_buffer + '▋';
                } else if (event.key.length === 1) {
                    state.search_buffer += event.key;
                    searchtext.textContent = '/ ' + state.search_buffer + '▋';
                }
                return;
            }
            if (state.menu_open) {
                event.preventDefault();
                var _items_ = state.menu_items[state.menu_kind];
                if      (event.key === 'Escape') { self.menuClose();  }
                else if (event.key === 'Enter')  { self.menuCommit(); }
                else if (event.key === 'ArrowDown' || event.key === 'j' ||
                         (event.key === 'W' && state.menu_kind === 'operation'    && !event.ctrlKey) ||
                         (event.key === 'G' && state.menu_kind === 'mode'         && !event.ctrlKey) ||
                         (event.key === 'L' && state.menu_kind === 'link_size')    ||
                         (event.key === 'O' && state.menu_kind === 'link_opacity') ||
                         (event.key === 'l' && state.menu_kind === 'link_shape'   && !event.ctrlKey) ||
                         (event.key === 'P' && state.menu_kind === 'node_size')) {
                    state.menu_index = (state.menu_index + 1) % _items_.length;
                    self.menuRender(); self.menuArmTimer();
                }
                else if (event.key === 'ArrowUp' || event.key === 'k' ||
                         (event.key === 'W' && state.menu_kind === 'operation'    && event.ctrlKey) ||
                         (event.key === 'G' && state.menu_kind === 'mode'         && event.ctrlKey) ||
                         (event.key === 'l' && state.menu_kind === 'link_size'    && event.ctrlKey) ||
                         (event.key === 'o' && state.menu_kind === 'link_opacity' && event.ctrlKey) ||
                         (event.key === 'p' && state.menu_kind === 'node_size'    && event.ctrlKey)) {
                    state.menu_index = (state.menu_index - 1 + _items_.length) % _items_.length;
                    self.menuRender(); self.menuArmTimer();
                }
                else if (event.key.length === 1) {
                    for (var _i_ = 0; _i_ < _items_.length; _i_++) {
                        if (_items_[_i_][0] === event.key) { state.menu_index = _i_; self.menuCommit(); break; }
                    }
                }
                return;
            }
            data.ctrlkey  = event.ctrlKey;
            data.shiftkey = event.shiftKey;
            data.x_mouse  = state.cur_mouse_x;
            data.y_mouse  = state.cur_mouse_y;

            if      (event.key == "a" && !event.ctrlKey) { data.key_op_finished = 'a'; } // Toggle link arrows (on | off)
            else if (event.key == "b" ||                                // Cycle background (none | background | background + labels)
                     event.key == "B") { data.key_op_finished = 'b';  }
            else if (event.key == "c") { if (event.ctrlKey) event.preventDefault(); data.key_op_finished = 'c';  } // (if selected) zoom to selected, else zoom to entire view; ctrl-c copies (suppress native copy so it can't clobber our clipboard write)
            else if (event.key == "C") { if (event.ctrlKey) event.preventDefault(); data.key_op_finished = 'C';  } // Zoom to selected + neighbors; ctrl-shift-c copies labels
            else if (event.key == "d") { data.key_op_finished = 'd';  } // Detect communities (louvain) & color nodes by community
            else if (event.key == "D") { data.key_op_finished = 'D';  } // Clear the community colors
            else if (event.key == "e") { if (event.ctrlKey) event.preventDefault(); data.key_op_finished = 'e';  } // Expand; ctrl-e evens out distribution (ctrl-e is browser search-bar focus)
            else if (event.key == "E") { data.key_op_finished = 'E';  } // Expand (w/ digraph)
            else if (event.key == "g") { state.layout_op        = true; // Mouse press is layout shape
                                         state.layout_line_flag = false; } 
            else if (event.key == "G") { state.menu_kind = 'mode';      self.menuOpen(); } // Open the layout-mode picker menu
            else if (event.key == "h") {
                if (data.keyboardhelp_x == -1000) { data.keyboardhelp_x =     5; }
                else                              { data.keyboardhelp_x = -1000; }
            }
            else if (event.key == "l" && !event.ctrlKey) { state.menu_kind = 'link_shape'; self.menuOpen(); } // Open the link-shape picker menu; 'l' cycles
            else if (event.key == "L" || (event.key == "l" && event.ctrlKey)) { if (event.ctrlKey) event.preventDefault(); state.menu_kind = 'link_size';    self.menuOpen(); } // Cycle link size
            else if (event.key == "O" || (event.key == "o" && event.ctrlKey)) { if (event.ctrlKey) event.preventDefault(); state.menu_kind = 'link_opacity'; self.menuOpen(); } // Cycle link opacity
            else if (event.key == "P" || (event.key == "p" && event.ctrlKey)) { if (event.ctrlKey) event.preventDefault(); state.menu_kind = 'node_size';    self.menuOpen(); } // Cycle node size
            else if (event.key == "n" ||                                // Select nodes with the same shape as the one under the mouse
                     event.key == "N") { data.key_op_finished = 'n';  }
            else if (event.key == "q") { data.key_op_finished = 'q';  } // Invert selection
            else if (event.key == "Q") { data.key_op_finished = 'Q';  } // Select common neighbors to selected nodes
            else if (event.key == "s") { if (event.ctrlKey) event.preventDefault(); data.key_op_finished = 's';  } // Set sticky labels; ctrl-s cycles label mode (ctrl-s is browser Save Page As)
            else if (event.key == "S") { if (event.ctrlKey) event.preventDefault(); data.key_op_finished = 'S';  } // Subtract selected from sticky labels; ctrl-shift-s cycles label mode
            else if (event.key == "t") { data.key_op_finished = 't';  } // Collapse selected to a single point
            else if (event.key == "T") { data.key_op_finished = 'T';  } // Horizontally collapse selected
            else if (event.key == "u") { data.key_op_finished = 'u';  } // Undo last layout
            else if (event.key == "w") { data.key_op_finished = 'w';  } // Apply layout operation
            else if (event.key == "W") { state.menu_kind = 'operation'; self.menuOpen(); } // Open the layout-operation picker menu
            else if (event.key == "x") { data.key_op_finished = 'x';  } // push the stack (remove the selected from the current graph)
            else if (event.key == "X" && event.ctrlKey) { data.key_op_finished = 'ctrl_shift_x'; } // collapse edges to one row each (push the stack)
            else if (event.key == "X") { data.key_op_finished = 'X';  } // pop the stack (add removed nodes back in)
            else if (event.key == "y") { state.layout_op        = true; // Mouse press is layout line
                                         state.layout_line_flag = true;  }
            else if (event.key == "Y") { state.layout_op        = true; // Mouse press is layout line
                                         state.layout_line_flag = true;  }
            else if (event.key == "z" ||                                // Select nodes with the same color as the one under the mouse
                     event.key == "Z") { data.key_op_finished = 'z';     }
            else if (event.key == "1" || event.key == "!") { data.key_op_finished = '1';  }
            else if (event.key == "2" || event.key == "@") { data.key_op_finished = '2';  }
            else if (event.key == "3" || event.key == "#") { data.key_op_finished = '3';  }
            else if (event.key == "4" || event.key == "$") { data.key_op_finished = '4';  }
            else if (event.key == "5" || event.key == "%") { data.key_op_finished = '5';  }
            else if (event.key == "6" || event.key == "^") { data.key_op_finished = '6';  }
            else if (event.key == "7" || event.key == "&") { data.key_op_finished = '7';  }
            else if (event.key == "8" || event.key == "*") { data.key_op_finished = '8';  }
            else if (event.key == "9" || event.key == "(") { data.key_op_finished = '9';  }
            else if (event.key == "0" || event.key == ")") { data.key_op_finished = '0';  }
            else if (event.key == "/") {
                state.search_mode   = true;
                state.search_buffer = '';
                searchtext.textContent = '/ ▋';
            }

            data.last_key = event.key;
        """,
        'myOnKeyUp':"""
            event.stopPropagation();
            if (state.menu_open) { return; }
            data.ctrlkey  = event.ctrlKey;
            data.shiftkey = event.shiftKey;
            if (event.key == "g" || event.key == "y" || event.key == "Y") { state.layout_op = state.layout_line_flag = false; }
        """,
        'myOnMouseMove':"""
            state.cur_mouse_x = event.offsetX;
            state.cur_mouse_y = event.offsetY;
            state.x1_drag     = event.offsetX;
            state.y1_drag     = event.offsetY;
            if (state.drag_op)               { self.myUpdateDragRect(); }
            if (state.move_op)               { selectionlayer.setAttribute("transform", "translate(" + (state.x1_drag - state.x0_drag) + "," + (state.y1_drag - state.y0_drag) + ")"); }
            if (state.unselected_move_op)    { selectionlayer.setAttribute("transform", "translate(" + (state.x1_drag - state.x0_drag) + "," + (state.y1_drag - state.y0_drag) + ")"); }
            if (state.layout_op_shape != "") { self.myUpdateLayoutOp(); }
        """,
        'downAllEntities':"""
            data.ctrlkey  = event.ctrlKey;
            data.shiftkey = event.shiftKey;
            if (event.button == 0) {
                    data.allentities_x0      = event.offsetX; 
                    data.allentities_y0      = event.offsetY; 
                    state.x0_drag            = event.offsetX;                
                    state.y0_drag            = event.offsetY;                
                    state.x1_drag            = event.offsetX;
                    state.y1_drag            = event.offsetY;
                    state.unselected_move_op = true;
                    var ex = event.offsetX, ey = event.offsetY;
                    selectionlayer.setAttribute("d", "M " + (ex-5) + " " + (ey-5) + " l 10 0 l 0 10 l -10 0 z");
                    selectionlayer.setAttribute("transform", "");
            }
        """,
        'downSelect':"""
            if (event.button == 0) {
                state.x0_drag  = event.offsetX;
                state.y0_drag  = event.offsetY;
                state.x1_drag  = event.offsetX;
                state.y1_drag  = event.offsetY;
                if (state.layout_op) { 
                    if (state.layout_line_flag) { 
                        if      (data.ctrlkey)  { state.layout_op_shape = "v-line"; }
                        else if (data.shiftkey) { state.layout_op_shape = "h-line"; }
                        else                    { state.layout_op_shape = "line";   }
                    }
                    else                        { state.layout_op_shape = data.layout_mode; }
                    self.myUpdateLayoutOp();
                } else               { state.drag_op         = true;             self.myUpdateDragRect(); }
            } else if (event.button == 1) {
                data.x0_middle = data.x1_middle = event.offsetX;
                data.y0_middle = data.y1_middle = event.offsetY;
            }
        """,
        'downMove':"""
            if (event.button == 0) {
                state.x0_drag  = state.x1_drag  = event.offsetX;
                state.y0_drag  = state.y1_drag  = event.offsetY;
                state.move_op  = true;
            } else if (event.button == 1) {
                data.x0_middle = data.x1_middle = event.offsetX; 
                data.y0_middle = data.y1_middle = event.offsetY;
            }
        """,
        'myUpdateLayoutOp':"""
            var dx = state.x1_drag - state.x0_drag,
                dy = state.y1_drag - state.y0_drag;
            var reset_circle = true, reset_sunflower = true, reset_rect = true, reset_line = true;
            if        (state.layout_op_shape == "circle")    { reset_circle = false;
                layoutcircle.setAttribute("cx", state.x0_drag);
                layoutcircle.setAttribute("cy", state.y0_drag);
                layoutcircle.setAttribute("r",  Math.sqrt(dx*dx + dy*dy));
            } else if (state.layout_op_shape == "sunflower") { reset_sunflower = false;
                layoutsunflower.setAttribute("cx", state.x0_drag);
                layoutsunflower.setAttribute("cy", state.y0_drag);
                layoutsunflower.setAttribute("r",  Math.sqrt(dx*dx + dy*dy));            
            } else if (state.layout_op_shape == "grid" || 
                       state.layout_op_shape == "grid (color)" || 
                       state.layout_op_shape == "grid (color, clouds)" ||
                       state.layout_op_shape == "rescale") { reset_rect = false;
                layoutrect.setAttribute("x", Math.min(state.x0_drag, state.x1_drag));
                layoutrect.setAttribute("y", Math.min(state.y0_drag, state.y1_drag));
                layoutrect.setAttribute("width",  Math.abs(dx));
                layoutrect.setAttribute("height", Math.abs(dy));
            } else if (state.layout_op_shape == "line")    { reset_line = false;
                layoutline.setAttribute("x1", state.x0_drag);
                layoutline.setAttribute("y1", state.y0_drag);
                layoutline.setAttribute("x2", state.x1_drag);
                layoutline.setAttribute("y2", state.y1_drag);
            } else if (state.layout_op_shape == "h-line")  { reset_line = false;
                layoutline.setAttribute("x1", state.x0_drag);
                layoutline.setAttribute("y1", state.y1_drag);
                layoutline.setAttribute("x2", state.x1_drag);
                layoutline.setAttribute("y2", state.y1_drag);
            } else if (state.layout_op_shape == "v-line")  { reset_line = false;
                layoutline.setAttribute("x1", state.x1_drag);
                layoutline.setAttribute("y1", state.y0_drag);
                layoutline.setAttribute("x2", state.x1_drag);
                layoutline.setAttribute("y2", state.y1_drag);
            } else { state.layout_op_shape == ""; }
            if (reset_circle)    { layoutcircle   .setAttribute("cx", -10); layoutcircle   .setAttribute("cy", -10); layoutcircle   .setAttribute("r",      5); }
            if (reset_sunflower) { layoutsunflower.setAttribute("cx", -10); layoutsunflower.setAttribute("cy", -10); layoutsunflower.setAttribute("r",      5); }
            if (reset_rect)      { layoutrect     .setAttribute("x",  -10); layoutrect     .setAttribute("y",  -10); layoutrect     .setAttribute("width",  5);  layoutrect.setAttribute("height",  5); }
            if (reset_line)      { layoutline     .setAttribute("x1", -10); layoutline     .setAttribute("y1", -10); layoutline     .setAttribute("x2",    -5);  layoutline.setAttribute("y2",     -5); }
        """,
        'myOnMouseUp':"""
            if (event.button == 0) {
                data.ctrlkey          = event.ctrlKey;
                data.shiftkey         = event.shiftKey;
                state.x1_drag         = event.offsetX;
                state.y1_drag         = event.offsetY;
                if (state.drag_op) {
                    state.shiftkey        = event.shiftKey;
                    state.drag_op         = false;
                    self.myUpdateDragRect();
                    data.drag_x0          = state.x0_drag; 
                    data.drag_y0          = state.y0_drag; 
                    data.drag_x1          = state.x1_drag; 
                    data.drag_y1          = state.y1_drag;
                    data.drag_op_finished = true;
                } else if (state.move_op) {
                    state.move_op         = false;
                    data.drag_x0          = state.x0_drag; 
                    data.drag_y0          = state.y0_drag; 
                    data.drag_x1          = state.x1_drag; 
                    data.drag_y1          = state.y1_drag;
                    data.move_op_finished = true;                    
                } else if (state.layout_op_shape != "") {
                    data.drag_x0          = state.x0_drag; 
                    data.drag_y0          = state.y0_drag; 
                    data.drag_x1          = state.x1_drag; 
                    data.drag_y1          = state.y1_drag;
                    data.layout_shape     = state.layout_op_shape;
                    state.layout_op_shape = "";
                    self.myUpdateLayoutOp();
                } else if (state.unselected_move_op) {
                    data.ctrlkey  = event.ctrlKey;
                    data.shiftkey = event.shiftKey;
                    data.drag_x0  = state.x0_drag;
                    data.drag_y0  = state.y0_drag;
                    data.drag_x1  = state.x1_drag;
                    data.drag_y1  = state.y1_drag;
                    data.unselected_move_op_finished = true;
                    state.unselected_move_op = false;
                }
            } else if (event.button == 1) {
                data.x1_middle          = event.offsetX; 
                data.y1_middle          = event.offsetY;
                data.middle_op_finished = true;                
            }
        """,
        'myOnMouseWheel':"""
            event.preventDefault();
            data.wheel_x = event.offsetX; data.wheel_y = event.offsetY; data.wheel_rots  = Math.round(10*event.deltaY);
            data.wheel_op_finished = true;
        """,
        'mod_inner':"""
            mod.innerHTML       = data.mod_inner;
            infostr.innerHTML   = data.info_str;
        """,
        'animation_inner':"""
            opanimation.innerHTML = data.animation_inner;
        """,
        'allentitiespath':"""
            allentitieslayer.setAttribute("d", data.allentitiespath);
        """,
        'selectionpath':"""
            selectionlayer.setAttribute("d", data.selectionpath);
            selectionlayer.setAttribute("transform", "");
        """,
        'info_str': """
            infostr.innerHTML = data.info_str;
        """,
        'myUpdateDragRect':"""
            if (state.drag_op) {
                x = Math.min(state.x0_drag, state.x1_drag); 
                y = Math.min(state.y0_drag, state.y1_drag);
                w = Math.abs(state.x1_drag - state.x0_drag)
                h = Math.abs(state.y1_drag - state.y0_drag)
                drag.setAttribute('x',x);     drag.setAttribute('y',y);
                drag.setAttribute('width',w); drag.setAttribute('height',h);
                if      (data.shftkey && data.ctrlkey)  drag.setAttribute('stroke','#0000ff');
                else if (data.shftkey)                  drag.setAttribute('stroke','#ff0000');
                else if (                data.ctrlkey)  drag.setAttribute('stroke','#00ff00');
                else                                    drag.setAttribute('stroke','#000000');
            } else {
                drag.setAttribute('x',-10);   drag.setAttribute('y',-10);
                drag.setAttribute('width',5); drag.setAttribute('height',5);
            }
        """
    }

    })
    _cls_ref_[0] = cls
    return cls(**kwargs)

_PLOT_TYPE_TO_WRAPPER_['LinkP'] = linkpi

try:
    from .spreadlinepi import spreadlinepi
    _PLOT_TYPE_TO_WRAPPER_['SpreadLinesP'] = spreadlinepi
except ImportError:
    # Circular import: spreadlinepi is mid-import (it imports this module first)
    # and registers itself in _PLOT_TYPE_TO_WRAPPER_ at the end of its own body.
    pass
