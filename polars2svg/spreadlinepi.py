from typing import Any
"""
Interactive wrapper for SpreadLinesP.

  spreadlinepi(spread, **kwargs)
      Selection highlights are baked into the SpreadLinesP render: highlighted
      node circles use a thicker stroke and higher fill-opacity.  A full
      re-render is triggered on each selection change, which keeps the
      highlights visually integrated with the rest of the chart (correct
      viewBox scaling, consistent with node color, etc.).

Supports:
  - Stack navigation (receives display() calls from peers)
  - Selection receive (receiveSelection from linkp / other components)
  - Click/drag on nodes   → select/deselect (ctrl=add, shift=subtract, both=intersect)
  - Click on empty space  → clear selection
  - 'c' key              → set selected nodes as new ego (single or set) → re-render
  - 'x' key              → remove selected nodes → pushStack
  - 'X' key              → popStack / clear selection
"""
import asyncio
import polars as pl
import param
from panel.reactive import ReactiveHTML

from .p2s_reactive_base import P2SReactiveHTML

from .interactive_controller import InteractionController, _gpu_error_overlay, _withGpuScripts_


# ─────────────────────────────────────────────────────────────────────────────
# Hit-test & filter helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_viewbox_coords(spread: Any, px: float, py: float) -> tuple:
    """Convert pixel mouse coordinates to SpreadLinesP viewBox coordinates.
    Accounts for preserveAspectRatio="xMidYMid meet" letterboxing: when the
    viewBox aspect ratio differs from wxh, the browser centers the content and
    leaves empty margins, so a straight linear mapping would be wrong."""
    w, h = spread.wxh
    vw   = spread.vx1 - spread.vx0
    vh   = spread.vy1 - spread.vy0
    if vw == 0 or vh == 0:
        return px, py
    scale = min(w / vw, h / vh)
    ox    = (w - vw * scale) / 2
    oy    = (h - vh * scale) / 2
    return spread.vx0 + (px - ox) / scale, spread.vy0 + (py - oy) / scale


def _nodes_at_xy(spread: Any, vx: float, vy: float) -> set:
    """Return the set of nodes at or nearest to (vx, vy).
    For a single node returns {that_node}; for a cloud returns all nodes sharing that center."""
    best_d, best_xy = float('inf'), None
    for n2xyrs in spread.bin_to_node_to_xyrepstat.values():
        for node, xyrs in n2xyrs.items():
            x, y, _rep, _state, _bi, _alt, _side, r = xyrs
            if r is None:
                r = spread.r_pref
            d = ((vx - x) ** 2 + (vy - y) ** 2) ** 0.5
            if d <= r * 1.5 and d < best_d:
                best_d, best_xy = d, (x, y)
    if best_xy is None:
        return set()
    return {str(node)
            for n2xyrs in spread.bin_to_node_to_xyrepstat.values()
            for node, xyrs in n2xyrs.items()
            if (xyrs[0], xyrs[1]) == best_xy}


def _expand_ego(spread: Any, nodes: set) -> set:
    """Translate '__EGO__' back to the real ego node names.
    When ego_is_set, the layout collapses all ego nodes to a virtual '__EGO__' token
    for internal routing.  Hit-testing returns '__EGO__' for cloud clicks, but peers
    (e.g. linkp) hold the real names, so we must expand before storing or broadcasting."""
    if '__EGO__' in nodes and spread.ego_is_set:
        return (nodes - {'__EGO__'}) | {str(n) for n in spread.ego}
    return nodes


def _apply_set_op(current: set, new_nodes: set, shiftkey: bool, ctrlkey: bool) -> set:
    """Apply ctrl/shift modifier set operation: replace, add, subtract, or intersect."""
    if shiftkey and ctrlkey: return current & new_nodes
    if shiftkey:             return current - new_nodes
    if ctrlkey:              return current | new_nodes
    return new_nodes


def _nodes_in_rect(spread: Any, vx0: float, vy0: float, vx1: float, vy1: float) -> set:
    """Return set of nodes whose circle overlaps the viewbox rectangle.
    Uses circle-rectangle intersection so boundary nodes aren't missed."""
    x0, x1 = min(vx0, vx1), max(vx0, vx1)
    y0, y1 = min(vy0, vy1), max(vy0, vy1)
    found = set()
    for n2xyrs in spread.bin_to_node_to_xyrepstat.values():
        for node, xyrs in n2xyrs.items():
            nx, ny = xyrs[0], xyrs[1]
            r = xyrs[7] if xyrs[7] is not None else spread.r_pref
            cx = max(x0, min(nx, x1))   # closest point on rect to circle center
            cy = max(y0, min(ny, y1))
            if (nx - cx) ** 2 + (ny - cy) ** 2 <= r * r:
                found.add(str(node))
    return found


def _filter_out_nodes(spread: Any, df: pl.DataFrame, nodes: set) -> pl.DataFrame:
    """Filter df to remove rows where any relationship column contains a node in nodes.

    Reads relationships_orig, not relationships: SpreadLinesP rewrites a tuple endpoint
    into a synthetic '|'-joined '__fm{i}__' column that exists only on its own render
    frame, and the only caller passes df_orig -- the caller's untouched columns.  The
    join is redone here (same separator) so the names compared against `nodes` are the
    ones the view actually selected."""
    _orig_cols_ = df.columns
    for _i_, _rel_ in enumerate(spread.relationships_orig):
        _fm_col_, _to_col_ = _rel_[0], _rel_[1]
        if isinstance(_fm_col_, tuple):
            _fm_col_ = f'__fm{_i_}__'
            if _fm_col_ not in df.columns: df = spread.p2s.createConcatColumn(df, _rel_[0], _fm_col_)
        if isinstance(_to_col_, tuple):
            _to_col_ = f'__to{_i_}__'
            if _to_col_ not in df.columns: df = spread.p2s.createConcatColumn(df, _rel_[1], _to_col_)
        df = df.filter(
            ~pl.col(_fm_col_).cast(pl.String).is_in(nodes) &
            ~pl.col(_to_col_).cast(pl.String).is_in(nodes)
        )
    return df.select(_orig_cols_)


# ─────────────────────────────────────────────────────────────────────────────
# Panel template / JS
#
# Both are plain static text on the class -- no f-strings, so no doubled braces.
# Per-instance values reach them two ways:
#   {{ param }}  jinja, substituted per instance by ReactiveHTML._get_template()
#   data.param   the same value read from the synced data model inside _scripts
# The GPU canvas wrapper and the ~14 KB WebGPU runtime live on SLPI_GPU, not
# here: an SVG view never calls into them and should not carry them.
# ─────────────────────────────────────────────────────────────────────────────

_SLPI_GPU_HEAD_ = (
    "<div id='gpuwrap' style='position:relative;width:{{ svg_w }}px;height:{{ svg_h }}px;'>"
    "<canvas id='gpucanvas' width='{{ svg_w }}' height='{{ svg_h }}'"
    " style='position:absolute;left:0;top:0;'></canvas>"
)

_SLPI_SVG_ROOT_ = (
    "<svg id='svgparentslpi' width='{{ svg_w }}' height='{{ svg_h }}' tabindex='0'"
    "{%- if use_webgpu %} style='position:absolute;left:0;top:0;'{% endif %}"
    " onkeydown=\"${script('myOnKeyDown')}\">"
    "<svg id='mod' x='0' y='0' width='{{ svg_w }}' height='{{ svg_h }}'>${mod_inner}</svg>"
    "<rect id='screen' x='0' y='0' width='{{ svg_w }}' height='{{ svg_h }}'"
    " style='fill:none;pointer-events:all;'"
    " onmouseover=\"${script('myOnMouseOver')}\""
    " onmousedown=\"${script('myOnMouseDown')}\""
    " onmousemove=\"${script('myOnMouseMove')}\""
    " onmouseup=\"${script('myOnMouseUp')}\""
    " onmouseleave=\"${script('myOnMouseLeave')}\"/>"
    "<rect id='drag_rect' x='0' y='0' width='0' height='0'"
    " style='fill:rgba(128,128,128,0.08);stroke:#000000;stroke-width:1;pointer-events:none;stroke-dasharray:4,2;'/>"
    "</svg>"
)

# The SVG class's template is the root alone -- no canvas.  SLPI_GPU wraps it.
_SLPI_TEMPLATE_     = _SLPI_SVG_ROOT_
_SLPI_GPU_TEMPLATE_ = _SLPI_GPU_HEAD_ + _SLPI_SVG_ROOT_ + "</div>"

_SLPI_SCRIPTS_ = {
    'render': (
        "state.dragging = false; state.x0_drag = state.y0_drag = state.x1_drag = state.y1_drag = 0;"
        "state.shiftkey = false; state.ctrlkey = false;"
        "mod.innerHTML = data.mod_inner;"
    ),
    'mod_inner':     "mod.innerHTML = data.mod_inner;",
    'myOnMouseOver': "svgparentslpi.focus();",
    'myOnMouseDown': (
        "state.x0_drag = state.x1_drag = event.offsetX;"
        "state.y0_drag = state.y1_drag = event.offsetY;"
        "state.shiftkey = event.shiftKey; state.ctrlkey = event.ctrlKey;"
        "state.dragging = true;"
        "data.drag_x0 = Math.round(event.offsetX); data.drag_y0 = Math.round(event.offsetY);"
        "self.myUpdateDragRect();"
    ),
    'myOnMouseMove': (
        "data.x_mouse = event.offsetX; data.y_mouse = event.offsetY;"
        "if (state.dragging) {"
        "  state.x1_drag = event.offsetX; state.y1_drag = event.offsetY;"
        "  state.shiftkey = event.shiftKey; state.ctrlkey = event.ctrlKey;"
        "  self.myUpdateDragRect();"
        "}"
    ),
    'myOnMouseUp': (
        "if (!state.dragging) return; state.dragging = false;"
        "data.drag_x1 = Math.round(event.offsetX); data.drag_y1 = Math.round(event.offsetY);"
        "data.ctrlkey = event.ctrlKey; data.shiftkey = event.shiftKey;"
        "self.myUpdateDragRect();"
        "data.drag_op_finished = !data.drag_op_finished;"
    ),
    'myOnMouseLeave': (
        "state.dragging = false; self.myUpdateDragRect();"
    ),
    'myUpdateDragRect': (
        "if (state.dragging) {"
        "  var x = Math.min(state.x0_drag, state.x1_drag);"
        "  var y = Math.min(state.y0_drag, state.y1_drag);"
        "  var w = Math.abs(state.x1_drag - state.x0_drag);"
        "  var h = Math.abs(state.y1_drag - state.y0_drag);"
        "  drag_rect.setAttribute('x', x); drag_rect.setAttribute('y', y);"
        "  drag_rect.setAttribute('width', w); drag_rect.setAttribute('height', h);"
        "  if      (state.shiftkey && state.ctrlkey) drag_rect.setAttribute('stroke', '#0000ff');"
        "  else if (state.shiftkey)                  drag_rect.setAttribute('stroke', '#ff0000');"
        "  else if (state.ctrlkey)                   drag_rect.setAttribute('stroke', '#00ff00');"
        "  else                                       drag_rect.setAttribute('stroke', '#000000');"
        "} else {"
        "  drag_rect.setAttribute('width', 0); drag_rect.setAttribute('height', 0);"
        "}"
    ),
    'myOnKeyDown':    "event.stopPropagation(); var k=event.key; if(k==='X'){data.key_op_finished='X';} else if(k==='x'){data.key_op_finished='x';} else if(k==='c'){data.key_op_finished='c';}",
}


# ─────────────────────────────────────────────────────────────────────────────
# Interactive wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SLPI(P2SReactiveHTML):
    """Panel view for SpreadLinesP.

    A single static class: size and GPU mode are ordinary params, so one
    compiled class serves every instance.  (This used to be built per call with
    ``type('SLPI', (ReactiveHTML,), {...})`` because the width/height were baked
    into ``_template``; they no longer are.)
    """

    # ── per-instance view state ──────────────────────────────────────────────
    svg_w:      Any = param.Integer(default=0)
    svg_h:      Any = param.Integer(default=0)
    use_webgpu: Any = param.Boolean(default=False)

    mod_inner:   Any = param.String(default='')
    # gpu_payload / gpu_error live on SLPI_GPU only.

    x_mouse:          Any = param.Integer(default=0)
    y_mouse:          Any = param.Integer(default=0)
    drag_x0:          Any = param.Integer(default=0)
    drag_y0:          Any = param.Integer(default=0)
    drag_x1:          Any = param.Integer(default=0)
    drag_y1:          Any = param.Integer(default=0)
    drag_op_finished: Any = param.Boolean(default=False)
    shiftkey:         Any = param.Boolean(default=False)
    ctrlkey:          Any = param.Boolean(default=False)
    key_op_finished:  Any = param.String(default='')

    _template = _SLPI_TEMPLATE_
    _scripts  = _SLPI_SCRIPTS_

    # Read by panelize() to decide what this view broadcasts; inherited by SLPI_GPU.
    _broadcasts_selection_ = True

    # Instance state filled in __init__.  Bare annotations only -- assigning here
    # would make them class attributes that shadow the per-instance values.
    _spread_base_:     Any
    _spread_:          Any
    _cache_:           dict
    selected_entities: set[str]
    lock:              Any
    mvc:               Any

    # ── Constructor ──────────────────────────────────────────────────────────
    def __init__(self, _spread_: Any, use_webgpu: bool = False, **kwargs: Any) -> None:
        _mvc_    = kwargs.pop('mvc', None)
        _w_, _h_ = _spread_.wxh
        super().__init__(
            svg_w=_w_, svg_h=_h_, use_webgpu=use_webgpu,
            **({'gpu_payload': _spread_.webgpu()} if use_webgpu else {}),
            **kwargs)
        # Assigned rather than passed to super(): `mod_inner` is a content
        # binding (${mod_inner} between tags), so ReactiveHTML treats it as a
        # child and rejects a str passed as a constructor kwarg.  The dynamic
        # class carried the SVG as the param *default* for the same reason.
        self.mod_inner = '' if use_webgpu else _spread_._repr_svg_()
        self.lock = asyncio.Lock()
        # The base plot every render_with() starts from.  'c' (change ego)
        # replaces it, which is why it is instance state rather than a constant.
        self._spread_base_     = _spread_
        self._spread_          = _spread_
        self._cache_           = {id(_spread_.df_orig): (_spread_.df_orig, _spread_)}
        self.selected_entities = set()
        if _mvc_ is None:
            self.mvc = InteractionController()
            self.mvc.addStack('default', _spread_.df_orig)
            self.mvc.view_stack[id(self)] = 'default'
            self.mvc.view_refs[id(self)]  = self
        else:
            self.mvc = _mvc_
            self.mvc.view_refs[id(self)] = self
        self.param.watch(self.applyDragOp, 'drag_op_finished')
        self.param.watch(self.applyKeyOp,  'key_op_finished')
        if use_webgpu:
            self.param.watch(self.applyGpuError, 'gpu_error')

    # ── View helpers ─────────────────────────────────────────────────────────
    def __renderView__(self, df: Any) -> Any:
        return self._spread_base_.render_with(df)

    def _highlighted_plot(self) -> Any:
        """The SpreadLinesP instance to render now (with current highlight_nodes)."""
        df = self._spread_.df
        if self.selected_entities:
            return self._spread_base_.render_with(df, highlight_nodes=self.selected_entities)
        return self._spread_

    def _highlighted_svg(self) -> str:
        """Re-render self._spread_.df with current highlight_nodes, return SVG."""
        return self._highlighted_plot()._repr_svg_()

    def _apply_render_(self) -> None:
        """Push the current plot to whichever backend is active (GPU canvas or SVG)."""
        _plot_ = self._highlighted_plot()
        if   not self.use_webgpu: self.mod_inner   = _plot_._repr_svg_()
        elif self.gpu_error:      self.mod_inner   = _gpu_error_overlay(self.gpu_error, self.svg_w, self.svg_h)
        else:                     self.gpu_payload = _plot_.webgpu()

    def __refreshView__(self) -> None:
        self._apply_render_()

    # WebGPU rendering failed in the browser -> surface the error in the overlay.
    # No automatic SVG fallback: the user must re-create the view with use_webgpu=False.
    async def applyGpuError(self, event: Any) -> None:
        if self.gpu_error:
            self.mod_inner = _gpu_error_overlay(self.gpu_error, self.svg_w, self.svg_h)

    # ── MVC callbacks ────────────────────────────────────────────────────────
    async def display(self, df: pl.DataFrame, dfs: list, dfs_index: int) -> None:
        async with self.lock:
            # entries are (df, spread); identity-guard the id() key against reuse
            entry = self._cache_.get(id(df))
            if entry is None or entry[0] is not df:
                entry = (df, self._spread_base_.render_with(df))
                self._cache_[id(df)] = entry
            self._spread_ = entry[1]
            self._apply_render_()
            keep = {id(d) for d in dfs}
            for k in list(self._cache_):
                if k not in keep:
                    del self._cache_[k]

    async def receiveSelection(self, entities: list) -> None:
        self.selected_entities = {str(e) for e in entities}
        self._apply_render_()

    async def _broadcastSelection(self) -> None:
        """Notify all mvc-registered views with receiveSelection, bypassing explicit link setup."""
        for _v_ in self.mvc.view_refs.values():
            if _v_ is not self and hasattr(_v_, 'receiveSelection'):
                await _v_.receiveSelection(self.selected_entities)

    # ── Interaction callbacks ─────────────────────────────────────────────────
    async def applyDragOp(self, event: Any) -> None:
        async with self.lock:
            x0, y0 = self.drag_x0, self.drag_y0
            x1, y1 = self.drag_x1, self.drag_y1
            shiftkey, ctrlkey = self.shiftkey, self.ctrlkey
        is_click = abs(x1 - x0) < 5 and abs(y1 - y0) < 5
        vx0, vy0 = _to_viewbox_coords(self._spread_, x0, y0)
        vx1, vy1 = _to_viewbox_coords(self._spread_, x1, y1)
        if is_click:
            nodes = _nodes_at_xy(self._spread_, vx0, vy0)
            nodes = _expand_ego(self._spread_, nodes)
            if nodes:
                self.selected_entities = _apply_set_op(
                    self.selected_entities, nodes, shiftkey, ctrlkey)
            else:
                self.selected_entities = set()
        else:
            nodes = _nodes_in_rect(self._spread_, vx0, vy0, vx1, vy1)
            nodes = _expand_ego(self._spread_, nodes)
            self.selected_entities = _apply_set_op(
                self.selected_entities, nodes, shiftkey, ctrlkey)
        self._apply_render_()
        await self._broadcastSelection()

    async def applyKeyOp(self, event: Any) -> None:
        async with self.lock:
            op = self.key_op_finished
            self.key_op_finished = ''
        if op == 'X':
            self.selected_entities = set()
            self._apply_render_()
            await self.mvc.popStack(self)
        elif op == 'x':
            if not self.selected_entities:
                return
            df = _filter_out_nodes(self._spread_, self._spread_.df_orig, self.selected_entities)
            if len(df) > 0:
                self.selected_entities = set()
                await self.mvc.pushStack(self, df)
        elif op == 'c':
            if not self.selected_entities:
                return
            new_ego = (next(iter(self.selected_entities))
                       if len(self.selected_entities) == 1
                       else set(self.selected_entities))
            new_spread = self._spread_base_.render_with(self._spread_.df_orig, ego=new_ego)
            self._spread_base_ = new_spread
            self._cache_ = {id(new_spread.df_orig): (new_spread.df_orig, new_spread)}
            self._spread_ = new_spread
            self._apply_render_()
            await self._broadcastSelection()


class SLPI_GPU(SLPI):
    """SLPI drawing into a WebGPU canvas -- adds the runtime the SVG class omits."""
    use_webgpu:  Any = param.Boolean(default=True)
    gpu_payload: Any = param.Dict(default={})
    gpu_error:   Any = param.String(default='')

    _template = _SLPI_GPU_TEMPLATE_
    _scripts  = _withGpuScripts_(_SLPI_SCRIPTS_)


def spreadlinepi(_spread_: Any, use_webgpu: bool = False, **kwargs: Any) -> Any:
    _cls_ = SLPI_GPU if use_webgpu else SLPI
    return _cls_(_spread_, use_webgpu=use_webgpu, **kwargs)


# Register with the panelize wrapper registry.  Registration lives here (not
# only in interactive_controller) so that importing this module first still
# yields a complete registry despite the module cycle between the two files.
from .interactive_controller import _PLOT_TYPE_TO_WRAPPER_
_PLOT_TYPE_TO_WRAPPER_['SpreadLinesP'] = spreadlinepi
