//
// p2s_interactivep - the browser half of the five generic components
//
// TIMEPI / HISTOPI / XYPI / CHORDPI / PIEPI, and their *_GPU subclasses, all share this
// one module.  Ported from _INTERACTIVEP_TEMPLATE_ + _INTERACTIVEP_SCRIPTS_
// (PLANNING.md W1) with the same substitutions as the earlier contracts: data.x ->
// model.x, an id'd template node -> a const in this closure with the same name, Panel's
// per-view `state` -> a plain object, and `self.helper()` -> a plain call.
//
// Three things the template did that a module has to do explicitly:
//
//   * `__ROOT__` was substituted per kind, because the root <svg>'s id doubled as the JS
//     variable name Panel declared for it.  The closure holds the element directly, so
//     the id is now just an id and arrives as the `svg_parent_id` param.
//   * `__KBD_HELP__`, the pre-laid-out help overlay, was substituted in per kind too.
//     It arrives as `kbd_help_svg`.
//   * `{% if has_search %}` decided whether #searchtext existed at all.  It is an
//     `if (model.has_search)` here, and every path that touches the node is already
//     guarded by the same flag -- as it had to be under ReactiveHTML, where Panel
//     likewise did not declare the variable for a node the template had omitted.
//
// updateBrushCursor and myUpdateDragRect below are deliberately local rather than shared
// fragments.  LINKPI has same-named versions and the plan expected to share them, but
// measured against each other they are 44% (drag rect) and 88% (brush cursor) similar.
// Parameterising around that much divergence costs more than the duplication saves.
//
// The PICKER MENU is no longer among them.  It was measured at 33-57% and left duplicated
// on the same reasoning, and F1 overturned that: the tooltip is a configuration-panel row
// on every component, the panel opens the row's picker (CP3), and this module's picker was
// a strict subset of LINKPI's -- one kind, no display strings, no guarded items, no
// menu_x.  Both now come from js/fragments/p2s_config_panel.js and this module gains the
// superset.  The divergence that justified duplicating the others was never here.
//
// This is an ENTRY module: it exports render().
//

export function render({ model, el }) {
  const W = model.svg_w;
  const H = model.svg_h;

  const state = {
    x0_drag: -10, y0_drag: -10, x1_drag: -5, y1_drag: -5,
    drag_op: false,
    last_brush_x: -999, last_brush_y: -999,
    cur_mouse_x: -999, cur_mouse_y: -999,
    brush_defs: [null, ['circle', 5], ['circle', 15], ['vertical', 1], ['vertical', 3],
                 ['horizontal', 1], ['horizontal', 3]],
    brush_names: ['', 'circ r=5', 'circ r=15', 'vert r=1', 'vert r=3', 'horiz r=1', 'horiz r=3'],
    // Python's, not a literal: the tooltip row's value list varies per view (the icon
    // state exists only when the view was built with icon=), so the menus are data now.
    // state.select_shape went with it -- model.select_shape is the single source, which
    // is what the shared picker and the panel both read.
    menu_items: model.menu_items,
    menu_open: false, menu_kind: '', menu_index: 0, menu_timer: null, menu_x: 8,
    panel_open: false, panel_row: 0, panel_pending: {}, panel_timer: null, panel_w: 0,
    search_mode: false, search_buffer: '',
    pending_mods: null,
    // F1.  seq_sent is what the browser has asked for, seq_drawn what it has answered or
    // abandoned; a payload is drawn only while the two say it is still wanted.
    tooltip_timer: null, tooltip_seq_sent: 0, tooltip_seq_drawn: 0,
  };

  // ── DOM, in the order the template listed it ───────────────────────────────

  const root = svgEl('svg', {
    id: model.svg_parent_id, width: W, height: H, tabindex: '0',
  });

  const mod             = svgEl('svg', { id: 'mod', width: W, height: H }, root);
  const brushindicator  = svgEl('g', { id: 'brushindicator', 'pointer-events': 'none' }, root);
  const brushmodelabel  = svgEl('g', { id: 'brushmodelabel', 'pointer-events': 'none' }, root);
  const keyboardhelp    = svgEl('g', {
    id: 'keyboardhelp', transform: 'translate(' + model.keyboardhelp_x + ' 0)',
  }, root);

  const drag = svgEl('rect', {
    id: 'drag', x: -10, y: -10, width: 5, height: 5,
    stroke: p2sInk(model, 'replace'), 'stroke-width': 2, fill: 'none',
  }, root);

  const dragoval = svgEl('ellipse', {
    id: 'dragoval', cx: -10, cy: -10, rx: 0, ry: 0,
    stroke: p2sInk(model, 'replace'), 'stroke-width': 2, fill: 'none', display: 'none',
  }, root);

  const screen = svgEl('rect', {
    id: 'screen', x: 0, y: 0, width: W, height: H, opacity: 0.05,
  }, root);

  const infostr = svgEl('text', {
    id: 'infostr', x: 5, y: H - 3, fill: p2sInk(model, 'ink'),
    'font-size': '10px', 'pointer-events': 'none',
  }, root);

  // #searchtext exists only for the kinds that bind '/'.  Everything that touches it is
  // guarded by model.has_search, exactly as it was when the node's absence also meant
  // the JS variable did not exist.
  let searchtext = null;
  if (model.has_search) {
    searchtext = svgEl('text', {
      id: 'searchtext', x: Math.floor(W / 2), y: H - 2, 'text-anchor': 'middle',
      fill: p2sInk(model, 'hint'), 'font-size': '11px', 'font-family': 'monospace',
      'pointer-events': 'none',
    }, root);
  }

  // Under the panel and the picker, over everything else: a tooltip follows the pointer
  // and would otherwise sit on top of the overlay the user is reading.
  const tooltip     = svgEl('g', { id: 'tooltip', 'pointer-events': 'none' }, root);
  // Before #pickermenu so that a picker opened FROM the panel draws over it rather than
  // under it; menuRender also shifts that picker clear of the panel (state.menu_x).
  const configpanel = svgEl('g', { id: 'configpanel', 'pointer-events': 'none' }, root);
  const pickermenu  = svgEl('g', { id: 'pickermenu', 'pointer-events': 'none' }, root);

  keyboardhelp.innerHTML = model.kbd_help_svg;

  // ── helpers ────────────────────────────────────────────────────────────────

  function updateBrushCursor() {
    var _bs_ = model.brush_state;
    if (_bs_ == 0) {
      brushindicator.innerHTML = '';
      brushmodelabel.innerHTML = '';
      return;
    }
    var _x_ = state.cur_mouse_x, _y_ = state.cur_mouse_y;
    var _d_ = state.brush_defs[_bs_], _r_ = _d_[1];
    var _s_ = 'stroke="rgba(100,150,255,0.8)" fill="none" pointer-events="none"';
    if      (_d_[0] == 'circle')     { brushindicator.innerHTML = '<circle cx="'+_x_+'" cy="'+_y_+'" r="'+_r_+'" '+_s_+' stroke-width="1.5"/>'; }
    else if (_d_[0] == 'vertical')   { brushindicator.innerHTML = '<line x1="'+_x_+'" y1="0" x2="'+_x_+'" y2="'+model.svg_h+'" '+_s_+' stroke-width="'+Math.max(1,_r_)+'"/>'; }
    else if (_d_[0] == 'horizontal') { brushindicator.innerHTML = '<line x1="0" y1="'+_y_+'" x2="'+model.svg_w+'" y2="'+_y_+'" '+_s_+' stroke-width="'+Math.max(1,_r_)+'"/>'; }
    var _nm_ = state.brush_names[_bs_];
    var _tw_ = _nm_.length * 6 + 10, _rx_ = model.svg_w - _tw_ - 3;
    brushmodelabel.innerHTML = '<rect x="'+_rx_+'" y="3" width="'+_tw_+'" height="15" rx="3" fill="rgba(100,150,255,0.3)" stroke="rgba(100,150,255,0.7)" stroke-width="0.5" pointer-events="none"/>'
        + '<text x="'+(_rx_+5)+'" y="14" font-size="10px" fill="rgba(40,60,200,1.0)" font-family="monospace" pointer-events="none">'+_nm_+'</text>';
  }

  function myUpdateDragRect() {
    // Names the pending set operation; palette-driven so the band stays visible on a
    // dark canvas.  Keys match _resolve_set_op() in Python.
    var _stroke_ = (model.shiftkey && model.ctrlkey) ? p2sInk(model, 'intersect')
                 : (model.shiftkey)                  ? p2sInk(model, 'subtract')
                 : (model.ctrlkey)                   ? p2sInk(model, 'add')
                 :                                     p2sInk(model, 'replace');
    if (state.drag_op && model.select_shape == 'oval') {
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
      var x = Math.min(state.x0_drag, state.x1_drag);
      var y = Math.min(state.y0_drag, state.y1_drag);
      var w = Math.abs(state.x1_drag - state.x0_drag);
      var h = Math.abs(state.y1_drag - state.y0_drag);
      drag.setAttribute('x',x);     drag.setAttribute('y',y);
      drag.setAttribute('width',w); drag.setAttribute('height',h);
      drag.setAttribute('stroke',_stroke_);
      dragoval.setAttribute('display','none');
    } else {
      drag.setAttribute('x',-10);   drag.setAttribute('y',-10);
      drag.setAttribute('width',5); drag.setAttribute('height',5);
      dragoval.setAttribute('display','none');
    }
  }

  // ── menus, the configuration panel and the tooltip ──
  //
  // All three come from js/fragments/p2s_config_panel.js and p2s_tooltip.js, which is
  // where LINKPI's versions live too.  The five kinds had their own two-value picker for
  // `select_shape`; the shared one is a superset of it, and the panel (F1, and
  // 20260921_config_panel_design.md for the design it follows) needs both.

  const MENU_PARAM_ = {
      select_shape: 'select_shape',
      tooltip:      'tooltip',
  };

  const MENU_HEADER_ = {
      // Unchanged from the local picker this replaces -- the browser tests address a
      // picker by the header text it draws.
      select_shape: 'selection shape:',
      tooltip:      'tooltip:',
  };

  // Render rows (interactive_render_rows.py) are the kinds with no param of their own:
  // every one of them lives in the single render_settings dict, and a kind is one when
  // it is a key there.  Their picker header is the row's own label, read off
  // config_panel_rows, because the kinds are per component and this module is shared.
  function isRenderKind(kind) {
      return model.render_settings && Object.prototype.hasOwnProperty.call(model.render_settings, kind);
  }
  function syncRenderHeaders() {
      for (const _row_ of (model.config_panel_rows || [])) {
          if (isRenderKind(_row_[1])) { MENU_HEADER_[_row_[1]] = _row_[2] + ':'; }
      }
  }
  syncRenderHeaders();

  function menuGetValue(kind) {
      return isRenderKind(kind) ? model.render_settings[kind] : model[MENU_PARAM_[kind]];
  }

  // The single write path for every choice a picker or a panel row can make.  An
  // explicit chain, matching LINKPI's, rather than MENU_PARAM_[kind]: the write side is
  // where a kind gets to be special, and spelling it out is what makes that visible.
  // A render row writes a whole new dict -- reassigning is what syncs a Dict param.
  function menuSetValue(kind, label) {
      if      (kind == 'select_shape') { model.select_shape = label; }
      else if (kind == 'tooltip')      { model.tooltip      = label; }
      else if (isRenderKind(kind)) {
          const _s_ = Object.assign({}, model.render_settings);
          _s_[kind] = label;
          model.render_settings = _s_;
      }
  }

  const _cp_ = p2sConfigPanel({
      model: model, state: state, menuNode: pickermenu, panelNode: configpanel,
      headers: MENU_HEADER_, params: MENU_PARAM_, setValue: menuSetValue,
      getValue: menuGetValue,
  });
  // Only the entry points this module still calls.  The rest of the fragment's surface
  // is reached through menuKeyDown / panelKeyDown, which own the two modal key blocks.
  const menuOpen  = _cp_.menuOpen,
        panelOpen = _cp_.panelOpen, panelRender = _cp_.panelRender;

  const _tt_ = p2sTooltip({ model: model, state: state, node: tooltip, w: W, h: H });

  // ── DOM handlers ───────────────────────────────────────────────────────────

  function myOnMouseOver() {
    model.has_focus = true;
    root.focus();
  }

  function myOnMouseOut() {
    model.has_focus          = false;
    brushindicator.innerHTML = '';
    _tt_.clear();
    if (model.brush_state > 0) { model.brush_leave_done = true; }
  }

  // key events don't have access to event.offsetX/Y
  function myOnKeyDown(event) {
    event.stopPropagation();
    if (model.has_search && state.search_mode) {
      if (event.key === 'Enter') {
        if (state.search_buffer) {
          model.search_str = state.search_buffer;
          model.search_op_finished = !model.search_op_finished;
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
    // Both modal blocks live in the shared fragment now and each reports whether it
    // consumed the event -- the same `return` the inlined versions ended on.  The picker
    // is tested first, because a picker opened FROM a panel row is on top of it and has
    // to see the keys.
    if (_cp_.menuKeyDown(event))  { return; }
    if (_cp_.panelKeyDown(event)) { return; }
    model.shiftkey = event.shiftKey;
    model.ctrlkey  = event.ctrlKey;
    model.x_mouse  = state.cur_mouse_x;
    model.y_mouse  = state.cur_mouse_y;
    if      (model.has_z_key     && (event.key == 'z' || event.key == 'Z')) { model.key_op_finished = "z"; }
    else if (model.has_time_keys && (event.key == 'u' || event.key == 'U')) { model.key_op_finished = "u"; }
    else if (model.has_time_keys && (event.key == 'e' || event.key == 'E')) { model.key_op_finished = "e"; }
    else if (event.key == 'r') {
      if (model.brush_state == 0) {
        var _seq_ = model.brush_seq;
        model.brush_state = _seq_[1];
      } else {
        model.brush_state = 0;
      }
      model.brushing_mode = model.brush_state > 0;
      model.brush_changed += 1;
      updateBrushCursor();
    }
    else if (event.key == 'R') {
      var _seq2_ = model.brush_seq;
      var _non0_ = _seq2_.filter(function(x) { return x > 0; });
      if (model.brush_state == 0) {
        model.brush_state = _non0_[0];
      } else {
        var _i2_ = _non0_.indexOf(model.brush_state);
        model.brush_state = _non0_[(_i2_ + 1) % _non0_.length];
      }
      model.brushing_mode = true;
      model.brush_changed += 1;
      updateBrushCursor();
    }
    else if (event.key == 'q') { model.key_op_finished = "q"; }
    // The settings panel, same key and same shape as LINKPI's.  'F' is NOT absorbed
    // into it the way LINKPI's twelve bindings were: that absorption paid for a keyspace
    // that had run out, and these five have most of the alphabet free.  Both doors reach
    // the one picker (CP3).
    else if (event.key == 'a' && !event.ctrlKey) { panelOpen(false); }
    else if (event.key == 'A' && !event.ctrlKey) { panelOpen(true);  }
    else if (event.key == 'F') { state.menu_kind = 'select_shape'; menuOpen(); }
    else if (event.key == 'h') {
      if (model.keyboardhelp_x == -1000) { model.keyboardhelp_x =     5; }
      else                               { model.keyboardhelp_x = -1000; }
    }
    else if (model.has_search && event.key == '/') {
      state.search_mode   = true;
      state.search_buffer = '';
      searchtext.textContent = '/ ▋';
    }
  }

  // Hold the modifiers while a key operation is still in flight.  applyKeyOp runs
  // asynchronously behind a lock and reads ctrlkey / shiftkey when it gets there, so
  // clearing them the instant the user let go made a *tapped* ctrl-<key> arrive with the
  // modifier already gone -- the handler took the unmodified branch, and ctrl-c zoomed
  // the view instead of copying (U3).
  //
  // The release is not discarded, it is deferred: the key_op_finished handler applies it
  // as soon as Python reports the operation done.  Simply skipping the clear would leave
  // the modifier stuck on until the next keydown, and the drag band -- which reads these
  // to colour itself -- would name the wrong set-operation.
  function myOnKeyUp(event) {
    if (model.key_op_finished === '') {
      model.shiftkey = event.shiftKey;
      model.ctrlkey  = event.ctrlKey;
    } else {
      state.pending_mods = [event.ctrlKey, event.shiftKey];
    }
  }

  function myOnMouseMove(event) {
    state.cur_mouse_x = event.offsetX;
    state.cur_mouse_y = event.offsetY;
    state.x1_drag     = event.offsetX;
    state.y1_drag     = event.offsetY;
    if (state.drag_op) { myUpdateDragRect(); }
    // F1.  Not throttled by distance the way the brush below is: the dwell timer is the
    // throttle, and re-arming it on every move is what makes it a dwell.  A drag in
    // progress suppresses it -- a tooltip is a pure read, and nothing is being read
    // while the pointer is pulling a rubber band.
    if (state.drag_op) { _tt_.clear(); }
    else               { _tt_.onMouseMove(event.offsetX, event.offsetY); }
    if (model.brush_state > 0) {
      updateBrushCursor();
      var _dx_ = event.offsetX - state.last_brush_x;
      var _dy_ = event.offsetY - state.last_brush_y;
      if (_dx_*_dx_ + _dy_*_dy_ >= 9) {
        state.last_brush_x = event.offsetX;
        state.last_brush_y = event.offsetY;
        model.x_mouse      = event.offsetX;
        model.y_mouse      = event.offsetY;
        model.brush_changed += 1;
      }
    }
  }

  function downSelect(event) {
    if (event.button == 0) {
      state.x0_drag  = event.offsetX;
      state.y0_drag  = event.offsetY;
      state.x1_drag  = event.offsetX;
      state.y1_drag  = event.offsetY;
      state.drag_op  = true;
      myUpdateDragRect();
    } else if (event.button == 1) {
      model.x0_middle = model.x1_middle = event.offsetX;
      model.y0_middle = model.y1_middle = event.offsetY;
    }
  }

  function myOnMouseUp(event) {
    if (event.button == 0) {
      state.x1_drag         = event.offsetX;
      state.y1_drag         = event.offsetY;
      if (state.drag_op) {
        state.shiftkey        = event.shiftKey;
        state.ctrlkey         = event.ctrlKey;
        state.drag_op         = false;
        myUpdateDragRect();
        model.drag_x0          = state.x0_drag;
        model.drag_y0          = state.y0_drag;
        model.drag_x1          = state.x1_drag;
        model.drag_y1          = state.y1_drag;
        model.drag_op_finished = true;
      }
    }
  }

  // ── wiring ─────────────────────────────────────────────────────────────────

  root.addEventListener('keydown', myOnKeyDown);
  root.addEventListener('keyup',   myOnKeyUp);
  screen.addEventListener('mouseover', myOnMouseOver);
  screen.addEventListener('mouseout',  myOnMouseOut);
  screen.addEventListener('mousedown', downSelect);
  screen.addEventListener('mousemove', myOnMouseMove);
  screen.addEventListener('mouseup',   myOnMouseUp);

  // Non-passive so it can preventDefault.  The template never bound a wheel handler --
  // the `myOnMouseWheel` entry in _scripts was unreachable, with no onwheel attribute,
  // no matching param and no self.myOnMouseWheel() call -- so this listener, attached in
  // `render`, was and remains the only wheel path.  The dead entry is not carried over.
  screen.addEventListener('wheel', function(event) {
    event.preventDefault();
    model.wheel_x = event.offsetX; model.wheel_y = event.offsetY;
    model.wheel_rots = Math.round(10*event.deltaY);
    model.wheel_op_finished = true;
  }, { passive: false });

  // On macOS ctrl+click is a secondary click -> the browser raises a contextmenu (popup)
  // during ctrl / shift-ctrl rectangular drags.  Swallow it on the panel so the
  // intersection/add selection survives.
  root.addEventListener('contextmenu', function(event) {
    event.preventDefault();
  });

  // ── param changes ──────────────────────────────────────────────────────────

  mod.innerHTML     = model.mod_inner;
  infostr.innerHTML = model.info_str;

  model.on('mod_inner', function() {
    mod.innerHTML     = model.mod_inner;
    infostr.innerHTML = model.info_str;
    // A tooltip drawn over the old plot describes marks that are no longer there.
    _tt_.clear();
  });
  model.on('info_str', function() {
    infostr.innerHTML = model.info_str;
  });
  model.on('key_op_finished', function() {
    if (model.key_op_finished === '' && state.pending_mods) {
      model.ctrlkey  = state.pending_mods[0];
      model.shiftkey = state.pending_mods[1];
      state.pending_mods = null;
    }
  });
  // ${keyboardhelp_x} was an attribute binding, so it is an initial setAttribute (done
  // above, at construction) plus this.
  model.on('keyboardhelp_x', function() {
    keyboardhelp.setAttribute('transform', 'translate(' + model.keyboardhelp_x + ' 0)');
  });

  // The configuration panel is a live display, so every value it shows re-renders it --
  // including one Python changed by itself.  config_panel_rows carries the row order AND
  // the per-row enabled flag.
  for (const _pp_ of ['config_panel_rows', 'select_shape', 'render_settings']) {
    model.on(_pp_, function() { syncRenderHeaders(); panelRender(); });
  }
  // The tooltip row's value list varies per view -- the icon state exists only when the
  // view was built with icon= -- so the snapshot in state has to be refreshed rather
  // than taken once.
  model.on('menu_items', function() {
    state.menu_items = model.menu_items;
    panelRender();
  });
  model.on('tooltip_payload', function() { _tt_.draw(); });
  // One listener, two jobs: the row's displayed value changes and the drawing that the
  // old mode left behind has to go with it.
  model.on('tooltip', function() { _tt_.clear(); panelRender(); });

  // `render` also reset these on every ReactiveHTML subtree rebuild.  It runs once per
  // mount now, so they are plain initialisation.
  model.has_focus     = false;
  model.shiftkey      = false;
  model.ctrlkey       = false;
  model.brush_state   = 0;
  model.brushing_mode = false;
  model.brush_changed = 0;

  return (typeof p2sGpuWrap === 'function') ? p2sGpuWrap(model, root) : root;
}
