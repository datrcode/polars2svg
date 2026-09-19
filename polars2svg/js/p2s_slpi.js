//
// p2s_slpi - the browser half of SLPI / SLPI_GPU (spread lines, ego network over time)
//
// Ported from _SLPI_TEMPLATE_ + _SLPI_SCRIPTS_ (PLANNING.md W1), with the same three
// mechanical substitutions as p2s_smallpi.js: data.x -> model.x, an id'd template node
// -> a const in this closure with the same name, and Panel's per-view `state` object ->
// a plain object created once.  `self.myUpdateDragRect()` -- ReactiveHTML's way of
// calling one script from another -- is simply a function call here.
//
// The drag rectangle is deliberately NOT shared with the other contracts yet.  LINKPI
// draws an oval as well as a rect and parks the band off-canvas when idle; the generic
// components encode a different set of modifier states.  Three similar-but-different
// implementations are worth less shared than they are separate, so Phase 4/5 decides
// whether a common one actually falls out.
//
// This is an ENTRY module: it exports render().
//

export function render({ model, el }) {
  const state = {
    dragging: false,
    x0_drag: 0, y0_drag: 0, x1_drag: 0, y1_drag: 0,
    shiftkey: false, ctrlkey: false,
  };
  const W = model.svg_w;
  const H = model.svg_h;

  const root = svgEl('svg', {
    id: 'svgparentslpi', width: W, height: H, tabindex: '0',
  });

  const mod = svgEl('svg', { id: 'mod', x: 0, y: 0, width: W, height: H }, root);

  // #screen before #drag_rect, exactly as the template listed them: the band is a
  // later sibling so it paints over the hit layer.
  const screen = svgEl('rect', {
    id: 'screen', x: 0, y: 0, width: W, height: H,
    style: 'fill:none;pointer-events:all;',
  }, root);

  const drag_rect = svgEl('rect', {
    id: 'drag_rect', x: 0, y: 0, width: 0, height: 0,
    style: 'fill:rgba(128,128,128,0.08);stroke:#000000;stroke-width:1;'
         + 'pointer-events:none;stroke-dasharray:4,2;',
  }, root);

  // ── handlers, straight from _SLPI_SCRIPTS_ ─────────────────────────────────

  function myUpdateDragRect() {
    if (state.dragging) {
      var x = Math.min(state.x0_drag, state.x1_drag);
      var y = Math.min(state.y0_drag, state.y1_drag);
      var w = Math.abs(state.x1_drag - state.x0_drag);
      var h = Math.abs(state.y1_drag - state.y0_drag);
      drag_rect.setAttribute('x', x); drag_rect.setAttribute('y', y);
      drag_rect.setAttribute('width', w); drag_rect.setAttribute('height', h);
      if      (state.shiftkey && state.ctrlkey) drag_rect.setAttribute('stroke', '#0000ff');
      else if (state.shiftkey)                  drag_rect.setAttribute('stroke', '#ff0000');
      else if (state.ctrlkey)                   drag_rect.setAttribute('stroke', '#00ff00');
      else                                      drag_rect.setAttribute('stroke', '#000000');
    } else {
      drag_rect.setAttribute('width', 0); drag_rect.setAttribute('height', 0);
    }
  }

  function myOnMouseOver() {
    root.focus();
  }

  function myOnMouseDown(event) {
    state.x0_drag = state.x1_drag = event.offsetX;
    state.y0_drag = state.y1_drag = event.offsetY;
    state.shiftkey = event.shiftKey; state.ctrlkey = event.ctrlKey;
    state.dragging = true;
    model.drag_x0 = Math.round(event.offsetX); model.drag_y0 = Math.round(event.offsetY);
    myUpdateDragRect();
  }

  function myOnMouseMove(event) {
    model.x_mouse = event.offsetX; model.y_mouse = event.offsetY;
    if (state.dragging) {
      state.x1_drag = event.offsetX; state.y1_drag = event.offsetY;
      state.shiftkey = event.shiftKey; state.ctrlkey = event.ctrlKey;
      myUpdateDragRect();
    }
  }

  function myOnMouseUp(event) {
    if (!state.dragging) return;
    state.dragging = false;
    model.drag_x1 = Math.round(event.offsetX); model.drag_y1 = Math.round(event.offsetY);
    model.ctrlkey = event.ctrlKey; model.shiftkey = event.shiftKey;
    myUpdateDragRect();
    model.drag_op_finished = !model.drag_op_finished;
  }

  function myOnMouseLeave() {
    state.dragging = false;
    myUpdateDragRect();
  }

  function myOnKeyDown(event) {
    event.stopPropagation();
    var k = event.key;
    if (k === 'X') { model.key_op_finished = 'X'; }
    else if (k === 'x') { model.key_op_finished = 'x'; }
    else if (k === 'c') { model.key_op_finished = 'c'; }
  }

  // ── wiring ─────────────────────────────────────────────────────────────────

  root.addEventListener('keydown', myOnKeyDown);
  screen.addEventListener('mouseover',  myOnMouseOver);
  screen.addEventListener('mousedown',  myOnMouseDown);
  screen.addEventListener('mousemove',  myOnMouseMove);
  screen.addEventListener('mouseup',    myOnMouseUp);
  screen.addEventListener('mouseleave', myOnMouseLeave);

  mod.innerHTML = model.mod_inner;
  model.on('mod_inner', function() { mod.innerHTML = model.mod_inner; });

  return (typeof p2sGpuWrap === 'function') ? p2sGpuWrap(model, root) : root;
}
