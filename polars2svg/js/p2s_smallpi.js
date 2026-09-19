//
// p2s_smallpi - the browser half of SMALLPI / SMALLPI_GPU
//
// Ported from _SMALLPI_TEMPLATE_ + _SMALLPI_SCRIPTS_ (PLANNING.md W1).  The handler
// bodies are the ReactiveHTML ones with three mechanical substitutions:
//
//   data.x                  ->  model.x
//   an id'd template node   ->  a const in this closure, same name
//   `state`, supplied by    ->  a plain object, same name, created once
//   Panel per view
//
// Two things the ReactiveHTML build needed and this does not.  `mod_inner` was bound
// as `${mod_inner}` *between* tags, which made it a Panel *child* -- the only way to
// get an SVG past panel's HTML sanitizer, at the cost of rebuilding the whole subtree
// on every write and destroying JS-only state with it.  There is no sanitizer on the
// ESM path and no child here, so it is an ordinary string param written into
// mod.innerHTML.  And `render` used to re-run on every one of those rebuilds, which is
// why the scripts re-seeded `state` defensively; it runs once per mount now.
//
// This is an ENTRY module: it exports render(), and the fragments it calls
// (svgEl/htmlEl, and p2sGpuWrap on the GPU composition) are concatenated ahead of it.
//

export function render({ model, el }) {
  const state = { dragging: false, sx: 0, sy: 0 };
  const W = model.svg_w;
  const H = model.svg_h;

  const root = svgEl('svg', {
    id: 'svgparentsmallpi', width: W, height: H, tabindex: '0',
  });

  const mod = svgEl('svg', { id: 'mod', x: 0, y: 0, width: W, height: H }, root);

  const selbox = svgEl('rect', {
    id: 'selbox', x: 0, y: 0, width: 0, height: 0, display: 'none',
    fill: 'rgba(100,100,255,0.08)', stroke: '#4488ff', 'stroke-width': 1,
    'stroke-dasharray': '4,2', 'pointer-events': 'none',
  }, root);

  const screen = svgEl('rect', {
    id: 'screen', x: 0, y: 0, width: W, height: H,
    style: 'fill:none;pointer-events:all;',
  }, root);

  // ── handlers, straight from _SMALLPI_SCRIPTS_ ──────────────────────────────

  function myOnMouseOver() {
    root.focus();
  }

  function myOnMouseDown(event) {
    state.sx = event.offsetX; state.sy = event.offsetY;
    state.dragging = true;
    model.drag_x0 = Math.round(event.offsetX);
    model.drag_y0 = Math.round(event.offsetY);
  }

  function myOnMouseMove(event) {
    model.x_mouse = event.offsetX; model.y_mouse = event.offsetY;
    if (model.brush_on) { model.brush_changed += 1; }
    if (state.dragging) {
      selbox.setAttribute('x',       Math.min(state.sx, event.offsetX));
      selbox.setAttribute('y',       Math.min(state.sy, event.offsetY));
      selbox.setAttribute('width',   Math.abs(event.offsetX - state.sx));
      selbox.setAttribute('height',  Math.abs(event.offsetY - state.sy));
      selbox.setAttribute('display', 'block');
    }
  }

  function myOnMouseUp(event) {
    if (!state.dragging) return;
    state.dragging = false;
    model.drag_x1 = Math.round(event.offsetX);
    model.drag_y1 = Math.round(event.offsetY);
    model.shiftkey = event.shiftKey;
    selbox.setAttribute('display', 'none');
    model.drag_op_finished = !model.drag_op_finished;
  }

  function myOnMouseLeave() {
    state.dragging = false;
    selbox.setAttribute('display', 'none');
    model.brush_leave_done = !model.brush_leave_done;
  }

  function myOnKeyDown(event) {
    var k = event.key;
    if (k === 'r') {
      model.brush_on = !model.brush_on;
      if (!model.brush_on) { model.key_op_finished = 'brush_off'; }
    } else if (k === 'q' && !event.shiftKey) {
      model.key_op_finished = 'q';
    } else if (k === 'Q' || (event.shiftKey && k === 'q')) {
      model.key_op_finished = 'Q';
    }
  }

  // ── wiring ─────────────────────────────────────────────────────────────────
  //
  // The template bound these as onmouseover="${script('...')}" attributes.  The order
  // of registration is the order the template listed them, and the harness installs
  // its own probes after render() returns, exactly as before.

  root.addEventListener('keydown', myOnKeyDown);
  screen.addEventListener('mouseover',  myOnMouseOver);
  screen.addEventListener('mousedown',  myOnMouseDown);
  screen.addEventListener('mousemove',  myOnMouseMove);
  screen.addEventListener('mouseup',    myOnMouseUp);
  screen.addEventListener('mouseleave', myOnMouseLeave);

  // `render` seeded this, and the `mod_inner` script re-ran it on every write.
  mod.innerHTML = model.mod_inner;
  model.on('mod_inner', function() { mod.innerHTML = model.mod_inner; });

  return (typeof p2sGpuWrap === 'function') ? p2sGpuWrap(model, root) : root;
}
