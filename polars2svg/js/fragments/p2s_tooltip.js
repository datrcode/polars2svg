//
// p2s_tooltip - the browser half of F1, per-element hover tooltips
//
// PLANNING.md section 7 F1.  Resting the pointer over a mark says what is under it,
// without a click, without arming a mode, and without changing selection, stack or any
// linked view.  It is a pure read.
//
// The round trip is the brush's, with the result rendered IN PLACE instead of broadcast
// to peers:
//
//   mousemove -> dwell timer -> tooltip_x/tooltip_y + tooltip_seq -> applyTooltipOp ->
//   recordsAt() -> tooltip_payload -> draw here
//
// Two renderings, chosen by the panel's `tooltip` row: `text` (the fallback) and `icon`
// (a user-supplied component re-rendered against the records under the cursor).  Both
// arrive as one payload and are drawn by the same code; only the body differs.
//
// The dwell timer is not an optimisation.  In icon mode every request is a full component
// render, so firing one per mousemove would queue renders faster than they complete.
//
// STALENESS.  A tooltip that resolves slowly and lands after the pointer has moved on is
// U7 exactly -- the defect the brush already carries a sequence ticket for -- so the
// ticket is here from the start rather than rediscovered.  Python echoes the seq it
// answered and a payload whose seq is not the latest one sent is dropped on arrival.
// The check is on THIS side as well as Python's because the two races are different:
// Python's ticket stops a stale recordsAt() from overwriting a fresh one, and this one
// stops a payload that was fresh when it was sent from being drawn over a pointer that
// has since left the mark.
//
// This is a FRAGMENT: no import, no export.  See polars2svg/p2s_esm.py.
//
// ctx:
//   model, state   the entry module's data model and its plain per-view state object
//   node           <g id="tooltip">, pointer-events: none
//   w, h           the canvas size, for edge flipping
//

function p2sTooltip(ctx) {
  const model = ctx.model, state = ctx.state, node = ctx.node;
  const NS_ = 'http://www.w3.org/2000/svg';
  const PAD_ = 4, LINE_H_ = 11, GAP_ = 12;

  function isOn() { return model.tooltip && model.tooltip !== 'off'; }

  // Clear the drawing AND supersede anything in flight.  Called on mouse-out, on the
  // mode going off, and on every re-render -- a tooltip drawn over a plot that has
  // since been replaced describes marks that are no longer there.
  function clear() {
      state.tooltip_seq_drawn = state.tooltip_seq_sent;
      if (state.tooltip_timer != null) { clearTimeout(state.tooltip_timer); state.tooltip_timer = null; }
      while (node.firstChild) { node.removeChild(node.firstChild); }
  }

  // Called from myOnMouseMove.  Re-arms the dwell on every move, so the request fires
  // once the pointer has been still for tooltip_delay_ms and not once per pixel.
  //
  // It also CLEARS on every move, which is deliberate: a box that lingered where the
  // pointer used to be is describing a mark the pointer has left.  The brush throttles
  // by distance instead (>= 3px of travel) because a brush that flickered off on every
  // twitch would drop the linked views it feeds; a tooltip feeds nothing, so hiding and
  // coming back is free.
  function onMouseMove(x, y) {
      if (!isOn()) { return; }
      clear();
      state.tooltip_timer = setTimeout(function() {
          state.tooltip_timer = null;
          state.tooltip_seq_sent += 1;
          model.tooltip_x   = Math.round(x);
          model.tooltip_y   = Math.round(y);
          model.tooltip_seq = state.tooltip_seq_sent;
      }, Math.max(0, model.tooltip_delay_ms));
  }

  // Called from the model.on('tooltip_payload') handler.
  function draw() {
      var _p_ = model.tooltip_payload;
      if (!_p_ || !_p_.seq) { return; }
      // Superseded while Python was working, or already answered.  `<` and not `!==`:
      // an out-of-order arrival must not resurrect an older payload either.
      if (_p_.seq <= state.tooltip_seq_drawn) { return; }
      state.tooltip_seq_drawn = _p_.seq;
      while (node.firstChild) { node.removeChild(node.firstChild); }
      if (!isOn() || _p_.empty) { return; }

      var _w_ = _p_.w, _h_ = _p_.h;
      // Flip at the canvas edge rather than clamp: a box pinned to the right edge sits
      // ON the mark it describes, which is the one thing it must not cover.
      var _x_ = (_p_.x + GAP_ + _w_ <= ctx.w) ? _p_.x + GAP_ : _p_.x - GAP_ - _w_;
      var _y_ = (_p_.y + GAP_ + _h_ <= ctx.h) ? _p_.y + GAP_ : _p_.y - GAP_ - _h_;
      // Both flips can still leave it off-canvas on a plot narrower than the box.
      _x_ = Math.max(0, Math.min(_x_, Math.max(0, ctx.w - _w_)));
      _y_ = Math.max(0, Math.min(_y_, Math.max(0, ctx.h - _h_)));

      var _g_ = document.createElementNS(NS_, 'g');
      _g_.setAttribute('transform', 'translate(' + _x_ + ',' + _y_ + ')');
      var _r_ = document.createElementNS(NS_, 'rect');
      _r_.setAttribute('x', 0); _r_.setAttribute('y', 0);
      _r_.setAttribute('width', _w_); _r_.setAttribute('height', _h_);
      _r_.setAttribute('rx', 3);
      _r_.setAttribute('fill', 'rgba(240,240,240,0.95)');
      _r_.setAttribute('stroke', '#888'); _r_.setAttribute('stroke-width', '1');
      _g_.appendChild(_r_);

      if (_p_.svg) {
          // Icon mode.  The payload is a whole <svg> element from the component's own
          // _repr_svg_(); nesting one inside the view's is what stack_controli already
          // does with the very same render_with() contract.
          var _holder_ = document.createElementNS(NS_, 'g');
          _holder_.setAttribute('transform', 'translate(' + PAD_ + ',' + PAD_ + ')');
          _holder_.innerHTML = _p_.svg;
          _g_.appendChild(_holder_);
      }
      if (_p_.lines && _p_.lines.length) {
          // Text mode, and the caption under an icon when the payload carries both.
          var _ty_ = (_p_.svg ? _p_.icon_h + PAD_ : 0) + PAD_ + LINE_H_ - 2;
          var _t_ = document.createElementNS(NS_, 'text');
          _t_.setAttribute('x', PAD_ + 2); _t_.setAttribute('y', _ty_);
          _t_.setAttribute('font-family', "'Courier New', monospace");
          _t_.setAttribute('font-size', '11px');
          _t_.setAttribute('fill', '#222');
          for (var _i_ = 0; _i_ < _p_.lines.length; _i_++) {
              var _s_ = document.createElementNS(NS_, 'tspan');
              _s_.setAttribute('x', PAD_ + 2);
              _s_.setAttribute('dy', _i_ === 0 ? 0 : LINE_H_);
              _s_.textContent = _p_.lines[_i_];
              _t_.appendChild(_s_);
          }
          _g_.appendChild(_t_);
      }
      node.appendChild(_g_);
  }

  return { onMouseMove: onMouseMove, draw: draw, clear: clear };
}
