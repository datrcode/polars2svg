//
// p2s_stack_control - the browser half of STACKCONTROLI (the stack navigator)
//
// Ported from _STACK_CONTROL_TEMPLATE_ + _STACK_CONTROL_SCRIPTS_ (PLANNING.md W1).
// The smallest contract in the project: five scripts, no drag, no canvas, no GPU
// variant.  It is also the only one with *two* things the template supplied that a
// module has to be given explicitly:
//
//   * `_STACK_HELP_SVG_`, a module constant concatenated straight into the template.
//     It arrives as the `kbd_help_svg` param now.  It costs the same on the wire --
//     the template shipped per view too -- and it keeps the markup out of the JS.
//   * `display="${help_display}"`, an *attribute* binding rather than a content one,
//     so it is an initial setAttribute plus a model.on(), not an innerHTML write.
//
// This is an ENTRY module: it exports render().
//

export function render({ model, el }) {
  const W = model.svg_w;
  const H = model.svg_h;

  // overflow:visible so the help overlay can spill over a neighbouring component
  // rather than being clipped away -- see the note above the original template.
  const root = svgEl('svg', {
    id: 'svgstackcontrol', width: W, height: H, tabindex: '0',
    style: 'overflow: visible;',
  });

  const mod = svgEl('svg', { id: 'mod', width: W, height: H }, root);

  const keyboardhelp = svgEl('g', {
    id: 'keyboardhelp', transform: 'translate(5 0)', display: model.help_display,
  }, root);
  keyboardhelp.innerHTML = model.kbd_help_svg;

  const screen = svgEl('rect', {
    id: 'screen', x: 0, y: 0, width: W, height: H, opacity: 0,
    style: 'cursor:pointer;',
  }, root);

  // ── handlers, straight from _STACK_CONTROL_SCRIPTS_ ────────────────────────

  function myOnClick(event) {
    model.click_y = Math.round(event.offsetY);
    model.click_op_finished = !model.click_op_finished;
  }

  // Grab focus on hover so the widget receives key events (matches the other
  // interactive components).
  function focusSelf() {
    root.focus();
  }

  // 'h' toggles the help overlay (JS-only, shown/hidden via display); 'c' collapses to
  // base+current; ctrl+shift+c rebases the visible dataframe as the new base.  Both ops
  // funnel through key_op_finished, which the Python watcher reads & resets.
  function myOnKeyDown(event) {
    event.stopPropagation();
    var k = event.key;
    if (k === 'h') {
      model.help_display = (model.help_display === 'none') ? 'inline' : 'none';
      event.preventDefault();
    } else if ((k === 'c' || k === 'C') && event.ctrlKey && event.shiftKey) {
      model.key_op_finished = 'rebase';
      event.preventDefault();
    } else if (k === 'c' && !event.ctrlKey && !event.shiftKey && !event.altKey && !event.metaKey) {
      model.key_op_finished = 'collapse';
      event.preventDefault();
    }
  }

  // ── wiring ─────────────────────────────────────────────────────────────────

  root.addEventListener('keydown', myOnKeyDown);
  screen.addEventListener('click',     myOnClick);
  screen.addEventListener('mouseover', focusSelf);

  mod.innerHTML = model.mod_inner;
  model.on('mod_inner', function() { mod.innerHTML = model.mod_inner; });

  // help_display is written from BOTH sides: the 'h' key sets it here, and the Python
  // watcher can too.  The attribute has to follow either way, which is what the
  // template's ${help_display} binding did.
  model.on('help_display', function() {
    keyboardhelp.setAttribute('display', model.help_display);
  });

  return root;
}
