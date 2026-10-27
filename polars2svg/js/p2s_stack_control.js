//
// p2s_stack_control - the browser half of STACKCONTROLI (the stack navigator)
//
// Ported from _STACK_CONTROL_TEMPLATE_ + _STACK_CONTROL_SCRIPTS_ (PLANNING.md W1).
// The smallest contract in the project: no drag, no canvas, no GPU variant.  It is also the only one with *two* things the template supplied that a
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

  // The full text of a name the render cut off (F18).  Drawn here rather than as a
  // <title>, because the screen above covers every element a <title> could sit on.
  const tip     = svgEl('g', { id: 'stacktip', display: 'none', 'pointer-events': 'none' }, root);
  const tipRect = svgEl('rect', { rx: 2, fill: '#fffff0', stroke: '#808080', 'stroke-width': 0.5 }, tip);
  const tipText = svgEl('text', { 'font-family': 'sans-serif', 'font-size': '10px', fill: '#202020' }, tip);

  // The naming dialog for `s` (F18, Q11): a text field over the first free shelf row,
  // which is where the new entry will be drawn, pre-filled with a generated name that the first typed character replaces
  // -- the in-SVG text entry linkpi's `/` search uses.  Enter commits, Esc cancels, and
  // Enter on an empty field does nothing.
  const dlg     = svgEl('g', { id: 'shelfname', display: 'none', 'pointer-events': 'none' }, root);
  const dlgRect = svgEl('rect', { rx: 2, fill: '#ffffff', stroke: '#404040', 'stroke-width': 1 }, dlg);
  const dlgText = svgEl('text', { 'font-family': 'sans-serif', 'font-size': '9px', fill: '#000000' }, dlg);
  const MAX_NAME = 64;                       // stack_shelf._MAX_SHELF_NAME_
  var naming = { open: false, buffer: '', fresh: false };

  function drawDialog() {
    // A fresh pre-fill is shown selected (inverted), so it is clear typing replaces it.
    dlgText.textContent = '';
    var shown = naming.buffer;
    if (naming.fresh) {
      var sel = svgEl('tspan', { fill: '#ffffff', 'font-style': 'italic' }, dlgText);
      sel.textContent = shown;
      dlgRect.setAttribute('fill', '#3060c0');
    } else {
      dlgText.textContent = shown + '\u258b';
      dlgRect.setAttribute('fill', '#ffffff');
    }
  }

  function openDialog() {
    var box = [];
    try { box = JSON.parse(model.shelf_dialog_box || '[]'); } catch (e) { box = []; }
    if (box.length != 4) return false;
    dlgRect.setAttribute('x', box[0]);  dlgRect.setAttribute('y', box[1]);
    dlgRect.setAttribute('width', box[2]); dlgRect.setAttribute('height', box[3]);
    dlgText.setAttribute('x', box[0] + 3);
    dlgText.setAttribute('y', box[1] + box[3] - 4);
    naming.open = true; naming.buffer = model.shelf_default_name; naming.fresh = true;
    drawDialog();
    dlg.setAttribute('display', 'inline');
    return true;
  }

  function closeDialog() {
    naming.open = false; naming.buffer = ''; naming.fresh = false;
    dlg.setAttribute('display', 'none');
  }

  // Every key goes to the dialog while it is open, so a typed 'c' cannot also collapse
  // the stack, and none reaches the page behind.
  function dialogKey(event) {
    var k = event.key;
    event.preventDefault();
    if (k === 'Enter') {
      if (naming.buffer.trim()) {
        model.shelf_name = naming.buffer;
        model.key_op_finished = 'shelve';
        closeDialog();
      }
    } else if (k === 'Escape') {
      closeDialog();
    } else if (k === 'Backspace') {
      naming.buffer = naming.fresh ? '' : naming.buffer.slice(0, -1);
      naming.fresh  = false;
      drawDialog();
    } else if (k.length === 1 && !event.ctrlKey && !event.metaKey) {
      naming.buffer = ((naming.fresh ? '' : naming.buffer) + k).slice(0, MAX_NAME);
      naming.fresh  = false;
      drawDialog();
    }
  }

  // ── handlers ──────────────────────────────────────────────────────────────

  // A click is reported on mouseup after a primary-button mousedown on the screen, with
  // its x (a trashcan is hit by x) and its modifiers (ctrl / shift / ctrl-shift pick the
  // set operation, F18).  mousedown/mouseup rather than 'click' because macOS turns a
  // ctrl-click into a secondary click: the button is still 0, but no 'click' follows.
  var _down_ = false;
  function myOnMouseDown(event) {
    _down_ = (event.button == 0);
  }
  function myOnMouseUp(event) {
    if (!_down_ || event.button != 0) return;
    _down_ = false;
    if (naming.open) { closeDialog(); return; }     // a click elsewhere cancels the name
    model.click_x    = Math.round(event.offsetX);
    model.click_y    = Math.round(event.offsetY);
    model.click_mods = (event.ctrlKey ? 'c' : '') + (event.shiftKey ? 's' : '');
    model.click_op_finished = !model.click_op_finished;
  }

  function myOnMouseMove(event) {
    var regions = [];
    try { regions = JSON.parse(model.hover_regions); } catch (e) { regions = []; }
    var x = event.offsetX, y = event.offsetY;
    for (var i = 0; i < regions.length; i++) {
      var r = regions[i];
      if (x >= r[0] && x < r[2] && y >= r[1] && y < r[3]) {
        tipText.textContent = r[4];
        var tw = tipText.getComputedTextLength ? tipText.getComputedTextLength() : 6 * r[4].length;
        var tx = Math.max(2, Math.min(r[0], W - tw - 6));
        tipText.setAttribute('x', tx + 3);
        tipText.setAttribute('y', r[3] + 11);
        tipRect.setAttribute('x', tx);
        tipRect.setAttribute('y', r[3] + 1);
        tipRect.setAttribute('width', tw + 6);
        tipRect.setAttribute('height', 13);
        tip.setAttribute('display', 'inline');
        return;
      }
    }
    tip.setAttribute('display', 'none');
  }
  function myOnMouseOut() {
    tip.setAttribute('display', 'none');
  }

  // Grab focus on hover so the widget receives key events (matches the other
  // interactive components).
  function focusSelf() {
    root.focus();
  }

  // 'h' toggles the help overlay (JS-only, shown/hidden via display); 'c' collapses to
  // base+current; ctrl+shift+c rebases the visible dataframe as the new base; 's' opens
  // the naming dialog that shelves it.  The ops funnel through key_op_finished, which
  // the Python watcher reads & resets.
  function myOnKeyDown(event) {
    event.stopPropagation();
    if (naming.open) { dialogKey(event); return; }
    var k = event.key;
    if (k === 's' && !event.ctrlKey && !event.altKey && !event.metaKey) {
      // Shelve the displayed frame.  When it cannot be shelved whatever it is called
      // (shelf off or full, already shelved), Python says why instead of a dialog.
      if (model.shelve_blocked) model.key_op_finished = 'shelve_blocked';
      else if (!openDialog()) model.key_op_finished = 'shelve_blocked';
      event.preventDefault();
    } else if (k === 'h') {
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
  screen.addEventListener('mousedown', myOnMouseDown);
  screen.addEventListener('mouseup',   myOnMouseUp);
  screen.addEventListener('mousemove', myOnMouseMove);
  screen.addEventListener('mouseout',  myOnMouseOut);
  screen.addEventListener('mouseover', focusSelf);

  // On macOS ctrl+click is a secondary click, so the browser raises a context menu over
  // the ctrl / ctrl-shift set-operation clicks.  Swallow it, as the other views do.
  root.addEventListener('contextmenu', function(event) {
    event.preventDefault();
  });

  mod.innerHTML = model.mod_inner;
  model.on('mod_inner', function() { mod.innerHTML = model.mod_inner; });

  // help_display is written from BOTH sides: the 'h' key sets it here, and the Python
  // watcher can too.  The attribute has to follow either way, which is what the
  // template's ${help_display} binding did.
  model.on('help_display', function() {
    keyboardhelp.setAttribute('display', model.help_display);
  });

  return p2sConsumeWheel(root);
}
