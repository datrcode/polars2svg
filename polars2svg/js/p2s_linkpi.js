//
// p2s_linkpi - the browser half of LINKPI / LINKPI_GPU
//
// The largest contract in the project and the last to move off ReactiveHTML
// (PLANNING.md W1): 26 scripts, ~37 KB, a 12.5 KB myOnKeyDown with ~50 bindings, and
// four layout-preview shapes on top of everything the other components do.
//
// The handler bodies were converted mechanically rather than retyped -- `data.x` ->
// `model.x`, `self.helper()` -> `helper()`, and the `var _self_ = self` that
// menuArmTimer needed to reach a sibling script -- because at this size a transcription
// slip is the likeliest way to break something no test covers.  The element names are
// unchanged for the same reason: every id'd node in the old template becomes a const of
// exactly that name, so the bodies did not have to be touched to find their elements.
// `svgparent` is therefore the root's const name here, where the smaller modules call it
// `root`.
//
// Two things `render` did that this does not need to:
//
//   * It guarded `state.cur_mouse_x === undefined` before seeding, because ReactiveHTML
//     re-ran it on every subtree rebuild and clobbering the cursor back to 0 made the
//     next key op collapse to the upper-left corner.  render() runs once per mount now,
//     so that is plain initialisation.
//   * `myOnMouseWheel` is not carried over.  It was an unreachable _scripts entry -- no
//     onwheel attribute, no matching param, no self. call -- and the wheel has always
//     been the three non-passive listeners attached below.
//
// This is an ENTRY module: it exports render().
//

export function render({ model, el }) {
  const W = model.svg_w;
  const H = model.svg_h;

  const state = {
    menu_items: model.menu_items,
    menu_open: false, menu_kind: '', menu_index: 0, menu_timer: null, menu_x: 8,
    // The configuration panel (20260921_config_panel_design.md).  panel_pending holds
    // the values cycled but not yet committed -- CP7: cycling renders every intermediate
    // state, and link shape passes THROUGH flowmap on its way to off, so the write is
    // debounced rather than issued on every space.
    panel_open: false, panel_row: 0, panel_pending: {}, panel_timer: null, panel_w: 0,
    x0_drag: -10, y0_drag: -10, x1_drag: -5, y1_drag: -5,
    cur_mouse_x: 0, cur_mouse_y: 0,
    drag_op: false, move_op: false, unselected_move_op: false,
    layout_op: false,          // true if next mouse button 1 press begins a layout
    layout_line_flag: false,   // true if the shape will be overridden by the line version
    layout_op_shape: '',       // trigger field for python to perform the layout operation
    search_mode: false, search_buffer: '',
    last_brush_x: -999, last_brush_y: -999,
    brush_defs: [null, ['circle', 5], ['circle', 15]],
    brush_names: ['', 'circ r=5', 'circ r=15'],
    pending_mods: null,
  };

  // ── DOM, in the order the template listed it ───────────────────────────────

  const svgparent = svgEl('svg', {
    id: 'svgparent', width: W, height: H, tabindex: '0', style: 'user-select:none;',
  });

  const mod = svgEl('svg', { id: 'mod', width: W, height: H }, svgparent);

  const keyboardhelp = svgEl('g', {
    id: 'keyboardhelp', transform: 'translate(' + model.keyboardhelp_x + ' 0)',
  }, svgparent);

  const drag = svgEl('rect', {
    id: 'drag', x: -10, y: -10, width: 5, height: 5,
    stroke: '#000000', 'stroke-width': 2, fill: 'none',
  }, svgparent);

  const layoutline = svgEl('line', {
    id: 'layoutline', x1: -10, y1: -10, x2: -10, y2: -10,
    stroke: '#000000', 'stroke-width': 2,
  }, svgparent);

  const layoutrect = svgEl('rect', {
    id: 'layoutrect', x: -10, y: -10, width: 10, height: 10,
    stroke: '#000000', 'stroke-width': 2,
  }, svgparent);

  const layoutcircle = svgEl('circle', {
    id: 'layoutcircle', cx: -10, cy: -10, r: 5, fill: 'none',
    stroke: '#000000', 'stroke-width': 6,
  }, svgparent);

  const layoutsunflower = svgEl('circle', {
    id: 'layoutsunflower', cx: -10, cy: -10, r: 5,
    stroke: '#000000', 'stroke-width': 2,
  }, svgparent);

  const screen = svgEl('rect', {
    id: 'screen', x: 0, y: 0, width: W, height: H, opacity: 0.05,
  }, svgparent);

  const infostr = svgEl('text', {
    id: 'infostr', x: 5, y: H - 2, fill: '#000000',
    'font-size': '10px', 'pointer-events': 'none',
  }, svgparent);

  const allentitieslayer = svgEl('path', {
    id: 'allentitieslayer', d: '', fill: '#000000', 'fill-opacity': 0.01, stroke: 'none',
  }, svgparent);

  const selectionlayer = svgEl('path', {
    id: 'selectionlayer', d: '', fill: '#ff0000', transform: '', stroke: 'none',
  }, svgparent);

  const selectedlabels = svgEl('g', { id: 'selectedlabels', 'pointer-events': 'none' }, svgparent);

  const searchtext = svgEl('text', {
    id: 'searchtext', x: Math.floor(W / 2), y: H - 2, 'text-anchor': 'middle',
    fill: '#0000cc', 'font-size': '11px', 'font-family': 'monospace',
    'pointer-events': 'none',
  }, svgparent);

  const brushindicator = svgEl('g', { id: 'brushindicator', 'pointer-events': 'none' }, svgparent);
  const brushmodelabel = svgEl('g', { id: 'brushmodelabel', 'pointer-events': 'none' }, svgparent);
  // Before #pickermenu so that a picker opened FROM the panel draws over it rather
  // than under it; menuRender also shifts that picker clear of the panel (state.menu_x).
  const configpanel    = svgEl('g', { id: 'configpanel', 'pointer-events': 'none' }, svgparent);
  const pickermenu     = svgEl('g', { id: 'pickermenu', 'pointer-events': 'none' }, svgparent);

  keyboardhelp.innerHTML = model.kbd_help_svg;

  // ── helpers and handlers, from _LINKPI_SCRIPTS_ ────────────────────────────

  function myOnMouseOver(event) {
      model.has_focus = true;
      svgparent.focus();
  }

  function updateBrushCursor(event) {
      var _bs_ = model.brush_state;
      if (_bs_ == 0) {
          brushindicator.innerHTML = '';
          brushmodelabel.innerHTML = '';
          return;
      }
      var _x_ = state.cur_mouse_x, _y_ = state.cur_mouse_y;
      var _d_ = state.brush_defs[_bs_], _r_ = _d_[1];
      var _s_ = 'stroke="rgba(100,150,255,0.8)" fill="none" pointer-events="none"';
      brushindicator.innerHTML = '<circle cx="'+_x_+'" cy="'+_y_+'" r="'+_r_+'" '+_s_+' stroke-width="1.5"/>';
      var _nm_ = state.brush_names[_bs_];
      var _tw_ = _nm_.length * 6 + 10, _rx_ = model.svg_w - _tw_ - 3;
      brushmodelabel.innerHTML = '<rect x="'+_rx_+'" y="3" width="'+_tw_+'" height="15" rx="3" fill="rgba(100,150,255,0.3)" stroke="rgba(100,150,255,0.7)" stroke-width="0.5" pointer-events="none"/>'
          + '<text x="'+(_rx_+5)+'" y="14" font-size="10px" fill="rgba(40,60,200,1.0)" font-family="monospace" pointer-events="none">'+_nm_+'</text>';
  }

  function myOnMouseOut(event) {
      // Ignore moves between this component's own hit layers.  #screen,
      // #allentitieslayer and #selectionlayer are siblings that cover each
      // other, so putting the pointer on a node fires mouseout on the layer
      // being left -- which this handler read as "the mouse left the
      // component".  The brush was therefore cleared at the very moment the
      // pointer reached something worth brushing, and an open picker menu
      // committed itself for the same reason (PLANNING.md U7).
      //
      // relatedTarget is where the pointer went; if that is still inside the
      // component, nothing has been left.  It is null when the pointer leaves
      // the window entirely, which is a real leave.
      if (event.relatedTarget && svgparent.contains(event.relatedTarget)) { return; }
      model.has_focus = false;
      brushindicator.innerHTML = '';
      if (model.brush_state > 0) { model.brush_leave_done = true; }
      if (state.menu_open) { menuCommit(); }
  }

  // Menu kind -> the param that holds its current value, and the header it draws.
  //
  // A table rather than the ternary chains these replace, because the config panel
  // (CP3) both re-enters these pickers and reads the same values to render its rows,
  // and four new kinds arrived with it -- three chains of twelve branches each is how
  // a kind ends up handled in two places and missed in the third.  The WRITE side
  // stays an explicit chain in menuSetValue: only it knows that committing a
  // background also bumps background_op_seq.
  const MENU_PARAM_ = {
      background:       'background_operation',
      operation:        'layout_operation',
      mode:             'layout_mode',
      link_size:        'link_size_choice',
      link_opacity:     'link_opacity_choice',
      link_shape:       'link_shape_choice',
      timing_spacing:   'timing_spacing_choice',
      node_size:        'node_size_choice',
      link_arrows:      'link_arrows_choice',
      timing_marks:     'timing_marks_choice',
      label_mode:       'label_mode_choice',
      background_state: 'background_state_choice',
  };

  const MENU_HEADER_ = {
      background:       'background:',
      operation:        'layout operation:',
      mode:             'layout mode:',
      link_size:        'link size:',
      link_opacity:     'link opacity:',
      link_shape:       'link shape:',
      timing_spacing:   'timing mark spacing (px):',
      node_size:        'node size:',
      link_arrows:      'link arrows:',
      timing_marks:     'timing marks:',
      label_mode:       'labels:',
      // NOT 'background:' -- that header belongs to the producer picker (shift-b), and
      // the browser tests address a picker by the header text it draws.
      background_state: 'background display:',
  };

  function menuOpen(event) {
      var _items_   = state.menu_items[state.menu_kind];
      var _current_ = model[MENU_PARAM_[state.menu_kind]];
      state.menu_index = 0;
      for (var _i_ = 0; _i_ < _items_.length; _i_++) {
          if (_items_[_i_][1] == _current_) { state.menu_index = _i_; break; }
      }
      state.menu_open = true;
      menuRender();
      menuArmTimer();
  }

  function menuRender(event) {
      if (!state.menu_open) { return; }
      var _items_  = state.menu_items[state.menu_kind];
      var _header_ = MENU_HEADER_[state.menu_kind];
      // state.menu_x is 8 for every keyboard entry point and is only moved when the
      // config panel opens a picker, so that the two overlays sit side by side.
      var _ox_     = state.menu_x;
      var _maxlen_ = _header_.length;
      for (var _i_ = 0; _i_ < _items_.length; _i_++) {
          _maxlen_ = Math.max(_maxlen_, (_items_[_i_][2] || _items_[_i_][1]).length + 4);
      }
      var _w_menu_ = _maxlen_ * 7 + 20,
          _h_menu_ = (_items_.length + 1) * 14 + 12,
          _style_  = 'font-family: \'Courier New\', monospace; font-size: 11px; fill: #222;';
      var _html_ = '<rect x="' + _ox_ + '" y="8" width="' + _w_menu_ + '" height="' + _h_menu_ + '"'
                 + ' fill="rgba(240,240,240,0.95)" stroke="#888" stroke-width="1" rx="3"/>'
                 + '<rect x="' + (_ox_ + 2) + '" y="' + (8 + 1 + (state.menu_index + 1) * 14) + '" width="' + (_w_menu_ - 4) + '" height="13"'
                 + ' fill="rgba(100,150,255,0.3)"/>'
                 + '<text x="' + (_ox_ + 10) + '" y="' + (8 + 12) + '" style="' + _style_ + ' font-weight: bold;">' + _header_ + '</text>';
      for (var _i_ = 0; _i_ < _items_.length; _i_++) {
          _html_ += '<text x="' + (_ox_ + 10) + '" y="' + (8 + 12 + (_i_ + 1) * 14) + '" style="' + _style_ + '">'
                  + '[' + _items_[_i_][0] + '] ' + (_items_[_i_][2] || _items_[_i_][1]) + '</text>';
      }
      pickermenu.innerHTML = _html_;
  }

  function menuCommit(event) {
      menuSetValue(state.menu_kind, state.menu_items[state.menu_kind][state.menu_index][1]);
      menuClose();
  }

  // The single write path for every choice a picker or a panel row can make.  Kept an
  // explicit chain rather than MENU_PARAM_[kind]: the background producer commits a
  // SEQUENCE number as well as a label -- re-picking the same producer has to re-run it,
  // and a param write of an unchanged value fires no watcher -- and that asymmetry is
  // easier to see spelled out than as a special case beside a table lookup.
  function menuSetValue(kind, label) {
      if      (kind == 'background')       { model.background_operation  = label;
                                             model.background_op_seq     = model.background_op_seq + 1; }
      else if (kind == 'operation')        { model.layout_operation      = label; }
      else if (kind == 'mode')             { model.layout_mode           = label; }
      else if (kind == 'link_size')        { model.link_size_choice      = label; }
      else if (kind == 'link_opacity')     { model.link_opacity_choice   = label; }
      else if (kind == 'link_shape')       { model.link_shape_choice     = label; }
      else if (kind == 'timing_spacing')   { model.timing_spacing_choice = label; }
      else if (kind == 'node_size')        { model.node_size_choice      = label; }
      else if (kind == 'link_arrows')      { model.link_arrows_choice    = label; }
      else if (kind == 'timing_marks')     { model.timing_marks_choice   = label; }
      else if (kind == 'label_mode')       { model.label_mode_choice     = label; }
      else if (kind == 'background_state') { model.background_state_choice = label; }
  }

  function menuClose(event) {
      if (state.menu_timer != null) { clearTimeout(state.menu_timer); }
      state.menu_timer     = null;
      state.menu_open      = false;
      state.menu_kind      = '';
      state.menu_x         = 8;
      pickermenu.innerHTML = '';
      // A picker opened from the panel hands control back to it, showing whatever was
      // just committed.  panelRender is a no-op when the panel is closed.
      panelRender();
  }

  function menuArmTimer(event) {
      if (state.menu_timer != null) { clearTimeout(state.menu_timer); }
      state.menu_timer = setTimeout(function() {
          if (!state.menu_open) { return; }
          var _sel_ = state.menu_items[state.menu_kind][state.menu_index];
          // Walking away from the menu must not start an expensive operation.  For a
          // guarded item the timeout closes without committing; everything else commits
          // as before.
          if (_sel_ && _sel_[3]) { menuClose(); } else { menuCommit(); }
      }, 2500);
  }


  // ── the configuration panel ────────────────────────────────────────────────
  //
  // 20260921_config_panel_design.md.  It is info_str made editable and given room
  // (CP2): a state display you leave open while you work, not a menu you open, use and
  // close.  Nine rows, one per persistent visual-encoding setting, each showing its
  // CURRENT value only (design section 5 -- shift-space retires the inline-all preview,
  // and the real value sets would size the panel to a ~500px rectangle over the graph).
  //
  // A row IS a menu kind.  model.config_panel_rows carries [mnemonic, kind, label,
  // enabled] and state.menu_items[kind] carries the values, their per-value mnemonics
  // and their display strings -- so `space` cycles the same list `Enter` shows in full
  // (CP3), and there is one source of truth per row rather than two that drift.
  //
  // Modal while open (CP5): this block returns before the binding chain, exactly as the
  // picker's does.  preventDefault() here is also what reclaims `space` from the
  // browser's scroll-the-page default.

  function panelRows() { return model.config_panel_rows || []; }

  // The value a row displays: what has been cycled but not yet committed, else what
  // Python currently holds.  Everything undecided lives in panel_pending, so a commit
  // that Python REFUSES -- the flowmap confirm gate resets link_shape_choice -- lands
  // back here as an ordinary param change and the row shows the real value again.
  function panelValue(kind) {
      var _p_ = state.panel_pending[kind];
      if (_p_ !== undefined) { return _p_; }
      var _v_ = model[MENU_PARAM_[kind]];
      return (_v_ === undefined || _v_ === null) ? '' : _v_;
  }

  function panelIndexOf(kind, value) {
      var _items_ = state.menu_items[kind] || [];
      for (var _i_ = 0; _i_ < _items_.length; _i_++) {
          if (_items_[_i_][1] == value) { return _i_; }
      }
      return -1;
  }

  function panelRender(event) {
      if (!state.panel_open) { configpanel.innerHTML = ''; return; }
      var _rows_   = panelRows();
      var _header_ = 'appearance:';
      var _lw_     = 0;
      for (var _i_ = 0; _i_ < _rows_.length; _i_++) { _lw_ = Math.max(_lw_, _rows_[_i_][2].length); }
      // Fixed-width label + dot leader, so the value column aligns and the panel reads
      // as a table rather than nine ragged lines.
      var _texts_ = [], _maxlen_ = _header_.length;
      for (var _i_ = 0; _i_ < _rows_.length; _i_++) {
          var _lead_ = _rows_[_i_][2] + ' ';
          while (_lead_.length < _lw_ + 5) { _lead_ += '.'; }
          var _txt_ = '[' + _rows_[_i_][0] + '] ' + _lead_ + ' ' + panelValue(_rows_[_i_][1]);
          _texts_.push(_txt_);
          _maxlen_ = Math.max(_maxlen_, _txt_.length + 4);
      }
      var _w_ = _maxlen_ * 7 + 20,
          _h_ = (_rows_.length + 1) * 14 + 12,
          _style_ = 'font-family: \'Courier New\', monospace; font-size: 11px; fill: #222;';
      var _html_ = '<rect x="8" y="8" width="' + _w_ + '" height="' + _h_ + '"'
                 + ' fill="rgba(240,240,240,0.95)" stroke="#888" stroke-width="1" rx="3"/>'
                 + '<rect x="10" y="' + (8 + 1 + (state.panel_row + 1) * 14) + '" width="' + (_w_ - 4) + '" height="13"'
                 + ' fill="rgba(100,150,255,0.3)"/>'
                 + '<text x="18" y="' + (8 + 12) + '" style="' + _style_ + ' font-weight: bold;">' + _header_ + '</text>';
      for (var _i_ = 0; _i_ < _texts_.length; _i_++) {
          // A disabled row is drawn greyed AND skipped by the cursor.  Either alone is
          // the failure the design names: space silently does nothing and the panel
          // looks broken.
          var _grey_ = _rows_[_i_][3] ? '' : ' fill: #999;';
          _html_ += '<text x="18" y="' + (8 + 12 + (_i_ + 1) * 14) + '" style="' + _style_ + _grey_ + '">'
                  + _texts_[_i_] + '</text>';
      }
      configpanel.innerHTML = _html_;
      state.panel_w = _w_;
  }

  function panelStep(delta) {
      var _rows_ = panelRows();
      if (_rows_.length === 0) { return; }
      var _i_ = state.panel_row;
      for (var _n_ = 0; _n_ < _rows_.length; _n_++) {
          _i_ = (_i_ + delta + _rows_.length) % _rows_.length;
          if (_rows_[_i_][3]) { state.panel_row = _i_; break; }
      }
      panelRender();
  }

  function panelCycle(delta) {
      var _rows_ = panelRows(), _row_ = _rows_[state.panel_row];
      if (!_row_ || !_row_[3]) { return; }
      var _items_ = state.menu_items[_row_[1]] || [];
      if (_items_.length === 0) { return; }
      var _i_ = panelIndexOf(_row_[1], panelValue(_row_[1]));
      // A value outside the row's list (a linkp built with something the menu does not
      // carry) starts the walk rather than being stepped from a position it does not have.
      _i_ = (_i_ < 0) ? 0 : (_i_ + delta + _items_.length) % _items_.length;
      state.panel_pending[_row_[1]] = _items_[_i_][1];
      panelRender();
      panelArmCommit();
  }

  // CP7.  The pickers navigate without rendering and commit once on Enter; the panel
  // renders live, so cycling link shape line -> curve -> flowmap -> off would render
  // flowmap ON THE WAY PAST -- a force layout whose cost grows faster than linearly
  // (linkp.py's own warning).  Debounced rather than Enter-to-commit because tapping
  // space and watching the graph is the interaction that makes the panel worth having.
  //
  // Deliberately NOT menuArmTimer's 2.5s walk-away auto-commit (CP6): that one COMMITS
  // AND CLOSES a transient picker, and a panel that closed itself while you looked at
  // the graph would be a bug.  This one only flushes; the panel stays up.
  function panelArmCommit() {
      if (state.panel_timer != null) { clearTimeout(state.panel_timer); }
      state.panel_timer = setTimeout(panelCommitPending, 300);
  }

  function panelCommitPending(event) {
      if (state.panel_timer != null) { clearTimeout(state.panel_timer); }
      state.panel_timer = null;
      var _pending_ = state.panel_pending;
      state.panel_pending = {};
      for (var _k_ in _pending_) { menuSetValue(_k_, _pending_[_k_]); }
      panelRender();
  }

  function panelOpen(from_end) {
      var _rows_ = panelRows();
      state.panel_open = true;
      state.panel_row  = (from_end && _rows_.length > 0) ? _rows_.length - 1 : 0;
      if (_rows_.length > 0 && !_rows_[state.panel_row][3]) { panelStep(from_end ? -1 : 1); }
      else                                                  { panelRender(); }
  }

  // esc flushes rather than discards: the debounce is 300ms and closing inside it is a
  // normal thing to do, so dropping the choice would read as the panel ignoring a
  // keystroke.  This is not the walk-away commit CP6 rejects -- it takes an explicit esc.
  function panelClose(event) {
      panelCommitPending();
      state.panel_open = false;
      panelRender();
  }

  function myOnKeyDown(event) {
      event.stopPropagation();
      if (state.search_mode) {
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
      if (state.menu_open) {
          event.preventDefault();
          var _items_ = state.menu_items[state.menu_kind];
          if      (event.key === 'Escape') { menuClose();  }
          else if (event.key === 'Enter')  { menuCommit(); }
          // !ctrlKey on W and G below is deliberate even though nothing tests ctrlKey
          // for them any more: it keeps the chords deleted in U10 *inert* rather than
          // silently cycling forward, which is the opposite of what they used to do.
          // Only shift-W and shift-G still have an entry point of their own.  The
          // 'press the opening key again to step down' clauses for link size / opacity
          // / shape / node size / timing spacing went with the bindings that opened
          // those pickers (the config panel absorbed them); ArrowDown / j reach every
          // menu, and those keys now fall through to the mnemonic scan and do nothing.
          else if (event.key === 'ArrowDown' || event.key === 'j' ||
                   (event.key === 'W' && state.menu_kind === 'operation'    && !event.ctrlKey) ||
                   (event.key === 'G' && state.menu_kind === 'mode'         && !event.ctrlKey)) {
              state.menu_index = (state.menu_index + 1) % _items_.length;
              menuRender(); menuArmTimer();
          }
          // ctrl-shift-W and ctrl-shift-G used to reverse-cycle here and are gone
          // (PLANNING.md U10).  ctrl-shift-W is a reserved chrome-level accelerator
          // off macOS -- it CLOSES THE BROWSER WINDOW, and preventDefault() cannot
          // reclaim what the page is never shown; ctrl-shift-G worked but went with
          // it so the pure-reverse chords are gone as a class rather than leaving one
          // survivor.  Nothing was lost: ArrowUp / k reverse every menu, which is what
          // the generic _interactivep components have always done.  The clauses that
          // remain below pair with a *ctrl entry point* (ctrl-l opens the link-size
          // picker and steps back through it), and the audit measured all four as
          // page-interceptable.
          // The four ctrl reverse-cycles that used to live here (ctrl-l / ctrl-o /
          // ctrl-a / ctrl-p) each paired with a ctrl ENTRY point, and both halves are
          // gone: in the panel, shift-space reverses (CP4), which is the shift-for-a-
          // variant idiom the rest of LINKPI uses.  ArrowUp / k still reverse any menu.
          else if (event.key === 'ArrowUp' || event.key === 'k') {
              state.menu_index = (state.menu_index - 1 + _items_.length) % _items_.length;
              menuRender(); menuArmTimer();
          }
          else if (event.key.length === 1) {
              for (var _i_ = 0; _i_ < _items_.length; _i_++) {
                  if (_items_[_i_][0] === event.key) {
                      state.menu_index = _i_;
                      // A guarded item is only SELECTED by its mnemonic; committing it
                      // takes a deliberate Enter.  'l' then '3' used to start the force
                      // layout in two keystrokes with nothing in between.
                      if (_items_[_i_][3]) { menuRender(); menuArmTimer(); }
                      else                 { menuCommit(); }
                      break;
                  }
              }
          }
          return;
      }
      // CP5 -- modal for v1, the same shape as the picker block above.  Non-modal (only
      // the panel's own keys captured, everything else falling through) is more useful,
      // but it re-opens the keyspace conflict the panel exists to close.
      if (state.panel_open) {
          event.preventDefault();          // 'space' would otherwise scroll the page
          var _prows_ = panelRows(), _prow_ = _prows_[state.panel_row];
          if      (event.key === 'Escape') { panelClose(); }
          else if (event.key === ' ')      { panelCycle(event.shiftKey ? -1 : 1); }
          else if (event.key === 'Enter')  {
              // Flush first: the picker opens on the CURRENT value, and a value cycled
              // within the last 300ms is not committed yet.
              panelCommitPending();
              if (_prow_ && _prow_[3]) {
                  state.menu_kind = _prow_[1];
                  state.menu_x    = state.panel_w + 16;
                  menuOpen();
              }
          }
          else if (event.key === 'a' || event.key === 'ArrowDown' || event.key === 'j') { panelStep(1);  }
          else if (event.key === 'A' || event.key === 'ArrowUp'   || event.key === 'k') { panelStep(-1); }
          else if (event.key.length === 1) {
              for (var _i_ = 0; _i_ < _prows_.length; _i_++) {
                  if (_prows_[_i_][0] === event.key && _prows_[_i_][3]) {
                      state.panel_row = _i_;
                      panelRender();
                      break;
                  }
              }
          }
          return;
      }
      model.ctrlkey  = event.ctrlKey;
      model.shiftkey = event.shiftKey;
      model.x_mouse  = state.cur_mouse_x;
      model.y_mouse  = state.cur_mouse_y;

      // 'a' is the panel key, and it is one the panel itself frees -- the old 'a'
      // (arrows x timing marks) and shift-a (spacing picker) are two of its rows now,
      // so taking it costs nothing.  ctrl-a is deliberately NOT bound: with the ctrl
      // entry points gone there is nothing to guard, so select-all goes back to the
      // browser.  'b' and 'l' are now unbound and free for whatever needs a left-hand
      // key next (the whole point of CP1).
      if      (event.key == "a" && !event.ctrlKey) { panelOpen(false); } // Open the appearance panel; 'a' again advances the row cursor
      else if (event.key == "A" && !event.ctrlKey) { panelOpen(true);  } // ...opening on the last row instead of the first
      else if (event.key == "B") { state.menu_kind = 'background'; menuOpen(); } // Open the background producer picker (committing runs it)
      else if (event.key == "c") { if (event.ctrlKey) event.preventDefault(); model.key_op_finished = 'c';  } // (if selected) zoom to selected, else zoom to entire view; ctrl-c copies (suppress native copy so it can't clobber our clipboard write)
      else if (event.key == "C") { if (event.ctrlKey) event.preventDefault(); model.key_op_finished = 'C';  } // Zoom to selected + neighbors; ctrl-shift-c copies labels
      else if (event.key == "d") { model.key_op_finished = 'd';  } // Detect communities (louvain) & color nodes by community
      else if (event.key == "D") { model.key_op_finished = 'D';  } // Clear the community colors
      else if (event.key == "e") { if (event.ctrlKey) event.preventDefault(); model.key_op_finished = 'e';  } // Expand (undirected); ctrl-e expands along reversed directed edges (preventDefault: ctrl-e is browser search-bar focus)
      else if (event.key == "E") { model.key_op_finished = 'E';  } // Expand (w/ digraph, forward)
      else if (event.key == "f") { model.key_op_finished = 'f';  } // Edge unfilter: re-add base rows on the currently-visible edges
      else if (event.key == "F") { model.key_op_finished = 'F';  } // Node expansion: re-add base rows incident to the currently-visible nodes
      else if (event.key == "Escape") { model.cancel_seq = model.cancel_seq + 1; } // Ask the running layout to stop and keep its best-so-far result
      else if (event.key == "g") { state.layout_op        = true; // Mouse press is layout shape
                                   state.layout_line_flag = false; }
      else if (event.key == "G") { state.menu_kind = 'mode';      menuOpen(); } // Open the layout-mode picker menu
      else if (event.key == "h") {
          if (model.keyboardhelp_x == -1000) { model.keyboardhelp_x =     5; }
          else                              { model.keyboardhelp_x = -1000; }
      }
      // l / shift-l / ctrl-l (link shape, link size), shift-o / ctrl-o (link opacity)
      // and shift-p / ctrl-p (node size) are panel rows now -- 'a', then the row's
      // mnemonic, then space; Enter still opens the very same picker.
      else if (event.key == "n" ||                                // Select nodes with the same shape as the one under the mouse
               event.key == "N") { model.key_op_finished = 'n';  }
      else if (event.key == "q") { model.key_op_finished = 'q';  } // Invert selection
      else if (event.key == "Q") { model.key_op_finished = 'Q';  } // Select common neighbors to selected nodes
      else if (event.key == "r") {                                // Toggle the radius brush on/off
          if (model.brush_state == 0) { model.brush_state = 1; }
          else                       { model.brush_state = 0; }
          model.brushing_mode = model.brush_state > 0;
          model.brush_changed += 1;
          updateBrushCursor();
      }
      else if (event.key == "R") {                                // Cycle the brush radius (r=5 | r=15)
          var _seq_ = [1, 2];
          if (model.brush_state == 0) { model.brush_state = _seq_[0]; }
          else {
              var _i_ = _seq_.indexOf(model.brush_state);
              model.brush_state = _seq_[(_i_ + 1) % _seq_.length];
          }
          model.brushing_mode = true;
          model.brush_changed += 1;
          updateBrushCursor();
      }
      else if (event.key == "s") { if (event.ctrlKey) event.preventDefault(); model.key_op_finished = 's';  } // Set sticky labels; ctrl-s cycles label mode (ctrl-s is browser Save Page As)
      else if (event.key == "S") { if (event.ctrlKey) event.preventDefault(); model.key_op_finished = 'S';  } // Subtract selected from sticky labels (label mode is the panel's 'labels' row now)
      else if (event.key == "t") { model.key_op_finished = 't';  } // Collapse selected to a single point
      else if (event.key == "T") { model.key_op_finished = 'T';  } // Horizontally collapse selected
      else if (event.key == "u") { model.key_op_finished = 'u';  } // Undo last layout
      // Vertical collapse.  It lives on an unmodified key because its old chord,
      // ctrl-t, is reserved as new-tab off macOS and preventDefault() cannot
      // reclaim a browser-chrome shortcut -- see PLANNING.md U2.
      else if (event.key == "v") { model.key_op_finished = 'v';  } // Vertically collapse selected
      else if (event.key == "w") { model.key_op_finished = 'w';  } // Apply layout operation
      else if (event.key == "W") { state.menu_kind = 'operation'; menuOpen(); } // Open the layout-operation picker menu
      else if (event.key == "x") { model.key_op_finished = 'x';  } // push the stack (remove the selected from the current graph)
      else if (event.key == "X" && event.ctrlKey) { model.key_op_finished = 'ctrl_shift_x'; } // collapse edges to one row each (push the stack)
      else if (event.key == "X") { model.key_op_finished = 'X';  } // pop the stack (add removed nodes back in)
      else if (event.key == "y") { state.layout_op        = true; // Mouse press is layout line
                                   state.layout_line_flag = true;  }
      else if (event.key == "Y") { state.layout_op        = true; // Mouse press is layout line
                                   state.layout_line_flag = true;  }
      else if (event.key == "z" ||                                // Select nodes with the same color as the one under the mouse
               event.key == "Z") { model.key_op_finished = 'z';     }
      else if (event.key == "1" || event.key == "!") { model.key_op_finished = '1';  }
      else if (event.key == "2" || event.key == "@") { model.key_op_finished = '2';  }
      else if (event.key == "3" || event.key == "#") { model.key_op_finished = '3';  }
      else if (event.key == "4" || event.key == "$") { model.key_op_finished = '4';  }
      else if (event.key == "5" || event.key == "%") { model.key_op_finished = '5';  }
      else if (event.key == "6" || event.key == "^") { model.key_op_finished = '6';  }
      else if (event.key == "7" || event.key == "&") { model.key_op_finished = '7';  }
      else if (event.key == "8" || event.key == "*") { model.key_op_finished = '8';  }
      else if (event.key == "9" || event.key == "(") { model.key_op_finished = '9';  }
      else if (event.key == "0" || event.key == ")") { model.key_op_finished = '0';  }
      else if (event.key == "/") {
          state.search_mode   = true;
          state.search_buffer = '';
          searchtext.textContent = '/ ▋';
      }

      model.last_key = event.key;
  }

  function myOnKeyUp(event) {
      event.stopPropagation();
      if (state.menu_open || state.panel_open) { return; }
      // Hold the modifiers while a key operation is still in flight.  applyKeyOp
      // runs asynchronously behind a lock and reads ctrlkey / shiftkey when it gets
      // there, so clearing them the instant the user let go made a *tapped*
      // ctrl-<key> arrive with the modifier already gone -- the handler took the
      // unmodified branch, and ctrl-c zoomed the view instead of copying (U3).
      //
      // The release is not discarded, it is deferred: the key_op_finished script
      // applies it as soon as Python reports the operation done.  Simply skipping
      // the clear would leave the modifier stuck on until the next keydown, and the
      // drag band -- which reads these to colour itself -- would name the wrong
      // set-operation.
      if (model.key_op_finished === '') {
          model.ctrlkey  = event.ctrlKey;
          model.shiftkey = event.shiftKey;
      } else {
          state.pending_mods = [event.ctrlKey, event.shiftKey];
      }
      if (event.key == "g" || event.key == "y" || event.key == "Y") { state.layout_op = state.layout_line_flag = false; }
  }

  function myOnMouseMove(event) {
      state.cur_mouse_x = event.offsetX;
      state.cur_mouse_y = event.offsetY;
      state.x1_drag     = event.offsetX;
      state.y1_drag     = event.offsetY;
      if (state.drag_op)               { myUpdateDragRect(); }
      if (state.move_op)               { var _tr_ = "translate(" + (state.x1_drag - state.x0_drag) + "," + (state.y1_drag - state.y0_drag) + ")";
                                         selectionlayer.setAttribute("transform", _tr_);
                                         selectedlabels.setAttribute("transform", _tr_); }
      if (state.unselected_move_op)    { selectionlayer.setAttribute("transform", "translate(" + (state.x1_drag - state.x0_drag) + "," + (state.y1_drag - state.y0_drag) + ")"); }
      if (state.layout_op_shape != "") { myUpdateLayoutOp(); }
      if (model.brush_state > 0) {
          updateBrushCursor();
          var _dx_ = event.offsetX - state.last_brush_x;
          var _dy_ = event.offsetY - state.last_brush_y;
          if (_dx_*_dx_ + _dy_*_dy_ >= 9) {   // throttle: re-brush only after ~3px of travel
              state.last_brush_x = event.offsetX;
              state.last_brush_y = event.offsetY;
              model.x_mouse       = event.offsetX;
              model.y_mouse       = event.offsetY;
              model.brush_changed += 1;
          }
      }
  }

  function downAllEntities(event) {
      model.ctrlkey  = event.ctrlKey;
      model.shiftkey = event.shiftKey;
      if (event.button == 0) {
              model.allentities_x0      = event.offsetX;
              model.allentities_y0      = event.offsetY;
              state.x0_drag            = event.offsetX;
              state.y0_drag            = event.offsetY;
              state.x1_drag            = event.offsetX;
              state.y1_drag            = event.offsetY;
              state.unselected_move_op = true;
              var ex = event.offsetX, ey = event.offsetY;
              selectionlayer.setAttribute("d", "M " + (ex-5) + " " + (ey-5) + " l 10 0 l 0 10 l -10 0 z");
              selectionlayer.setAttribute("transform", "");
      }
  }

  function downSelect(event) {
      if (event.button == 0) {
          state.x0_drag  = event.offsetX;
          state.y0_drag  = event.offsetY;
          state.x1_drag  = event.offsetX;
          state.y1_drag  = event.offsetY;
          if (state.layout_op) {
              if (state.layout_line_flag) {
                  if      (model.ctrlkey)  { state.layout_op_shape = "v-line"; }
                  else if (model.shiftkey) { state.layout_op_shape = "h-line"; }
                  else                    { state.layout_op_shape = "line";   }
              }
              else                        { state.layout_op_shape = model.layout_mode; }
              myUpdateLayoutOp();
          } else               { state.drag_op         = true;             myUpdateDragRect(); }
      } else if (event.button == 1) {
          model.x0_middle = model.x1_middle = event.offsetX;
          model.y0_middle = model.y1_middle = event.offsetY;
      }
  }

  function downMove(event) {
      if (event.button == 0) {
          state.x0_drag  = state.x1_drag  = event.offsetX;
          state.y0_drag  = state.y1_drag  = event.offsetY;
          state.move_op  = true;
      } else if (event.button == 1) {
          model.x0_middle = model.x1_middle = event.offsetX;
          model.y0_middle = model.y1_middle = event.offsetY;
      }
  }

  function myUpdateLayoutOp(event) {
      var dx = state.x1_drag - state.x0_drag,
          dy = state.y1_drag - state.y0_drag;
      var reset_circle = true, reset_sunflower = true, reset_rect = true, reset_line = true;
      if        (state.layout_op_shape == "circle" ||
                 state.layout_op_shape == "circle (color)") { reset_circle = false;
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
  }

  function myOnMouseUp(event) {
      if (event.button == 0) {
          model.ctrlkey          = event.ctrlKey;
          model.shiftkey         = event.shiftKey;
          state.x1_drag         = event.offsetX;
          state.y1_drag         = event.offsetY;
          if (state.drag_op) {
              state.shiftkey        = event.shiftKey;
              state.drag_op         = false;
              myUpdateDragRect();
              model.drag_x0          = state.x0_drag;
              model.drag_y0          = state.y0_drag;
              model.drag_x1          = state.x1_drag;
              model.drag_y1          = state.y1_drag;
              model.drag_op_finished = true;
          } else if (state.move_op) {
              state.move_op         = false;
              model.drag_x0          = state.x0_drag;
              model.drag_y0          = state.y0_drag;
              model.drag_x1          = state.x1_drag;
              model.drag_y1          = state.y1_drag;
              model.move_op_finished = true;
          } else if (state.layout_op_shape != "") {
              model.drag_x0          = state.x0_drag;
              model.drag_y0          = state.y0_drag;
              model.drag_x1          = state.x1_drag;
              model.drag_y1          = state.y1_drag;
              model.layout_shape     = state.layout_op_shape;
              state.layout_op_shape = "";
              myUpdateLayoutOp();
          } else if (state.unselected_move_op) {
              model.ctrlkey  = event.ctrlKey;
              model.shiftkey = event.shiftKey;
              model.drag_x0  = state.x0_drag;
              model.drag_y0  = state.y0_drag;
              model.drag_x1  = state.x1_drag;
              model.drag_y1  = state.y1_drag;
              model.unselected_move_op_finished = true;
              state.unselected_move_op = false;
          }
      } else if (event.button == 1) {
          model.x1_middle          = event.offsetX;
          model.y1_middle          = event.offsetY;
          model.middle_op_finished = true;
      }
  }

  function renderSelectedLabels(event) {
      while (selectedlabels.firstChild) { selectedlabels.removeChild(selectedlabels.firstChild); }
      // New geometry is absolute, so it cancels any translate a node-move left behind.
      selectedlabels.setAttribute("transform", "");
      var _d_ = model.selection_labels;
      if (!_d_ || !_d_.labels || _d_.labels.length === 0) { return; }
      var _NS_ = "http://www.w3.org/2000/svg", _h_ = _d_.h, _ls_ = _d_.labels;
      // Backdrops first, in one pass, so a neighbouring label's backdrop can never
      // land on top of this one's text (the java original's clearStr, which drew each
      // string onto its own cleared box).
      for (var _i_ = 0; _i_ < _ls_.length; _i_++) {
          var _e_ = _ls_[_i_];
          var _r_ = document.createElementNS(_NS_, "rect");
          _r_.setAttribute("x",      _e_.x - _e_.w / 2 - 2);
          _r_.setAttribute("y",      _e_.y - _h_);
          _r_.setAttribute("width",  _e_.w + 4);
          _r_.setAttribute("height", _e_.lines.length * _h_ + 3);
          _r_.setAttribute("fill",   _d_.bg);
          _r_.setAttribute("fill-opacity", "0.75");
          selectedlabels.appendChild(_r_);
      }
      for (var _i_ = 0; _i_ < _ls_.length; _i_++) {
          var _e_ = _ls_[_i_];
          var _t_ = document.createElementNS(_NS_, "text");
          _t_.setAttribute("x", _e_.x); _t_.setAttribute("y", _e_.y);
          _t_.setAttribute("text-anchor", "middle");
          _t_.setAttribute("font-size", _h_ + "px");
          _t_.setAttribute("fill", _d_.fg);
          for (var _j_ = 0; _j_ < _e_.lines.length; _j_++) {
              var _s_ = document.createElementNS(_NS_, "tspan");
              _s_.setAttribute("x", _e_.x);
              _s_.setAttribute("dy", _j_ === 0 ? 0 : _h_);
              _s_.textContent = _e_.lines[_j_];
              _t_.appendChild(_s_);
          }
          selectedlabels.appendChild(_t_);
      }
  }

  function myUpdateDragRect(event) {
      if (state.drag_op) {
          // `var`, which the ReactiveHTML original did without: a _scripts body ran as a
          // sloppy-mode function, so these four became implicit globals and worked.  An
          // ES module is strict mode automatically, where the same assignment throws
          // ReferenceError -- which is exactly what it did, taking out all five rubber-band
          // tests.  Declaring them is also what the code always meant.
          var x = Math.min(state.x0_drag, state.x1_drag);
          var y = Math.min(state.y0_drag, state.y1_drag);
          var w = Math.abs(state.x1_drag - state.x0_drag);
          var h = Math.abs(state.y1_drag - state.y0_drag);
          drag.setAttribute('x',x);     drag.setAttribute('y',y);
          drag.setAttribute('width',w); drag.setAttribute('height',h);
          // shiftkey, not shftkey: the misspelling read undefined, so both shift
          // branches were dead and the band drew the wrong colour for two of the
          // four set-operations -- black for subtract and green for intersect,
          // i.e. it named the operation the user was *not* about to perform
          // (PLANNING.md U8).  The operations themselves were always correct;
          // myOnMouseUp reads event.shiftKey off the event, which is why nothing
          // else noticed.
          if      (model.shiftkey && model.ctrlkey)  drag.setAttribute('stroke','#0000ff');
          else if (model.shiftkey)                  drag.setAttribute('stroke','#ff0000');
          else if (                model.ctrlkey)  drag.setAttribute('stroke','#00ff00');
          else                                    drag.setAttribute('stroke','#000000');
      } else {
          drag.setAttribute('x',-10);   drag.setAttribute('y',-10);
          drag.setAttribute('width',5); drag.setAttribute('height',5);
      }
  }

  // ── param-change handlers ──────────────────────────────────────────────────

  model.on('key_op_finished', function() {
      // Python empties this when it has finished the operation; that is the moment
      // a modifier release deferred by myOnKeyUp can safely be applied.  If a
      // re-render has wiped state in between the release is simply lost, and the
      // next keydown sets the modifiers correctly anyway.
      if (model.key_op_finished === '' && state.pending_mods) {
          model.ctrlkey  = state.pending_mods[0];
          model.shiftkey = state.pending_mods[1];
          state.pending_mods = null;
      }
  });

  model.on('mod_inner', function() {
      mod.innerHTML       = model.mod_inner;
      infostr.innerHTML   = model.info_str;
  });

  model.on('allentitiespath', function() {
      allentitieslayer.setAttribute("d", model.allentitiespath);
  });

  model.on('selectionpath', function() {
      selectionlayer.setAttribute("d", model.selectionpath);
      selectionlayer.setAttribute("transform", "");
  });

  model.on('selection_labels', function() {
      renderSelectedLabels();
  });

  model.on('info_str', function() {
      infostr.innerHTML = model.info_str;
  });

  // The configuration panel is a live display, so every value it shows re-renders it --
  // including one Python changed by itself (a stack pop, a refused flowmap commit).
  // config_panel_rows carries the row order AND the per-row enabled flag, which Python
  // recomputes from the current layer.
  const _PANEL_PARAMS_ = [
      'config_panel_rows',
      'link_arrows_choice', 'timing_marks_choice', 'label_mode_choice',
      'background_state_choice', 'timing_spacing_choice', 'link_shape_choice',
      'link_size_choice', 'link_opacity_choice', 'node_size_choice',
  ];
  for (const _pp_ of _PANEL_PARAMS_) {
      model.on(_pp_, function() { panelRender(); });
  }

  // menu_items was fixed at construction until the panel arrived; the 'labels' row's
  // value list is the one that genuinely varies at runtime (linkLabelsAvailable() gains
  // or loses the two link-label states as the stack is navigated), so the snapshot in
  // state has to be refreshed rather than taken once.
  model.on('menu_items', function() {
      state.menu_items = model.menu_items;
      panelRender();
  });
  // ── wiring, matching the template's on* attributes ─────────────────────────
  //
  // Three hit layers stack over the plot and share every handler except mousedown:
  // #screen starts a rubber band, #allentitieslayer moves an unselected node, and
  // #selectionlayer moves the selection.

  svgparent.addEventListener('keydown', myOnKeyDown);
  svgparent.addEventListener('keyup',   myOnKeyUp);

  for (const layer of [screen, allentitieslayer, selectionlayer]) {
    layer.addEventListener('mouseover', myOnMouseOver);
    layer.addEventListener('mouseout',  myOnMouseOut);
    layer.addEventListener('mousemove', myOnMouseMove);
    layer.addEventListener('mouseup',   myOnMouseUp);
  }
  screen.addEventListener('mousedown',           downSelect);
  allentitieslayer.addEventListener('mousedown', downAllEntities);
  selectionlayer.addEventListener('mousedown',   downMove);

  // Non-passive so it can preventDefault.  One function on all three layers, exactly as
  // `render` attached it.
  const _wheelFn_ = function(event) {
    event.preventDefault();
    model.wheel_x = event.offsetX; model.wheel_y = event.offsetY;
    model.wheel_rots = Math.round(10*event.deltaY);
    model.wheel_op_finished = true;
  };
  screen.addEventListener('wheel',           _wheelFn_, { passive: false });
  allentitieslayer.addEventListener('wheel', _wheelFn_, { passive: false });
  selectionlayer.addEventListener('wheel',   _wheelFn_, { passive: false });

  // On macOS ctrl+click is a secondary click, so the browser raises a contextmenu
  // (popup) during ctrl / shift-ctrl rectangular drags, which interrupts the drag and
  // loses the selection (U1).
  svgparent.addEventListener('contextmenu', function(event) {
    event.preventDefault();
  });

  // ── first paint ────────────────────────────────────────────────────────────

  mod.innerHTML     = model.mod_inner;
  infostr.innerHTML = model.info_str;
  allentitieslayer.setAttribute('d', model.allentitiespath);
  selectionlayer.setAttribute('d', model.selectionpath);
  renderSelectedLabels();

  // ${keyboardhelp_x} was an attribute binding; the initial value is set above.
  model.on('keyboardhelp_x', function() {
    keyboardhelp.setAttribute('transform', 'translate(' + model.keyboardhelp_x + ' 0)');
  });

  // `render` also reset these on every ReactiveHTML subtree rebuild; once per mount now.
  model.has_focus          = false;
  model.shiftkey           = false;
  model.ctrlkey            = false;
  model.middle_op_finished = false;
  model.move_op_finished   = false;
  model.brush_state        = 0;
  model.brushing_mode      = false;
  model.brush_changed      = 0;

  return (typeof p2sGpuWrap === 'function') ? p2sGpuWrap(model, svgparent) : svgparent;
}
