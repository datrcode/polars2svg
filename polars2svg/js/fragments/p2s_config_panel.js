//
// p2s_config_panel - the picker menu and the configuration panel, shared by every view
//
// 20260921_config_panel_design.md designed this for LINKPI and it was built there, inside
// p2s_linkpi.js.  F1 (the tooltip) is a panel row on EVERY component, so the panel had to
// become shared infrastructure; this file is that extraction, moved rather than rewritten,
// and the LINKPI browser tests are what say the move was faithful.
//
// The picker menu comes with it, because the two are coupled in both directions: a panel
// row's Enter opens the row's picker (CP3), and menuClose() hands control back to the
// panel.  Splitting them would have meant an interface between two halves of one state
// machine.  The five generic components had a smaller picker of their own (one kind,
// `select_shape`, no display strings, no guarded items, no menu_x); it is a strict subset
// of this one, so they move onto this and lose nothing.
//
// This is deliberately the OPPOSITE call to the one p2s_interactivep.js records for
// updateBrushCursor / myUpdateDragRect / menuOpen, which measured 33-57% similar between
// the two modules and were left duplicated.  Those had already diverged; this had not
// diverged at all, because the generic half did not exist yet.  Parameterising around
// divergence costs more than it saves -- writing the second copy identical on purpose and
// then keeping two of them in step costs more still.
//
// This is a FRAGMENT: no import, no export.  See polars2svg/p2s_esm.py.
//
// ctx:
//   model, state      the entry module's data model and its plain per-view state object
//   menuNode          <g id="pickermenu">
//   panelNode         <g id="configpanel">, or null for a view with no panel
//   headers           {kind: header text} for the picker
//   params            {kind: model param name holding that kind's current value}
//   setValue          (kind, label) -> void; the single write path into the data model
//

function p2sConfigPanel(ctx) {
  const model = ctx.model, state = ctx.state;
  const menuNode = ctx.menuNode, panelNode = ctx.panelNode;
  const STYLE_ = 'font-family: \'Courier New\', monospace; font-size: 11px; fill: #222;';

  // ── the picker menu ────────────────────────────────────────────────────────

  function menuOpen(event) {
      var _items_   = state.menu_items[state.menu_kind];
      var _current_ = model[ctx.params[state.menu_kind]];
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
      var _header_ = ctx.headers[state.menu_kind];
      // state.menu_x is 8 for every keyboard entry point and is only moved when the
      // config panel opens a picker, so that the two overlays sit side by side.
      var _ox_     = state.menu_x;
      var _maxlen_ = _header_.length;
      for (var _i_ = 0; _i_ < _items_.length; _i_++) {
          _maxlen_ = Math.max(_maxlen_, (_items_[_i_][2] || _items_[_i_][1]).length + 4);
      }
      var _w_menu_ = _maxlen_ * 7 + 20,
          _h_menu_ = (_items_.length + 1) * 14 + 12;
      var _html_ = '<rect x="' + _ox_ + '" y="8" width="' + _w_menu_ + '" height="' + _h_menu_ + '"'
                 + ' fill="rgba(240,240,240,0.95)" stroke="#888" stroke-width="1" rx="3"/>'
                 + '<rect x="' + (_ox_ + 2) + '" y="' + (8 + 1 + (state.menu_index + 1) * 14) + '" width="' + (_w_menu_ - 4) + '" height="13"'
                 + ' fill="rgba(100,150,255,0.3)"/>'
                 + '<text x="' + (_ox_ + 10) + '" y="' + (8 + 12) + '" style="' + STYLE_ + ' font-weight: bold;">' + _header_ + '</text>';
      for (var _i_ = 0; _i_ < _items_.length; _i_++) {
          _html_ += '<text x="' + (_ox_ + 10) + '" y="' + (8 + 12 + (_i_ + 1) * 14) + '" style="' + STYLE_ + '">'
                  + '[' + _items_[_i_][0] + '] ' + (_items_[_i_][2] || _items_[_i_][1]) + '</text>';
      }
      menuNode.innerHTML = _html_;
  }

  function menuCommit(event) {
      ctx.setValue(state.menu_kind, state.menu_items[state.menu_kind][state.menu_index][1]);
      menuClose();
  }

  function menuClose(event) {
      if (state.menu_timer != null) { clearTimeout(state.menu_timer); }
      state.menu_timer   = null;
      state.menu_open    = false;
      state.menu_kind    = '';
      state.menu_x       = 8;
      menuNode.innerHTML = '';
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

  // The picker's modal key block.  Returns true when it consumed the event, which is
  // also the caller's signal to stop -- the entry modules' myOnKeyDown `return`s on it
  // exactly as it did when this block was written out inline.
  function menuKeyDown(event) {
      if (!state.menu_open) { return false; }
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
      // Each of the three "press the opening key again to step down" clauses pairs with
      // a picker that still has a bare-key entry point of its own: shift-W, shift-G and
      // -- on the generic components -- shift-F.  The clauses for the pickers the panel
      // absorbed went with their bindings.
      else if (event.key === 'ArrowDown' || event.key === 'j' ||
               (event.key === 'W' && state.menu_kind === 'operation'    && !event.ctrlKey) ||
               (event.key === 'G' && state.menu_kind === 'mode'         && !event.ctrlKey) ||
               (event.key === 'F' && state.menu_kind === 'select_shape' && !event.ctrlKey)) {
          state.menu_index = (state.menu_index + 1) % _items_.length;
          menuRender(); menuArmTimer();
      }
      // ctrl-shift-W and ctrl-shift-G used to reverse-cycle here and are gone
      // (PLANNING.md U10).  ctrl-shift-W is a reserved chrome-level accelerator
      // off macOS -- it CLOSES THE BROWSER WINDOW, and preventDefault() cannot
      // reclaim what the page is never shown; ctrl-shift-G worked but went with
      // it so the pure-reverse chords are gone as a class rather than leaving one
      // survivor.  Nothing was lost: ArrowUp / k reverse every menu, which is what
      // the generic _interactivep components have always done.
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
      return true;
  }

  // ── the configuration panel ────────────────────────────────────────────────
  //
  // 20260921_config_panel_design.md.  It is info_str made editable and given room
  // (CP2): a state display you leave open while you work, not a menu you open, use and
  // close.  One row per persistent visual-encoding setting, each showing its CURRENT
  // value only (design section 5 -- shift-space retires the inline-all preview, and the
  // real value sets would size the panel to a ~500px rectangle over the plot).
  //
  // A row IS a menu kind.  model.config_panel_rows carries [mnemonic, kind, label,
  // enabled] and state.menu_items[kind] carries the values, their per-value mnemonics
  // and their display strings -- so `space` cycles the same list `Enter` shows in full
  // (CP3), and there is one source of truth per row rather than two that drift.
  //
  // Modal while open (CP5): the key block returns true before the binding chain, exactly
  // as the picker's does.  preventDefault() there is also what reclaims `space` from the
  // browser's scroll-the-page default.

  function panelRows() { return model.config_panel_rows || []; }

  // The value a row displays: what has been cycled but not yet committed, else what
  // Python currently holds.  Everything undecided lives in panel_pending, so a commit
  // that Python REFUSES -- the flowmap confirm gate resets link_shape_choice -- lands
  // back here as an ordinary param change and the row shows the real value again.
  function panelValue(kind) {
      var _p_ = state.panel_pending[kind];
      if (_p_ !== undefined) { return _p_; }
      var _v_ = model[ctx.params[kind]];
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
      if (panelNode === null) { return; }
      if (!state.panel_open) { panelNode.innerHTML = ''; return; }
      var _rows_   = panelRows();
      var _header_ = 'appearance:';
      var _lw_     = 0;
      for (var _i_ = 0; _i_ < _rows_.length; _i_++) { _lw_ = Math.max(_lw_, _rows_[_i_][2].length); }
      // Fixed-width label + dot leader, so the value column aligns and the panel reads
      // as a table rather than a set of ragged lines.
      var _texts_ = [], _maxlen_ = _header_.length;
      for (var _i_ = 0; _i_ < _rows_.length; _i_++) {
          var _lead_ = _rows_[_i_][2] + ' ';
          while (_lead_.length < _lw_ + 5) { _lead_ += '.'; }
          var _txt_ = '[' + _rows_[_i_][0] + '] ' + _lead_ + ' ' + panelValue(_rows_[_i_][1]);
          _texts_.push(_txt_);
          _maxlen_ = Math.max(_maxlen_, _txt_.length + 4);
      }
      var _w_ = _maxlen_ * 7 + 20,
          _h_ = (_rows_.length + 1) * 14 + 12;
      var _html_ = '<rect x="8" y="8" width="' + _w_ + '" height="' + _h_ + '"'
                 + ' fill="rgba(240,240,240,0.95)" stroke="#888" stroke-width="1" rx="3"/>'
                 + '<rect x="10" y="' + (8 + 1 + (state.panel_row + 1) * 14) + '" width="' + (_w_ - 4) + '" height="13"'
                 + ' fill="rgba(100,150,255,0.3)"/>'
                 + '<text x="18" y="' + (8 + 12) + '" style="' + STYLE_ + ' font-weight: bold;">' + _header_ + '</text>';
      for (var _i_ = 0; _i_ < _texts_.length; _i_++) {
          // A disabled row is drawn greyed AND skipped by the cursor.  Either alone is
          // the failure the design names: space silently does nothing and the panel
          // looks broken.
          var _grey_ = _rows_[_i_][3] ? '' : ' fill: #999;';
          _html_ += '<text x="18" y="' + (8 + 12 + (_i_ + 1) * 14) + '" style="' + STYLE_ + _grey_ + '">'
                  + _texts_[_i_] + '</text>';
      }
      panelNode.innerHTML = _html_;
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
      // A value outside the row's list (a plot built with something the menu does not
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
  // space and watching the plot is the interaction that makes the panel worth having.
  //
  // Deliberately NOT menuArmTimer's 2.5s walk-away auto-commit (CP6): that one COMMITS
  // AND CLOSES a transient picker, and a panel that closed itself while you looked at
  // the plot would be a bug.  This one only flushes; the panel stays up.
  function panelArmCommit() {
      if (state.panel_timer != null) { clearTimeout(state.panel_timer); }
      state.panel_timer = setTimeout(panelCommitPending, 300);
  }

  function panelCommitPending(event) {
      if (state.panel_timer != null) { clearTimeout(state.panel_timer); }
      state.panel_timer = null;
      var _pending_ = state.panel_pending;
      state.panel_pending = {};
      for (var _k_ in _pending_) { ctx.setValue(_k_, _pending_[_k_]); }
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

  // CP5 -- modal, the same shape as the picker block above.  Non-modal (only the panel's
  // own keys captured, everything else falling through) is more useful, but it re-opens
  // the keyspace conflict the panel exists to close.  Returns true when consumed.
  function panelKeyDown(event) {
      if (!state.panel_open) { return false; }
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
      return true;
  }

  return {
      menuOpen: menuOpen, menuRender: menuRender, menuCommit: menuCommit,
      menuClose: menuClose, menuArmTimer: menuArmTimer, menuKeyDown: menuKeyDown,
      panelOpen: panelOpen, panelClose: panelClose, panelRender: panelRender,
      panelKeyDown: panelKeyDown, panelCommitPending: panelCommitPending,
  };
}
