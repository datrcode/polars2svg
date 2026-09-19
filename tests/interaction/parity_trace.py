'''Recording what a component *does*, so a port can be shown to preserve it.

The JSComponent migration (PLANNING.md **W1**) replaces the whole browser half of each
view: the template becomes imperative DOM construction, the `_scripts` dict becomes one ES
module, and `data.x` becomes `model.x`.  The Python half does not move.  Cutting each
contract over in place -- rather than running both implementations behind a switch -- means
there is no second implementation left to compare against at the moment it matters, so the
comparison has to be recorded **before** the port and committed.

Two recordings, because neither covers the other:

**The param trace.**  Every JS->Python hop in this codebase is a param write plus a
`param.watch` -- 48 distinct params, no `send_event`, no `send_msg`.  So the sequence of
param writes a gesture provokes *is* the component's observable behaviour as far as Python
is concerned, and a matching sequence is a strong statement that the port preserved it.

**The DOM digest.**  A param trace is structurally blind to everything that never crosses
into Python: the picker menu, the brush cursor and its mode label, the search buffer, the
rubber-band rectangle, the selected-node labels.  Those are pure JS-only state -- exactly
the half the port rewrites, and exactly the half `settle()` exists to protect.  The digest
is deliberately *structural* (ids, tags, child counts, short text) rather than raw
`outerHTML`: an ESM view legitimately emits different whitespace and attribute order, and a
golden that fails on those would be abandoned within a day.

Nothing here asserts.  `test_param_trace_parity.py` records and compares.
'''
import json
import os
import re
import time
from contextlib import contextmanager

import param

#: Payload params: **Python computes these and pushes them down**; JavaScript only reads
#: them.  Recorded as "written" rather than by value, for two reasons.
#:
#: The first is that their value is not evidence about this migration.  The port rewrites
#: the JS->Python direction and the DOM construction; it does not touch the render code
#: that produces an SVG or a hit-test path.  What the trace needs from them is *that* the
#: gesture caused one to be pushed -- the bytes are the golden-image suite's job.
#:
#: The second is measured.  Recording a digest made `linkpi_wheel_multiplicity` fail
#: against its own re-recording: one wheel notch resolves to a zoom factor via a rounded
#: delta, and a coordinate that lands on 123.45 in one run and 123.46 in the next changes
#: the path string and therefore the hash -- float jitter in Python arithmetic, reported as
#: a parity failure in JavaScript.
_PAYLOAD_PARAMS_ = frozenset({'mod_inner', 'selectionpath', 'allentitiespath', 'gpu_payload',
                              'selection_labels', 'menu_items', 'kbd_help_svg'})

#: Anything longer than this is treated the same way, whatever it is called -- a value that
#: large is a payload by definition and does not belong in a golden literally.
_PAYLOAD_OVER_ = 200

#: Pointer-position params, recorded as presence only: no value, no count.
#:
#: They are written from `myOnMouseMove`, so both are a function of how many `mousemove`
#: events the browser chose to deliver rather than of anything the component decides.
#: Measured across re-recordings of these same ten tests, and they are the *only* two that
#: moved: the drag gestures recorded `('x_mouse', 252, 1)` one run and
#: `('x_mouse', 252, 2)` the next, and a hover recorded 55 then 25 because
#: `InteractivePage.hover` nudges the pointer in a retry loop until focus lands.
#:
#: Presence still carries signal -- a port that stopped tracking the pointer drops the
#: entry entirely -- and everything else keeps its exact value and write count.
_POINTER_PARAMS_ = frozenset({'x_mouse', 'y_mouse'})

#: Components mint per-render element ids, so the same render differs between runs.
#: Both separators occur and both have to be covered -- the underscore form (`xyp_519746971`,
#: `smallp_1129274803`) and the hyphen form (`plotClip-319619177`, `.rect-group-1075334182`,
#: which also appears inside a <style> block and so reaches textContent).  Recorded goldens
#: were unstable against their own next run until the hyphen form was included; the digit
#: count varies too, so a normalisation that misses one cannot be patched up by comparing
#: lengths instead.
_MINTED_ID_RE_ = re.compile(r'([A-Za-z][A-Za-z0-9_]*?)([-_])\d{5,}')

#: Set to rewrite the goldens instead of comparing against them.
RECORD_ENV = 'P2S_PARITY_RECORD'


def recording() -> bool:
    return bool(os.environ.get(RECORD_ENV))


def _stable(text: str) -> str:
    '''Replace per-render minted ids with a placeholder, keeping the separator so that
    `plotClip-N` and `xyp_N` stay distinguishable.'''
    return _MINTED_ID_RE_.sub(r'\1\2N', text)


def _norm_value(name: str, value):
    '''One param value, as it goes into a golden.'''
    if name in _PAYLOAD_PARAMS_ or (isinstance(value, (str, dict, list))
                                    and len(value) > _PAYLOAD_OVER_):
        return f'{type(value).__name__}:written'
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, str):
        return _stable(value)
    if isinstance(value, (int, bool)) or value is None:
        return value
    return repr(value)[:120]


def p2s_param_names(view) -> list:
    '''The params polars2svg itself declares on this view.

    Panel's own base params (`loading`, `margin`, `stylesheets`, ...) are excluded: they
    change for reasons that have nothing to do with the component's behaviour, and they
    would make every golden a record of Panel's internals rather than of the view's.
    '''
    _names_ = set()
    for _klass_ in type(view).__mro__:
        if not getattr(_klass_, '__module__', '').startswith('polars2svg'):
            continue
        for _n_, _obj_ in vars(_klass_).items():
            if isinstance(_obj_, param.Parameter):
                _names_.add(_n_)
    return sorted(_names_)


#: How much element text a golden keeps.  Enough to see an info line, a brush mode label
#: or a menu header; not so much that a container's whole plot text lands in the file.
_TEXT_KEEP_ = 120

#: How much is fetched from the browser before normalisation.  It must exceed _TEXT_KEEP_
#: by more than normalisation can shrink, because minted ids differ in *digit count*
#: between runs -- slicing first and normalising second moved the cut by a character and
#: made every container text differ from its own re-recording.
_TEXT_FETCH_ = 600

#: Collect every id'd element under the root, with just enough shape to notice that a menu
#: opened, a label appeared or a cursor was cleared.
_DOM_DIGEST_JS_ = '''
(rootEl) => {
  const out = {};
  const walk = (el) => {
    if (el.id) {
      out[el.id] = {
        tag:  el.tagName.toLowerCase(),
        kids: el.children.length,
        text: (el.textContent || '').trim().replace(/\\s+/g, ' ').slice(0, __FETCH__),
      };
    }
    for (const c of el.children) walk(c);
  };
  walk(rootEl);
  return out;
}
'''.replace('__FETCH__', str(_TEXT_FETCH_))


class ParityTrace:
    '''Records param writes and a DOM digest per named gesture.

    Usage::

        trace = ParityTrace(ip)
        with trace.gesture('hover-centre'):
            ip.hover(200, 150)
        trace.compare_or_record('linkpi')
    '''

    def __init__(self, ip, view=None):
        self.ip      = ip
        self.view    = view if view is not None else ip.app.view(0)
        self.names   = p2s_param_names(self.view)
        self.results = []

    # ── capture ──────────────────────────────────────────────────────────────

    @contextmanager
    def gesture(self, name: str):
        '''Record one gesture's **net effect** on the view's params, plus the DOM after.

        Net effect, not the write sequence, and that is a deliberate limit.  A drag
        delivers however many `mousemove` events the browser felt like delivering, so
        `x_mouse`/`y_mouse` receive a different number of intermediate writes on every
        run -- measured, not feared: recording each write made four of these nine tests
        fail against their own re-recording.  The value a gesture *leaves behind* is
        deterministic; how many times it passed through on the way there is not.

        What this therefore cannot see is a handler firing twice where it should fire
        once, when the second firing restores the first's value.  That is a real risk
        for the port (three wheel listeners are registered per LINKPI render today), so
        it is checked directly and per-param by `write_count()` rather than smuggled
        into the general oracle -- see `test_one_wheel_event_finishes_one_operation`.
        '''
        _final_, _counts_ = {}, {}

        def _record(*events):
            for _e_ in events:
                _final_[_e_.name]  = _norm_value(_e_.name, _e_.new)
                _counts_[_e_.name] = _counts_.get(_e_.name, 0) + 1

        _watcher_ = self.view.param.watch(_record, self.names)
        try:
            yield
            # A write is only observable once the round trip has completed; every
            # harness gesture is asynchronous on the Python side.
            self.ip.wait_until_idle()
            self._wait_for_write_quiescence(_counts_)
        finally:
            self.view.param.unwatch(_watcher_)

        self.counts = _counts_
        self.results.append({
            'gesture': name,
            # Sorted by name: a dict of net values has no meaningful order, and sorting
            # keeps diffs small and readable.  Ordering *between* gestures is preserved
            # by the list this is appended to.
            #
            # Each entry is [name, net value, write count].  The count is not decoration:
            # Python's watchers *reset* the trigger params after handling them, so
            # `drag_op_finished` nets to False whether the drag was reported or not.  A
            # net-only golden passed a mutant whose mouseup stopped telling Python the
            # drag had finished -- the count is what catches it.
            'writes':  [[_k_, 'tracked', None] if _k_ in _POINTER_PARAMS_
                        else [_k_, _final_[_k_], _counts_[_k_]]
                        for _k_ in sorted(_final_)],
            'dom':     self.dom_digest(),
        })

    #: How long the param writes must stop arriving before a gesture counts as finished,
    #: and the ceiling on waiting for that.
    WRITE_QUIET_S  = 0.4
    WRITE_BUDGET_S = 6.0

    def _wait_for_write_quiescence(self, counts: dict) -> None:
        '''Wait until the gesture has stopped provoking param writes.

        `wait_until_idle()` is not sufficient on its own, and this is the fix for a flake
        that took three sightings to pin down.  It waits for the controller lock to be
        free -- but a keystroke whose Python operation has not *yet* acquired the lock
        leaves it trivially free, so the gesture returns, the watcher is removed, and the
        writes land where nothing is listening.  They then go missing from the recording,
        or (for a non-final gesture) turn up attributed to the next one.

        Observed on `test_linkpi_search_parity` in roughly one whole-suite run in three,
        never in isolation: the commit keystroke starts a search whose writes arrive a
        few milliseconds after the lock check. The first attempt at a fix -- confirming
        each typed character reached the on-screen echo -- addressed a different,
        inferred cause and did not stop it.

        Watching the write stream itself is the general form: whatever the gesture sets
        off, this waits for it to finish rather than for a proxy.
        '''
        _seen_ = -1
        _deadline_ = time.monotonic() + self.WRITE_BUDGET_S
        while time.monotonic() < _deadline_:
            _now_ = sum(counts.values())
            if _now_ == _seen_:
                return
            _seen_ = _now_
            time.sleep(self.WRITE_QUIET_S)

    def quiesce(self) -> None:
        '''Let setup done *outside* a gesture finish crossing into Python.

        A test that hovers before its first gesture -- to take focus, or to put the
        pointer on bare canvas -- provokes param writes of its own (`myOnMouseOver` sets
        `has_focus`). Those writes are asynchronous, so whether they land before the first
        `gesture()` starts watching or after it is pure timing, and if they land after
        they are recorded against a gesture that did not cause them.

        Seen as `linkpi_search`'s `search-open` intermittently gaining or losing
        `('has_focus', True, 1)` -- a gesture whose only action is a keystroke, and
        `myOnKeyDown` does not touch `has_focus`. Call this after any setup that touches
        the component and before the first gesture.
        '''
        self.ip.wait_until_idle()
        _counts_ = {}
        _watcher_ = self.view.param.watch(
            lambda *evs: [_counts_.__setitem__(e.name, _counts_.get(e.name, 0) + 1) for e in evs],
            self.names)
        try:
            self._wait_for_write_quiescence(_counts_)
        finally:
            self.view.param.unwatch(_watcher_)

    def write_count(self, name: str) -> int:
        '''How many times the most recent gesture wrote `name`.  For the targeted
        multiplicity checks the net-effect trace deliberately cannot make.'''
        return getattr(self, 'counts', {}).get(name, 0)

    def dom_digest(self) -> dict:
        _raw_ = self.ip.root.evaluate(_DOM_DIGEST_JS_)
        _sfx_ = self.ip.suffix
        _out_ = {}
        for _id_, _rec_ in _raw_.items():
            # ReactiveHTML suffixes every template id per model (`mod-p1015`); an ESM view
            # emits it bare.  Strip it so a golden is about the component, not the model id.
            _key_ = _id_[:-len(_sfx_)] if _sfx_ and _id_.endswith(_sfx_) else _id_
            _rec_['text'] = _stable(_rec_['text'])[:_TEXT_KEEP_]
            _out_[_stable(_key_)] = _rec_
        return _out_

    # ── compare / record ─────────────────────────────────────────────────────

    def payload(self) -> dict:
        return {'params_watched': self.names, 'gestures': self.results}

    def compare_or_record(self, name: str) -> None:
        from pathlib import Path
        _path_ = Path(__file__).parent / 'parity' / f'{name}.json'
        _now_  = self.payload()
        if recording():
            _path_.parent.mkdir(parents=True, exist_ok=True)
            _path_.write_text(json.dumps(_now_, indent=2, sort_keys=False) + '\n',
                              encoding='utf-8')
            return
        if not _path_.is_file():
            raise AssertionError(
                f'no parity golden at {_path_}. Record one against the CURRENT '
                f'implementation before porting this component:\n'
                f'  {RECORD_ENV}=1 ./.venv/bin/python -m pytest '
                f'tests/interaction/test_param_trace_parity.py --interaction -q')
        _want_ = json.loads(_path_.read_text(encoding='utf-8'))
        _diff_ = describe_diff(_want_, _now_)
        if _diff_:
            raise AssertionError(
                f'{name}: behaviour differs from the recorded golden ({_path_.name}).\n'
                + '\n'.join(_diff_)
                + '\n\nIf the change is intended, say so in the CHANGELOG and re-record. '
                  'Do NOT re-record to make a port go green -- the golden is the only '
                  'statement that the port preserved behaviour.')


def describe_diff(want: dict, got: dict) -> list:
    '''Human-readable differences, most useful first.  Empty when they agree.'''
    _out_ = []
    _wg_ = {_g_['gesture']: _g_ for _g_ in want.get('gestures', [])}
    _gg_ = {_g_['gesture']: _g_ for _g_ in got.get('gestures', [])}

    _missing_ = [_k_ for _k_ in _wg_ if _k_ not in _gg_]
    _extra_   = [_k_ for _k_ in _gg_ if _k_ not in _wg_]
    if _missing_:
        _out_.append(f'  gestures missing from this run: {_missing_}')
    if _extra_:
        _out_.append(f'  gestures not in the golden: {_extra_}')

    for _name_ in [_g_['gesture'] for _g_ in want.get('gestures', []) if _g_['gesture'] in _gg_]:
        _w_, _g_ = _wg_[_name_], _gg_[_name_]
        _ww_ = [tuple(_x_) for _x_ in _w_['writes']]
        _gw_ = [tuple(_x_) for _x_ in _g_['writes']]
        if _ww_ != _gw_:
            _only_w_ = [_x_ for _x_ in _ww_ if _x_ not in _gw_]
            _only_g_ = [_x_ for _x_ in _gw_ if _x_ not in _ww_]
            _out_.append(f'  [{_name_}] param writes differ')
            if _only_w_:
                _out_.append(f'      expected but absent: {_only_w_}')
            if _only_g_:
                _out_.append(f'      present but unexpected: {_only_g_}')
        if _w_.get('write_counts') != _g_.get('write_counts'):
            _out_.append(f'  [{_name_}] write counts differ: '
                         f'golden={_w_.get("write_counts")} run={_g_.get("write_counts")} '
                         f'-- a handler fired a different number of times')
        for _id_ in sorted(set(_w_['dom']) | set(_g_['dom'])):
            _a_, _b_ = _w_['dom'].get(_id_), _g_['dom'].get(_id_)
            if _a_ != _b_:
                _out_.append(f'  [{_name_}] #{_id_}: golden={_a_} run={_b_}')
    return _out_
