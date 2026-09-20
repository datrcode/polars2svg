"""Browser-driven interaction harness for the Panel/ReactiveHTML components.

The Python suite tests the *model*; this package tests the *interaction layer* --
the ~3,800 lines of JavaScript that live as Python strings inside the ``type()``
class factories in ``polars2svg/interactive_controller.py`` and that no Python
test can execute.  See PLANNING.md section 2.1 (V1).

Five facts about the runtime shape this file; each was measured, not assumed, and
each is a trap that silently produces an empty selector rather than an error:

1. **Bokeh refuses the websocket from ``127.0.0.1``.**  ``pn.serve`` allows only
   the origin it advertises, which is ``localhost:<port>``.  Connect to
   ``127.0.0.1`` and the handshake is rejected with a 403, the ReactiveHTML model
   never renders, and the page looks merely slow.  ``serve_panel()`` passes both
   origins and hands back a ``localhost`` URL.

2. **The component renders inside three nested (open) shadow roots.**  Playwright's
   CSS engine pierces them, so locators work -- but ``document.querySelector``
   inside ``page.evaluate`` does *not*.  Hand-written JS therefore either goes
   through ``locator.evaluate()``, which is handed the element itself, or does its
   own shadow-piercing lookup (``_DEEP_FIND_JS``) when it must re-resolve on every
   poll.

3. **Panel rewrites every template id with a per-model suffix.**  ``id="svgparent"``
   in the ``_template`` string reaches the DOM as ``id="svgparent-p1015"``.  Nothing
   in the page has the bare id, so ``#svgparent`` matches zero elements.  The
   harness resolves the suffix once from the root and applies it to every lookup.

   This is a ``ReactiveHTML`` behaviour, and the JSComponent migration (PLANNING.md
   **W1**) removes it: an ESM view builds its own DOM, so its ids reach the browser
   exactly as written.  The harness handles both -- the root locator matches the bare
   id *or* a suffixed one, and :attr:`InteractivePage.suffix` then resolves to ``''``.

   **Inner ids are therefore no longer unique across a page, and every lookup here is
   scoped to the component root because of it.**  The suffix used to make ``#mod``
   unique document-wide; with two ported components on one page (the stack control
   beside its plot) there are two bare ``#mod`` elements, and a page-level locator
   matches both -- Playwright reports a strict-mode violation, which is the good case.
   The bad case was :meth:`wait_for_mod_change`, which resolved the id document-wide and
   could have polled the wrong component indefinitely.  Shadow roots scope ids for
   ``getElementById``, but *not* for Playwright's CSS engine, which pierces them (fact
   2) -- so scoping has to be explicit: ``self.root.locator(...)``, never
   ``self.page.locator(...)``, for anything inside a component.

4. **Hovering focuses; clicking would not be safe; and the focus must be waited
   for.**  ``myOnMouseOver`` calls ``svgparent.focus()``, so a plain ``mouse.move()``
   over the plot both sets ``state.cur_mouse_x/y`` *and* gives the SVG keyboard
   focus.  A click would also focus, but ``downSelect`` starts a drag-select on
   mousedown, so clicking to focus would silently begin a gesture -- hover, never
   click, unless the click is the thing under test.

   The handler runs *after* ``mouse.move()`` returns, so a keystroke issued in that
   window lands on whatever had focus before and the component never sees it.  This
   is the one race that survived the first build of this harness: invisible on an
   idle machine (18 consecutive clean runs) and fatal on a loaded one (3 of 4 runs
   failed at load ~10).  :meth:`InteractivePage.hover` therefore ends with an
   auto-retrying focus assertion, and every mouse move in a test must go through it.

5. **Panel rebuilds the whole subtree on every re-render, so nothing survives it.**
   After one keystroke, element handles captured beforehand -- the interaction root
   included -- report ``isConnected === false``.  Three consequences the harness
   handles and any new test must respect: a wait anchored to a handle watches a
   detached node forever; a JS listener installed on the root is gone (probes are
   re-installed before every keystroke); and **keyboard focus is lost**, so a second
   key press needs a fresh hover first (:meth:`InteractivePage.press_at`).

   This is the ``mod_inner``-as-child rebuild, and it is also ``ReactiveHTML``-only:
   an ESM view has no ``Child`` params, so ``render()`` runs once per mount and the
   DOM it builds is never replaced under it.  A ported component therefore makes
   :meth:`settle` a no-op, the probe re-installation redundant and the re-hover in
   :meth:`press_at` unnecessary -- all three stay, because they cost a few
   milliseconds on an idle page and are what the un-ported components still need.

Assertions live at the DOM layer on purpose (PLANNING.md 2.1, "Decide before
writing test #1").  The bug class this suite exists to catch is *handler fires,
does nothing, reports success* -- a test that reads controller state has to
already know the right answer to catch it, while a test that counts rendered
selection marks catches it as a plain mismatch.
"""
from __future__ import annotations

import os
import re
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable, Iterator

import panel as pn
from playwright.sync_api import Locator, Page, expect

from polars2svg.interactive_controller import panelize

# How long a Python-side operation may take to come back as a DOM change.  Layout
# operations run off the event loop and can be slow; the unit of waiting is always
# an auto-retrying expect(), so this is a ceiling and not a sleep -- a passing test
# never spends it, and raising it costs nothing but the time a genuine failure takes
# to report.
#
# 30s rather than something tighter because these tests share a machine.  DT runs
# several Claude sessions at once (see CLAUDE.md), and a full suite alongside a
# browser suite puts the box at load ~10; the two flaky incidents seen while building
# this were both timeout-shaped and both under that load, neither reproducible on an
# idle machine.  Override with P2S_INTERACTION_TIMEOUT_MS when debugging.
DEFAULT_TIMEOUT_MS = int(os.environ.get('P2S_INTERACTION_TIMEOUT_MS', 30_000))

#: Per-attempt ceiling on Panel server startup, then a retry on a fresh port.
#:
#: Deliberately short.  Startup is well under a second, so anything approaching this is
#: not a slow server but a *dead* one: the bind failed with "Address already in use" in
#: the server thread, where nothing here can see it, and the only symptom is a port
#: that never starts listening.  Waiting that out generously just makes each collision
#: expensive -- it turned a 60s file into 87s and, often enough, into a timeout that
#: surfaced as an unexplained fixture error.  Short timeout, more attempts.
STARTUP_TIMEOUT_S = 4.0

#: Resolve an id anywhere in the document, descending through open shadow roots.
#: Prepended to page.wait_for_function bodies, which run in a bare page context where
#: ``document.querySelector`` stops at the first shadow boundary.
_DEEP_FIND_JS = """
    const __p2sDeepFind = (id) => {
        const find = (root) => {
            const direct = root.querySelector('[id="' + id + '"]');
            if (direct) { return direct; }
            for (const e of root.querySelectorAll('*')) {
                if (e.shadowRoot) {
                    const hit = find(e.shadowRoot);
                    if (hit) { return hit; }
                }
            }
            return null;
        };
        return find(document);
    };
"""


#: Ports this process has already handed to a server.  The OS hands the *most
#: recently freed* port back out again very willingly, so two function-scoped servers
#: created back to back get the same number and the second one dies in its thread with
#: "Address already in use" -- which a bare connect check then mistakes for a server
#: merely never started, and the retry can collide all over again.  Never reusing a
#: number within the process removes the common case; the retry still covers the rest.
_ISSUED_PORTS: set[int] = set()


def _free_port() -> int:
    for _ in range(50):
        with socket.socket() as _s_:
            _s_.bind(('127.0.0.1', 0))
            _port_ = int(_s_.getsockname()[1])
        if _port_ not in _ISSUED_PORTS:
            _ISSUED_PORTS.add(_port_)
            return _port_
    raise RuntimeError('could not find an unused port in 50 attempts')


def _wait_until_serving(port: int, timeout_s: float = STARTUP_TIMEOUT_S) -> None:
    """Poll until the port answers an actual HTTP request, not merely a connect.

    A TCP connect is not enough, and settling for one is what made the suite fail about
    one run in two once it grew past 200 servers.  ``_free_port()`` can hand back a port
    a just-stopped server still holds in TIME_WAIT; bokeh's bind then fails with
    EADDRINUSE inside its own thread, but the lingering socket still *accepts* -- so the
    old check declared the server ready and the failure surfaced much later as an
    inscrutable ``Page.goto: Timeout`` against a URL nothing was serving.

    Asking for a response tells a real Panel server from a leftover socket.
    """
    import urllib.error
    import urllib.request

    _deadline_ = time.monotonic() + timeout_s
    while time.monotonic() < _deadline_:
        try:
            with urllib.request.urlopen(f'http://localhost:{port}/', timeout=0.5) as _r_:
                if _r_.status == 200:
                    return
        except urllib.error.HTTPError:
            return                      # answering at all is what matters
        except Exception:
            pass
        time.sleep(0.05)
    raise TimeoutError(f'nothing served HTTP on port {port} within {timeout_s}s')


@dataclass
class ServedApp:
    """A running Panel server plus the objects the test needs to reason about it."""
    url: str
    container: Any
    views: dict[int, Any]
    _stop: Callable[[], None]

    def stop(self) -> None:
        self._stop()

    def view(self, index: int = 0) -> Any:
        """The nth ReactiveHTML view (LINKPI, XYPI, ...) in creation order.

        Provided for diagnostics and for the few assertions that genuinely belong on
        the Python side (e.g. proving a websocket round-trip completed).  Prefer DOM
        assertions -- see the module docstring.
        """
        return list(self.views.values())[index]


def serve_panel(layout: Any, *, attempts: int = 6, **panelize_kwargs: Any) -> ServedApp:
    """panelize() the layout and serve it on a free port, returning once it accepts
    connections.

    Threaded rather than a subprocess: the JS ``state.*`` variables are re-initialised
    by the ``render`` script on every page load, so per-test browser pages already
    isolate them, and a thread keeps the Python view objects reachable for
    diagnostics.  Switch to a subprocess only if Python-side state bleed appears.

    Retries on a fresh port because picking one is unavoidably racy: ``_free_port()``
    has to *close* the probe socket before ``pn.serve`` can bind it, and anything on
    the machine -- including the next test in this same run -- can take it in between.
    A lost race shows up as a server that never listens, which without the retry
    surfaces as an opaque fixture-setup error one run in some tens.

    When it happens you will see a ``PytestUnhandledThreadExceptionWarning`` carrying
    an ``OSError: [Errno 48] Address already in use`` from a ``get_server`` thread, on
    an otherwise passing run.  That is this, already handled: the bind fails inside the
    server thread where nothing here can catch it, the port never starts listening, and
    the loop below moves on to the next one.  It is noise, not a failure -- and it is
    deliberately not filtered out, because suppressing the warning class would also
    hide a genuine exception escaping a thread.
    """
    _last_error_: Exception | None = None
    for _attempt_ in range(attempts):
        _container_ = panelize(layout, **panelize_kwargs)
        _port_      = _free_port()
        _server_    = pn.serve(_container_, port=_port_, show=False, threaded=True, verbose=False,
                               websocket_origin=[f'localhost:{_port_}', f'127.0.0.1:{_port_}'])
        try:
            _wait_until_serving(_port_, timeout_s=STARTUP_TIMEOUT_S)
        except TimeoutError as _e_:
            _last_error_ = _e_
            _server_.stop()
            continue
        return ServedApp(url=f'http://localhost:{_port_}',
                         container=_container_,
                         views=dict(_container_.mvc.view_refs),
                         _stop=_server_.stop)
    raise RuntimeError(f'panel server failed to start on {attempts} separate ports') from _last_error_


class InteractivePage:
    """Page object for one served interactive component.

    Every element lookup goes through :meth:`el`, which applies the per-model id
    suffix; every wait goes through an auto-retrying ``expect``.  There are no
    ``wait_for_timeout()`` calls in this class, and tests must not add any --
    PLANNING.md 2.1 phase 1 step 3 is explicit that a fixed sleep is where this kind
    of harness turns flaky.
    """

    #: id of the interaction root in the ``_template`` string, before Panel suffixes it.
    #: LINKPI uses the bare name; the generic ``_interactivep`` components append their
    #: kind (``svgparentxypi``, ``svgparenttimepi``, ...), and SMALLPI uses
    #: ``svgparentsmallpi``.  Pass ``root_id`` to address one of those.
    ROOT_ID = 'svgparent'

    #: How long the controller lock must stay free before it counts as idle.
    #: See wait_until_idle() -- a stack operation releases and re-takes it.
    IDLE_SETTLE_S = 0.25

    #: How long the DOM must go unrebuilt before settle() calls the page quiet.
    #: There is no longer a fixed-time rebuild to outlast -- see settle().
    REBUILD_QUIET_S = 0.6

    def __init__(self, page: Page, plot: Any, index: int = 0,
                 timeout_ms: int = DEFAULT_TIMEOUT_MS, root_id: str | None = None) -> None:
        self.page       = page
        self.plot       = plot
        self.timeout_ms = timeout_ms
        self.root_id    = root_id or self.ROOT_ID
        #: Roughly when the page was loaded -- this object is built right after
        #: goto(), and settle() needs an age to reason about.
        self.created_at = time.monotonic()
        # Exact-match the id prefix rather than a bare "starts with svgparent": on a
        # multi-component page `[id^="svgparent"]` also matches svgparentxypi and
        # friends, so addressing "the LINKPI" by index would silently depend on the
        # order Panel happened to emit the models in.
        #
        # Both spellings are matched because both occur: `ReactiveHTML` suffixes the id
        # per model (`svgparent-p1015`, harness fact 3) while an ESM view emits it bare.
        # The exact-match arm is listed first but CSS `,` is unordered, so the `-` in the
        # prefix arm is what keeps `svgparent` from also matching `svgparentxypi`.
        _bare_ = self.root_id
        self.root = page.locator(f'[id="{_bare_}"], [id^="{_bare_}-"]').nth(index)
        self.root.wait_for(state='attached', timeout=timeout_ms)
        _root_id_ = self.root.get_attribute('id') or ''
        #: Panel's per-model id suffix, e.g. ``-p1015`` -- ``''`` for an ESM view, which
        #: leaves every ``[id="...{suffix}"]`` lookup below correct without a branch.
        self.suffix = _root_id_[len(self.root_id):]
        self._install_probes()

    def settle(self, budget_s: float = 8.0) -> bool:
        """Wait until the component is not in the middle of re-rendering.

        **This no longer waits out a fixed delay.**  It used to: a one-off rebuild
        landed ~2.05s after load with no interaction of any kind, destroying every
        JS-only variable -- an open picker lost ``menu_open`` / ``menu_index``, the
        search buffer emptied, ``brush_state`` reset to 0 -- so this method held every
        caller until the page was old enough for that to be behind it.

        The cause was PLANNING.md U5, and it is fixed: ``info_str`` was bound into
        ``_template`` as a content ``${info_str}``, which registers a param as a
        ReactiveHTML *child*, and panel rebuilds the subtree on every write to a child.
        Python rewrites ``info_str`` after almost every operation, so almost every
        operation rebuilt.  With the binding gone nothing rebuilds on its own; a fresh
        page now sits idle indefinitely without one (verified: root never replaced over
        5s, menu still open until its own 2.5s auto-commit timer).

        What remains is genuine: ``mod_inner`` is still a child, because unbinding it
        would put it through panel's HTML_SANITIZER, which strips SVG to the empty
        string.  So a real plot redraw still rebuilds, and a test that acts while one is
        in flight can still lose JS-only state.  This waits for that -- a quiet window
        with no rebuild observed -- and returns immediately on an idle page.

        Returns True if a rebuild was observed while waiting.
        """
        _deadline_ = time.monotonic() + budget_s
        _seen_     = False
        _last_     = time.monotonic()
        self.root.evaluate("(rootEl) => { rootEl.__p2s_settle_mark__ = true; }")
        while time.monotonic() < _deadline_:
            if not self.root.evaluate("(rootEl) => rootEl.__p2s_settle_mark__ === true"):
                _seen_ = True
                _last_ = time.monotonic()
                self._install_probes()          # the listener went with the old root
                self.root.evaluate("(rootEl) => { rootEl.__p2s_settle_mark__ = true; }")
            if (time.monotonic() - _last_) >= self.REBUILD_QUIET_S:
                return _seen_
            time.sleep(0.05)
        return _seen_

    # ── element access ────────────────────────────────────────────────────────    # ── element access ────────────────────────────────────────────────────────

    def el(self, template_id: str) -> Locator:
        """Locator for a ``_template`` id, with Panel's per-model suffix applied."""
        return self.root.locator(f'[id="{template_id}{self.suffix}"]')

    def within(self, template_id: str, css: str) -> Locator:
        return self.root.locator(f'[id="{template_id}{self.suffix}"] {css}')

    @property
    def mod(self) -> Locator:
        """The inner SVG the component renders into (``data.mod_inner``)."""
        return self.el('mod')

    # ── coordinates ───────────────────────────────────────────────────────────

    def node_screen_xy(self, node: Any) -> tuple[int, int]:
        """Screen (SVG-local) coordinates of a node, from the component's own
        ``df_node``.  Keyed by ``__first__``, which is the *stringified* node id --
        integer-id graphs included."""
        _rows_ = self.plot.df_node.filter(
            self.plot.df_node['__first__'] == str(node))
        if len(_rows_) != 1:
            raise LookupError(f'expected exactly one df_node row for node {node!r}, got {len(_rows_)}')
        return int(_rows_['__sx__'][0]), int(_rows_['__sy__'][0])

    def empty_point(self, clearance: int = 25) -> tuple[int, int]:
        """A canvas point far enough from every node to miss ``#allentitieslayer``.

        A rubber-band selection has to start on bare canvas: the hit layers stack, and
        pressing on a node reaches ``downAllEntities`` (move-unselected) instead, with
        no complaint.  Layouts are random per process, so the point is searched for
        rather than hardcoded -- a fixed corner is empty until the day it isn't, and
        the test that breaks then looks like a selection bug.
        """
        _w_, _h_ = self.plot.wxh
        _nodes_ = [(int(_r_['__sx__']), int(_r_['__sy__']))
                   for _r_ in self.plot.df_node.iter_rows(named=True)]
        for _y_ in range(clearance, _h_ - clearance, 10):
            for _x_ in range(clearance, _w_ - clearance, 10):
                if all(abs(_x_ - _nx_) > clearance or abs(_y_ - _ny_) > clearance
                       for _nx_, _ny_ in _nodes_):
                    return _x_, _y_
        raise LookupError('no point on this canvas is clear of every node')

    def first_mark_xy(self) -> tuple[float, float]:
        """Centre of the smallest drawn mark in ``#mod``.

        For the generic components there is no ``df_node`` to read positions out of,
        and a point picked by arithmetic easily lands on blank canvas -- where the
        colour and shape bindings correctly do nothing, which looks exactly like a
        binding that is not wired up.  Aiming at something provably drawn removes that
        ambiguity.

        Smallest-area rather than first-in-document because the components disagree
        about what a mark *is*: linkp draws circles, xyp draws small rects, and every
        plot's chrome (background, border, clip rect) is a rect too -- but always a
        big one.
        """
        _marks_ = self.root.locator(
            f'[id="mod{self.suffix}"] circle, [id="mod{self.suffix}"] rect'
        ).evaluate_all("""els => els.map(e => {
            const b = e.getBBox();
            return {cx: b.x + b.width / 2, cy: b.y + b.height / 2,
                    area: Math.max(b.width, 1) * Math.max(b.height, 1)};
        })""")
        if not _marks_:
            raise LookupError('nothing is drawn in #mod')
        _m_ = min(_marks_, key=lambda _d_: _d_['area'])
        return float(_m_['cx']), float(_m_['cy'])

    def _page_xy(self, x: float, y: float) -> tuple[float, float]:
        """SVG-local (x, y) -> page coordinates, waiting out a rebuild if one is in flight.

        ``bounding_box()`` returns None -- rather than waiting -- for an element that is
        attached but has no layout box yet, which is exactly what the root looks like
        during one of Panel's subtree rebuilds.  A hover issued in that window used to
        raise "no bounding box", intermittently, in whichever test happened to move the
        mouse just after an operation re-rendered.
        """
        _deadline_ = time.monotonic() + self.timeout_ms / 1000
        while time.monotonic() < _deadline_:
            self.root.wait_for(state='visible', timeout=self.timeout_ms)
            _box_ = self.root.bounding_box()
            if _box_ is not None and _box_['width'] > 0 and _box_['height'] > 0:
                return _box_['x'] + x, _box_['y'] + y
            time.sleep(0.02)
        raise RuntimeError('interaction root never got a bounding box (not rendered?)')

    # ── input ─────────────────────────────────────────────────────────────────

    def park_pointer(self, budget_s: float = 5.0) -> None:
        """Move the pointer clear of the component, so ``has_focus`` starts out False.

        Playwright's virtual cursor sits at viewport ``(0, 0)`` until something moves
        it, and Panel lays the component flush into that corner -- measured, not
        assumed: the root's bounding box is ``(0, 0, 400, 300)``.  A browser that
        dispatches a ``mouseover`` when content appears under a stationary cursor
        therefore runs ``myOnMouseOver`` at mount, and ``has_focus`` is already true
        before the test has touched anything.  Chromium on the Linux CI runner does
        exactly that; the macOS build does not, which is the whole reason this was
        only ever seen in CI.

        Harmless under ``ReactiveHTML``, whose ``render`` script re-ran on every
        subtree rebuild and reset ``data.has_focus`` to false.  An ESM ``render`` runs
        once per mount -- see the note beside the initialisers at the foot of
        ``p2s_interactivep.js`` -- so the stale true survives, the first hover's
        ``model.has_focus = true`` is a no-op, param emits no event, and a trace of
        that hover records **no write at all**.  For four of the parity corpora that
        one write is the entire first gesture, so the gesture recorded nothing.

        Call after :meth:`settle` and before the first gesture in any test whose
        recording includes the focus transition.  Where the pointer was never over the
        component it finds ``has_focus`` already false and returns at once, so it
        costs nothing on the platforms that do not show the problem.
        """
        self.root.wait_for(state='visible', timeout=self.timeout_ms)
        _box_ = self.root.bounding_box()
        _vp_  = self.page.viewport_size
        if _box_ is not None and _vp_ is not None:
            _pad_ = 40.0
            # Right of the root first, then below, then left, then above: one of the
            # four is on the page for any box smaller than the viewport, and a fixed
            # choice is not safe -- a component laid out at the corner has two sides
            # with nothing beyond them.
            for _x_, _y_ in ((_box_['x'] + _box_['width'] + _pad_, _box_['y']),
                             (_box_['x'], _box_['y'] + _box_['height'] + _pad_),
                             (_box_['x'] - _pad_, _box_['y']),
                             (_box_['x'], _box_['y'] - _pad_)):
                _inside_ = (_box_['x'] <= _x_ < _box_['x'] + _box_['width']
                            and _box_['y'] <= _y_ < _box_['y'] + _box_['height'])
                if not _inside_ and 0 <= _x_ < _vp_['width'] and 0 <= _y_ < _vp_['height']:
                    self.page.mouse.move(_x_, _y_)
                    break
            else:
                raise RuntimeError(
                    'no point in the viewport is clear of the interaction root')

        # Wait for the mouseout to cross into Python rather than sleeping on it: the
        # point of parking is that the *next* gesture sees a real False -> True
        # transition, and that is only guaranteed once the False has landed.
        _app_  = getattr(self, 'app', None)
        _view_ = _app_.view() if _app_ is not None else None
        if _view_ is None or 'has_focus' not in _view_.param:
            return
        _deadline_ = time.monotonic() + budget_s
        while time.monotonic() < _deadline_:
            if not _view_.has_focus:
                return
            time.sleep(0.02)
        raise AssertionError(
            f'has_focus never cleared after parking the pointer ({budget_s}s)')

    def hover(self, x: float, y: float) -> None:
        """Move the mouse to SVG-local (x, y) **and make sure focus lands**.

        ``event.offsetX/offsetY`` on every mouse target in the template
        (``#screen``, ``#allentitieslayer``, ``#selectionlayer``) equals the SVG-local
        coordinate -- verified, not assumed -- so no correction is applied.

        Focus is not belt-and-braces here, it is the whole point.  It arrives via the
        ``myOnMouseOver`` handler, which runs *after* ``mouse.move()`` has returned;
        press a key in that window and it lands on whatever had focus before, the
        component sees nothing, and the test fails instantly rather than timing out.
        On an idle machine the window is too small to notice and on a loaded one it
        opens wide enough to fail most runs.

        And a single move plus a wait is not enough either.  If Panel rebuilds the
        subtree just after the move -- which it does on its own about two seconds after
        load (U5) -- focus goes with the old element and *nothing brings it back*,
        because the pointer is already where it needs to be and the browser sends no
        further mouseover.  Waiting on that is waiting forever.  So the move is
        repeated, with a one-pixel nudge to guarantee a fresh event, until the root
        reports itself focused.
        """
        _px_, _py_ = self._page_xy(x, y)
        self.page.mouse.move(_px_, _py_)
        _deadline_ = time.monotonic() + self.timeout_ms / 1000
        while time.monotonic() < _deadline_:
            if self.root.evaluate(
                    "(rootEl) => rootEl === rootEl.getRootNode().activeElement"):
                return
            # Nudge and come back: re-moving to the same point dispatches nothing.
            self.page.mouse.move(_px_ + 1, _py_ + 1)
            self.page.mouse.move(_px_, _py_)
            time.sleep(0.05)
        raise AssertionError(
            f'the interaction root never took focus after hovering ({x}, {y})')

    def hover_node(self, node: Any) -> tuple[int, int]:
        _xy_ = self.node_screen_xy(node)
        self.hover(*_xy_)
        return _xy_

    def press(self, key: str, *, shift: bool = False, ctrl: bool = False,
              wait_idle: bool = True) -> None:
        """Press a key on the focused SVG.

        ``key`` is the literal ``event.key`` the handler branches on, so an
        uppercase binding is written as ``press('Z')`` -- the modifier flags are for
        ctrl (and for a shift that is *not* already implied by the character).

        **Waits for the controller to go idle first**, because a key arriving while
        the previous operation still holds the lock is silently dropped rather than
        queued (D4).  Without this, a test only works if the machine happens to finish
        the previous operation in time: removing a single completion wait from a
        working three-keystroke sequence made it fail every run.  Tests that press
        deliberately early -- there is one, studying the drop itself -- pass
        ``wait_idle=False``.
        """
        if wait_idle and getattr(self, 'app', None) is not None:
            self.wait_until_idle()
        # Re-install the probe: the listener lives on the root element, and Panel
        # replaces that element on every re-render, taking the listener with it.  The
        # JS guard makes this a no-op when the current root already carries it.
        self._install_probes()
        _mods_ = [m for m, on in (('Control', ctrl), ('Shift', shift)) if on]
        self.page.keyboard.press('+'.join([*_mods_, key]))

    @contextmanager
    def holding(self, *, ctrl: bool = False, shift: bool = False) -> Iterator[None]:
        """Hold modifiers down across the whole operation, the way a person does.

        Use this -- not ``press(ctrl=True)`` -- whenever the assertion depends on the
        modifier reaching **Python**, because the two are not equivalent.

        ``press('Control+c')`` sends keydown Control, keydown c, keyup c, keyup
        Control back to back.  The keydown sets ``data.ctrlkey = true`` and
        ``data.key_op_finished``; the *keyup of Control* then sets
        ``data.ctrlkey = false`` (``myOnKeyUp`` writes it unconditionally).  Both
        cross the websocket, while ``applyKeyOp`` runs asynchronously behind a lock --
        so the handler frequently reads ``self.ctrlkey`` as False and takes the
        unmodified branch.  Measured: ctrl-c arrives with ``ctrlkey False`` and zooms
        the view instead of copying (see PLANNING.md U3).

        Holding the modifier across the round-trip removes the race from the test and
        matches what a person's timing does -- they are still holding ctrl when the
        server processes the keystroke.  It does not remove the race from the
        component; that is a real defect, and ``test_keyup_clears_the_modifier_...``
        pins the behaviour so a fix is visible.
        """
        _mods_ = [m for m, on in (('Control', ctrl), ('Shift', shift)) if on]
        for _m_ in _mods_:
            self.page.keyboard.down(_m_)
        try:
            yield
        finally:
            for _m_ in reversed(_mods_):
                self.page.keyboard.up(_m_)

    def wait_until_idle(self, budget_s: float = 15.0) -> None:
        """Block until the controller is between operations.

        Necessary before *any* second keystroke, because a key arriving while an
        operation still holds the lock is **silently dropped** rather than queued
        (D4 -- deliberate, so a long layout cannot bank a replay of everything typed
        during it).  ``applyKeyOp`` opens with ``if self._busy_(): return``, so the
        keystroke leaves no trace at all: no error, no effect, and the "busy --
        ignored" notice it tries to show is itself invisible (U4).

        That combination is why this is a harness primitive rather than a note in one
        test.  Measured: pressing 'X' straight after an 'x' stack push loses the 'X'
        every single run; the same sequence with a few hundred ms between them works.

        Python-side by necessity -- the lock has no DOM representation -- and precise
        rather than a sleep: it polls the same flag ``_busy_()`` reads.
        """
        _app_ = getattr(self, 'app', None)
        if _app_ is None:
            raise RuntimeError('wait_until_idle() needs the ServedApp; set page.app in the fixture')
        _view_ = _app_.view()
        # Not every view has a lock: the stack control is an interactive-only leaf with
        # no async operations of its own -- it mutates someone else's stack -- so it has
        # no busy state to wait on.  Treat "no lock" as "never busy" and fall back to
        # render quiescence alone.
        _lock_ = getattr(_view_, 'lock', None)
        _deadline_ = time.monotonic() + budget_s
        _free_since_ = None
        _last_mod_   = None
        while time.monotonic() < _deadline_:
            _mod_ = self.mod_html()
            if (_lock_ is not None and _lock_.locked()) or _mod_ != _last_mod_:
                # Quiescence, not merely an unlocked instant: the render must also have
                # stopped moving.  Tuning the window alone was not enough -- 250ms of
                # free lock still lost the pop about one run in three, because a stack
                # operation's fan-out re-renders after the lock has already been given
                # back.  Requiring #mod to hold still as well keys the wait to the end
                # of the work rather than to a guess about how long it takes.
                _last_mod_   = _mod_
                _free_since_ = None
            else:
                # A single "not locked" reading is not readiness.  A stack operation
                # releases the lock at the end of applyKeyOp and then RE-acquires it:
                # `await self.mvc.pushStack(...)` runs outside the `async with`, and
                # the display() it fans out to takes the lock again.  Sampling the gap
                # between the two lets the next keystroke through just in time to be
                # dropped by the second acquisition -- which is exactly how 'x' then
                # 'X' loses the pop.  So require the lock to stay free.
                _free_since_ = _free_since_ or time.monotonic()
                if time.monotonic() - _free_since_ >= self.IDLE_SETTLE_S:
                    return
            time.sleep(0.02)
        raise TimeoutError(f'controller still busy after {budget_s}s')

    @contextmanager
    def holding_key(self, key: str) -> Iterator[None]:
        """Hold an ordinary key down across a gesture.

        The layout gestures need this: ``g`` / ``y`` / ``Y`` set ``state.layout_op`` on
        keydown and ``myOnKeyUp`` clears it again on release, so they are hold-to-arm.
        ``press()`` sends down *and* up, which arms and disarms before the mouse has
        moved -- the drag then falls through to an ordinary rubber band (see U6).
        """
        if getattr(self, 'app', None) is not None:
            self.wait_until_idle()
        self._install_probes()
        self.page.keyboard.down(key)
        try:
            yield
        finally:
            self.page.keyboard.up(key)

    def press_at(self, node: Any, key: str, *, shift: bool = False, ctrl: bool = False,
                 wait_idle: bool = True) -> None:
        """Hover a node, then press a key -- the correct order, made the easy one.

        Focus does **not** survive a re-render: Panel rebuilds the subtree, the new
        ``#svgparent`` is not focused, and the ``render`` script resets
        ``data.has_focus`` to false.  So a second keystroke after an operation that
        re-rendered is silently swallowed unless the mouse is moved back over the plot
        first.  Any test pressing more than one key must re-hover between them.
        """
        self.hover_node(node)
        self.press(key, shift=shift, ctrl=ctrl, wait_idle=wait_idle)

    # ── mouse gestures ────────────────────────────────────────────────────────
    #
    # Which handler a press reaches is decided by what is under the pointer, because
    # the template stacks three transparent hit layers in this order:
    #
    #   #screen           full canvas          -> downSelect        (drag-select)
    #   #allentitieslayer every node's marker  -> downAllEntities   (move unselected)
    #   #selectionlayer   the selected markers -> downMove          (move selected)
    #
    # Later siblings paint on top, so empty canvas gives a rubber-band selection, an
    # unselected node gives an unselected-move, and a selected node gives a move.
    # Aim accordingly; a "drag-select" that starts on a node is a move, silently.

    def mouse_down(self, x: float, y: float, *, button: str = 'left') -> None:
        """Move to (x, y) and press.  Waits for idle first, as press() does."""
        if getattr(self, 'app', None) is not None:
            self.wait_until_idle()
        self.hover(x, y)
        self.page.mouse.down(button=button)

    def mouse_move_to(self, x: float, y: float, steps: int = 4) -> None:
        """Move the pointer, in steps, so mousemove handlers actually run.

        A single jump fires one mousemove; the drag rectangle and the layout shape are
        both drawn from that handler, so a one-step drag tests less than it looks.
        """
        self.page.mouse.move(*self._page_xy(x, y), steps=steps)

    def mouse_up(self, *, button: str = 'left') -> None:
        self.page.mouse.up(button=button)

    def drag(self, x0: float, y0: float, x1: float, y1: float, *,
             button: str = 'left') -> None:
        self.mouse_down(x0, y0, button=button)
        self.mouse_move_to(x1, y1)
        self.mouse_up(button=button)

    def wheel(self, x: float, y: float, delta_y: float) -> None:
        """Scroll over the plot.  The handler is registered {passive: false} so it can
        preventDefault; a negative delta is a zoom in."""
        if getattr(self, 'app', None) is not None:
            self.wait_until_idle()
        self.hover(x, y)
        self.page.mouse.wheel(0, delta_y)

    def drag_rect(self) -> dict[str, str]:
        """The rubber-band rectangle's current geometry and stroke.

        Parked off-canvas at (-10, -10) 5x5 when no drag is in progress, which is how
        'the band was cleared' is distinguished from 'the band is a small square'.
        """
        _el_ = self.el('drag')
        return {_a_: (_el_.get_attribute(_a_) or '')
                for _a_ in ('x', 'y', 'width', 'height', 'stroke')}

    def node_positions(self) -> dict[str, tuple[float, float]]:
        """Rendered node centres, read back out of the SVG.

        The point of reading the *drawing* rather than ``plot.pos`` is that a move
        which updates the model and never repaints is the failure this suite exists to
        catch.
        """
        return {_c_['id']: (float(_c_['cx']), float(_c_['cy']))
                for _c_ in self.root.locator(
                    f'[id="mod{self.suffix}"] circle').evaluate_all(
                        "els => els.map(e => ({id: e.getAttribute('id') || "
                        "(e.getAttribute('cx') + ',' + e.getAttribute('cy')), "
                        "cx: e.getAttribute('cx'), cy: e.getAttribute('cy')}))")}

    # ── observation (DOM, auto-retrying) ──────────────────────────────────────

    def info_text(self) -> str:
        return self.el('infostr').text_content() or ''

    def selected_count(self) -> int:
        """Selection count as the component itself reports it in ``#infostr``.

        A page that has never refreshed reads as 0: the regex simply does not match.
        That used to be the *normal* state rather than an edge case -- while
        ``info_str`` was bound as a child it never reached the browser at all and the
        line stayed on its ``" | | grid"`` param default for the life of the page
        (PLANNING.md U5).  Four search-mode tests were quietly relying on that.
        """
        _m_ = re.match(r'\s*(\d+) Selected', self.info_text())
        return int(_m_.group(1)) if _m_ else 0

    def selection_mark_count(self) -> int:
        """Number of marks drawn in ``#selectionlayer``.

        This is the *rendered* selection -- the thing a user sees -- rather than the
        controller's idea of it.  The no-selection fallback path is a single mark at
        (-100,-100), which :meth:`has_no_selection` distinguishes.
        """
        _d_ = self.el('selectionlayer').get_attribute('d') or ''
        return _d_.count('M ')

    def has_no_selection(self) -> bool:
        _d_ = self.el('selectionlayer').get_attribute('d') or ''
        return _d_.strip().startswith('M -100 -100')

    def expect_selected(self, n: int) -> None:
        """Wait for the component to report *and draw* a selection of n entities.

        Both halves are *waited* for, and the second one has to be.  ``__refreshView__``
        writes ``info_str`` before ``selectionpath``, so the count can reach the browser
        and be applied before the marks do -- asserting on the drawing the instant the
        text matches is a race, and it fires perhaps one full-suite run in several.

        It was not always.  While ``info_str`` was a ReactiveHTML child (PLANNING.md
        U5), writing it rebuilt the whole subtree, and the rebuild re-ran ``render``,
        which reset ``#selectionlayer`` from ``data.selectionpath`` -- so the two could
        not be seen disagreeing.  Fixing U5 made ``info_str`` update in place and left
        this ordering visible.  Anything else in this class that reads one param's
        rendering after waiting on another's is the same hazard.
        """
        expect(self.el('infostr')).to_contain_text(f'{n} Selected', timeout=self.timeout_ms)
        _deadline_ = time.monotonic() + self.timeout_ms / 1000.0
        while time.monotonic() < _deadline_:
            _drawn_ = 0 if self.has_no_selection() else self.selection_mark_count()
            if _drawn_ == n:
                return
            time.sleep(0.05)
        if n == 0:
            assert self.has_no_selection(), (
                f'#infostr reports 0 selected but #selectionlayer still draws '
                f'{self.selection_mark_count()} mark(s)')
        else:
            _marks_ = self.selection_mark_count()
            assert _marks_ == n, (
                f'#infostr reports {n} selected but #selectionlayer draws {_marks_} mark(s) '
                f'-- the handler ran and the render disagrees')

    def expect_focused(self) -> None:
        expect(self.root).to_be_focused(timeout=self.timeout_ms)

    def expect_info_contains(self, text: str) -> None:
        expect(self.el('infostr')).to_contain_text(text, timeout=self.timeout_ms)

    def expect_menu_open(self, header_text: str) -> None:
        """Wait for the picker menu overlay to show a given header.

        ``#pickermenu`` is written entirely by JS (``menuRender``) and never crosses
        into Python until a commit, so it is the only observable the menu state
        machine has.  Note ``menuArmTimer`` auto-commits after 2.5s -- assert with
        this rather than with a sleep, or the menu closes underneath the test.
        """
        expect(self.el('pickermenu')).to_contain_text(header_text, timeout=self.timeout_ms)

    #: The picker menu highlights the selected row with a rect at
    #: ``y = 8 + 1 + (menu_index + 1) * 14`` (menuRender).  Inverting that is the only
    #: way to observe ``state.menu_index``, which is one of the 29 JS-only variables
    #: and never crosses into Python -- the whole reason the menu had no coverage.
    MENU_ROW_H  = 14
    MENU_ROW_Y0 = 9
    MENU_HILITE = 'rgba(100,150,255,0.3)'

    def menu_index(self) -> int:
        """The highlighted row, recovered from the highlight rect's y offset."""
        _y_ = self.root.locator(
            f'[id="pickermenu{self.suffix}"] rect[fill="{self.MENU_HILITE}"]'
        ).get_attribute('y', timeout=self.timeout_ms)
        if _y_ is None:
            raise AssertionError('no highlight rect -- the picker menu is not open')
        return round((int(_y_) - self.MENU_ROW_Y0) / self.MENU_ROW_H) - 1

    def expect_menu_index(self, index: int) -> None:
        """Wait for a given row to be highlighted.

        Expressed as an attribute assertion so it auto-retries: the menu is redrawn
        by JS on every cycle keystroke, and reading the index eagerly would race it.
        """
        _y_ = self.MENU_ROW_Y0 + (index + 1) * self.MENU_ROW_H
        expect(self.root.locator(
            f'[id="pickermenu{self.suffix}"] rect[fill="{self.MENU_HILITE}"]'
        )).to_have_attribute('y', str(_y_), timeout=self.timeout_ms)

    def menu_is_open(self) -> bool:
        return bool((self.el('pickermenu').inner_html() or '').strip())

    def menu_text(self) -> str:
        return self.el('pickermenu').text_content() or ''

    def expect_menu_closed(self) -> None:
        expect(self.el('pickermenu')).to_be_empty(timeout=self.timeout_ms)

    def mod_html(self) -> str:
        return self.mod.inner_html()

    def marks(self) -> Locator:
        """Every drawn primitive in ``#mod`` -- the plot's marks plus its chrome.

        Used as a *count*, which is the stable way to say "this view redrew with more
        or fewer things in it".  Comparing innerHTML works but is momentary: a brush
        that paints and clears again inside one poll interval reads as unchanged.
        """
        return self.root.locator(f'[id="mod{self.suffix}"] circle, '
                                 f'[id="mod{self.suffix}"] rect, '
                                 f'[id="mod{self.suffix}"] path')

    def expect_marks_to_change_from(self, count: int) -> None:
        expect(self.marks()).not_to_have_count(count, timeout=self.timeout_ms)

    def expect_marks(self, count: int) -> None:
        expect(self.marks()).to_have_count(count, timeout=self.timeout_ms)

    def wait_for_mod_change(self, before: str) -> str:
        """Block until ``#mod`` differs from the snapshot, then return the new html.

        The sync point for operations with no other DOM-visible signal: the chain is
        JS -> websocket -> Python -> re-render -> new ``mod_inner``, and this waits on
        its last link.  Polling a DOM property rather than sleeping keeps it honest.

        The poll holds **no element handle**, because none survives.  Panel rebuilds
        the whole ReactiveHTML subtree when a binding updates -- measured: after one
        keystroke both the previously-captured ``#svgparent`` and ``#mod`` handles
        report ``isConnected === false``.  A wait anchored to either of them watches a
        detached node that will never see another update, and times out while the page
        in front of you has plainly changed.  (This fails *silently sometimes*, which is
        worse: an update applied to the old subtree just before the swap is visible on
        the stale handle, so a length-changing operation can pass while a colour-only
        one on the same code path hangs.)

        So the lookup is redone on every tick rather than held as a handle.  It is done
        through :meth:`mod_html`, which resolves ``#mod`` **within this component's
        root**, and that scoping is load-bearing now: a page can hold two components
        whose inner ids are both bare (see fact 3), so a document-wide search for
        ``#mod`` -- which is what this used to do -- could poll the other one and wait
        forever on a node nothing was going to change.
        """
        _deadline_ = time.monotonic() + self.timeout_ms / 1000
        while time.monotonic() < _deadline_:
            _now_ = self.mod_html()
            if _now_ != before:
                return _now_
            time.sleep(0.05)
        raise AssertionError(
            f'#mod never changed within {self.timeout_ms}ms of the operation')

    # ── preventDefault observation ────────────────────────────────────────────

    def _install_probes(self) -> None:
        """Record keydown events *after* the component's handler has run.

        The listener goes on ``#svgparent`` itself, not on an ancestor, because
        ``myOnKeyDown`` opens with ``event.stopPropagation()`` -- nothing above the
        root ever sees the event, so the obvious bubble-from-the-parent probe records
        precisely nothing and looks like a focus failure.  Two listeners on the *same*
        element both run regardless: ``stopPropagation()`` only stops the climb, and
        the template's inline ``onkeydown`` attribute is registered first, so a
        listener added here runs after it and reads the settled ``defaultPrevented``.
        (``stopImmediatePropagation()`` would defeat this; the handler does not use it.)

        ``defaultPrevented`` is the browser's own answer to "was the default
        suppressed?" -- half of what PLANNING.md 2.1 item 1 asks for.  The other half
        (the handler still fired) is an ordinary DOM assertion.
        """
        self.root.evaluate("""(rootEl) => {
            if (rootEl.__p2s_probe_installed__) { return; }
            rootEl.__p2s_probe_installed__ = true;
            const bucket = [];
            rootEl.__p2s_keydowns__ = bucket;
            rootEl.addEventListener('keydown', (ev) => {
                bucket.push({key: ev.key, ctrlKey: ev.ctrlKey, shiftKey: ev.shiftKey,
                             altKey: ev.altKey, metaKey: ev.metaKey,
                             defaultPrevented: ev.defaultPrevented});
            });
        }""")

    def context_menu_prevented(self, x: float = 200, y: float = 150) -> bool:
        """Does a secondary click over the plot get its browser popup suppressed?

        Dispatched rather than clicked, and read from a listener added *after* the
        component's own, so the answer is the browser's ``defaultPrevented`` flag
        rather than an inference.  A real ctrl-left-drag produces the same event on
        macOS -- which is the gesture U1 is about -- but dispatching keeps the
        assertion about the guard instead of about mouse emulation.
        """
        return bool(self.root.evaluate("""(rootEl) => {
            const target = rootEl.querySelector('[id^="screen"]') || rootEl;
            const ev = new MouseEvent('contextmenu', {bubbles: true, cancelable: true});
            target.dispatchEvent(ev);
            return ev.defaultPrevented;
        }"""))

    def keydowns(self) -> list[dict[str, Any]]:
        return list(self.root.evaluate('(rootEl) => rootEl.__p2s_keydowns__ || []'))

    def last_keydown(self) -> dict[str, Any]:
        _events_ = self.keydowns()
        if not _events_:
            raise AssertionError('no keydown reached the interaction root -- '
                                 'the SVG almost certainly did not have focus')
        return _events_[-1]

    def clear_keydowns(self) -> None:
        self._install_probes()
        self.root.evaluate('(rootEl) => { if (rootEl.__p2s_keydowns__) rootEl.__p2s_keydowns__.length = 0; }')
