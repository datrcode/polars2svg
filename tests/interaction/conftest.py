"""Collection gate and fixtures for the browser-driven interaction suite.

**Opt-in.**  These tests boot a Panel server and drive a real Chromium, so they are
an order of magnitude slower than the rest of the suite and depend on tooling the
package does not otherwise need.  ``pytest tests/`` therefore skips them; run them
with ``--interaction`` (or ``P2S_INTERACTION=1``):

    .venv/bin/python -m pytest tests/interaction --interaction -q

Setup, once:

    # NOTE the venv hazard: a bare `uv pip install` in these repos resolves
    # VIRTUAL_ENV to racetrack's .venv.  Always pass --python explicitly.
    uv pip install --python "$PWD/.venv/bin/python" pytest-playwright
    ./.venv/bin/python -m playwright install chromium

**Style departure, deliberate.**  Every other test file in this repo is
``unittest.TestCase`` (enforced for ``tests/*.py`` by ``tests/test_meta.py``, whose
glob is non-recursive and does not reach this directory).  Browser tests cannot
follow that convention: ``page``, ``browser`` and friends are pytest fixtures, and
a ``unittest.TestCase`` method cannot receive one.  Tests here are plain pytest
functions taking fixtures.
"""
import os
import sys
from typing import Any, Callable, Iterator

import pytest

try:
    import playwright  # noqa: F401
    _PLAYWRIGHT_AVAILABLE_ = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE_ = False
    # Nothing in this directory can even be imported without playwright (the harness
    # imports its sync API at module scope), so skip collection rather than erroring.
    collect_ignore_glob = ['test_*.py', 'interaction_harness.py']


# The --interaction flag itself is declared in tests/conftest.py; see the note there.

def _is_interaction_item(item: pytest.Item) -> bool:
    return os.path.dirname(str(item.fspath)) == os.path.dirname(os.path.abspath(__file__))


def pytest_itemcollected(item: pytest.Item) -> None:
    """Apply the ``interaction`` marker as each item is collected.

    Deliberately *not* done in pytest_collection_modifyitems below: that one is
    ``trylast``, and pytest's own ``-m`` deselection is itself a
    pytest_collection_modifyitems implementation.  Marking from there lands after the
    filter has already run, and `pytest tests/ -m interaction` quietly deselects
    everything.  pytest_itemcollected fires during collection, before any filtering.
    """
    if _is_interaction_item(item):
        item.add_marker(pytest.mark.interaction)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config,
                                  items: list[pytest.Item]) -> None:
    """Gate the browser tests, and **move them to the end of the run**.

    The reordering is not cosmetic, and removing it breaks 159 unrelated tests.
    Playwright's *sync* API runs its asyncio event loop in the calling thread and
    switches greenlets around it, and pytest-playwright's ``playwright`` fixture is
    session-scoped -- so from the first browser test onward the main thread has a
    running loop for the rest of the session.  Every later test that calls
    ``asyncio.run()`` (``test_interactive_controller.py`` and friends -- 159 of them)
    then dies with "asyncio.run() cannot be called from a running event loop".

    ``tests/interaction`` sorts before ``tests/test_*.py``, so by default the browser
    tests run *first* and poison everything after them.  Running them last confines
    the loop to the tail of the session, where nothing follows.  ``trylast`` so the
    ordering survives any other plugin that reorders (none is installed today, but
    pytest-randomly or -p xdist would each disturb it).
    """
    _enabled_ = config.getoption('--interaction') or os.environ.get('P2S_INTERACTION') == '1'
    _skip_    = pytest.mark.skip(reason='browser-driven; pass --interaction (or set P2S_INTERACTION=1) to run')

    _mine_, _others_ = [], []
    for item in items:
        if not _is_interaction_item(item):
            _others_.append(item)
            continue
        if not _enabled_:
            item.add_marker(_skip_)
        _mine_.append(item)

    items[:] = _others_ + _mine_


if _PLAYWRIGHT_AVAILABLE_:
    import polars as pl

    from polars2svg import BackgroundShape, Polars2SVG
    from interaction_harness import InteractivePage, serve_panel

    @pytest.fixture
    def served() -> Iterator[Callable[..., Any]]:
        """Factory fixture: ``served(layout, **panelize_kwargs)`` -> ServedApp.

        Function-scoped, so every test gets a fresh server and therefore a fresh
        controller: the Python-side state (``selected_entities``, the dataframe
        stack, cached positions) lives on the view object and would otherwise carry
        between tests.  The JS-only ``state.*`` variables are re-initialised by the
        ``render`` script on each page load, so they are already isolated.
        """
        _apps_ = []

        def _factory(layout: Any, **kwargs: Any) -> Any:
            _app_ = serve_panel(layout, **kwargs)
            _apps_.append(_app_)
            return _app_

        yield _factory
        for _app_ in _apps_:
            _app_.stop()

    @pytest.fixture
    def two_color_linkp() -> Any:
        """A five-node graph with **integer** node ids in two colour groups.

        Integer ids on purpose: ``color_nodes_final`` is keyed by the *stringified*
        node name while ``pos`` is keyed by the original id, so an integer-id graph
        is the case where a colour-based selection has to cross that boundary.  That
        is the exact shape of the ``z`` bug PLANNING.md 2.1 names as the harness's
        validation case.
        """
        _p2s_ = Polars2SVG()
        _df_  = pl.DataFrame({'fm':  [1, 2, 3, 4, 5],
                              'to':  [2, 3, 1, 5, 4],
                              'grp': ['x', 'x', 'x', 'y', 'y']})
        _lp_  = _p2s_.linkp(_df_, relationships=[('fm', 'to')], node_color='grp', wxh=(400, 300))
        _lp_._repr_svg_()          # materialise pos / df_node / color_nodes_final
        return _lp_

    @pytest.fixture
    def labelled_linkp() -> Any:
        """The two-colour graph with node *labels* that differ from the node ids.

        ctrl-c copies ids and ctrl-shift-c copies labels; with labels absent the two
        produce identical text and a test cannot tell them apart -- so the label set
        is deliberately unlike the ids.
        """
        _p2s_ = Polars2SVG()
        _df_  = pl.DataFrame({'fm':  [1, 2, 3, 4, 5],
                              'to':  [2, 3, 1, 5, 4],
                              'grp': ['x', 'x', 'x', 'y', 'y']})
        _lp_  = _p2s_.linkp(_df_, relationships=[('fm', 'to')], node_color='grp',
                            wxh=(400, 300),
                            node_labels={1: 'alpha', 2: 'bravo', 3: 'charlie',
                                         4: 'delta', 5: 'echo'})
        _lp_._repr_svg_()
        return _lp_

    @pytest.fixture
    def labelled_page(page: Any, served: Callable[..., Any],
                      labelled_linkp: Any) -> InteractivePage:
        _app_ = served([[labelled_linkp]])
        page.goto(_app_.url, wait_until='load')
        _ip_ = InteractivePage(page, labelled_linkp)
        _ip_.app = _app_
        return _ip_

    @pytest.fixture
    def search_linkp() -> Any:
        """Node names chosen so substring and regex matches are unambiguous.

        'alp' matches alpha and alpine and nothing else; 'beta' matches one node;
        '^a' as a regex matches the same two as 'alp'.  Numeric ids would make
        'matched the right nodes' and 'matched some nodes' hard to tell apart.
        """
        _p2s_ = Polars2SVG()
        _df_  = pl.DataFrame({'fm': ['alpha', 'alpine', 'beta',  'gamma'],
                              'to': ['alpine', 'beta',  'gamma', 'alpha']})
        _lp_  = _p2s_.linkp(_df_, relationships=[('fm', 'to')], wxh=(400, 300))
        _lp_._repr_svg_()
        return _lp_

    @pytest.fixture
    def search_page(page: Any, served: Callable[..., Any],
                    search_linkp: Any) -> InteractivePage:
        _app_ = served([[search_linkp]])
        page.goto(_app_.url, wait_until='load')
        _ip_ = InteractivePage(page, search_linkp)
        _ip_.app = _app_
        return _ip_

    @pytest.fixture
    def brush_ready_page(linkpi_page: Any) -> Any:
        """A settled, focused LINKPI ready for a brush keystroke.

        The brush cursor is drawn from JS-only state and positioned from
        ``state.cur_mouse_x/y``, so the page is settled first (U5 used to erase the
        cursor; it no longer does, but a redraw still can) and the pointer must
        already be over the plot before the toggle --
        otherwise the first cursor is drawn at the seeded origin instead of under the
        pointer.
        """
        linkpi_page.settle()
        linkpi_page.hover(200, 150)
        return linkpi_page

    @pytest.fixture
    def quad_linkp() -> Any:
        """Four nodes pinned to a 2x2 grid, well inside the canvas.

        Mouse gestures need to know where things are.  With a random layout a node
        lands in a corner about as often as not, and a drag from it runs off the
        canvas -- where ``myOnMouseUp`` never fires, so the gesture is abandoned and
        the test fails for a reason that has nothing to do with the binding.  Pinning
        the positions makes every rectangle in the mouse suite exact: which nodes a
        box contains is arithmetic, not luck.

        ``view_window`` is pinned as well, and that is the load-bearing half: without
        it the layout auto-fits whatever positions it is given back out to the canvas
        edges, so the pinned coordinates land in the corners again and nothing is
        gained.  With both pinned the nodes sit at (101, 76) / (298, 76) / (101, 223) /
        (298, 223) on a 400x300 canvas -- comfortably inside, with the corners clear
        for a press on bare canvas.  Tests read the real values from ``df_node``
        rather than trusting these numbers.
        """
        _p2s_ = Polars2SVG()
        _df_  = pl.DataFrame({'fm': ['nw', 'ne', 'se', 'sw'],
                              'to': ['ne', 'se', 'sw', 'nw']})
        _lp_  = _p2s_.linkp(_df_, relationships=[('fm', 'to')], wxh=(400, 300),
                            view_window=(0.0, 0.0, 1.0, 1.0),
                            pos={'nw': (0.25, 0.75), 'ne': (0.75, 0.75),
                                 'sw': (0.25, 0.25), 'se': (0.75, 0.25)})
        _lp_._repr_svg_()
        return _lp_

    @pytest.fixture
    def quad_page(page: Any, served: Callable[..., Any],
                  quad_linkp: Any) -> InteractivePage:
        _app_ = served([[quad_linkp]])
        page.goto(_app_.url, wait_until='load')
        _ip_ = InteractivePage(page, quad_linkp)
        _ip_.app = _app_
        return _ip_

    # ── the generic _interactivep components ──────────────────────────────────
    #
    # xyp / timep / histop / chordp / piep all go through _interactivep(), which gives
    # them one shared keydown handler plus a per-kind config: whether z / the time keys
    # / '/' search exist at all, and which brush shapes R cycles through.  Their
    # interaction root is `svgparent<kind>`, hence the `root_id` on InteractivePage.

    def _grid_df() -> Any:
        """Points on a coarse grid, so a rectangle covers a knowable number of them."""
        return pl.DataFrame({
            'x':   [10.0, 30.0, 50.0, 70.0, 90.0] * 4,
            'y':   [float(_v_) for _v_ in ([10] * 5 + [30] * 5 + [70] * 5 + [90] * 5)],
            'cat': ['a', 'b', 'a', 'b', 'a'] * 4,
        })

    @pytest.fixture
    def xypi_page(page: Any, served: Callable[..., Any]) -> InteractivePage:
        """XYPI -- the generic component with the most optional bindings enabled:
        z (colour filter) and the time keys are both on for it."""
        _p2s_ = Polars2SVG()
        _xyp_ = _p2s_.xyp(_grid_df(), 'x', 'y', color='cat', wxh=(400, 300))
        _xyp_._repr_svg_()
        _app_ = served([[_xyp_]])
        page.goto(_app_.url, wait_until='load')
        _ip_ = InteractivePage(page, _xyp_, root_id='svgparentxypi')
        _ip_.app = _app_
        return _ip_

    @pytest.fixture
    def histopi_page(page: Any, served: Callable[..., Any]) -> InteractivePage:
        """HISTOPI -- brush sequence [0,1,2,5,6], so R reaches *horizontal* bands."""
        _p2s_ = Polars2SVG()
        _hp_  = _p2s_.histop(_grid_df(), 'cat', wxh=(400, 300))
        _hp_._repr_svg_()
        _app_ = served([[_hp_]])
        page.goto(_app_.url, wait_until='load')
        _ip_ = InteractivePage(page, _hp_, root_id='svgparenthistopi')
        _ip_.app = _app_
        return _ip_

    @pytest.fixture
    def timepi_page(page: Any, served: Callable[..., Any]) -> InteractivePage:
        """TIMEPI -- brush sequence [0,1,2,3,4], so R reaches *vertical* bands."""
        import datetime
        _p2s_ = Polars2SVG()
        _df_  = pl.DataFrame({
            'ts':  [datetime.datetime(2024, 1, 1) + datetime.timedelta(days=_i_ * 3)
                    for _i_ in range(20)],
            'cat': ['a', 'b'] * 10,
        })
        _tp_ = _p2s_.timep(_df_, 'ts', color='cat', wxh=(400, 300))
        _tp_._repr_svg_()
        _app_ = served([[_tp_]])
        page.goto(_app_.url, wait_until='load')
        _ip_ = InteractivePage(page, _tp_, root_id='svgparenttimepi')
        _ip_.app = _app_
        return _ip_

    @pytest.fixture
    def linked_pair(page: Any, served: Callable[..., Any]) -> Any:
        """A LINKPI and an XYPI on one page, over the same dataframe.

        The brush exists to be cross-component: brushing in one view broadcasts the
        records under the cursor and every linked view redraws to show them.  That
        fan-out needs two components, and no Python test reaches the JS that starts it.
        """
        _p2s_ = Polars2SVG()
        _df_  = pl.DataFrame({
            'fm': ['a', 'b', 'c', 'd', 'e'],
            'to': ['b', 'c', 'd', 'e', 'a'],
            'x':  [10.0, 30.0, 50.0, 70.0, 90.0],
            'y':  [10.0, 30.0, 50.0, 70.0, 90.0],
        })
        # Positions pinned, and centrally, for the same reason quad_linkp pins its own:
        # a random layout puts a node in a corner often enough, and a test that then
        # hovers "just next to it" walks off the canvas -- where there is no hit layer,
        # so nothing takes focus and the failure reads as a focus bug.
        _lp_  = _p2s_.linkp(_df_, relationships=[('fm', 'to')], wxh=(400, 300),
                            view_window=(0.0, 0.0, 1.0, 1.0),
                            pos={'a': (0.3, 0.3), 'b': (0.5, 0.35), 'c': (0.7, 0.5),
                                 'd': (0.55, 0.65), 'e': (0.35, 0.6)})
        _xyp_ = _p2s_.xyp(_df_, 'x', 'y', wxh=(400, 300))
        _lp_._repr_svg_(); _xyp_._repr_svg_()
        _app_ = served([[_lp_], [_xyp_]])
        page.goto(_app_.url, wait_until='load')
        _link_ = InteractivePage(page, _lp_)
        _xy_   = InteractivePage(page, _xyp_, root_id='svgparentxypi')
        _link_.app = _xy_.app = _app_
        return _link_, _xy_

    @pytest.fixture
    def multi_edge_page(page: Any, served: Callable[..., Any]) -> InteractivePage:
        """A graph whose edges carry several rows each.

        ctrl-shift-X collapses every edge to one row; on a frame that already has one
        row per edge it correctly does nothing, so the binding cannot be observed
        there at all.
        """
        _p2s_ = Polars2SVG()
        _df_  = pl.DataFrame({'fm': ['a', 'a', 'a', 'b', 'b', 'c'],
                              'to': ['b', 'b', 'b', 'c', 'c', 'a']})
        _lp_  = _p2s_.linkp(_df_, relationships=[('fm', 'to')], wxh=(400, 300))
        _lp_._repr_svg_()
        _app_ = served([[_lp_]])
        page.goto(_app_.url, wait_until='load')
        _ip_ = InteractivePage(page, _lp_)
        _ip_.app = _app_
        return _ip_

    # ── the three components the suite reached last ───────────────────────────
    #
    # SMALLPI, SLPI and the stack control each build their own ReactiveHTML class with
    # its own keydown handler, none of which share the LINKPI or _interactivep code
    # paths.  Their interaction roots are `svgparentsmallpi`, `svgparentslpi` and
    # `svgstackcontrol`.

    @pytest.fixture
    def smallpi_page(page: Any, served: Callable[..., Any]) -> InteractivePage:
        """SMALLPI -- small multiples of an xy template, one tile per category."""
        _p2s_ = Polars2SVG()
        _df_  = pl.DataFrame({'x':   [10.0, 30.0, 50.0, 70.0, 90.0] * 4,
                              'y':   [10.0, 30.0, 50.0, 70.0, 90.0] * 4,
                              'cat': ['a', 'b', 'c', 'd'] * 5})
        _xyp_ = _p2s_.xyp(_df_, 'x', 'y', wxh=(96, 96))
        _sm_  = _p2s_.smallp(_df_, 'cat', _xyp_, wxh=(384, 384))
        _sm_._repr_svg_()
        _app_ = served([[_sm_]])
        page.goto(_app_.url, wait_until='load')
        _ip_ = InteractivePage(page, _sm_, root_id='svgparentsmallpi')
        _ip_.app = _app_
        return _ip_

    @pytest.fixture
    def slpi_page(page: Any, served: Callable[..., Any]) -> InteractivePage:
        """SLPI -- the spread-lines (ego network over time) interactive wrapper."""
        _p2s_ = Polars2SVG()
        _df_  = pl.DataFrame({'fm':   ['a', 'b', 'c', 'a', 'd', 'b'],
                              'to':   ['b', 'a', 'a', 'c', 'a', 'c'],
                              'time': [1, 1, 1, 2, 2, 3],
                              'w':    [3, 1, 2, 4, 1, 2]})
        _sp_  = _p2s_.spreadlinesp(_df_, [('fm', 'to')], ego='a', time='time',
                                   wxh=(600, 300))
        _sp_._repr_svg_()
        _app_ = served([[_sp_]])
        page.goto(_app_.url, wait_until='load')
        _ip_ = InteractivePage(page, _sp_, root_id='svgparentslpi')
        _ip_.app = _app_
        return _ip_

    @pytest.fixture
    def stack_control_page(page: Any, served: Callable[..., Any]) -> Any:
        """An xy plot with the stack control beside it.

        The control is an *interactive-only* leaf -- it has no static twin -- so it is
        built explicitly and placed in the layout rather than being wrapped from a
        plain component.  It needs to be tall enough for the MLX/CUDA header, the
        current icon and two skip labels, or it refuses to build.

        Returns (xy_page, control_page); the control acts on the plot's stack, so both
        halves are needed to see anything happen.
        """
        _p2s_ = Polars2SVG()
        _df_  = pl.DataFrame({'x':   [10.0, 30.0, 50.0, 70.0, 90.0] * 4,
                              'y':   [10.0, 30.0, 50.0, 70.0, 90.0] * 4,
                              'cat': ['a', 'b', 'c', 'd'] * 5})
        _xyp_ = _p2s_.xyp(_df_, 'x', 'y', color='cat', wxh=(256, 256))
        _xyp_._repr_svg_()
        _sc_  = _p2s_.stack_controli(_xyp_, wxh=(160, 340))
        _app_ = served([[_xyp_, _sc_]])
        page.goto(_app_.url, wait_until='load')
        _xy_ = InteractivePage(page, _xyp_, root_id='svgparentxypi')
        _ct_ = InteractivePage(page, _xyp_, root_id='svgstackcontrol')
        _xy_.app = _ct_.app = _app_
        return _xy_, _ct_

    # ── WebGPU (PLANNING.md V2) ───────────────────────────────────────────────
    #
    # The GPU path needs a browser Playwright does not use by default.  Its bundled
    # `chromium-headless-shell` exposes `navigator.gpu` but `requestAdapter()` returns
    # null, so every GPU render would fail and every test would pass for the wrong
    # reason.  The full `chromium` build -- installed alongside it by
    # `playwright install chromium`, so nothing extra to download -- gets a real
    # adapter and device.
    #
    # `navigator.gpu` also requires a secure context: it is simply absent on
    # about:blank, which makes a naive capability probe report "no WebGPU" on a
    # browser that has it.  Everything here is served over http://localhost, which
    # counts as secure.

    def _gpu_df() -> Any:
        return pl.DataFrame({'x':   [10.0, 30.0, 50.0, 70.0, 90.0] * 4,
                             'y':   [15.0, 35.0, 55.0, 75.0, 95.0] * 4,
                             'cat': ['a', 'b', 'c', 'd'] * 5,
                             'n':   list(range(20))})

    @pytest.fixture
    def gpu_xyp() -> Any:
        _lp_ = Polars2SVG().xyp(_gpu_df(), 'x', 'y', color='cat', wxh=(400, 300))
        _lp_._repr_svg_()
        return _lp_

    @pytest.fixture
    def gpu_timep() -> Any:
        import datetime
        _df_ = pl.DataFrame({
            'ts':  [datetime.datetime(2024, 1, 1) + datetime.timedelta(days=_i_ * 3)
                    for _i_ in range(20)],
            'cat': ['a', 'b'] * 10})
        _c_ = Polars2SVG().timep(_df_, 'ts', color='cat', wxh=(400, 300))
        _c_._repr_svg_()
        return _c_

    @pytest.fixture
    def gpu_histop() -> Any:
        _c_ = Polars2SVG().histop(_gpu_df(), 'cat', wxh=(400, 300))
        _c_._repr_svg_()
        return _c_

    @pytest.fixture
    def gpu_piep() -> Any:
        _c_ = Polars2SVG().piep(_gpu_df(), 'cat', wxh=(300, 300))
        _c_._repr_svg_()
        return _c_

    @pytest.fixture
    def gpu_chordp() -> Any:
        _df_ = pl.DataFrame({'fm': ['a', 'b', 'c', 'd', 'a'],
                             'to': ['b', 'c', 'd', 'a', 'c']})
        _c_ = Polars2SVG().chordp(_df_, [('fm', 'to')], wxh=(300, 300))
        _c_._repr_svg_()
        return _c_

    @pytest.fixture
    def gpu_linkp() -> Any:
        _df_ = pl.DataFrame({'fm': ['a', 'b', 'c', 'd'], 'to': ['b', 'c', 'd', 'a']})
        _c_ = Polars2SVG().linkp(_df_, relationships=[('fm', 'to')], wxh=(400, 300))
        _c_._repr_svg_()
        return _c_

    @pytest.fixture
    def gpu_spreadlines() -> Any:
        """A spreadlines with a dashed context zigzag and several node circles.

        Shaped for V2's two spreadlines claims: enough time columns that the context
        line is a multi-segment polyline (so its dash phase has joins to cross), and
        enough nodes that a "did the picture match the geometry" check has something
        to check.
        """
        _df_ = pl.DataFrame({'fm':   ['a', 'b', 'c', 'a', 'd', 'b', 'c', 'd'],
                             'to':   ['b', 'a', 'a', 'c', 'a', 'c', 'b', 'a'],
                             'time': [1, 1, 1, 2, 2, 3, 3, 4],
                             'w':    [3, 1, 2, 4, 1, 2, 1, 3]})
        _c_ = Polars2SVG().spreadlinesp(_df_, [('fm', 'to')], ego='a', time='time',
                                        wxh=(600, 320))
        _c_._repr_svg_()
        return _c_

    @pytest.fixture
    def gpu_link_labels() -> Any:
        """A three-edge triangle with curved links and a label on each.

        Pinned positions and view window so the edge chords have three clearly distinct
        angles -- that is what makes "the label is drawn along its edge" a real
        assertion rather than a coincidence, and the labels are uppercase so their ink
        is a solid, easily separated blob against the blue curves.
        """
        _df_ = pl.DataFrame({'fm':  ['alpha', 'bravo', 'charlie'],
                             'to':  ['bravo', 'charlie', 'alpha'],
                             'rel': ['CALLS', 'READS', 'WRITES']})
        _c_ = Polars2SVG().linkp(_df_, relationships=[('fm', 'to', 'rel')], wxh=(460, 340),
                                 link_shape='curve', draw_link_labels=True,
                                 view_window=(0.0, 0.0, 1.0, 1.0),
                                 pos={'alpha': (0.15, 0.25), 'bravo': (0.85, 0.30),
                                      'charlie': (0.5, 0.85)})
        _c_._repr_svg_()
        return _c_

    @pytest.fixture
    def gpu_dashed_polyline() -> Any:
        """A long dashed polyline built for exactly one measurement: **V2 item 2**,
        whether the dash phase runs across the joins of a flattened polyline.

        Every number here was chosen against that measurement and none of them is
        arbitrary, so change them only with the probe in hand:

        * **25 points, gently rising, evenly spaced.** Straight and monotonic, so there
          are no sharp joins, no curvature and no self-overlap to confuse a sample --
          but still 24 separate line primitives, which is what gives the phase 23 joins
          to cross.  The rise is deliberate rather than a flat line: axis-aligned
          geometry is exactly where a rasteriser is most likely to have a special case.
        * **760x300, giving ~33px segments.** Long enough that the samples inside one
          segment outnumber the two ends that have to be skipped (see the test).
        * **dasharray [14, 15] -- deliberately not equal, and coprime-ish with the
          segment length.** Period 29 against a 33.2px segment steps the phase 4.2px per
          vertex, so successive segments start at very different places in the pattern.
          That is the whole point: if the phase were reset per segment, every segment
          would instead start at 0, and the two hypotheses are told apart by exactly
          that spread.  An equal on/off, or a period that divided the segment length,
          would make them agree and the test would prove nothing.

        ``draw_context=False`` keeps the picture to the dashed line alone, so nothing
        else can put ink near a sample point.
        """
        _n_ = 25
        _df_ = pl.DataFrame({'t': [float(_i_) for _i_ in range(_n_)],
                             'v': [float(_i_) * 0.35 for _i_ in range(_n_)],
                             's': ['a'] * _n_})
        _c_ = Polars2SVG().xyp(_df_, 't', 'v', wxh=(760, 300), draw_context=False,
                               line=('s', [14, 15]))
        _c_._repr_svg_()
        return _c_

    @pytest.fixture
    def gpu_dashed_background() -> Any:
        """The same measurement as `gpu_dashed_polyline`, through the *other* dash
        implementation.

        There are two, and they share no code.  `xyp` accumulates the phase with a
        polars `cum_sum().over('__line__')` (xyp.py ~3092); everything that goes through
        a path -- background records, `pathToDL`, spreadlinesp's context line -- uses
        `strokePolylineDL` (p2s_displaylist.py).  A test of one says nothing about the
        other, which is why this exists: mutating `strokePolylineDL` leaves the xyp
        fixture entirely green.

        Same 25 points, dasharray and canvas as the polyline fixture, so the geometry
        argument in its docstring carries over unchanged (~33.9px segments against a
        period of 29).  The dataframe is two points only and `dot_size=None` keeps them
        unpainted: the background path must be the sole ink on the canvas, or a stray
        mark near a sample point reads as a dash.
        """
        _n_ = 25
        _d_ = 'M ' + ' L '.join(f'{_i_} {_i_ * 0.35}' for _i_ in range(_n_))
        _df_ = pl.DataFrame({'t': [0.0, float(_n_ - 1)], 'v': [0.0, (_n_ - 1) * 0.35]})
        _c_ = Polars2SVG().xyp(_df_, 't', 'v', wxh=(760, 300), draw_context=False,
                               dot_size=None,
                               background={'ctx': BackgroundShape(_d_, stroke='#0044cc',
                                                                  stroke_width=2,
                                                                  dash='14 15')})
        _c_._repr_svg_()
        return _c_

    # Launch policy for the GPU browser -- three cases, measured 2026-09-10 on a Ryzen
    # 9 7900X / RTX 3090 with the NVIDIA Vulkan ICD installed.
    #
    # macOS needs nothing: the full `chromium` build gets a Metal adapter on a bare
    # launch, which is why the suite was written with no flags at all.
    #
    # Linux needs flags, and *which* flags depends on whether there is a display,
    # because that is what decides whether the real driver is reachable:
    #
    #   headed, with an X display -> the **real GPU**.  `--use-angle=vulkan
    #     --use-vulkan=native` reports `vendor: nvidia, architecture: ampere` and paints
    #     a full canvas.  This is the only configuration in which the WGSL has ever run
    #     on a non-Apple hardware driver, so prefer it whenever a display exists -- Xvfb
    #     counts, and needs no root (see PLANNING.md 2.1).
    #
    #   headless -> **no** hardware adapter, and no flag recovers it.  `requestAdapter()`
    #     returns null under `--use-vulkan=native`, `--enable-features=Vulkan,
    #     VulkanFromANGLE`, `--ignore-gpu-blocklist`, `--disable-gpu-sandbox`,
    #     `--headless=old` and their combinations: headless chromium has no surface to
    #     present to, so Dawn never brings the driver up.  SwiftShader is all that is
    #     left, and reaching it takes two steps rather than one.
    #     `--enable-unsafe-swiftshader` alone is a *trap*: it yields an adapter, a
    #     device, and working offscreen renders -- a render to a texture reads back the
    #     right pixels -- so a probe that stops at `requestAdapter()` reports success,
    #     and then the canvas destroys the device on the first present:
    #
    #         ERROR:shared_image_factory.cc: Could not find SharedImageBackingFactory
    #         with params: usage: ...WebgpuSwapChainTexture...
    #         ERROR:shared_image_stub.cc: SharedImageStub: Unable to create shared image
    #
    #     which fails all 14 tests blank instead of skipping -- strictly worse than the
    #     skip it replaced.  Routing compositing through ANGLE-on-Vulkan-on-SwiftShader
    #     supplies that backing factory and the canvas becomes drawable and
    #     screenshot-readable.  It is a CPU device: it exercises the WGSL, not a driver.
    #
    # None of these may be applied off Linux -- macOS has no Vulkan, so forcing
    # `--use-angle=vulkan` there would break the Metal path that works today.
    _WEBGPU_HEADED_ = False
    if not sys.platform.startswith('linux'):
        _WEBGPU_FLAGS_ = []
    elif os.environ.get('DISPLAY'):
        _WEBGPU_HEADED_ = True
        _WEBGPU_FLAGS_ = ['--enable-unsafe-webgpu', '--use-gl=angle', '--use-angle=vulkan',
                          '--use-vulkan=native', '--enable-features=Vulkan',
                          '--ignore-gpu-blocklist']
    else:
        _WEBGPU_FLAGS_ = ['--enable-unsafe-webgpu', '--enable-unsafe-swiftshader',
                          '--use-gl=angle', '--use-angle=vulkan', '--use-vulkan=swiftshader',
                          '--enable-features=Vulkan', '--ignore-gpu-blocklist']

    @pytest.fixture(scope='session')
    def webgpu_browser(playwright: Any) -> Any:
        """A Chromium with a working WebGPU device, or a skip explaining why not."""
        try:
            _browser_ = playwright.chromium.launch(channel='chromium',
                                                   headless=not _WEBGPU_HEADED_,
                                                   args=_WEBGPU_FLAGS_)
        except Exception as _e_:
            pytest.skip(f'the full chromium build is unavailable ({_e_})')
        yield _browser_
        _browser_.close()

    @pytest.fixture
    def webgpu_page(webgpu_browser: Any, served: Callable[..., Any]) -> Any:
        """Factory: ``webgpu_page(component)`` -> (page, view, canvas locator).

        Serves the component through ``panelize(..., use_webgpu=True)``, which puts the
        plot on a ``<canvas>`` and leaves the interaction SVG transparent on top of it.
        Skips -- rather than failing -- when the browser cannot actually present a GPU
        canvas, so the suite stays honest on a machine that cannot run the shaders.

        **The probe below paints, and it has to.**  Asking `requestAdapter()` and
        stopping there is the trap PLANNING.md V2 records under
        `--enable-unsafe-swiftshader`: an adapter is handed out, a device is created,
        and *offscreen* work is correct -- a render to a texture reads back the right
        pixels -- while the first canvas present destroys the device.  Nothing surfaces
        in JS beyond a device-lost with reason `destroyed`, so `gpu_error` stays empty
        and the component reports success while drawing nothing.

        That is not hypothetical.  It is what a GitHub `ubuntu-latest` runner does, and
        it turned the first CI run of this suite into **13 blank failures instead of 13
        skips** -- every WebGPU test asserting an ink ratio of 0.0000 against a browser
        that never had a usable canvas.  A capability probe has to exercise the
        capability the tests need, which is presenting to a canvas, not obtaining a
        device.
        """
        _pages_ = []

        def _factory(component: Any) -> Any:
            _app_  = served([[component]], use_webgpu=True)
            _page_ = webgpu_browser.new_page()
            _pages_.append(_page_)
            _page_.goto(_app_.url, wait_until='load')
            _why_ = _page_.evaluate("""async () => {
                    if (!navigator.gpu) { return 'navigator.gpu is absent'; }
                    let device;
                    try {
                        const adapter = await navigator.gpu.requestAdapter();
                        if (!adapter) { return 'requestAdapter() returned null'; }
                        device = await adapter.requestDevice();
                    } catch (e) { return 'no device: ' + e; }

                    // Present to a real canvas -- the step that fails where merely
                    // getting a device does not.
                    try {
                        const cvs = document.createElement('canvas');
                        cvs.width = cvs.height = 16;
                        document.body.appendChild(cvs);
                        const ctx = cvs.getContext('webgpu');
                        if (!ctx) { return 'canvas.getContext("webgpu") returned null'; }
                        ctx.configure({device: device,
                                       format: navigator.gpu.getPreferredCanvasFormat(),
                                       alphaMode: 'opaque'});
                        const enc  = device.createCommandEncoder();
                        const pass = enc.beginRenderPass({colorAttachments: [{
                            view: ctx.getCurrentTexture().createView(),
                            clearValue: {r: 1, g: 0, b: 0, a: 1},
                            loadOp: 'clear', storeOp: 'store'}]});
                        pass.end();
                        device.queue.submit([enc.finish()]);
                        await device.queue.onSubmittedWorkDone();
                    } catch (e) { return 'canvas present failed: ' + e; }

                    // device.lost resolves rather than rejecting, so race it against a
                    // tick: the loss from a failed present has already been queued by
                    // the time the submit above settles.
                    const lost = await Promise.race([
                        device.lost.then(i => 'device lost: ' + i.reason + ' ' + i.message),
                        new Promise(r => setTimeout(() => r(''), 250))]);
                    return lost;
                }""")
            if _why_:
                pytest.skip(f'this browser cannot present a WebGPU canvas ({_why_})')
            _page_.wait_for_selector('[id^="gpucanvas"]', timeout=30_000)
            return _page_, _app_.view(), _page_.locator('[id^="gpucanvas"]')

        yield _factory
        for _p_ in _pages_:
            _p_.close()

    @pytest.fixture
    def hub_page(page: Any, served: Callable[..., Any]) -> InteractivePage:
        """A star: one hub of degree 8 surrounded by eight degree-1 leaves.

        The digit bindings split at 7 -- 1-6 select an *exact* degree, while 7-0 select
        ranges (7-20, 21-50, 51-100, 101-10000).  Neither of the other fixtures has a
        node above degree 2, so the range branch could not be told apart from the exact
        one on them: every digit would select nothing and every test would pass.
        """
        _p2s_ = Polars2SVG()
        _df_  = pl.DataFrame({'fm': ['hub'] * 8,
                              'to': [f'leaf{_i_}' for _i_ in range(8)]})
        _lp_  = _p2s_.linkp(_df_, relationships=[('fm', 'to')], wxh=(400, 300))
        _lp_._repr_svg_()
        _app_ = served([[_lp_]])
        page.goto(_app_.url, wait_until='load')
        _ip_ = InteractivePage(page, _lp_)
        _ip_.app = _app_
        return _ip_

    @pytest.fixture
    def chain_linkp() -> Any:
        """A directed chain 1 -> 2 -> 3 -> 4, with **one colour per node**.

        One colour per node makes ``z`` select exactly one node, which is what the
        expansion keys need in order to be observable at all: on the two-cycle graph
        of ``two_color_linkp`` every node's predecessor closure is the whole cycle,
        so ``ctrl-e`` provably runs and provably changes nothing -- a test asserting
        on the selection count there would pass whether the binding worked or not.

        A dict ``node_color`` is what buys the one-node-per-colour property; a column
        would colour by row and leave the interior nodes in two groups at once.
        """
        _p2s_ = Polars2SVG()
        _df_  = pl.DataFrame({'fm': [1, 2, 3], 'to': [2, 3, 4]})
        _lp_  = _p2s_.linkp(_df_, relationships=[('fm', 'to')], wxh=(400, 300),
                            node_color={1: '#e41a1c', 2: '#377eb8',
                                        3: '#4daf4a', 4: '#984ea3'})
        _lp_._repr_svg_()
        return _lp_

    @pytest.fixture
    def chain_page(page: Any, served: Callable[..., Any],
                   chain_linkp: Any) -> InteractivePage:
        """A served, loaded LINKPI on the one-colour-per-node directed chain."""
        _app_ = served([[chain_linkp]])
        page.goto(_app_.url, wait_until='load')
        _ip_ = InteractivePage(page, chain_linkp)
        _ip_.app = _app_
        return _ip_

    @pytest.fixture
    def linkpi_page(page: Any, served: Callable[..., Any],
                    two_color_linkp: Any) -> InteractivePage:
        """A served, loaded, ready-to-drive LINKPI on the two-colour integer graph."""
        _app_ = served([[two_color_linkp]])
        page.goto(_app_.url, wait_until='load')
        _ip_ = InteractivePage(page, two_color_linkp)
        _ip_.app = _app_
        return _ip_
