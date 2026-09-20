'''The parity oracle for the JSComponent migration (PLANNING.md **W1**).

Each test drives a fixed gesture corpus at one component and compares what happened -- the
param writes it provoked and the shape of the DOM afterwards -- against a golden recorded
from the `ReactiveHTML` implementation **before** that component was ported.  See
`parity_trace.py` for why both halves are recorded and what is normalised away.

The goldens are the point.  Because each contract cuts over in place, there is no second
implementation to diff against at port time; these files are the only durable statement of
what the component did beforehand.  So:

* **Record once, against the pre-port implementation.**  Committed alongside it.
* **Never re-record to make a port pass.**  A diff here is either a real regression or an
  intended change that belongs in the CHANGELOG with its reasoning.
* A gesture that provokes no param writes at all is still worth recording -- "nothing
  crossed into Python" is a fact a port can break.

Corpora are per contract because the components genuinely differ: only linkp has nodes to
move or a picker to open, only the generic components take a `z` colour filter, and
stack_controli is a click target with no canvas.
'''
import time

import pytest
from parity_trace import ParityTrace


def _type_search(ip, text, budget_s=8.0):
    '''Type into the `/` search buffer, confirming each character landed.

    `press()` already waits for the controller to go idle, but the harness documents
    (fact D4) that a key arriving while an operation still holds the lock is *silently
    dropped* rather than queued -- no error, no effect.  Under the sustained load of the
    full interaction suite that window widens, and `test_linkpi_search_parity` was seen
    to fail once in a whole-suite run while passing 12/12 in isolation and in 3 runs of
    this file alone.

    The diff from that run was not captured, so a dropped keystroke is the *inferred*
    cause, not a confirmed one -- it is the only mechanism in this gesture that is
    load-sensitive, and the parity goldens record exact values, which makes them stricter
    than the surrounding tests about it.  Waiting for the on-screen echo to catch up
    removes the race whether or not it was the culprit, and costs nothing when idle.
    '''
    _buf_ = ''
    for _ch_ in text:
        ip.press(_ch_)
        _buf_ += _ch_
        _deadline_ = time.monotonic() + budget_s
        while time.monotonic() < _deadline_:
            if ip.el('searchtext').text_content().startswith(f'/ {_buf_}'):
                break
            time.sleep(0.05)
        else:
            raise AssertionError(
                f'the search echo never showed {_buf_!r} '
                f'(saw {ip.el("searchtext").text_content()!r}) -- keystroke dropped')


# ── linkp: the largest contract, and the one with the most JS-only state ─────

def test_linkpi_parity(linkpi_page):
    _ip_ = linkpi_page
    _ip_.settle()
    # The pointer starts inside the component, where the CI browser fires a mouseover
    # at mount; park it clear so the first gesture's has_focus write is a real one.
    _ip_.park_pointer()
    _trace_ = ParityTrace(_ip_)
    _ex_, _ey_ = _ip_.empty_point()

    with _trace_.gesture('hover-empty'):
        _ip_.hover(_ex_, _ey_)

    with _trace_.gesture('drag-select'):
        _ip_.drag(_ex_, _ey_, _ex_ + 60, _ey_ + 60)

    with _trace_.gesture('shift-drag-select'):
        with _ip_.holding(shift=True):
            _ip_.drag(_ex_, _ey_, _ex_ + 60, _ey_ + 60)

    with _trace_.gesture('ctrl-drag-select'):
        with _ip_.holding(ctrl=True):
            _ip_.drag(_ex_, _ey_, _ex_ + 60, _ey_ + 60)

    with _trace_.gesture('wheel-in'):
        _ip_.wheel(_ex_, _ey_, -120)

    with _trace_.gesture('wheel-out'):
        _ip_.wheel(_ex_, _ey_, 120)

    with _trace_.gesture('brush-on'):
        _ip_.press('r')

    with _trace_.gesture('brush-cycle'):
        _ip_.press('R')

    # Moving *while the brush is on* is what actually exercises it: `brush_changed` is
    # written from the mousemove handler, not from the 'r' that turns the brush on.  A
    # corpus that only toggled the brush let a mutant which severed `brush_changed`
    # entirely pass unnoticed.
    with _trace_.gesture('brush-move'):
        _ip_.hover(_ex_ + 25, _ey_ + 25)

    with _trace_.gesture('brush-off'):
        _ip_.press('r')

    with _trace_.gesture('escape'):
        _ip_.press('Escape')

    _trace_.compare_or_record('linkpi')


def test_linkpi_menu_parity(linkpi_page):
    '''The picker menu is the densest JS-only state in the project -- `menu_open`,
    `menu_index`, `menu_kind` and a 2.5s auto-commit timer, none of which reach Python
    until a commit.  A port that rebuilt the DOM at the wrong moment would lose it, which
    is precisely what U5 used to do.'''
    _ip_ = linkpi_page
    _ip_.settle()
    _trace_ = ParityTrace(_ip_)
    _ip_.hover(*_ip_.empty_point())
    # The hover is outside the gestures below but writes params of its own; let them
    # land before anything starts attributing writes to a gesture.
    _trace_.quiesce()

    with _trace_.gesture('menu-open-layout'):
        _ip_.press('l', ctrl=True)

    with _trace_.gesture('menu-next'):
        _ip_.press('j')

    with _trace_.gesture('menu-prev'):
        _ip_.press('k')

    with _trace_.gesture('menu-close'):
        _ip_.press('Escape')

    _trace_.compare_or_record('linkpi_menu')


def test_linkpi_search_parity(search_page):
    '''`/` opens a search buffer that lives entirely in JS until Enter commits it.'''
    _ip_ = search_page
    _ip_.settle()
    _trace_ = ParityTrace(_ip_)
    _ip_.hover(*_ip_.empty_point())
    # The hover is outside the gestures below but writes params of its own; let them
    # land before anything starts attributing writes to a gesture.
    _trace_.quiesce()

    with _trace_.gesture('search-open'):
        _ip_.press('/')

    with _trace_.gesture('search-type'):
        _type_search(_ip_, 'ab')

    with _trace_.gesture('search-commit'):
        _ip_.press('Enter')

    _trace_.compare_or_record('linkpi_search')


# ── the generic interactivep contract ────────────────────────────────────────

@pytest.mark.parametrize('fixture_name', ['xypi_page', 'histopi_page', 'timepi_page'])
def test_generic_component_parity(request, fixture_name):
    _ip_ = request.getfixturevalue(fixture_name)
    _ip_.settle()
    # The pointer starts inside the component, where the CI browser fires a mouseover
    # at mount; park it clear so the first gesture's has_focus write is a real one.
    _ip_.park_pointer()
    _trace_ = ParityTrace(_ip_)
    _w_, _h_ = _ip_.plot.wxh
    _cx_, _cy_ = _w_ // 2, _h_ // 2

    with _trace_.gesture('hover-centre'):
        _ip_.hover(_cx_, _cy_)

    with _trace_.gesture('drag-select'):
        _ip_.drag(_cx_ - 40, _cy_ - 40, _cx_ + 40, _cy_ + 40)

    with _trace_.gesture('shift-drag-select'):
        with _ip_.holding(shift=True):
            _ip_.drag(_cx_ - 40, _cy_ - 40, _cx_ + 40, _cy_ + 40)

    with _trace_.gesture('wheel-in'):
        _ip_.wheel(_cx_, _cy_, -120)

    with _trace_.gesture('brush-on'):
        _ip_.press('r')

    with _trace_.gesture('brush-off'):
        _ip_.press('r')

    with _trace_.gesture('escape'):
        _ip_.press('Escape')

    _trace_.compare_or_record(fixture_name.replace('_page', ''))


# ── the three remaining contracts ────────────────────────────────────────────

def test_smallpi_parity(smallpi_page):
    _ip_ = smallpi_page
    _ip_.settle()
    _trace_ = ParityTrace(_ip_)
    _w_, _h_ = _ip_.plot.wxh
    _cx_, _cy_ = _w_ // 2, _h_ // 2

    with _trace_.gesture('hover-centre'):
        _ip_.hover(_cx_, _cy_)

    with _trace_.gesture('drag-select'):
        _ip_.drag(_cx_ - 60, _cy_ - 60, _cx_ + 60, _cy_ + 60)

    with _trace_.gesture('escape'):
        _ip_.press('Escape')

    _trace_.compare_or_record('smallpi')


def test_slpi_parity(slpi_page):
    _ip_ = slpi_page
    _ip_.settle()
    _trace_ = ParityTrace(_ip_)
    _w_, _h_ = _ip_.plot.wxh
    _cx_, _cy_ = _w_ // 2, _h_ // 2

    with _trace_.gesture('hover-centre'):
        _ip_.hover(_cx_, _cy_)

    with _trace_.gesture('drag-select'):
        _ip_.drag(_cx_ - 60, _cy_ - 40, _cx_ + 60, _cy_ + 40)

    with _trace_.gesture('escape'):
        _ip_.press('Escape')

    _trace_.compare_or_record('slpi')


# ── multiplicity, which the net-effect trace deliberately cannot see ─────────

def test_one_wheel_event_finishes_one_operation(linkpi_page):
    '''One wheel notch must complete exactly one operation.

    The parity trace records a gesture's net effect, so a handler that fired twice and
    left the same value behind would slip past it.  That is not hypothetical here:
    LINKPI's `render` registers a `wheel` listener on three separate hit layers, and
    under `ReactiveHTML` `render` re-runs on every subtree rebuild -- so a duplicate
    registration is exactly the kind of thing this codebase can grow.

    Recorded as a number rather than asserted at 1, because whatever it is today is the
    thing the port has to preserve.  If it is >1 today that is a pre-existing defect the
    port would *fix*, which belongs in the CHANGELOG as a fix and not as a parity
    failure -- see the plan's R-WHEEL.
    '''
    _ip_ = linkpi_page
    _ip_.settle()
    # The pointer starts inside the component, where the CI browser fires a mouseover
    # at mount; park it clear so the first gesture's has_focus write is a real one.
    _ip_.park_pointer()
    _trace_ = ParityTrace(_ip_)
    _ex_, _ey_ = _ip_.empty_point()

    with _trace_.gesture('single-wheel'):
        _ip_.wheel(_ex_, _ey_, -120)

    _counts_ = {_n_: _trace_.write_count(_n_)
                for _n_ in ('wheel_op_finished', 'wheel_rots', 'wheel_x', 'wheel_y')}
    _trace_.results[-1]['write_counts'] = _counts_
    _trace_.compare_or_record('linkpi_wheel_multiplicity')


def test_stack_control_parity(stack_control_page):
    '''stack_controli is a click target with a help overlay rather than a canvas, and it
    is the one view that is not `app.view(0)` -- it is found through the mvc registry,
    the same way test_remaining_components.py does it.'''
    _xy_, _ct_ = stack_control_page
    _view_ = next(_v_ for _v_ in _ct_.app.container.mvc.view_refs.values()
                  if type(_v_).__name__ == 'STACKCONTROLI')
    _trace_ = ParityTrace(_ct_, _view_)

    with _trace_.gesture('toggle-help-on'):
        _ct_.hover(80, 170)
        _ct_.press('h')

    with _trace_.gesture('toggle-help-off'):
        _ct_.hover(80, 170)
        _ct_.press('h')

    with _trace_.gesture('click-frame'):
        _ct_.hover(80, 170)
        _ct_.page.mouse.click(*_ct_._page_xy(80, 170))

    _trace_.compare_or_record('stack_controli')
