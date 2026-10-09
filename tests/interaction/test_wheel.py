"""The wheel stops at every interactive view (DT, 2026-10-08).

With the pointer over any interactive view, the wheel must not scroll the notebook.
Only LINKPI does anything with it (zoom; ``test_mouse_gestures.py``).  On every other
view it is consumed and does nothing: whether a wheel zoom fits the stack methodology
the views obey is DT's deferred decision (PLANNING.md §7 F2).

``defaultPrevented`` read by a listener on the *document* is the browser's own answer
to "will this scroll the page": the wheel is a composed event, so it leaves each view's
shadow root and reaches the document after every listener inside has run.

Two probes per view, because they fail differently:

* a real wheel over the middle of the canvas -- what a user does; and
* a wheel dispatched on the view's root ``<svg>`` -- standing in for everything that is
  not ``#screen`` (the configuration panel and the help can overhang the canvas).  A
  listener on ``#screen`` alone let both of those through on the views that had one,
  and smallpi, spreadlinepi and the stack control had no wheel listener at all.
"""
import unittest

import pytest

#: (fixture, which half of a two-view fixture, or None)
VIEWS = [('xypi_page',          None),
         ('histopi_page',       None),
         ('timepi_page',        None),
         ('quad_page',          None),   # linkpi
         ('smallpi_page',       None),
         ('slpi_page',          None),
         ('stack_control_page', 1)]


def _view(request, fixture, half):
    _page_ = request.getfixturevalue(fixture)
    return _page_[half] if half is not None else _page_


def _real_wheel_is_prevented(ip) -> bool:
    _box_ = ip.root.bounding_box()
    ip.page.evaluate("""() => { window.__p2s_wheel__ = [];
        document.addEventListener('wheel', ev => window.__p2s_wheel__.push(ev.defaultPrevented)); }""")
    ip.wheel(_box_['width'] / 2, _box_['height'] / 2, 120)
    ip.page.wait_for_function('() => window.__p2s_wheel__.length > 0')
    _seen_ = ip.page.evaluate('() => window.__p2s_wheel__')
    return all(_seen_)


def _root_wheel_is_prevented(ip) -> bool:
    return ip.root.evaluate("""(rootEl) => {
        const ev = new WheelEvent('wheel', {deltaY: 120, bubbles: true, cancelable: true, composed: true});
        rootEl.dispatchEvent(ev);
        return ev.defaultPrevented;
    }""")


@pytest.mark.parametrize('fixture,half', VIEWS, ids=[_f_ for _f_, _ in VIEWS])
def test_a_wheel_over_the_canvas_does_not_scroll_the_page(request, fixture, half):
    assert _real_wheel_is_prevented(_view(request, fixture, half))


@pytest.mark.parametrize('fixture,half', VIEWS, ids=[_f_ for _f_, _ in VIEWS])
def test_a_wheel_anywhere_in_the_view_does_not_scroll_the_page(request, fixture, half):
    assert _root_wheel_is_prevented(_view(request, fixture, half))


def test_the_wheel_does_nothing_to_a_generic_view(xypi_page):
    """The generic views' wheel params are a placeholder: nothing watches them, so the
    drawing must not change."""
    _before_ = xypi_page.el('mod').inner_html()
    _box_    = xypi_page.root.bounding_box()
    xypi_page.wheel(_box_['width'] / 2, _box_['height'] / 2, -240)
    xypi_page.wait_until_idle()
    assert xypi_page.el('mod').inner_html() == _before_


if __name__ == '__main__':
    unittest.main()
