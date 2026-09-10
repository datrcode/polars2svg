"""The GPU render path -- PLANNING.md **V2**: "No WGSL ever executes under pytest."

`tests/webgpu_test_utils.py` decodes the real float32 instance buffers and
parity-checks them against geometry parsed from the SVG.  That is genuine numeric
verification -- but of the *display list*, not of the picture it produces.  Nothing in
the Python suite has ever compiled a shader, and the plan names Playwright as the
intended fix.  These are the first tests to run one.

Two practical findings shape the file, both measured rather than assumed:

* **The default browser cannot do it.** Playwright's bundled
  `chromium-headless-shell` exposes `navigator.gpu` but `requestAdapter()` returns
  null; every render would fail and every lenient test would pass. The `chromium`
  channel -- already installed alongside it -- gets a real device. And
  `navigator.gpu` is absent entirely outside a secure context, so a probe on
  about:blank reports "no WebGPU" on a browser that has it.
* **A WebGPU canvas cannot be read back with `drawImage`.** Copying one into a 2D
  canvas yields fully transparent pixels even when the canvas is visibly drawn --
  0 opaque pixels against a screenshot of the same element showing 189 colours. So
  the pixels come from an element screenshot instead.

The assertions stop short of image goldens on purpose, for the reason
`tests/svg_test_utils.py` already skips the PNG-RMS ones off macOS: font metrics and
antialiasing make exact images a per-platform liability. What is asserted instead is
that ink lands *where the SVG says it should*, which is the divergence V2 is actually
about.
"""
import io
import math
import unittest

import pytest

pytest.importorskip('PIL', reason='pixel assertions need Pillow')
from PIL import Image                                    # noqa: E402

#: Components with a webgpu() display list.  chordp/piep/smallp/spreadlinesp are built
#: by their own fixtures below; these three share the grid dataframe.
BACKGROUND_TOLERANCE = 12


def _canvas_image(canvas):
    return Image.open(io.BytesIO(canvas.screenshot())).convert('RGBA')


def _ink_ratio(img):
    """Fraction of pixels that differ from the most common (background) colour."""
    _colours_ = img.getcolors(maxcolors=1 << 20) or []
    if not _colours_:
        return 0.0
    _total_ = sum(_n_ for _n_, _c_ in _colours_)
    _bg_ = max(_colours_)[1]
    _ink_ = sum(_n_ for _n_, _c_ in _colours_
                if max(abs(_a_ - _b_) for _a_, _b_ in zip(_c_, _bg_)) > BACKGROUND_TOLERANCE)
    return _ink_ / _total_


# ── the GPU path runs at all ─────────────────────────────────────────────────

def test_the_gpu_path_reports_no_error(webgpu_page, gpu_xyp):
    """`gpu_error` is set by the JS on *any* failure from the runtime, shader
    compilation included -- so an empty one is the plainest statement that the WGSL
    compiled and ran."""
    _page_, _view_, _canvas_ = webgpu_page(gpu_xyp)
    assert _view_.gpu_error == '', f'the GPU runtime reported: {_view_.gpu_error!r}'


def test_gpu_mode_does_not_fall_back_to_svg(webgpu_page, gpu_xyp):
    """There is deliberately no automatic SVG fallback: in GPU mode `mod_inner` stays
    empty, and a failure shows an error overlay instead.  If this ever fills in, a
    'GPU' test could be silently checking the SVG renderer."""
    _page_, _view_, _canvas_ = webgpu_page(gpu_xyp)
    assert _view_.mod_inner == '', 'GPU mode fell back to the SVG renderer'


def test_the_canvas_is_actually_drawn_on(webgpu_page, gpu_xyp):
    """Ink on the canvas, not merely a canvas element of the right size."""
    _page_, _view_, _canvas_ = webgpu_page(gpu_xyp)
    _ratio_ = _ink_ratio(_canvas_image(_canvas_))
    assert _ratio_ > 0.005, f'the GPU canvas is effectively blank (ink ratio {_ratio_:.4f})'


@pytest.mark.parametrize('fixture', ['gpu_xyp', 'gpu_timep', 'gpu_histop',
                                     'gpu_piep', 'gpu_chordp', 'gpu_linkp'])
def test_every_component_renders_on_the_gpu(request, webgpu_page, fixture):
    """One shader run per component -- the sweep V2 asks for.

    Each of these has a `webgpu()` display list that until now was only ever decoded
    in Python.  A shader that fails to compile for any one of them sets `gpu_error`;
    one that compiles but draws nothing leaves the canvas blank.
    """
    _component_ = request.getfixturevalue(fixture)
    _page_, _view_, _canvas_ = webgpu_page(_component_)

    assert _view_.gpu_error == '', f'{fixture}: {_view_.gpu_error!r}'
    _ratio_ = _ink_ratio(_canvas_image(_canvas_))
    assert _ratio_ > 0.002, f'{fixture}: the GPU canvas is blank (ink ratio {_ratio_:.4f})'


# ── the parity V2 is really about ────────────────────────────────────────────

def test_the_gpu_draws_marks_where_the_svg_puts_them(webgpu_page, gpu_xyp):
    """Ink lands at the coordinates the SVG renderer independently computes.

    This is the divergence V2 names: the display list is verified numerically, but
    nothing checks that the picture agrees with it.  The SVG render is the reference --
    it is what the rest of the suite already trusts -- and each of its mark centres is
    sampled on the GPU canvas.  Not an image comparison: no golden, nothing that
    thrashes on font metrics, just "something was painted here".
    """
    _marks_ = _svg_mark_centres(gpu_xyp)
    assert len(_marks_) >= 3, f'the SVG reference drew too few marks to test ({len(_marks_)})'

    _page_, _view_, _canvas_ = webgpu_page(gpu_xyp)
    _img_ = _canvas_image(_canvas_)
    _bg_ = max(_img_.getcolors(maxcolors=1 << 20))[1]

    _hits_ = sum(1 for _x_, _y_ in _marks_ if _painted_near(_img_, _x_, _y_, _bg_))
    assert _hits_ == len(_marks_), (
        f'the GPU painted at only {_hits_} of the {len(_marks_)} interior positions '
        f'the SVG renderer computed -- the picture and the display list disagree')


def _painted_near(img, x, y, bg, radius=4):
    """Is anything non-background within a few pixels of (x, y)?

    A small radius rather than an exact hit: the two renderers round and antialias
    differently, and V2 is about marks being in the right *place*, not about them
    being pixel-identical -- which is the assertion that would make this a golden.
    """
    _w_, _h_ = img.size
    for _dy_ in range(-radius, radius + 1):
        for _dx_ in range(-radius, radius + 1):
            _px_, _py_ = int(x) + _dx_, int(y) + _dy_
            if 0 <= _px_ < _w_ and 0 <= _py_ < _h_:
                _c_ = img.getpixel((_px_, _py_))
                if max(abs(_a_ - _b_) for _a_, _b_ in zip(_c_, bg)) > BACKGROUND_TOLERANCE:
                    return True
    return False


def _svg_mark_centres(component, edge_margin=12):
    """Centres of the data marks in the component's own SVG render.

    xyp emits each dot as ``<rect x=".." y=".." fill=".."/>`` with **no width or
    height** -- those come from a ``<style>`` rule on the enclosing group
    (``width: 1px; height: 1px``).  So a mark is precisely a rect that carries a
    position and no dimensions; every piece of chrome (background, plot frame, clip
    rect, border) states its size explicitly.  Filtering on "small" instead finds
    nothing at all, which is how the first draft of this returned zero marks.

    Marks within ``edge_margin`` of the canvas edge are dropped: at 1px they sit right
    on the axis furniture, and whether a renderer clips them is not what this is
    testing.
    """
    import re
    _svg_ = component._repr_svg_()
    _w_, _h_ = component.wxh
    _out_ = []
    for _m_ in re.finditer(r'<rect\b[^>]*/?>', _svg_):
        _attrs_ = dict(re.findall(r'(\w[\w-]*)="([^"]*)"', _m_.group(0)))
        if 'x' not in _attrs_ or 'y' not in _attrs_:
            continue
        if 'width' in _attrs_ or 'height' in _attrs_:
            continue                                   # chrome always states its size
        try:
            _x_, _y_ = float(_attrs_['x']), float(_attrs_['y'])
        except ValueError:
            continue
        if edge_margin <= _x_ <= _w_ - edge_margin and edge_margin <= _y_ <= _h_ - edge_margin:
            _out_.append((_x_, _y_))
    return sorted(set(_out_))


# ── V2's specific claims ─────────────────────────────────────────────────────
#
# The sweep above proves shaders run.  These go after the individual concerns the item
# names, which are about the *picture* rather than the buffer.
#
# Coordinates need care here in a way they did not for xy.  spreadlinesp emits a
# **viewBox** (`48.5 100 316.5 116` on a 600x320 canvas), so its internal coordinates
# are not canvas coordinates: they are scaled and centred by the default
# `preserveAspectRatio="xMidYMid meet"`.  Sampling the raw numbers hits arbitrary
# pixels -- and passes, by chance, which is how the first draft of this file managed to
# "verify" the dash pattern at points nowhere near the dashed line.  `_to_canvas()`
# below does the mapping, and `test_spreadlines_honours_the_viewbox` pins that the GPU
# applies the same one.


def _to_canvas(component, canvas_w, canvas_h):
    """viewBox coordinates -> canvas pixels, under `xMidYMid meet` (the SVG default).

    Returns identity when the component emits no viewBox, which is the xy case.
    """
    import re
    _m_ = re.search(r'viewBox="([-\d.\s]+)"', component._repr_svg_())
    if _m_ is None:
        return lambda _x_, _y_: (_x_, _y_)
    _vx_, _vy_, _vw_, _vh_ = (float(_v_) for _v_ in _m_.group(1).split())
    _s_ = min(canvas_w / _vw_, canvas_h / _vh_)
    _tx_, _ty_ = (canvas_w - _vw_ * _s_) / 2, (canvas_h - _vh_ * _s_) / 2
    return lambda _x_, _y_: ((_x_ - _vx_) * _s_ + _tx_, (_y_ - _vy_) * _s_ + _ty_)


def test_spreadlines_draws_its_nodes_where_the_svg_puts_them(webgpu_page, gpu_spreadlines):
    """V2's third claim: the 2026-08-06 `spreadlinesp` instrumentation -- the largest
    single change ever made to this half of the package -- was verified against decoded
    buffers with no shader in the loop.

    Its nodes are circles at known viewBox centres, so the same parity check the xy path
    gets applies once those are mapped onto the canvas.
    """
    _page_, _view_, _canvas_ = webgpu_page(gpu_spreadlines)
    _img_ = _canvas_image(_canvas_)
    _map_ = _to_canvas(gpu_spreadlines, *_img_.size)
    _centres_ = [_map_(_x_, _y_) for _x_, _y_ in _svg_circle_centres(gpu_spreadlines)]
    assert len(_centres_) >= 4, f'too few nodes to test ({len(_centres_)})'

    _bg_ = max(_img_.getcolors(maxcolors=1 << 20))[1]
    _hits_ = sum(1 for _x_, _y_ in _centres_ if _painted_near(_img_, _x_, _y_, _bg_))
    assert _hits_ == len(_centres_), (
        f'the GPU painted {_hits_} of the {len(_centres_)} node positions the SVG '
        f'renderer computed')


def test_spreadlines_honours_the_viewbox(webgpu_page, gpu_spreadlines):
    """The GPU applies the same `xMidYMid meet` mapping the SVG viewBox implies.

    Worth its own test because it is a whole-picture property that a per-primitive
    buffer check cannot see: every instance could be correct and the picture still be
    stretched or off-centre.  Measured by contradiction -- under the wrong (stretch)
    mapping the node centres miss almost everything, and 9 of 9 dashed-path points land
    on ink under the right one against 2 of 9 under the wrong one.
    """
    _page_, _view_, _canvas_ = webgpu_page(gpu_spreadlines)
    _img_ = _canvas_image(_canvas_)
    _w_, _h_ = _img_.size
    _bg_ = max(_img_.getcolors(maxcolors=1 << 20))[1]

    import re
    _vb_ = re.search(r'viewBox="([-\d.\s]+)"', gpu_spreadlines._repr_svg_())
    assert _vb_ is not None, 'this fixture no longer emits a viewBox; the test is moot'
    _vx_, _vy_, _vw_, _vh_ = (float(_v_) for _v_ in _vb_.group(1).split())

    # Sampled along the dashed path, not at the node centres: the nodes sit near the
    # vertical middle, where the two mappings differ by about two pixels and a
    # tolerant hit test cannot tell them apart (measured: 8 hits either way).  The
    # dashed line spans the full height, where they diverge by ~50px.
    _probe_ = _dashed_path_points(gpu_spreadlines)
    assert len(_probe_) >= 5, 'no dashed path to probe the mapping with'
    _meet_ = _to_canvas(gpu_spreadlines, _w_, _h_)
    _stretch_ = lambda _x_, _y_: ((_x_ - _vx_) * _w_ / _vw_, (_y_ - _vy_) * _h_ / _vh_)

    _meet_hits_ = sum(1 for _c_ in _probe_
                      if _painted_near(_img_, *_meet_(*_c_), _bg_, radius=3))
    _stretch_hits_ = sum(1 for _c_ in _probe_
                         if _painted_near(_img_, *_stretch_(*_c_), _bg_, radius=3))
    assert _meet_hits_ > _stretch_hits_, (
        f'the GPU picture fits a non-uniform stretch ({_stretch_hits_} of {len(_probe_)} '
        f'points on ink) at least as well as the uniform meet the viewBox asks for '
        f'({_meet_hits_}) -- the aspect handling has changed')


def test_the_dashed_context_line_renders_as_dashes(webgpu_page, gpu_spreadlines):
    """The dash shader runs on the *real* component: the path is drawn, and drawn broken.

    A deliberately weak claim, and it stays weak on purpose.  V2's second item -- is the
    phase carried across the joins? -- is now asserted properly by
    `test_the_dash_phase_runs_across_the_joins_of_a_polyline`, on a purpose-built fixture
    whose geometry is *chosen* for it.  This one keeps `spreadlinesp`'s own context
    zigzag in the picture, and what it buys is that a shipping component's dashed path
    reaches the shader at all -- with the rest of the plot drawn around it, where the
    purpose-built fixture deliberately has nothing else on the canvas.

    Why not measure the phase here too: the numbers are whatever the component happens
    to produce rather than anything picked, and they would drift silently with its
    styling.  Measured 2026-09-10: 8 dashed segments (against 24), 42.5px each, and a
    period of 11.37 with **equal** on and off -- which is exactly the case the other
    test refuses, because an equal split narrows the gap between the carried and reset
    hypotheses that the assertion depends on.

    Two oracles were tried here first and both are dead ends worth not re-attempting.
    Counting dashes does not discriminate (continuous predicts ~30 along this path,
    per-segment ~32).  Rasterising the SVG as a reference gave 9 runs where the GPU gave
    55 on the same path -- svglib and Chromium disagree about dash rendering far more
    than the two phase behaviours differ from each other.
    """
    _page_, _view_, _canvas_ = webgpu_page(gpu_spreadlines)
    _img_ = _canvas_image(_canvas_)
    _map_ = _to_canvas(gpu_spreadlines, *_img_.size)
    _pts_ = [_map_(_x_, _y_) for _x_, _y_ in _dashed_path_points(gpu_spreadlines)]
    assert len(_pts_) >= 5, f'the dashed context line has too few vertices ({len(_pts_)})'

    _bg_ = max(_img_.getcolors(maxcolors=1 << 20))[1]
    _samples_ = _walk_path(_img_, _pts_, _bg_)
    assert any(_samples_), 'the dashed context line did not render at all'
    assert not all(_samples_), 'the dashed context line rendered solid -- no dashing'
    _runs_ = sum(1 for _i_ in range(1, len(_samples_)) if _samples_[_i_] and not _samples_[_i_ - 1])
    assert _runs_ >= 5, f'only {_runs_} dash runs along the path; it is barely broken'


def _ink_across(img, x, y, nx, ny, bg, reach=3):
    """Is the line inked at (x, y)?  Sampled **across** the line, never along it.

    The distinction is the whole reason this exists rather than `_painted_near`.  A
    square neighbourhood of radius r reaches r pixels *along* the path as well as
    across it, which blurs the dash boundary the sample is trying to resolve -- and at
    a join it reaches the end cap, which paints regardless of the dash phase.  Walking
    the normal keeps the sample sharp in arc length while still tolerating a thin,
    antialiased, sub-pixel-positioned line.
    """
    _w_, _h_ = img.size
    for _k_ in range(-reach, reach + 1):
        _px_, _py_ = int(round(x + nx * _k_)), int(round(y + ny * _k_))
        if 0 <= _px_ < _w_ and 0 <= _py_ < _h_:
            _c_ = img.getpixel((_px_, _py_))
            if max(abs(_a_ - _b_) for _a_, _b_ in zip(_c_, bg)) > BACKGROUND_TOLERANCE:
                return True
    return False


def _dashed_segments(component):
    """(x0, y0, x1, y1, dash_phase, on, off) for every dashed line the GPU is given.

    Straight off the display list -- the same instance buffer the shader is handed --
    so the test compares the picture against what was actually submitted rather than
    against a re-derivation of it.
    """
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from webgpu_test_utils import decode_buffer            # tests/webgpu_test_utils.py

    _X0_, _Y0_, _X1_, _Y1_, _ON_, _OFF_, _PH_ = 0, 1, 2, 3, 9, 10, 11
    return [(float(_r_[_X0_]), float(_r_[_Y0_]), float(_r_[_X1_]), float(_r_[_Y1_]),
             float(_r_[_PH_]), float(_r_[_ON_]), float(_r_[_OFF_]))
            for _r_ in decode_buffer(component.webgpu(), 'line') if float(_r_[_ON_]) > 0.0]


@pytest.mark.parametrize('fixture_name,implementation', [
    ('gpu_dashed_polyline',   "xyp's polars cum_sum (xyp.py ~3092)"),
    ('gpu_dashed_background', 'strokePolylineDL (p2s_displaylist.py)'),
])
def test_the_dash_phase_runs_across_the_joins_of_a_polyline(request, webgpu_page,
                                                            fixture_name, implementation):
    """V2's second claim, finally asserted: the dash pattern is continuous across joins.

    Run against **both** dash implementations, which share no code -- `xyp` accumulates
    with a polars `cum_sum`, everything path-shaped uses `strokePolylineDL`.  Verified
    by mutation that a test of one says nothing about the other.

    `strokePolylineDL` hands each segment the running arc length as its `dash_phase`,
    so a flattened polyline dashes as one path rather than restarting at every vertex.
    The display-list half of that is covered in Python (`TestDashPhaseContinuity`).
    What had never been checked is whether the **shader honours the field** -- and a
    shader that ignored it would draw a perfectly plausible dashed line, which is why
    the weaker "it draws, and draws broken" assertion next door cannot see it.

    **The oracle is a correlation, and it calibrates itself.**  Every sample carries the
    phase the display list declares for that point; ink is expected inside the `on`
    span and absent inside the `off` span.  The same pixels are then scored a second
    time against the *reset* hypothesis -- phase measured from each segment's own start
    -- and the test asserts the first correlation is overwhelming and the second is
    noise.  That second half is what makes a green here mean something: if the fixture
    geometry ever made the two hypotheses agree, the contrast gap would collapse and
    this would fail rather than passing vacuously.

    Measured on macOS/Metal, 2026-09-10: carried +1.000 (deep-on 1.000, deep-off 0.000
    over 425 samples); reset +0.011 (0.506 / 0.495).  A hundredfold separation.

    Two things the earlier attempt got wrong, both worth keeping:

    * **Sampling at the vertices cannot work, and not because the phase is wrong.**
      That was the plan recorded in V2 -- with the phase carried, roughly half the
      vertices should fall in a gap.  They do, and they are inked anyway: each segment
      is drawn with end caps, so a join paints whatever the pattern says.  The vertex
      samples were the *only* ink in the `off` buckets.  Hence JOIN_SKIP.
    * **A square sampling neighbourhood blurs the thing being measured.**  See
      `_ink_across`.
    """
    #: Pixels of each segment discarded at either end -- see the docstring.  Measured:
    #: cap ink reaches ~1px past the vertex, and the neighbouring arc position is clean.
    _JOIN_SKIP_ = 2
    #: Keeps samples clear of the dash boundaries themselves, where antialiasing makes
    #: "inked" a genuinely fuzzy question rather than a wrong answer.
    _MARGIN_ = 3.0

    _component_ = request.getfixturevalue(fixture_name)
    _page_, _view_, _canvas_ = webgpu_page(_component_)
    _segs_ = _dashed_segments(_component_)
    assert len(_segs_) >= 20, f'the fixture should give a long polyline; got {len(_segs_)} segments'

    _on_, _off_ = _segs_[0][5], _segs_[0][6]
    _period_ = _on_ + _off_
    assert _on_ != _off_, 'an equal on/off would weaken the reset/carried contrast'

    # Precondition, checked before a single pixel is read, because it fails first and
    # says why.  If the phases are all equal the two hypotheses below are the *same
    # hypothesis*, and the correlation would be perfect against both -- a green that
    # meant nothing.  All-zero here is the actual regression: the accumulation gone.
    _declared_ = {round(_s_[4] % _period_, 1) for _s_ in _segs_}
    assert len(_declared_) >= 4, (
        f'the display list gave {len(_segs_)} segments only {len(_declared_)} distinct '
        f'dash phases ({sorted(_declared_)}) -- the arc length is not being accumulated, '
        f'so this test cannot tell a carried phase from a reset one.  That is a Python-'
        f'side defect: see TestDashPhaseContinuity in tests/test_webgpu_components.py.')

    _img_ = _canvas_image(_canvas_)
    _bg_ = max(_img_.getcolors(maxcolors=1 << 20))[1]

    _samples_ = []                       # (phase_if_carried, phase_if_reset, inked)
    for _x0_, _y0_, _x1_, _y1_, _ph0_, _, _ in _segs_:
        _len_ = math.hypot(_x1_ - _x0_, _y1_ - _y0_)
        if _len_ < 4 * _JOIN_SKIP_:
            continue
        _ux_, _uy_ = (_x1_ - _x0_) / _len_, (_y1_ - _y0_) / _len_
        for _k_ in range(_JOIN_SKIP_, int(_len_) - _JOIN_SKIP_ + 1):
            _inked_ = _ink_across(_img_, _x0_ + _ux_ * _k_, _y0_ + _uy_ * _k_,
                                  -_uy_, _ux_, _bg_)
            _samples_.append(((_ph0_ + _k_) % _period_, _k_ % _period_, _inked_))

    assert len(_samples_) >= 200, f'too few usable samples ({len(_samples_)}) to correlate'

    def _contrast_(_idx_):
        _in_on_  = [_s_[2] for _s_ in _samples_ if _MARGIN_ < _s_[_idx_] < _on_ - _MARGIN_]
        _in_off_ = [_s_[2] for _s_ in _samples_ if _on_ + _MARGIN_ < _s_[_idx_] < _period_ - _MARGIN_]
        assert _in_on_ and _in_off_, 'the fixture produced no samples on one side of the pattern'
        _f_on_  = sum(_in_on_) / len(_in_on_)
        _f_off_ = sum(_in_off_) / len(_in_off_)
        return _f_on_, _f_off_, _f_on_ - _f_off_

    _carried_on_, _carried_off_, _carried_ = _contrast_(0)
    _reset_on_, _reset_off_, _reset_ = _contrast_(1)

    assert _carried_on_ >= 0.95, (
        f'only {_carried_on_:.3f} of the samples the display list marks as inside a dash '
        f'are actually inked -- the dashes are not landing where the phase says '
        f'({implementation})')
    assert _carried_off_ <= 0.05, (
        f'{_carried_off_:.3f} of the samples the display list marks as inside a gap are '
        f'inked -- the shader is not using dash_phase, or not the way the display list '
        f'means it ({implementation})')
    assert _carried_ - _reset_ >= 0.5, (
        f'the pixels do not distinguish the two hypotheses: carried={_carried_:+.3f} '
        f'vs reset={_reset_:+.3f}, so a pass here would prove nothing.  Either the '
        f'fixture geometry has drifted until the two coincide -- check that the segment '
        f'length is still coprime-ish with the dash period, see the fixture docstring -- '
        f'or the sampling has stopped resolving the pattern.')


def _walk_path(img, pts, bg, per_px=2):
    """Ink / no-ink samples stepped along a polyline in canvas coordinates."""
    import math
    _out_ = []
    for _i_ in range(len(pts) - 1):
        _a_, _b_ = pts[_i_], pts[_i_ + 1]
        _n_ = max(2, int(math.dist(_a_, _b_) * per_px))
        for _k_ in range(_n_):
            _t_ = _k_ / _n_
            _out_.append(_painted_near(img, _a_[0] + (_b_[0] - _a_[0]) * _t_,
                                       _a_[1] + (_b_[1] - _a_[1]) * _t_, bg, radius=0))
    return _out_


def test_link_labels_are_drawn_as_straight_runs_along_their_edges(webgpu_page,
                                                                  gpu_link_labels):
    """V2's first claim: the GPU approximates `<textPath>` as a straight run along the
    curve's midpoint tangent -- "deliberate, never looked at side by side".

    Looked at now, and the approximation holds: each label's ink forms one blob near
    its edge's midpoint, with a principal axis within a few degrees of that edge's
    chord (measured -51.3 / 50.1 / 0.6 against chords of -51.5 / 49.2 / -3.1).  The
    three chords are deliberately far apart, so matching all three is not something a
    mis-rotated or unrotated label could do.

    What is *not* asserted is that the baseline is straight rather than curved, which
    is the other half of the approximation.  It is not measurable here: the labels span
    35-41px and the glyphs are ~10px tall, so perpendicular residuals (~4.3px) are
    dominated by cap height and would look the same either way.  A curve steep enough
    to separate them would need a much longer label -- noted in PLANNING.md V2.

    Worth recording alongside this: the SVG twin rasterises with **no labels at all**,
    because svglib does not implement `<textPath>`.  The repo's PNG-RMS goldens go
    through svglib, so link labels have never been covered by an image test in either
    renderer; this is the first.
    """
    _page_, _view_, _canvas_ = webgpu_page(gpu_link_labels)
    _img_ = _canvas_image(_canvas_)
    _chords_ = _edge_chords(gpu_link_labels)
    assert len(_chords_) == 3, f'expected three edges, found {len(_chords_)}'

    _clusters_ = _text_clusters(_img_)
    assert len(_clusters_) == len(_chords_), (
        f'found {len(_clusters_)} label blobs for {len(_chords_)} labelled edges; '
        f'the GPU may not be drawing link labels at all')

    for _cx_, _cy_, _angle_, _n_ in _clusters_:
        _best_ = min(_chords_, key=lambda _c_: math.hypot(_c_[0] - _cx_, _c_[1] - _cy_))
        _delta_ = abs((_angle_ - _best_[2] + 90) % 180 - 90)
        assert _delta_ <= 12, (
            f'a label near ({_cx_:.0f}, {_cy_:.0f}) is drawn at {_angle_:.1f} deg while '
            f'its edge runs at {_best_[2]:.1f} deg -- labels are not following their edges')
        assert math.hypot(_best_[0] - _cx_, _best_[1] - _cy_) < 60, (
            'a label blob is nowhere near the midpoint of any edge')


def _edge_chords(component):
    """(mid_x, mid_y, angle) for every pair of nodes, in canvas coordinates.

    Every pair rather than only the real edges: the component draws one label per edge
    and this fixture's graph is a triangle, so the two sets coincide -- and deriving
    pairs from the drawn circles avoids reaching into the component's internals.
    """
    _cs_ = _svg_circle_centres(component)
    _out_ = []
    for _i_ in range(len(_cs_)):
        for _j_ in range(_i_ + 1, len(_cs_)):
            (_x0_, _y0_), (_x1_, _y1_) = _cs_[_i_], _cs_[_j_]
            _ang_ = (math.degrees(math.atan2(_y1_ - _y0_, _x1_ - _x0_)) + 90) % 180 - 90
            _out_.append(((_x0_ + _x1_) / 2, (_y0_ + _y1_) / 2, _ang_))
    return _out_


def _text_clusters(img, min_pixels=40, gap=4):
    """Blobs of dark (text) pixels, each as (cx, cy, principal-axis angle, n).

    Text is near-neutral dark; the links are saturated blue, so a low-saturation dark
    filter separates them without needing to know either colour.
    """
    _w_, _h_ = img.size
    _pts_ = set()
    for _y_ in range(20, _h_ - 30):            # skip the border and the info line
        for _x_ in range(20, _w_ - 20):
            _r_, _g_, _b_ = img.getpixel((_x_, _y_))[:3]
            if _r_ < 140 and _g_ < 140 and _b_ < 140 and abs(_r_ - _b_) < 60:
                _pts_.add((_x_, _y_))

    _out_ = []
    while _pts_:
        _seed_ = _pts_.pop()
        _comp_, _stack_ = [_seed_], [_seed_]
        while _stack_:
            _x_, _y_ = _stack_.pop()
            for _dx_ in range(-gap, gap + 1):
                for _dy_ in range(-gap, gap + 1):
                    _q_ = (_x_ + _dx_, _y_ + _dy_)
                    if _q_ in _pts_:
                        _pts_.discard(_q_)
                        _comp_.append(_q_)
                        _stack_.append(_q_)
        if len(_comp_) < min_pixels:
            continue
        _n_ = len(_comp_)
        _mx_ = sum(_p_[0] for _p_ in _comp_) / _n_
        _my_ = sum(_p_[1] for _p_ in _comp_) / _n_
        _sxx_ = sum((_p_[0] - _mx_) ** 2 for _p_ in _comp_)
        _syy_ = sum((_p_[1] - _my_) ** 2 for _p_ in _comp_)
        _sxy_ = sum((_p_[0] - _mx_) * (_p_[1] - _my_) for _p_ in _comp_)
        _theta_ = math.degrees(0.5 * math.atan2(2 * _sxy_, _sxx_ - _syy_))
        _out_.append((_mx_, _my_, _theta_, _n_))
    return _out_


# ── geometry helpers: the SVG as a source of coordinates, never of pixels ────

def _dashed_path_points(component):
    """Vertices of the dashed context path, in the SVG's own (viewBox) coordinates."""
    import re
    _m_ = re.search(r'<path[^>]*stroke-dasharray="[^"]*"[^>]*>', component._repr_svg_())
    if _m_ is None:
        return []
    _d_ = re.search(r'd="([^"]+)"', _m_.group(0))
    if _d_ is None:
        return []
    _toks_ = _d_.group(1).replace(',', ' ').split()
    _pts_, _i_ = [], 0
    while _i_ < len(_toks_):
        if _toks_[_i_] in ('M', 'L'):
            _pts_.append((float(_toks_[_i_ + 1]), float(_toks_[_i_ + 2])))
            _i_ += 3
        else:
            try:
                _pts_.append((float(_toks_[_i_]), float(_toks_[_i_ + 1])))
                _i_ += 2
            except (ValueError, IndexError):
                _i_ += 1                       # a curve/arc command: not a plain vertex
    return _pts_


def _svg_circle_centres(component):
    """Circle centres in the SVG's own coordinates (map them before sampling)."""
    import re
    _out_ = []
    for _m_ in re.finditer(r'<circle\b[^>]*>', component._repr_svg_()):
        _a_ = dict(re.findall(r'(\w[\w-]*)="([^"]*)"', _m_.group(0)))
        try:
            _out_.append((float(_a_['cx']), float(_a_['cy'])))
        except (KeyError, ValueError):
            continue
    return sorted(set(_out_))


if __name__ == '__main__':
    unittest.main()
