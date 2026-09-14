import re
import os
import io
import logging
import platform
import unittest
from typing import TYPE_CHECKING

if TYPE_CHECKING:                     # annotation only -- PIL stays a lazy import
    from PIL import Image

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), 'golden')


def capture_log_warnings(fn):
    """Run fn() and return log records emitted by the polars2svg logger."""
    logger  = logging.getLogger('polars2svg_logger')
    records = []
    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)
    handler = _Capture()
    logger.addHandler(handler)
    try:
        fn()
    finally:
        logger.removeHandler(handler)
    return records
GOLDEN_PNG_DIR = os.path.join(os.path.dirname(__file__), 'golden_png')

def assert_valid_svg(test_case, svg):
    """Assert that svg is a non-empty string containing an <svg> root element."""
    test_case.assertIn('<svg',   svg)
    test_case.assertIn('</svg>', svg)


def assert_timing_metrics_populated(test_case, obj, keys=('__parseInput__', '__renderSVG__')):
    """Assert that each key is present in obj.timing_metrics."""
    for key in keys:
        test_case.assertIn(key, obj.timing_metrics)


def assert_ordered_keys(test_case, actual, expected):
    """Assert that a sequence of ordering keys matches expected (converts to list for readable diffs)."""
    test_case.assertEqual(list(actual), list(expected))


def normalize_svg(svg):
    '''Canonicalize an SVG string so identical visual output always compares equal.

    Two sources of non-determinism are neutralized:

    1. Random IDs — the renderer embeds a random 32-bit integer (_randid_) in:
         - CSS class names : rect-group-{randid}, circle-group-{randid}
         - Clip path ID    : plotClip-{randid}
         - Gradient IDs    : lines_{randid}_...
         - LinkP <textPath> paths : p2sll{randid}_{n}
         - SpreadLinesP cloud symbols : cloud_{randid}, cloud_outline_{randid}
         - SpreadLinesP clip paths    : ccl_{randid}_{bin}
       These are replaced with the fixed token TESTID.  Only the random half is
       replaced in the LinkP and SpreadLinesP-clip cases: the trailing _{n} / _{bin}
       distinguishes the elements within one render and must survive.

    2. Dot element order — __renderDots__() uses group_by() whose row order is
       non-deterministic across runs.  The individual <rect> and <circle>
       elements inside the plot <g> are sorted lexicographically so the joined
       string is stable without requiring a sort in the production pipeline.

    LinkP renders deterministically (sorted() at source), so no element sorting
    is needed here -- but its 'curve' link labels do carry random <textPath> ids,
    which is why p2sll appears in the substitution above.
    '''
    # 1. Replace random IDs
    svg = re.sub(r'(plotClip-|lines_|smallp_|xyp_|histop_|timep_|chordp_|p2sll|ccl_'
                 r'|cloud_outline_|cloud_|(?:rect|circle)-group-)(\d+)', r'\1TESTID', svg)

    # 2. Sort dot elements within the plot group
    def _sort_plot_group_(m):
        elements = sorted(re.findall(r'<(?:rect|circle)\b[^>]*/>', m.group(2)))
        return m.group(1) + ''.join(elements) + m.group(3)

    svg = re.sub(
        r'(<g class="(?:rect|circle)-group-TESTID"[^>]*>)(.*?)(</g>)',
        _sort_plot_group_,
        svg,
        flags=re.DOTALL,
    )

    return svg

def assert_svg_matches_golden(svg, name):
    '''Compare a normalized SVG string to the stored golden file at
    tests/golden/<name>.svg.

    With UPDATE_GOLDEN=1 set in the environment the golden file is written or
    overwritten and the assertion is skipped.  Without it the normalized SVG must
    match the golden file exactly, and a missing golden is a failure -- it is
    never created for you.

    Workflow:
      - Normal run : SVG is compared to the golden; test fails on any diff.
      - Missing golden : the test FAILS.  It is not created for you -- see below.
      - New or intentionally changed golden : set UPDATE_GOLDEN=1 to write it.

    A missing golden used to be written silently and the test passed.  That is a
    comfortable default exactly once (the very first run) and a trap every time after:
    any checkout without the golden files re-baselines the whole suite against whatever
    that machine happens to produce, and reports green.  It bit a working copy whose
    goldens were absent, minting all 68 from a machine whose numeric backend differs from
    the one the goldens were built on.  Writing now takes the explicit flag.
    '''
    normalized = normalize_svg(svg)
    path = os.path.join(GOLDEN_DIR, name + '.svg')
    if os.environ.get('UPDATE_GOLDEN'):
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(path, 'w') as f:
            f.write(normalized)
        return
    assert os.path.exists(path), (
        f'No golden file for {name!r}: {path}\n'
        f'If this is a new case, create it with UPDATE_GOLDEN=1 and commit the file. '
        f'If it should already exist, your checkout is missing it -- do NOT regenerate, '
        f'because a fresh baseline records whatever this machine produces rather than '
        f'what the golden was meant to capture.'
    )
    with open(path) as f:
        golden = f.read()
    assert normalized == golden, (
        f'SVG output does not match golden file: {path}\n'
        f'Run with UPDATE_GOLDEN=1 to regenerate.'
    )


# The rasterizer now lives in the package (polars2svg.export) so it can back the
# public save()/savePNG() API; rasterize_svg() below routes onto it so there is a
# single implementation.  _fixSVGForRasterize_ used to be re-exported here under
# its historical name for tests that predated the move -- nothing imports it any
# more, so it went with the 2026-09-14 lint sweep.
from polars2svg.export import svgToPNGBytes


def rasterize_svg(svg):
    '''Render an SVG string to a PIL RGB Image using svglib + reportlab.'''
    from PIL import Image
    return Image.open(io.BytesIO(svgToPNGBytes(svg))).convert('RGB')




# --- Two thresholds, two different questions --------------------------------
#
# A PNG golden answers two questions that used to share one number, and sharing
# it is how 14 goldens went stale in production without a single test failing:
#
#   1. "Is the render still visually correct?"  -> VISUAL tolerance (5.0).
#      Generous on purpose: a genuine anti-aliasing shift should not fail a
#      suite over a change no human can see.
#
#   2. "Does this golden still correspond to what the code emits?" -> DRIFT
#      tolerance (0.0).  On the platform that generated it, with the rasterizer
#      that generated it, rasterization is deterministic: a current golden
#      re-renders BIT-IDENTICALLY.  That is not a tuned threshold, it is a
#      property -- all 72 goldens in this repo measure exactly 0.0.
#
# Any nonzero RMS therefore means the stored golden predates a rendering change.
# Under a single 5.0 gate that reads as "pass", and the drift is invisible until
# it silently eats the budget that question 1 was relying on.  Production sat at
# 3.37/5.0 (67% consumed) on smallp_small_wxh_single_slot_all_remainder from a
# 2026-08-05 font change, while still reporting green.
#
# The escape hatch is for a rasterizer upgrade (reportlab/svglib/Pillow are not
# pinned to an exact version).  If one lands, EVERY golden drifts at once, which
# is a legible signal rather than a mystery -- regenerate deliberately with
# UPDATE_GOLDEN=1, or set P2S_PNG_GOLDEN_DRIFT to triage before doing so.
GOLDEN_PNG_VISUAL_TOLERANCE = 5.0
GOLDEN_PNG_DRIFT_TOLERANCE  = 0.0


def goldenPngDriftTolerance() -> float:
    '''The RMS above which a stored PNG golden is considered stale.

    Defaults to GOLDEN_PNG_DRIFT_TOLERANCE; P2S_PNG_GOLDEN_DRIFT overrides it
    (see the note above -- that is for surviving a rasterizer upgrade, not for
    quieting a golden that genuinely needs regenerating).
    '''
    raw = os.environ.get('P2S_PNG_GOLDEN_DRIFT')
    if raw is None or raw.strip() == '':
        return GOLDEN_PNG_DRIFT_TOLERANCE
    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f'P2S_PNG_GOLDEN_DRIFT must be a float (RMS out of 255), got {raw!r}'
        )


def pngRMS(img_a: 'Image.Image', img_b: 'Image.Image') -> float:
    '''Root-mean-square difference between two PIL images, in levels out of 255.

    The single RMS implementation in the project: the golden assertion below,
    the cross-repo comparator (tools/compare_png_goldens.py) and their tests all
    route through it, so "how different are these two images" cannot acquire two
    subtly different answers.

    Raises ValueError on a size mismatch -- differently sized renders have no
    meaningful RMS, and silently reporting one would be worse than failing.
    '''
    import numpy as np
    from PIL import ImageChops
    a = img_a.convert('RGB')
    b = img_b.convert('RGB')
    if a.size != b.size:
        raise ValueError(f'image size mismatch: {a.size} vs {b.size}')
    diff = ImageChops.difference(a, b)
    return float(np.sqrt(np.mean(np.array(diff, dtype=float) ** 2)))


def pngRMSForFiles(path_a: str, path_b: str) -> float:
    '''pngRMS() for two PNG paths.  Raises ValueError on a size mismatch.'''
    from PIL import Image
    with Image.open(path_a) as a, Image.open(path_b) as b:
        return pngRMS(a, b)


def assert_image_matches_golden(svg, name, tolerance=None, drift_tolerance=None):
    '''Rasterize svg and compare against a stored PNG golden in tests/golden_png/.

    Comparison uses RMS pixel difference against TWO thresholds (see the note
    above this function for why one was not enough):

      - `tolerance` (default 5.0) -- a visual regression.  The render no longer
        looks like the golden.
      - `drift_tolerance` (default 0.0) -- a STALE golden.  The render is
        visually indistinguishable from the golden but no longer bit-identical
        to it, which means the golden predates a rendering change and was never
        regenerated.  Reported separately because it is a different problem with
        a different fix, and because letting it pass is what allowed production
        to accumulate 14 stale goldens unnoticed.

    Set drift_tolerance=tolerance to opt one call site out of staleness checking.

    Same UPDATE_GOLDEN=1 workflow as assert_svg_matches_golden.

    Platform note: the PNG goldens are rasterized bitmaps and are therefore
    tied to the rasterization backend that produced them. They were generated
    on macOS; on other platforms the backend (Linux uses rlPyCairo) produces
    minor anti-aliasing differences that exceed `tolerance` even when the SVG
    output is byte-identical. The cross-platform guarantee is the exact-string
    check in assert_svg_matches_golden (which passes everywhere); this bitmap
    check is a macOS-only belt-and-suspenders, so skip it off macOS (like the
    machine-local perf baseline). Set P2S_FORCE_PNG_GOLDEN=1 to run/regenerate
    it anyway (e.g. to rebuild the goldens on a new host) -- note that forcing
    it on a foreign rasterizer will report every golden as drifted, which is
    accurate: they are not that machine's goldens.
    '''
    if platform.system() != 'Darwin' and not os.environ.get('P2S_FORCE_PNG_GOLDEN'):
        raise unittest.SkipTest(
            'PNG-RMS golden comparison is macOS-only (rasterizer-specific); '
            'the exact-SVG-string golden covers cross-platform rendering. '
            'Set P2S_FORCE_PNG_GOLDEN=1 to force.'
        )
    if tolerance is None:
        tolerance = GOLDEN_PNG_VISUAL_TOLERANCE
    if drift_tolerance is None:
        drift_tolerance = goldenPngDriftTolerance()
    img  = rasterize_svg(svg)
    path = os.path.join(GOLDEN_PNG_DIR, name + '.png')
    if os.environ.get('UPDATE_GOLDEN'):
        os.makedirs(GOLDEN_PNG_DIR, exist_ok=True)
        img.save(path)
        return
    assert os.path.exists(path), (
        f'No golden PNG for {name!r}: {path}\n'
        f'If this is a new case, create it with UPDATE_GOLDEN=1 and commit the file. '
        f'If it should already exist, your checkout is missing it -- do NOT regenerate.'
    )
    from PIL import Image
    ref = Image.open(path).convert('RGB')
    try:
        rms = pngRMS(img, ref)
    except ValueError as e:
        raise AssertionError(
            f'Rendered image {name!r} does not match the golden PNG geometry: {e}\n'
            f'Golden: {path}\n'
            f'A size change is always a real rendering change -- regenerate with '
            f'UPDATE_GOLDEN=1 only if it is intended.'
        )
    assert rms <= tolerance, (
        f'Rendered image {name!r} differs from golden PNG: '
        f'RMS={rms:.4f} > tolerance={tolerance}\n'
        f'Golden: {path}\n'
        f'This is a VISUAL regression, not just drift.\n'
        f'Run with UPDATE_GOLDEN=1 to regenerate.'
    )
    assert rms <= drift_tolerance, (
        f'Golden PNG {name!r} is STALE: RMS={rms:.4f} > drift tolerance='
        f'{drift_tolerance} (visual tolerance={tolerance}, '
        f'{100.0 * rms / tolerance:.1f}% consumed).\n'
        f'Golden: {path}\n'
        f'The render is visually indistinguishable from the golden but no longer '
        f'identical to it, so the stored file predates a rendering change and was '
        f'never regenerated. It is passing on borrowed tolerance and hides the next, '
        f'real regression.\n'
        f'Fix: regenerate with UPDATE_GOLDEN=1 and commit the PNG.\n'
        f'If instead the rasterizer was just upgraded, every golden will report this '
        f'at once -- regenerate them all, or set P2S_PNG_GOLDEN_DRIFT to triage.'
    )
