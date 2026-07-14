import re
import os
import io
import logging
import platform
import unittest

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
       These are replaced with the fixed token TESTID.

    2. Dot element order — __renderDots__() uses group_by() whose row order is
       non-deterministic across runs.  The individual <rect> and <circle>
       elements inside the plot <g> are sorted lexicographically so the joined
       string is stable without requiring a sort in the production pipeline.

    LinkP renders deterministically (sorted() at source), so no element sorting
    is needed here.
    '''
    # 1. Replace random IDs
    svg = re.sub(r'(plotClip-|lines_|smallp_|xyp_|histop_|timep_|chordp_|(?:rect|circle)-group-)(\d+)', r'\1TESTID', svg)

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

    On the first run (or when UPDATE_GOLDEN=1 is set in the environment) the
    golden file is written/overwritten and the assertion is skipped.  On
    subsequent runs the normalized SVG must match the golden file exactly.

    Workflow:
      - First run  : golden files are created automatically; test passes.
      - Normal run : SVG is compared to the golden; test fails on any diff.
      - Intentional change : set UPDATE_GOLDEN=1 to regenerate the goldens.
    '''
    normalized = normalize_svg(svg)
    path = os.path.join(GOLDEN_DIR, name + '.svg')
    if os.environ.get('UPDATE_GOLDEN') or not os.path.exists(path):
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(path, 'w') as f:
            f.write(normalized)
        return
    with open(path) as f:
        golden = f.read()
    assert normalized == golden, (
        f'SVG output does not match golden file: {path}\n'
        f'Run with UPDATE_GOLDEN=1 to regenerate.'
    )


# The rasterizer now lives in the package (polars2svg.export) so it can back the
# public save()/savePNG() API.  Re-exported here under the historical names so
# existing tests keep working through a single implementation.
from polars2svg.export import _fixSVGForRasterize_ as _fix_svg_for_rasterize
from polars2svg.export import svgToPNGBytes


def rasterize_svg(svg):
    '''Render an SVG string to a PIL RGB Image using svglib + reportlab.'''
    from PIL import Image
    return Image.open(io.BytesIO(svgToPNGBytes(svg))).convert('RGB')


def assert_image_matches_golden(svg, name, tolerance=5.0):
    '''Rasterize svg and compare against a stored PNG golden in tests/golden_png/.

    Comparison uses RMS pixel difference; values up to `tolerance` (out of 255)
    are accepted to allow for minor antialiasing variation.  Set tolerance=0 for
    a pixel-exact check.

    Same UPDATE_GOLDEN=1 workflow as assert_svg_matches_golden.

    Platform note: the PNG goldens are rasterized bitmaps and are therefore
    tied to the rasterization backend that produced them. They were generated
    on macOS; on other platforms the backend (Linux uses rlPyCairo) produces
    minor anti-aliasing differences that exceed `tolerance` even when the SVG
    output is byte-identical. The cross-platform guarantee is the exact-string
    check in assert_svg_matches_golden (which passes everywhere); this bitmap
    check is a macOS-only belt-and-suspenders, so skip it off macOS (like the
    machine-local perf baseline). Set P2S_FORCE_PNG_GOLDEN=1 to run/regenerate
    it anyway (e.g. to rebuild the goldens on a new host).
    '''
    if platform.system() != 'Darwin' and not os.environ.get('P2S_FORCE_PNG_GOLDEN'):
        raise unittest.SkipTest(
            'PNG-RMS golden comparison is macOS-only (rasterizer-specific); '
            'the exact-SVG-string golden covers cross-platform rendering. '
            'Set P2S_FORCE_PNG_GOLDEN=1 to force.'
        )
    import numpy as np
    from PIL import ImageChops
    img = rasterize_svg(svg)
    path = os.path.join(GOLDEN_PNG_DIR, name + '.png')
    if os.environ.get('UPDATE_GOLDEN') or not os.path.exists(path):
        os.makedirs(GOLDEN_PNG_DIR, exist_ok=True)
        img.save(path)
        return
    from PIL import Image
    ref = Image.open(path).convert('RGB')
    diff = ImageChops.difference(img, ref)
    rms = float(np.sqrt(np.mean(np.array(diff, dtype=float) ** 2)))
    assert rms <= tolerance, (
        f'Rendered image {name!r} differs from golden PNG: '
        f'RMS={rms:.2f} > tolerance={tolerance}\n'
        f'Run with UPDATE_GOLDEN=1 to regenerate.'
    )
