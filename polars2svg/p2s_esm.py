#
# p2s_esm - the JavaScript assets for the Panel views, and how they are composed
#
# The component JavaScript used to live entirely in Python string literals -- ~58 KB of
# it, invisible to every linter, type checker and editor the project runs (ruff, mypy and
# bandit are all Python-only).  It lives in ``polars2svg/js/`` now, and this module reads
# and concatenates it.
#
# **Why concatenation and not ``import``.**  The obvious shape -- one ES module per
# component, ``import``ing shared helpers from siblings -- does not work in a notebook,
# which is polars2svg's primary target.  ``ReactiveESM._render_esm()`` (panel/custom.py)
# emits a *URL* for the module only when the path is the class's ``_bundle_path`` AND a
# real server session context exists; otherwise it inlines ``esm_path.read_text()`` and
# the browser loads it from a blob URL, which has no server-relative base for a relative
# specifier to resolve against.  So a sibling ``import`` would resolve under
# ``panel serve`` and fail under Jupyter -- an asymmetry that would ship broken to every
# notebook user while every test passed.  ``panel compile`` / ``_esm_shared`` is the
# supported answer and it needs node + esbuild, which this project deliberately does not
# have.  Concatenation costs the same bytes on the wire and behaves identically in both.
#
# The consequence, which ``tests/test_js_assets.py`` enforces: files under ``js/fragments/``
# are *fragments*, not modules.  They may not contain ``import`` or ``export``.  Sharing
# across separately-compiled component classes stays on the existing global-with-guard
# idiom (``if (!window.__P2S_GPU__) { ... }``), which is also what lets the GPU runtime be
# injected into a plain ``<script>`` by ``p2s_webgpu_runtime.standalone_html()``.
#
import functools
import os
from pathlib import Path

#: Where the .js assets live.  Shipped in the wheel: hatchling's wheel target includes
#: non-.py files under the package directory, and tools/diff_packaging_prod.sh diffs
#: polars2svg/ recursively, so these are in scope for the production reconciliation.
_JS_DIR = Path(__file__).parent / 'js'

#: Set to disable the read cache, so that editing a .js file is picked up without
#: restarting the interpreter.  ``panel serve --dev`` watches ``_esm`` only when it is a
#: Path (it is a composed string here), so this is the hook for that workflow.
_NO_CACHE_ENV_ = 'P2S_JS_NO_CACHE'


def _read_js(name: str) -> str:
    '''Read one asset, uncached.  ``name`` is relative to ``polars2svg/js/`` and is
    confined to it -- these names are module-internal literals today, and a traversal
    guard keeps that true if one ever becomes data.'''
    _path_ = (_JS_DIR / name).resolve()
    if not _path_.is_relative_to(_JS_DIR.resolve()):
        raise ValueError(f'p2s_esm: {name!r} resolves outside {_JS_DIR}')
    if not _path_.is_file():
        raise FileNotFoundError(f'p2s_esm: no such JS asset: {name!r} (looked in {_JS_DIR})')
    return _path_.read_text(encoding='utf-8')


@functools.cache
def _read_js_cached(name: str) -> str:
    return _read_js(name)


def js_text(name: str) -> str:
    '''The text of one JS asset, e.g. ``js_text('fragments/p2s_gpu_runtime.js')``.'''
    if os.environ.get(_NO_CACHE_ENV_):
        return _read_js(name)
    return _read_js_cached(name)


def esm(*names: str) -> str:
    '''Concatenate assets into one ES module, in the order given.

    The entry module -- the one exporting ``render`` -- goes last, so that the fragments
    it calls are already defined.  Fragments are joined with a newline so that a file
    without a trailing newline cannot weld its last line onto the next file's first.
    '''
    return '\n'.join(js_text(_n_) for _n_ in names)


def js_dir() -> Path:
    '''The asset directory, for the tests that walk it.'''
    return _JS_DIR
