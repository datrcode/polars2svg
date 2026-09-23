'''Static checks over polars2svg/js/.

The project's JavaScript moved out of Python string literals and into real .js files so
that an editor could see it.  Nothing else in the toolchain can: ruff, mypy and bandit are
Python-only, and `tools/preflight.sh` runs no JS linter.  These tests are the whole of the
automated review those files get, so they cover the invariants that would otherwise fail
silently in a browser:

* no orphan assets, and no dangling references to assets that do not exist;
* fragments under js/fragments/ are fragments -- no `import`, no `export`.  A sibling import
  resolves under `panel serve` and fails in a notebook (see the header of p2s_esm.py), so
  it is precisely the mistake that local testing would not catch;
* an entry module exports exactly one `render`, which is what panel calls;
* encoding hygiene, because these files are concatenated and shipped verbatim.
'''
import ast
import re
import unittest
from pathlib import Path

import view_js_utils

from polars2svg import p2s_esm

_PKG_DIR_ = Path(p2s_esm.__file__).parent
_JS_DIR_  = p2s_esm.js_dir()

#: The functions whose string arguments name a JS asset.
_LOADER_FNS_ = {'js_text', 'esm'}


def _asset_files():
    '''Every .js asset, as a path relative to polars2svg/js/ with forward slashes.'''
    if not _JS_DIR_.is_dir():
        return set()
    return {_p_.relative_to(_JS_DIR_).as_posix()
            for _p_ in _JS_DIR_.rglob('*.js') if _p_.is_file()}


def _referenced_names():
    '''Every asset name passed to js_text()/esm() anywhere in the package source.

    Parsed rather than grepped so that a multi-line esm(...) call is read correctly and a
    name inside a comment is not.
    '''
    _names_ = set()
    for _py_ in _PKG_DIR_.rglob('*.py'):
        try:
            _tree_ = ast.parse(_py_.read_text(encoding='utf-8'), filename=str(_py_))
        except SyntaxError:                                     # pragma: no cover
            continue
        for _node_ in ast.walk(_tree_):
            if not isinstance(_node_, ast.Call):
                continue
            _fn_ = _node_.func
            _fn_name_ = (_fn_.attr if isinstance(_fn_, ast.Attribute) else
                         _fn_.id   if isinstance(_fn_, ast.Name) else None)
            if _fn_name_ not in _LOADER_FNS_:
                continue
            for _arg_ in _node_.args:
                if isinstance(_arg_, ast.Constant) and isinstance(_arg_.value, str):
                    _names_.add(_arg_.value)
    return _names_


def _esm_bundles():
    """Every esm(...) call in the package source, as an ordered list of asset names.

    _referenced_names() flattens these into a set, which answers "is this asset used"
    but not "is this asset used *together with* the one that defines what it calls".
    A fragment-supplied helper needs the second question: p2sInk() lives in
    fragments/p2s_dom.js and is called from three entry modules, so a bundle that
    omits the fragment is a ReferenceError at render time, not an import error.
    """
    _bundles_ = []
    for _py_ in _PKG_DIR_.rglob('*.py'):
        try:
            _tree_ = ast.parse(_py_.read_text(encoding='utf-8'), filename=str(_py_))
        except SyntaxError:                                     # pragma: no cover
            continue
        for _node_ in ast.walk(_tree_):
            if not isinstance(_node_, ast.Call):
                continue
            _fn_ = _node_.func
            _fn_name_ = (_fn_.attr if isinstance(_fn_, ast.Attribute) else
                         _fn_.id   if isinstance(_fn_, ast.Name) else None)
            if _fn_name_ != 'esm':
                continue
            _names_ = [_a_.value for _a_ in _node_.args
                       if isinstance(_a_, ast.Constant) and isinstance(_a_.value, str)]
            if _names_:
                _bundles_.append((f'{_py_.name}:{_node_.lineno}', _names_))
    return _bundles_


class TestAssetsAndReferencesAgree(unittest.TestCase):
    '''Every asset is used, and every use resolves.'''

    def test_no_dangling_references(self):
        _missing_ = sorted(_n_ for _n_ in _referenced_names()
                           if not (_JS_DIR_ / _n_).is_file())
        self.assertEqual(_missing_, [],
                         f'js_text()/esm() names with no file in {_JS_DIR_}: {_missing_}')

    def test_no_orphan_assets(self):
        '''A file nothing composes is dead weight in the wheel, and usually a rename that
        only got done on one side.'''
        _orphans_ = sorted(_asset_files() - _referenced_names())
        self.assertEqual(_orphans_, [],
                         f'.js assets that no js_text()/esm() call names: {_orphans_}')


class TestFragmentsAreFragments(unittest.TestCase):
    '''js/fragments/*.js are concatenated into a module, and also injected into a plain
    <script> by p2s_webgpu_runtime.standalone_html().  Either way an import/export is a
    syntax error or a silent no-op, and a relative import would work under panel serve
    while failing in a notebook.'''

    def _fragments(self):
        _dir_ = _JS_DIR_ / 'fragments'
        return sorted(_dir_.rglob('*.js')) if _dir_.is_dir() else []

    def test_the_fragment_directory_is_not_empty(self):
        '''Guard the guard.  Every check below iterates the fragment directory, so a
        rename that missed this file would scan nothing and pass in silence -- which is
        how the directory came to be called `fragments` in the first place: it was
        `parts`, and `.gitignore` line 19 ignores `parts/` (it is setuptools' build
        directory in the standard Python template).  The asset was therefore in neither
        the wheel nor the sdist, and nothing said so.'''
        self.assertTrue(self._fragments(), f'no .js fragments found under {_JS_DIR_}/fragments')

    def test_no_import_or_export_statements(self):
        for _p_ in self._fragments():
            with self.subTest(asset=_p_.name):
                for _i_, _line_ in enumerate(_p_.read_text(encoding='utf-8').splitlines(), 1):
                    _code_ = _line_.split('//')[0].strip()
                    self.assertFalse(
                        _code_.startswith(('import ', 'import{', 'import(',
                                           'export ', 'export{', 'export(')),
                        f'{_p_.name}:{_i_} is a fragment but has a module statement: {_line_.strip()!r}')


class TestNoImplicitGlobals(unittest.TestCase):
    '''An ES module is strict mode automatically; a `_scripts` body was not.

    This is the one hazard of the migration that is invisible until the code runs, and it
    bit exactly once: `myUpdateDragRect` assigned `x`, `y`, `w` and `h` without `var`.
    Under `ReactiveHTML` each script ran as a sloppy-mode function, so those became
    implicit globals and worked for years. Ported verbatim into a module they throw
    `ReferenceError`, which took out all five of LINKPI's rubber-band tests and nothing
    else -- the failure is silent until the exact handler runs.

    `node --check` does not catch it (it is a runtime error, not a syntax one) and no
    linter in this project reads JavaScript at all, so this is the check.
    '''

    #: Globals the modules may legitimately reach for.
    ALLOWED = frozenset({
        'Math', 'window', 'document', 'console', 'JSON', 'Object', 'Array', 'String',
        'Number', 'Boolean', 'Date', 'RegExp', 'Error', 'parseInt', 'parseFloat', 'isNaN',
        'setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'atob', 'btoa',
        'navigator', 'undefined', 'requestAnimationFrame', 'performance', 'globalThis',
        'Uint8Array', 'Uint16Array', 'Uint32Array', 'Float32Array', 'ArrayBuffer',
        'Promise', 'Map', 'Set', 'TextEncoder', 'structuredClone',
        'GPUBufferUsage', 'GPUTextureUsage', 'GPUShaderStage', 'GPUMapMode',
        # supplied by the render() signature or by a concatenated fragment
        'model', 'state', 'el', 'event', 'view',
        'svgEl', 'htmlEl', 'p2sGpuWrap', 'p2sGpuDraw',
    })

    @staticmethod
    def _strip(src):
        '''Blank out comments and string literals so their contents cannot match.

        view_js_utils.strip_noise, not a local sequence of regexes.  The local version
        ran the `//` pass before the string pass, so `'http://www.w3.org/2000/svg'` lost
        its closing quote and its statement's `;` to the comment stripper and everything
        after it was read in the wrong state -- which silently dropped the first
        declarator of the following line.  A single left-to-right pass cannot make that
        mistake, and this test is now the only caller that needed convincing.
        '''
        return view_js_utils.strip_noise(src)

    @classmethod
    def _declared(cls, clean):
        _out_ = set(re.findall(r'\bfunction\s+([A-Za-z_$][\w$]*)', clean))
        for _m_ in re.finditer(r'function[^(]*\(([^)]*)\)', clean):
            _out_ |= {re.sub(r'[^\w$]', '', _p_) for _p_ in _m_.group(1).split(',') if _p_.strip()}
        # Every declarator in a list, not just the first: `var a = 1, b = 2;`
        for _m_ in re.finditer(r'\b(?:var|let|const)\s+([^;{]*)', clean):
            for _part_ in _m_.group(1).split(','):
                _n_ = re.match(r'\s*([A-Za-z_$][\w$]*)', _part_)
                if _n_:
                    _out_.add(_n_.group(1))
        _out_ |= set(re.findall(r'catch\s*\(\s*([A-Za-z_$][\w$]*)', clean))
        return _out_

    def test_every_assignment_targets_a_declared_name(self):
        for _name_ in sorted(_asset_files()):
            with self.subTest(asset=_name_):
                _clean_ = self._strip(p2s_esm.js_text(_name_))
                _declared_ = self._declared(_clean_)
                _bad_ = {}
                for _m_ in re.finditer(r'(?<![.\w$])([A-Za-z_$][\w$]*)\s*=(?![=>])', _clean_):
                    _id_ = _m_.group(1)
                    if _id_ not in _declared_ and _id_ not in self.ALLOWED:
                        _bad_.setdefault(_id_, _clean_[:_m_.start()].count('\n') + 1)
                self.assertEqual(
                    dict(_bad_), {},
                    f'{_name_} assigns to undeclared names {sorted(_bad_)} -- an implicit '
                    f'global, which throws ReferenceError in a module. Add `var`/`let`, or '
                    f'add the name to ALLOWED if it is a real browser global.')


class TestEntryModules(unittest.TestCase):
    '''A .js directly under js/ (not in fragments/) is an entry module: the thing a
    JSComponent names as its _esm, which panel calls render() on.'''

    def _entries(self):
        return sorted(_p_ for _p_ in _JS_DIR_.glob('*.js')) if _JS_DIR_.is_dir() else []

    def test_each_entry_module_exports_exactly_one_render(self):
        for _p_ in self._entries():
            with self.subTest(asset=_p_.name):
                _n_ = _p_.read_text(encoding='utf-8').count('export function render(')
                self.assertEqual(_n_, 1,
                                 f'{_p_.name} has {_n_} `export function render(` -- panel needs exactly one')


class TestFragmentHelpersAreInEveryBundleThatCallsThem(unittest.TestCase):
    """A helper defined in one fragment and called from another asset only works if the
    two are concatenated into the same module.

    Nothing else checks this. test_every_assignment_targets_a_declared_name catches an
    implicit global *write*; a call to a function that was never concatenated in is a
    read, so it passes every static check here and fails in the browser -- where only
    the opt-in Playwright suite would see it.
    """

    # Helpers that MUST be concatenated in: called bare, so an absent definition is a
    # ReferenceError the moment the line runs.
    REQUIRED = {
        'p2sInk': 'fragments/p2s_dom.js',
        'svgEl':  'fragments/p2s_dom.js',
        'htmlEl': 'fragments/p2s_dom.js',
    }

    # Helpers that are OPTIONAL by design: present only in the GPU bundles, and every
    # call site tests `typeof ... === 'function'` first.  `typeof` on an undeclared
    # identifier is the one reference that does not throw, which is what makes the
    # pattern legal -- so the guard is the whole contract, and the second test below
    # checks it rather than taking this comment's word for it.
    OPTIONAL = {
        'p2sGpuWrap': 'fragments/p2s_gpu_mount.js',
    }

    def test_required_helper_is_bundled_with_every_caller(self):
        for _where_, _names_ in _esm_bundles():
            _text_ = '\n'.join(p2s_esm.js_text(_n_) for _n_ in _names_)
            for _helper_, _home_ in self.REQUIRED.items():
                if not re.search(rf'\b{_helper_}\s*\(', _text_):
                    continue                       # this bundle never calls it
                with self.subTest(bundle=_where_, helper=_helper_):
                    self.assertRegex(
                        _text_, rf'function\s+{_helper_}\s*\(',
                        f'{_where_} calls {_helper_}() but does not include {_home_}, '
                        f'so the name is undefined at render time. Add the fragment to '
                        f'that esm(...) call.')

    def test_optional_helper_is_only_ever_called_behind_a_typeof_guard(self):
        for _name_ in sorted(_asset_files()):
            _text_ = p2s_esm.js_text(_name_)
            for _helper_ in self.OPTIONAL:
                if f'function {_helper_}' in _text_:
                    continue                       # this is the file that defines it
                for _m_ in re.finditer(rf'\b{_helper_}\s*\(', _text_):
                    _line_ = _text_[:_m_.start()].count('\n') + 1
                    _stmt_ = _text_[max(0, _m_.start() - 200):_m_.start()]
                    with self.subTest(asset=_name_, line=_line_):
                        self.assertRegex(
                            _stmt_, rf"typeof\s+{_helper_}\s*===\s*'function'",
                            f'{_name_}:{_line_} calls {_helper_}() without a '
                            f"`typeof {_helper_} === 'function'` guard. It is absent from "
                            f'the non-GPU bundles, so an unguarded call is a ReferenceError '
                            f'there. Either guard it or add its fragment to those bundles.')

    def test_each_helper_is_defined_exactly_once_per_bundle(self):
        # Two copies would be a silent redeclaration; `function` hoisting makes the last
        # one win rather than throwing, so the wrong definition could quietly take over.
        for _where_, _names_ in _esm_bundles():
            _text_ = '\n'.join(p2s_esm.js_text(_n_) for _n_ in _names_)
            for _helper_ in {**self.REQUIRED, **self.OPTIONAL}:
                _n_ = len(re.findall(rf'function\s+{_helper_}\s*\(', _text_))
                if _n_:
                    with self.subTest(bundle=_where_, helper=_helper_):
                        self.assertEqual(_n_, 1, f'{_where_} defines {_helper_}() {_n_} times')


class TestAssetsAreShippable(unittest.TestCase):
    '''The .js files are only useful if they reach the installed package.

    They are data files in a tree that was 100% Python until now, and hatchling selects
    what to ship with a VCS-aware file list -- so an asset git ignores is silently absent
    from both the wheel and the sdist, with a green test suite the whole way (the source
    tree still has the file, and that is what the tests import).  This is not a
    hypothetical: the directory was called `parts/` for an afternoon, which `.gitignore`
    line 19 ignores, and the first build after that produced artifacts with no JS in them
    at all.

    Checked here by asking git directly, which is cheap and needs no build.  A real
    `uv build` check is in the plan's verification section for DT to run before a port.
    '''

    def test_no_asset_is_ignored_by_git(self):
        import subprocess
        _assets_ = sorted(_asset_files())
        self.assertTrue(_assets_, 'no JS assets found at all')
        _rel_ = [f'polars2svg/js/{_a_}' for _a_ in _assets_]
        _repo_ = _PKG_DIR_.parent
        _r_ = subprocess.run(['git', 'check-ignore', *_rel_],
                             cwd=_repo_, capture_output=True, text=True)
        # check-ignore exits 0 and lists paths when any ARE ignored, 1 when none are.
        _ignored_ = [_l_ for _l_ in _r_.stdout.splitlines() if _l_.strip()]
        self.assertEqual(_ignored_, [],
                         f'these JS assets are gitignored and would not ship: {_ignored_}')


class TestEncodingHygiene(unittest.TestCase):
    '''These files are read as utf-8 and concatenated verbatim.'''

    def test_assets_are_clean_utf8(self):
        for _name_ in sorted(_asset_files()):
            with self.subTest(asset=_name_):
                _raw_ = (_JS_DIR_ / _name_).read_bytes()
                self.assertFalse(_raw_.startswith(b'\xef\xbb\xbf'), f'{_name_} has a UTF-8 BOM')
                self.assertNotIn(b'\t', _raw_, f'{_name_} contains a tab')
                self.assertNotIn(b'\r', _raw_, f'{_name_} has CRLF line endings')
                _raw_.decode('utf-8')           # raises on invalid utf-8


class TestLoader(unittest.TestCase):
    '''p2s_esm itself.'''

    def test_esm_joins_in_order_with_a_separator(self):
        '''Fragments are newline-joined so a file with no trailing newline cannot weld its
        last line onto the next file's first.'''
        _names_ = sorted(_asset_files())
        if len(_names_) < 1:
            self.skipTest('no JS assets yet')
        _one_ = p2s_esm.esm(_names_[0])
        self.assertEqual(_one_, p2s_esm.js_text(_names_[0]))
        if len(_names_) >= 2:
            _two_ = p2s_esm.esm(_names_[0], _names_[1])
            self.assertEqual(_two_,
                             p2s_esm.js_text(_names_[0]) + '\n' + p2s_esm.js_text(_names_[1]))

    def test_missing_asset_raises_rather_than_returning_empty(self):
        '''An empty _esm would render a blank component and look like a CSS problem.'''
        with self.assertRaises(FileNotFoundError):
            p2s_esm.js_text('fragments/definitely_not_here.js')

    def test_names_cannot_escape_the_js_directory(self):
        with self.assertRaises(ValueError):
            p2s_esm.js_text('../p2s_esm.py')

    def test_the_no_cache_env_var_picks_up_an_edited_file(self):
        '''`panel serve --dev` watches `_esm` only when it is a Path; ours is a composed
        string, so this env var is the documented way to iterate on a .js file without
        restarting the interpreter.  It is only useful if it actually re-reads.'''
        import os
        import tempfile
        import unittest.mock

        # Redirected to a temp tree rather than writing into polars2svg/js/: a failure
        # part-way through would otherwise leave a stray .js in the package, which
        # test_no_orphan_assets would then fail on and `uv build` would ship.
        with tempfile.TemporaryDirectory() as _td_:
            _root_ = Path(_td_)
            (_root_ / 'fragments').mkdir()
            _asset_ = _root_ / 'fragments' / 'probe.js'
            _asset_.write_text('// one\n', encoding='utf-8')
            with unittest.mock.patch.object(p2s_esm, '_JS_DIR', _root_):
                p2s_esm._read_js_cached.cache_clear()
                self.assertEqual(p2s_esm.js_text('fragments/probe.js'), '// one\n')
                _asset_.write_text('// two\n', encoding='utf-8')
                self.assertEqual(p2s_esm.js_text('fragments/probe.js'), '// one\n',
                                 'the cache is not caching -- every read hits the disk')
                os.environ['P2S_JS_NO_CACHE'] = '1'
                try:
                    self.assertEqual(p2s_esm.js_text('fragments/probe.js'), '// two\n',
                                     'P2S_JS_NO_CACHE did not bypass the cache')
                finally:
                    del os.environ['P2S_JS_NO_CACHE']
                    p2s_esm._read_js_cached.cache_clear()

    def test_gpu_runtime_is_reachable_through_the_loader(self):
        '''The one asset Phase 1 moved: p2s_webgpu_runtime.P2S_GPU_JS is now this file.'''
        from polars2svg.p2s_webgpu_runtime import P2S_GPU_JS
        self.assertEqual(P2S_GPU_JS, p2s_esm.js_text('fragments/p2s_gpu_runtime.js'))
        self.assertIn('window.__P2S_GPU__', P2S_GPU_JS)


#: A synthetic ESM module standing in for a ported component, so that the ESM half of the
#: seam is exercised now rather than first being used during a port.  It deliberately
#: includes the things that break a naive brace matcher: braces inside string literals,
#: a brace in a // comment, an escaped quote, and a nested block.
_FAKE_ESM_ = '''
function svgEl(tag, attrs, parent) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  return el;
}

function menuCommit(state, model) {
  // a closing brace } in a comment must not end this function
  const label = "a } inside a string";
  const other = 'and a \\' escaped quote with } too';
  if (state.menu_open) {
    model.timing_spacing_choice = label;
  }
  return other;
}

export function render({ model, el }) {
  const root = svgEl('svg', {id: 'svgparent'}, el);
  model.on('mod_inner', () => { root.innerHTML = model.mod_inner; });
  return root;
}
'''


class _FakeEsmView:
    _esm = _FAKE_ESM_


class _FakeReactiveView:
    _template = '<svg id="svgparent"><g id="mod"></g></svg>'
    _scripts  = {'render':      'mod.innerHTML = data.mod_inner',
                 'menuCommit':  'data.timing_spacing_choice = _sel_[3]'}


class TestViewJsSeam(unittest.TestCase):
    '''tests/view_js_utils.py is what lets ~45 assertions survive the JSComponent port.
    Both of its branches are tested here; only the ReactiveHTML one is reachable through
    a real component today.'''

    def test_reactive_branch_reads_scripts(self):
        self.assertIn('data.mod_inner', view_js_utils.component_js(_FakeReactiveView))
        self.assertIn('_sel_[3]', view_js_utils.component_script(_FakeReactiveView, 'menuCommit'))
        self.assertEqual(view_js_utils.script_names(_FakeReactiveView), {'render', 'menuCommit'})
        self.assertTrue(view_js_utils.has_script(_FakeReactiveView, 'render'))
        self.assertFalse(view_js_utils.has_script(_FakeReactiveView, 'nope'))

    def test_esm_branch_reads_the_module(self):
        self.assertIn('createElementNS', view_js_utils.component_js(_FakeEsmView))
        self.assertEqual(view_js_utils.script_names(_FakeEsmView),
                         {'svgEl', 'menuCommit', 'render'})
        self.assertTrue(view_js_utils.has_script(_FakeEsmView, 'menuCommit'))

    def test_esm_param_change_handlers_count_as_scripts(self):
        '''A _scripts key named after a param becomes a model.on(...) callback, not a
        function declaration -- has_script has to find it either way or every
        param-change assertion would break on port.'''
        self.assertTrue(view_js_utils.has_script(_FakeEsmView, 'mod_inner'))
        self.assertFalse(view_js_utils.has_script(_FakeEsmView, 'info_str'))

    def test_accepts_an_instance_as_well_as_a_class(self):
        self.assertEqual(view_js_utils.component_js(_FakeReactiveView()),
                         view_js_utils.component_js(_FakeReactiveView))

    def test_missing_script_raises_rather_than_returning_empty(self):
        '''An empty string would make assertNotIn pass for the wrong reason.'''
        for _view_ in (_FakeReactiveView, _FakeEsmView):
            with self.subTest(view=_view_.__name__), self.assertRaises(KeyError):
                view_js_utils.component_script(_view_, 'no_such_handler')


class TestBraceMatcher(unittest.TestCase):
    '''component_script()'s ESM branch brace-matches.  The component JS has no template
    literals, block comments or regex literals, so skipping '/"' strings and // comments
    is sufficient -- these tests are what hold that assumption.'''

    def test_extracts_only_the_named_function(self):
        _body_ = view_js_utils.component_script(_FakeEsmView, 'menuCommit')
        self.assertTrue(_body_.startswith('function menuCommit('))
        self.assertTrue(_body_.rstrip().endswith('}'))
        self.assertIn('state.menu_open', _body_)
        self.assertNotIn('createElementNS', _body_, 'leaked into the neighbouring function')
        self.assertNotIn('export function render', _body_, 'ran past the closing brace')

    def test_braces_in_strings_and_comments_do_not_terminate_it(self):
        _body_ = view_js_utils.component_script(_FakeEsmView, 'menuCommit')
        self.assertIn('a } inside a string', _body_)
        self.assertIn('escaped quote with } too', _body_)
        self.assertIn('must not end this function', _body_)

    def test_exported_function_is_reachable(self):
        _body_ = view_js_utils.component_script(_FakeEsmView, 'render')
        self.assertIn("model.on('mod_inner'", _body_)
        self.assertNotIn('menuCommit', _body_)

    def test_it_extracts_cleanly_from_every_shipped_esm_module(self):
        '''The assumption, checked behaviourally against the JS that actually ships.

        This used to assert the *proxy* -- that no component JS contained a backtick or a
        `/*`.  That became false the moment the first module shipped: p2s_gpu_runtime.js
        holds the WGSL shader in a 180-line template literal, and the porting notes in
        every module quote identifiers in `//` comments.  Neither breaks the matcher, so
        the proxy was condemning working code.

        What matters is whether a body still comes out whole, so that is what is asserted:
        for every function each module defines, the extract must start at its own header,
        end on a balanced brace, and not run into the next function.
        '''
        from polars2svg import p2s_esm
        for _name_ in sorted(_asset_files()):
            _src_ = p2s_esm.js_text(_name_)

            class _Mod:
                _esm = _src_

            _names_ = view_js_utils.script_names(_Mod)
            for _fn_ in sorted(_names_):
                with self.subTest(asset=_name_, function=_fn_):
                    _body_ = view_js_utils.component_script(_Mod, _fn_)
                    self.assertTrue(_body_.startswith(f'function {_fn_}('),
                                    f'{_name_}:{_fn_} extract does not start at its header')
                    self.assertEqual(_body_.count('{'), _body_.count('}'),
                                     f'{_name_}:{_fn_} extract has unbalanced braces -- the '
                                     f'matcher ran past the end or stopped early')
                    # A nested function legitimately appears inside its parent; a leak is
                    # the extract reaching one defined at the SAME level or shallower.
                    # Indentation is what tells those apart, and it has to: from the
                    # extracted text alone they are identical.  This used to exempt the
                    # name `render`, which was the only container that existed; the
                    # shared fragments' factories (p2sConfigPanel, p2sTooltip) are
                    # containers of exactly the same shape, and a hard-coded list of them
                    # would have to be edited every time one is added.
                    _own_col_ = view_js_utils.declaration_column(_src_, _fn_)
                    _leaked_  = [
                        _o_ for _o_ in sorted(_names_)
                        if _o_ != _fn_
                        and f'function {_o_}(' in _body_
                        and not _body_.startswith(f'function {_o_}(')
                        and view_js_utils.declaration_column(_src_, _o_) <= _own_col_
                    ]
                    self.assertEqual(_leaked_, [],
                                     f'{_name_}:{_fn_} extract leaked into {_leaked_}')


class TestMenuTablesAgree(unittest.TestCase):
    """A menu kind has to appear in all three tables, and nothing checked that.

    The config panel's design note says why the tables exist at all: "three chains of
    twelve branches each is how a kind ends up handled in two places and missed in the
    third". Turning two of the chains into tables made the omission *quieter*, not
    impossible -- adding F1's `tooltip` kind, it was written into `menuSetValue` and left
    out of `MENU_PARAM_`, and the symptom was a panel row that drew a blank value. A
    browser test caught it; this one catches it in a second, which is the difference
    between noticing and hunting.

    Read out of the shipped JS rather than from Python, because the tables are JS.
    """

    #: `const NAME_ = { key: ..., }` -- the keys only.
    @staticmethod
    def _table_keys(src, name):
        _clean_ = view_js_utils.strip_noise(src)
        _at_ = _clean_.find(f'const {name} = {{')
        if _at_ < 0:
            return None
        _depth_, _end_ = 0, -1
        for _i_ in range(_clean_.index('{', _at_), len(_clean_)):
            if _clean_[_i_] == '{':
                _depth_ += 1
            elif _clean_[_i_] == '}':
                _depth_ -= 1
                if _depth_ == 0:
                    _end_ = _i_
                    break
        return set(re.findall(r'^\s*([A-Za-z_]\w*)\s*:', _clean_[_at_:_end_], re.M))

    @staticmethod
    def _set_value_kinds(src):
        _body_ = view_js_utils._extract_function(src, 'menuSetValue')
        return set(re.findall(r"kind\s*==\s*'([A-Za-z_]\w*)'", _body_ or ''))

    def test_every_module_with_menus_covers_each_kind_three_times(self):
        from polars2svg import p2s_esm
        _seen_ = 0
        for _name_ in sorted(_asset_files()):
            _src_ = p2s_esm.js_text(_name_)
            _params_ = self._table_keys(_src_, 'MENU_PARAM_')
            if _params_ is None:
                continue
            _seen_ += 1
            with self.subTest(asset=_name_):
                _headers_ = self._table_keys(_src_, 'MENU_HEADER_') or set()
                _writes_  = self._set_value_kinds(_src_)
                self.assertEqual(_params_, _headers_,
                                 f'{_name_}: MENU_PARAM_ and MENU_HEADER_ disagree -- '
                                 f'a kind with no header draws an undefined picker title')
                self.assertEqual(_params_, _writes_,
                                 f'{_name_}: MENU_PARAM_ and menuSetValue disagree -- '
                                 f'a kind missing from menuSetValue cannot be committed, '
                                 f'and one missing from MENU_PARAM_ draws a blank value')
        self.assertEqual(_seen_, 2,
                         'expected exactly two modules to carry menu tables '
                         '(p2s_linkpi.js and p2s_interactivep.js)')


if __name__ == '__main__':
    unittest.main()
