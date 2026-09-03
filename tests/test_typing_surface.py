import unittest

import ast
import importlib.util
import inspect
import tomllib
from enum import Enum
from pathlib import Path

import polars2svg
from polars2svg import (
    Polars2SVG,
    LandmarkMDSLayout,
    PivotMDSLayout,
)


#
# The package is typed: it ships a py.typed marker so
# downstream type checkers read the hints, and the public surface (constructor,
# component factory methods, tField, panelize, exported layout classes) carries
# return annotations. These tests guard the marker's presence and that the
# annotations are not accidentally stripped.
#
class TestPyTypedMarker(unittest.TestCase):

    def test_py_typed_ships_beside_package(self):
        # py.typed must live inside the installed package directory (that is what
        # PEP 561 tools look for and what hatchling bundles into the wheel).
        pkg_dir = Path(inspect.getfile(polars2svg)).parent
        self.assertTrue((pkg_dir / 'py.typed').is_file(),
                        f'py.typed marker missing from {pkg_dir}')

    def test_py_typed_locatable_via_importlib(self):
        # Same check via the loader, independent of how the package was installed.
        spec = importlib.util.find_spec('polars2svg')
        self.assertIsNotNone(spec and spec.origin)
        pkg_dir = Path(spec.origin).parent
        self.assertTrue((pkg_dir / 'py.typed').is_file())


class TestPublicSurfaceAnnotations(unittest.TestCase):

    FACTORY_RETURNS = {
        'xyp':          'XYp',
        'histop':       'Histop',
        'timep':        'Timep',
        'linkp':        'LinkP',
        'chordp':       'ChP',
        'piep':         'Piep',
        'spreadlinesp': 'SpreadLinesP',
        'smallp':       'Smallp',
        'tile':         'Tile',
    }

    def test_factory_methods_have_return_annotation(self):
        for name, expected in self.FACTORY_RETURNS.items():
            with self.subTest(method=name):
                ann = inspect.signature(getattr(Polars2SVG, name)).return_annotation
                self.assertIsNot(ann, inspect.Signature.empty,
                                 f'{name}() is missing a return annotation')
                # Annotation is the class object itself (evaluated at def time).
                self.assertEqual(getattr(ann, '__name__', ann), expected)

    def test_init_annotated_returns_none(self):
        ann = inspect.signature(Polars2SVG.__init__).return_annotation
        self.assertIs(ann, None)

    def test_tfield_annotated(self):
        sig = inspect.signature(Polars2SVG.tField)
        self.assertIsNot(sig.return_annotation, inspect.Signature.empty)
        self.assertIsNot(sig.parameters['column'].annotation, inspect.Parameter.empty)

    def test_panelize_annotated(self):
        sig = inspect.signature(Polars2SVG.panelize)
        self.assertIsNot(sig.return_annotation, inspect.Signature.empty)
        self.assertIsNot(sig.parameters['stack'].annotation, inspect.Parameter.empty)

    def test_exported_layouts_results_return_dict(self):
        for cls in (LandmarkMDSLayout, PivotMDSLayout):
            with self.subTest(cls=cls.__name__):
                ann = inspect.signature(cls.results).return_annotation
                self.assertIs(ann, dict,
                              f'{cls.__name__}.results() should be annotated -> dict')


#
# The typing ratchet.
#
# polars2svg was annotated module by module over 2026-09-02/03 and strict mypy
# checking is now the package-wide default (see the [tool.mypy] comment in
# pyproject.toml).  These tests are the half of that ratchet the test suite
# owns: they assert that annotation coverage only ever improves, and that the
# dynamic attribute surface stays declared.  mypy owns the other half, and now
# owns it for every module rather than for a graduating list.
#
class TestEnumMemberDeclarations(unittest.TestCase):
    '''Polars2SVG.__init__ flattens six nested Enums onto the instance with
    setattr() loops.  A class-level declaration block mirrors them so type
    checkers can see p2s.SCALARp / p2s.PT_DoWp / p2s.BARCHARTp.  Nothing at
    runtime keeps the two in sync -- these tests do.'''

    @staticmethod
    def _bound_enum_members():
        # Ground truth: what the setattr() loops actually put on an instance.
        p = Polars2SVG()
        out = {}
        for _name_ in dir(p):
            try: _val_ = getattr(p, _name_)
            except Exception: continue
            if isinstance(_val_, Enum): out[_name_] = type(_val_)
        return out

    @staticmethod
    def _declared_enum_members():
        _ann_ = Polars2SVG.__dict__.get('__annotations__', {})
        return {k: v for k, v in _ann_.items()
                if isinstance(v, type) and issubclass(v, Enum)}

    def test_every_bound_member_is_declared(self):
        _bound_, _declared_ = self._bound_enum_members(), self._declared_enum_members()
        _missing_ = sorted(set(_bound_) - set(_declared_))
        self.assertEqual(_missing_, [],
                         'enum members bound by __init__ but not declared on the class '
                         f'(add them to the declaration block in polars2svg.py): {_missing_}')

    def test_no_stale_declarations(self):
        _bound_, _declared_ = self._bound_enum_members(), self._declared_enum_members()
        _stale_ = sorted(set(_declared_) - set(_bound_))
        self.assertEqual(_stale_, [],
                         f'declared enum members that __init__ no longer binds: {_stale_}')

    def test_declared_types_match(self):
        _bound_, _declared_ = self._bound_enum_members(), self._declared_enum_members()
        for _name_ in sorted(set(_bound_) & set(_declared_)):
            with self.subTest(member=_name_):
                self.assertIs(_declared_[_name_], _bound_[_name_])

    def test_declarations_are_annotations_only(self):
        # Bare annotations must not become class attributes -- an accidental
        # `SCALARp = FieldTypeP.SCALARp` would shadow instance state.
        for _name_ in self._declared_enum_members():
            with self.subTest(member=_name_):
                self.assertNotIn(_name_, Polars2SVG.__dict__,
                                 f'{_name_} is assigned at class level; it must be a '
                                 'bare annotation (no value)')


class TestAnnotationCoverageRatchet(unittest.TestCase):
    '''Per-module ceiling on *unannotated* functions.  A function counts as
    annotated when it has a return annotation and every parameter (bar self/cls)
    is annotated.

    The rule: these numbers may fall, never rise.  Annotating a function lowers
    one; adding an already-typed function changes nothing; adding an UNTYPED
    function raises one and fails here.  That is the enforcement behind "new code
    arrives typed".

    When you annotate a module, lower its number.  Strict mypy checking no longer
    has to be switched on per module -- it is the global default in
    pyproject.toml, and the only module exempt from it is the one in
    PERMANENTLY_RELAXED below.'''

    # The one module deliberately outside the annotation target.  It builds its
    # widget classes at runtime with type('LINKPI', (ReactiveHTML,), {...}) and
    # keeps its state in Panel/param descriptors, so there is nothing static to
    # check; it also sits behind the `interactive` extra.  Its ceiling below still
    # applies -- it cannot get worse -- it is just not expected to reach zero.
    # See the matching note in pyproject.toml.
    PERMANENTLY_RELAXED = frozenset({'interactive_controller'})

    MAX_UNANNOTATED = {
        'interactive_controller':                 145,
        'xyp':                                    0,
        'linkp':                                  0,
        'chordp':                                 0,
        'piep':                                   0,
        'polars2svg':                             0,
        'spreadlinesp':                           0,
        'ncp_layout':                             0,
        'p2s_graph_mixin':                        0,
        'p2s_displaylist':                        0,
        'flow_field_background':                  0,
        'timep':                                  0,
        'od_flow_layout':                         0,
        'histop':                                 0,
        'p2s_geometry_mixin':                     0,
        'p2s_background_mixin':                   0,
        'smallp':                                 0,
        'p2s_legend_mixin':                       0,
        'spreadlinepi':                           0,
        'stack_control':                          0,
        'p2s_colors_mixin':                       0,
        'p2s_interactive_mixin':                  0,
        'p2s_text_mixin':                         0,
        'circle_packer':                          0,
        'tile':                                   0,
        'p2s_time_mixin':                         0,
        'udist_scatterplots_via_sectors_tile_opt':0,
        'p2s_polars_mixin':                       0,
        'p2s_render_mixin':                       0,
        '_seriation':                             0,
        'export':                                 0,
        'p2s_component_color_mixin':              0,
        'p2s_glyph_atlas':                        0,
        'mds_at_scale':                           0,
        'p2s_bin_component_mixin':                0,
        'tfdp_layout':                            0,
        'interactive_treatments':                 0,
        'p2s_font_metrics':                       0,
        'p2s_webgpu_runtime':                     0,
        '__init__':                               0,
        'exceptions':                             0,
        'laguerre_voronoi':                       0,
        'layout_budget':                          0,
        'layout_protocol':                        0,
    }

    @staticmethod
    def _unannotated(path):
        _tree_ = ast.parse(path.read_text())
        _n_ = 0
        for _node_ in ast.walk(_tree_):
            if not isinstance(_node_, (ast.FunctionDef, ast.AsyncFunctionDef)): continue
            _a_  = _node_.args
            _ps_ = [*_a_.posonlyargs, *_a_.args, *_a_.kwonlyargs]
            if _a_.vararg: _ps_.append(_a_.vararg)
            if _a_.kwarg:  _ps_.append(_a_.kwarg)
            _ps_ = [x for x in _ps_ if x.arg not in ('self', 'cls')]
            if _node_.returns is None or any(x.annotation is None for x in _ps_): _n_ += 1
        return _n_

    def _measure(self):
        _pkg_ = Path(inspect.getfile(polars2svg)).parent
        return {f.stem: self._unannotated(f) for f in sorted(_pkg_.glob('*.py'))}

    def test_no_module_gains_unannotated_functions(self):
        for _mod_, _count_ in sorted(self._measure().items()):
            with self.subTest(module=_mod_):
                _ceiling_ = self.MAX_UNANNOTATED.get(_mod_)
                self.assertIsNotNone(_ceiling_,
                                     f'new module {_mod_!r} is not in MAX_UNANNOTATED -- add it '
                                     f'with its current count ({_count_}), or 0 if fully typed')
                self.assertLessEqual(
                    _count_, _ceiling_,
                    f'{_mod_}.py has {_count_} unannotated functions, ceiling is {_ceiling_}. '
                    'New and modified functions must be fully annotated (return type + every '
                    'parameter). If you annotated functions here, lower the ceiling instead.')

    def test_ceilings_are_not_stale(self):
        # A ceiling left above the real count hides a later regression.
        _measured_ = self._measure()
        _slack_ = {m: (self.MAX_UNANNOTATED[m], c) for m, c in _measured_.items()
                   if m in self.MAX_UNANNOTATED and c < self.MAX_UNANNOTATED[m]}
        self.assertEqual(_slack_, {},
                         'these ceilings are above the real count -- lower them to lock the '
                         f'progress in {{module: (ceiling, actual)}}: {_slack_}')

    def test_removed_modules_are_dropped_from_the_table(self):
        _measured_ = self._measure()
        _gone_ = sorted(set(self.MAX_UNANNOTATED) - set(_measured_))
        self.assertEqual(_gone_, [], f'MAX_UNANNOTATED lists modules that no longer exist: {_gone_}')


class TestRatchetConfigConsistency(unittest.TestCase):
    '''The two halves of the ratchet must agree.  During the migration that meant
    "a module listed in [[tool.mypy.overrides]] must be fully annotated"; strict
    checking is now the package-wide default, so it runs the other way: every
    module is held to disallow_untyped_defs unless the config relaxes it by name,
    and the relaxed set may only ever be PERMANENTLY_RELAXED.'''

    @staticmethod
    def _mypy_cfg():
        _toml_ = Path(__file__).resolve().parent.parent / 'pyproject.toml'
        if not _toml_.is_file(): return None
        return tomllib.loads(_toml_.read_text()).get('tool', {}).get('mypy', {})

    def test_strict_checking_is_the_global_default(self):
        _cfg_ = self._mypy_cfg()
        if _cfg_ is None: self.skipTest('pyproject.toml not present (installed-wheel test run)')
        for _flag_ in ('disallow_untyped_defs', 'disallow_incomplete_defs', 'check_untyped_defs'):
            with self.subTest(setting=_flag_):
                self.assertTrue(_cfg_.get(_flag_),
                                f'[tool.mypy] {_flag_} must be on: strict checking is the floor '
                                'for every module, not a list modules opt into')
        self.assertFalse(_cfg_.get('disable_error_code'),
                         'no error code may be disabled package-wide -- the migration-era '
                         f'disable list is gone: {_cfg_.get("disable_error_code")}')

    def test_relaxed_set_is_exactly_the_documented_one(self):
        _cfg_ = self._mypy_cfg()
        if _cfg_ is None: self.skipTest('pyproject.toml not present (installed-wheel test run)')
        _relaxed_ = []
        for _o_ in _cfg_.get('overrides', []):
            _m_ = _o_.get('module', [])
            _relaxed_.extend([_m_] if isinstance(_m_, str) else _m_)
        _stems_ = {n.rsplit('.', 1)[-1] for n in _relaxed_}
        self.assertEqual(_stems_, set(TestAnnotationCoverageRatchet.PERMANENTLY_RELAXED),
                         'the [[tool.mypy.overrides]] block may only carry the permanently '
                         f'relaxed module(s); found: {sorted(_stems_)}')

    def test_strictly_checked_modules_are_fully_annotated(self):
        _cfg_ = self._mypy_cfg()
        if _cfg_ is None: self.skipTest('pyproject.toml not present (installed-wheel test run)')
        _relaxed_ = []
        for _o_ in _cfg_.get('overrides', []):
            _m_ = _o_.get('module', [])
            _relaxed_.extend([_m_] if isinstance(_m_, str) else _m_)
        _stems_ = {n.rsplit('.', 1)[-1] for n in _relaxed_}
        _ceilings_ = TestAnnotationCoverageRatchet.MAX_UNANNOTATED
        _strict_ = sorted(set(_ceilings_) - _stems_)
        self.assertTrue(_strict_, 'no strictly checked modules found -- MAX_UNANNOTATED is empty?')
        for _name_ in _strict_:
            with self.subTest(module=_name_):
                self.assertEqual(_ceilings_[_name_], 0,
                                 f'{_name_} is held to disallow_untyped_defs by [tool.mypy] '
                                 'but is not fully annotated')


class TestComponentAttrDeclarations(unittest.TestCase):
    '''Every component assigns its parameters onto the instance with setattr()
    (Polars2SVG.assignScratchDefaults / assignKwargsWithDefaults), which no checker
    can follow, so each component class carries a block of bare annotations
    mirroring its `_defaults_`.  Nothing at runtime keeps the two in step.

    `assertParamSpecMatches()` already guarantees `_VALID_KWARGS == _defaults_` at
    construction time; these tests add the third corner of the triangle.'''

    @staticmethod
    def _components():
        from polars2svg.xyp    import XYp
        from polars2svg.linkp  import LinkP
        from polars2svg.chordp import ChP
        from polars2svg.histop import Histop
        from polars2svg.timep  import Timep
        from polars2svg.piep   import Piep
        from polars2svg.smallp import Smallp
        from polars2svg.tile   import Tile
        from polars2svg.spreadlinesp import SpreadLinesP
        return [XYp, LinkP, ChP, Histop, Timep, SpreadLinesP, Piep, Smallp, Tile]

    @staticmethod
    def _source_facts(cls):
        '''(own bare annotations, _defaults_ keys, attrs assigned as self.X = ...)'''
        _pkg_  = Path(inspect.getfile(polars2svg)).parent
        _mod_  = cls.__module__.rsplit('.', 1)[-1]
        _tree_ = ast.parse((_pkg_ / f'{_mod_}.py').read_text())
        _c_    = next(n for n in _tree_.body
                      if isinstance(n, ast.ClassDef) and n.name == cls.__name__)
        _own_  = {n.target.id for n in _c_.body
                  if isinstance(n, ast.AnnAssign) and n.value is None
                  and isinstance(n.target, ast.Name)}
        _defaults_, _assigned_ = set(), set()
        for _n_ in ast.walk(_c_):
            if isinstance(_n_, ast.Assign):
                for _t_ in _n_.targets:
                    if (isinstance(_t_, ast.Name) and _t_.id in ('_defaults_', '_DEFAULTS_')
                            and isinstance(_n_.value, ast.Dict)):
                        _defaults_ |= {k.value for k in _n_.value.keys
                                       if isinstance(k, ast.Constant)}
                    for _e_ in (_t_.elts if isinstance(_t_, ast.Tuple) else [_t_]):
                        if (isinstance(_e_, ast.Attribute) and isinstance(_e_.value, ast.Name)
                                and _e_.value.id == 'self'):
                            _assigned_.add(_e_.attr)
            if isinstance(_n_, ast.AnnAssign) and isinstance(_n_.target, ast.Attribute):
                _assigned_.add(_n_.target.attr)
        return _own_, _defaults_, _assigned_

    def test_every_declaration_is_a_real_attribute(self):
        # Catches a typo, or a declaration left behind when something is renamed.
        # A declaration is legitimate if it is a constructor parameter, or an
        # attribute the class actually assigns (svg, xy_list, ... are built during
        # render rather than passed in, and need declaring for the same reason).
        for _cls_ in self._components():
            with self.subTest(component=_cls_.__name__):
                _own_, _, _assigned_ = self._source_facts(_cls_)
                _stale_ = sorted(_own_ - set(_cls_._VALID_KWARGS) - _assigned_)
                self.assertEqual(_stale_, [],
                                 f'{_cls_.__name__} declares attributes that are neither in '
                                 f'_VALID_KWARGS nor ever assigned: {_stale_}')

    def test_every_parameter_is_declared_or_assigned(self):
        # The "new parameter must be declared" guard.  A _defaults_ key reaches the
        # instance only through setattr(), so unless it is also written directly as
        # self.X = ... it needs a declaration -- here or on a base mixin.
        for _cls_ in self._components():
            with self.subTest(component=_cls_.__name__):
                _own_, _defaults_, _assigned_ = self._source_facts(_cls_)
                _inherited_ = set()
                for _b_ in _cls_.__mro__:
                    _inherited_ |= set(_b_.__dict__.get('__annotations__', {}))
                _undeclared_ = sorted(_defaults_ - _inherited_ - _assigned_)
                self.assertEqual(_undeclared_, [],
                                 f'{_cls_.__name__} has _defaults_ parameters with no '
                                 f'declaration and no direct assignment: {_undeclared_}')

    def test_declarations_are_annotations_only(self):
        for _cls_ in self._components():
            _own_, _, _ = self._source_facts(_cls_)
            for _name_ in sorted(_own_):
                with self.subTest(component=_cls_.__name__, attr=_name_):
                    self.assertNotIn(_name_, _cls_.__dict__,
                                     f'{_cls_.__name__}.{_name_} is assigned at class level; '
                                     'it must be a bare annotation (no value)')


class TestKwargsTypedDicts(unittest.TestCase):
    '''Each component's parameter surface is `**kwargs`, so a caller gets no
    completion and no misspelling check from the signature alone.  Each factory
    method is therefore typed `**kwargs: Unpack[<Component>Kwargs]`.

    The TypedDict is the third statement of the same fact as `_VALID_KWARGS`
    (validated at runtime) and `_defaults_` (the values) -- `assertParamSpecMatches()`
    already ties those two together; this ties in the third.'''

    @staticmethod
    def _pairs():
        from polars2svg.xyp    import XYp,    XYpKwargs
        from polars2svg.linkp  import LinkP,  LinkPKwargs
        from polars2svg.chordp import ChP,    ChPKwargs
        from polars2svg.histop import Histop, HistopKwargs
        from polars2svg.timep  import Timep,  TimepKwargs
        from polars2svg.piep   import Piep,   PiepKwargs
        from polars2svg.smallp import Smallp, SmallpKwargs
        from polars2svg.tile   import Tile,   TileKwargs
        from polars2svg.spreadlinesp import SpreadLinesP, SpreadLinesPKwargs
        return [(XYp, XYpKwargs), (LinkP, LinkPKwargs), (ChP, ChPKwargs),
                (Histop, HistopKwargs), (Timep, TimepKwargs),
                (SpreadLinesP, SpreadLinesPKwargs), (Piep, PiepKwargs),
                (Smallp, SmallpKwargs), (Tile, TileKwargs)]

    def test_keys_match_valid_kwargs_exactly(self):
        for _cls_, _td_ in self._pairs():
            with self.subTest(component=_cls_.__name__):
                _declared_ = set(_td_.__annotations__)
                _valid_    = set(_cls_._VALID_KWARGS)
                self.assertEqual(
                    _declared_, _valid_,
                    f'{_td_.__name__} has drifted from {_cls_.__name__}._VALID_KWARGS; '
                    f'only in the TypedDict: {sorted(_declared_ - _valid_)}; '
                    f'only in _VALID_KWARGS: {sorted(_valid_ - _declared_)}')

    def test_every_key_is_optional(self):
        # total=False -- callers pass any subset, which is what **kwargs means.
        for _cls_, _td_ in self._pairs():
            with self.subTest(component=_cls_.__name__):
                self.assertEqual(_td_.__required_keys__, frozenset(),
                                 f'{_td_.__name__} must be declared total=False')
                self.assertEqual(set(_td_.__optional_keys__), set(_td_.__annotations__))

    def test_factories_unpack_the_matching_typeddict(self):
        # The wiring that makes it reach a caller: p2s.xyp(...) must advertise
        # XYpKwargs, not a bare **kwargs.
        import typing
        for _cls_, _td_ in self._pairs():
            _factory_ = _cls_.__name__.lower().replace('chp', 'chordp').replace('linkp', 'linkp')
            _factory_ = {'XYp': 'xyp', 'LinkP': 'linkp', 'ChP': 'chordp', 'Histop': 'histop',
                         'Timep': 'timep', 'SpreadLinesP': 'spreadlinesp', 'Piep': 'piep',
                         'Smallp': 'smallp', 'Tile': 'tile'}[_cls_.__name__]
            with self.subTest(factory=_factory_):
                _fn_ = getattr(Polars2SVG, _factory_)
                try:
                    _hints_ = typing.get_type_hints(_fn_)
                except NameError:
                    # chordp() alone: ChP/ChPKwargs are imported under TYPE_CHECKING
                    # because chordp.py imports scipy at module level, so resolving
                    # its hints at runtime would need the 'layouts' extra.  Its
                    # return annotation has always been a string for the same
                    # reason -- check the raw annotation instead.
                    _raw_ = _fn_.__annotations__.get('kwargs')
                    self.assertEqual(_raw_, f'Unpack[{_td_.__name__}]',
                                     f'{_factory_}() should unpack {_td_.__name__}')
                    continue
                self.assertIn('kwargs', _hints_, f'{_factory_}() has no kwargs annotation')
                # PEP 692: the annotation is Unpack[TD]; the TypedDict is its arg.
                _unpacked_ = typing.get_args(_hints_['kwargs'])
                self.assertEqual(len(_unpacked_), 1,
                                 f'{_factory_}() kwargs is {_hints_["kwargs"]!r}, '
                                 f'expected Unpack[{_td_.__name__}]')
                self.assertIs(_unpacked_[0], _td_,
                              f'{_factory_}() should unpack {_td_.__name__}')

    def test_exported_from_package_root(self):
        # Callers annotate their own dicts with these, so they are public API.
        for _cls_, _td_ in self._pairs():
            with self.subTest(component=_cls_.__name__):
                self.assertIs(getattr(polars2svg, _td_.__name__, None), _td_,
                              f'{_td_.__name__} is not exported from polars2svg/__init__.py')


class TestAnnotationSanity(unittest.TestCase):
    '''Two whole-package invariants, both learned the hard way while annotating.

    The annotations were seeded from a runtime trace of the test suite, and a
    trace has two blind spots that these close: it never sees a branch the tests
    do not take, and `sys.setprofile` reports a *return of None* when a call
    leaves via an exception -- so a function whose failure paths all `raise` looks
    like it returns None. Both produced wrong annotations that mypy could not
    catch, because a too-narrow annotation is only wrong against code that does
    not run.'''

    @staticmethod
    def _functions():
        _pkg_ = Path(inspect.getfile(polars2svg)).parent
        for _f_ in sorted(_pkg_.glob('*.py')):
            _tree_ = ast.parse(_f_.read_text())
            for _n_ in ast.walk(_tree_):
                if isinstance(_n_, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield _f_.name, _n_

    @staticmethod
    def _own_nodes(fn):
        '''Nodes belonging to fn itself -- nested defs and lambdas are theirs.'''
        _out_ = []
        def _walk_(node):
            for _c_ in ast.iter_child_nodes(node):
                if isinstance(_c_, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)): continue
                _out_.append(_c_)
                _walk_(_c_)
        _walk_(fn)
        return _out_

    @classmethod
    def _own_returns(cls, fn):
        '''`return` statements belonging to fn itself, not to a nested def.'''
        return [_n_ for _n_ in cls._own_nodes(fn) if isinstance(_n_, ast.Return)]

    @classmethod
    def _none_bound_names(cls, fn):
        '''Locals fn assigns the None literal to, `a, b = None, x` included.'''
        _out_ = set()
        for _a_ in cls._own_nodes(fn):
            if not isinstance(_a_, ast.Assign): continue
            for _t_ in _a_.targets:
                _ts_ = list(_t_.elts) if isinstance(_t_, ast.Tuple) else [_t_]
                _vs_ = list(_a_.value.elts) if isinstance(_a_.value, ast.Tuple) else [_a_.value]
                if len(_ts_) != len(_vs_): continue
                for _tt_, _vv_ in zip(_ts_, _vs_):
                    if (isinstance(_tt_, ast.Name) and isinstance(_vv_, ast.Constant)
                            and _vv_.value is None):
                        _out_.add(_tt_.id)
        return _out_

    def test_none_defaults_are_optional(self):
        # `x: int = None` is a lie the checker will not flag on its own.
        _bad_ = []
        for _file_, _n_ in self._functions():
            _a_ = _n_.args
            _pos_ = [*_a_.posonlyargs, *_a_.args]
            _pairs_ = list(zip(_pos_[len(_pos_) - len(_a_.defaults):], _a_.defaults))
            _pairs_ += [(x, d) for x, d in zip(_a_.kwonlyargs, _a_.kw_defaults) if d is not None]
            for _arg_, _d_ in _pairs_:
                if not (isinstance(_d_, ast.Constant) and _d_.value is None): continue
                if _arg_.annotation is None: continue
                _ann_ = ast.unparse(_arg_.annotation)
                if 'None' in _ann_ or _ann_ == 'Any' or _ann_.startswith('Optional'): continue
                _bad_.append(f'{_file_}:{_n_.lineno} {_n_.name}({_arg_.arg}: {_ann_} = None)')
        self.assertEqual(_bad_, [], 'parameters defaulting to None must be Optional: ' + str(_bad_))

    def test_no_parameter_is_annotated_exactly_none(self):
        # `x: None` is what the trace produces for a parameter the tests only ever
        # passed None to. It is not a type -- nothing but None can be passed --
        # and it is almost always a stand-in for `Something | None`. 29 of these
        # shipped before this check existed; they are now `Any`.
        _bad_ = []
        for _file_, _n_ in self._functions():
            _a_ = _n_.args
            for _arg_ in [*_a_.posonlyargs, *_a_.args, *_a_.kwonlyargs, _a_.vararg, _a_.kwarg]:
                if _arg_ is None or _arg_.annotation is None: continue
                if ast.unparse(_arg_.annotation) == 'None':
                    _bad_.append(f'{_file_}:{_n_.lineno} {_n_.name}({_arg_.arg}: None)')
        self.assertEqual(_bad_, [],
                         'parameters annotated exactly `None` (use `Any`, or the real '
                         'type unioned with None): ' + str(_bad_))

    def test_optional_returns_have_a_none_path(self):
        # The inverse: `-> X | None` on a function that always returns an X or
        # raises. Usually a trace artifact -- the profiler logs an exception as a
        # None return.
        _bad_ = []
        for _file_, _n_ in self._functions():
            if _n_.returns is None: continue
            _ann_ = ast.unparse(_n_.returns)
            if 'None' not in _ann_ or _ann_ in ('None', 'Any'): continue
            _rs_ = self._own_returns(_n_)
            # A None path is a bare `return`, `return None`, or any returned
            # expression that mentions None -- `return x or None`,
            # `return a if b else None`.  Anything less strict flags those as
            # artifacts, which they are not.
            #
            # Two shapes yield None with no literal in the `return` itself, and
            # both are real rather than trace artifacts: a one-argument
            # `.get(key)`, which is None when the key is absent, and the
            # accumulator pattern -- `best = None` ... `return best` -- where the
            # None is the initial binding.  Missing either produced a false
            # positive on code whose callers demonstrably guard for None.
            _none_names_ = self._none_bound_names(_n_)
            def _yields_none_(r):
                if r.value is None: return True
                if any(isinstance(c, ast.Constant) and c.value is None
                       for c in ast.walk(r.value)): return True
                if (isinstance(r.value, ast.Call) and isinstance(r.value.func, ast.Attribute)
                        and r.value.func.attr == 'get' and len(r.value.args) == 1): return True
                return isinstance(r.value, ast.Name) and r.value.id in _none_names_
            _nones_ = [r for r in _rs_ if _yields_none_(r)]
            _last_ = _n_.body[-1]
            _falls_off_ = not isinstance(_last_, (ast.Return, ast.Raise, ast.While,
                                                  ast.For, ast.Try, ast.If, ast.With))
            if not _nones_ and not _falls_off_:
                _bad_.append(f'{_file_}:{_n_.lineno} {_n_.name}() -> {_ann_}')
        self.assertEqual(_bad_, [],
                         'these return Optional but have no path that yields None: ' + str(_bad_))

    def test_none_returns_do_not_return_values(self):
        # And the other direction: `-> None` on a function that returns something.
        _bad_ = []
        for _file_, _n_ in self._functions():
            if _n_.returns is None or ast.unparse(_n_.returns) != 'None': continue
            _vals_ = [r for r in self._own_returns(_n_)
                      if r.value is not None
                      and not (isinstance(r.value, ast.Constant) and r.value.value is None)]
            if _vals_:
                _bad_.append(f'{_file_}:{_n_.lineno} {_n_.name}() returns a value at '
                             f'{sorted({r.lineno for r in _vals_})}')
        self.assertEqual(_bad_, [], 'annotated -> None but returns a value: ' + str(_bad_))


class TestPermanentlyRelaxedModule(unittest.TestCase):
    '''`interactive_controller` is excluded from the annotation target by decision,
    not by neglect.  These tests keep the decision visible and bounded.'''

    def test_relaxed_module_exists(self):
        _pkg_ = Path(inspect.getfile(polars2svg)).parent
        for _mod_ in TestAnnotationCoverageRatchet.PERMANENTLY_RELAXED:
            with self.subTest(module=_mod_):
                self.assertTrue((_pkg_ / f'{_mod_}.py').is_file(),
                                f'{_mod_} is listed as permanently relaxed but does not exist')

    def test_relaxed_module_is_still_capped(self):
        # Relaxed means "not expected to reach zero", not "unbounded".
        for _mod_ in TestAnnotationCoverageRatchet.PERMANENTLY_RELAXED:
            with self.subTest(module=_mod_):
                self.assertIn(_mod_, TestAnnotationCoverageRatchet.MAX_UNANNOTATED,
                              f'{_mod_} must still carry a ceiling')

    def test_relaxed_module_is_not_in_the_strict_ratchet(self):
        _root_ = Path(__file__).resolve().parent.parent
        _toml_ = _root_ / 'pyproject.toml'
        if not _toml_.is_file():
            self.skipTest('pyproject.toml not present (installed-wheel test run)')
        _cfg_ = tomllib.loads(_toml_.read_text())
        _strict_ = []
        for _o_ in _cfg_.get('tool', {}).get('mypy', {}).get('overrides', []):
            if not _o_.get('disallow_untyped_defs'): continue
            _m_ = _o_.get('module', [])
            _strict_.extend([_m_] if isinstance(_m_, str) else _m_)
        _stems_ = {n.rsplit('.', 1)[-1] for n in _strict_}
        _clash_ = sorted(TestAnnotationCoverageRatchet.PERMANENTLY_RELAXED & _stems_)
        self.assertEqual(_clash_, [],
                         f'these are both permanently relaxed and strictly checked: {_clash_}')

    def test_every_other_module_reached_zero(self):
        # The completion claim for phase 3, asserted rather than described.
        _remaining_ = {m: c for m, c in TestAnnotationCoverageRatchet.MAX_UNANNOTATED.items()
                       if c > 0 and m not in TestAnnotationCoverageRatchet.PERMANENTLY_RELAXED}
        self.assertEqual(_remaining_, {},
                         'these modules are neither fully annotated nor permanently '
                         f'relaxed: {_remaining_}')


if __name__ == '__main__':
    unittest.main()
