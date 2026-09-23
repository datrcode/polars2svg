'''The seam between the tests and however a Panel view happens to carry its JavaScript.

Roughly forty assertions reach into `type(view)._scripts[...]` and `cls._template` as
strings.  Both are `ReactiveHTML` concepts: an ESM/`JSComponent` view has neither -- it
has one `_esm` module that builds its own DOM and defines its handlers as ordinary
functions (PLANNING.md **W1**).  Routing every assertion through here means the migration
changes these three functions instead of forty call sites.

The granularity matters and is worth the brace matcher below.  The cheap move would be to
flatten every assertion to a grep over the whole module, but then an assertion *about*
`menuCommit` passes because the string it wants happens to appear in `myOnKeyDown`.
`component_script()` keeps each assertion pointed at the handler it names, under either
API.

One thing this seam deliberately does *not* hide: JavaScript reads a param as `data.x`
under `ReactiveHTML` and `model.x` under an ESM component.  Only three assertions in the
suite spell that out, so they are left to say what the source really says and will be
updated with their component -- a helper that rewrote one accessor into the other would
be a test lying about the code it reads.
'''
import re as _re


def _cls(view):
    '''Accept either a view instance or its class -- call sites pass both.'''
    return view if isinstance(view, type) else type(view)


def _esm_of(view):
    _esm_ = getattr(_cls(view), '_esm', None)
    return _esm_ if isinstance(_esm_, str) and _esm_.strip() else None


def _scripts_of(view):
    _s_ = getattr(_cls(view), '_scripts', None)
    return _s_ if isinstance(_s_, dict) else {}


def component_js(view) -> str:
    '''Every line of JavaScript the view ships, as one string.

    Use for "does this component mention X anywhere" checks.  Prefer
    `component_script()` when the assertion is about one named handler.
    '''
    _esm_ = _esm_of(view)
    if _esm_ is not None:
        return _esm_
    return '\n'.join(str(_v_) for _v_ in _scripts_of(view).values())


def component_markup(view) -> str:
    '''The source that describes the view's DOM.

    Under `ReactiveHTML` that is the `_template` string, and assertions match the markup
    as written (`id="selectedlabels"`).  An ESM view has no markup: it builds the DOM
    imperatively, so the same id appears as a JS string literal (`'selectedlabels'`) in
    the module.  Assertions that spell out markup syntax therefore have to be relaxed to
    the bare name when their component is ported -- there is no way to preserve both, and
    routing them through here is what makes that a per-port edit rather than a hunt.
    '''
    _esm_ = _esm_of(view)
    if _esm_ is not None:
        return _esm_
    _tmpl_ = getattr(_cls(view), '_template', None)
    return _tmpl_ if isinstance(_tmpl_, str) else ''


def rendered_markup(view) -> str:
    '''The view's markup *after* `ReactiveHTML` runs `_template` through jinja.

    A handful of assertions check the rendered form as well as the source, to prove a
    `{% if %}` resolved the way the class intended.  An ESM view has no jinja -- the
    equivalent branch is `if (model.use_webgpu)` and only a browser can observe which way
    it went -- so this degrades to the source, and the real check becomes a DOM assertion
    in tests/interaction/ when the component is ported.
    '''
    _get_ = getattr(view, '_get_template', None)
    if _esm_of(view) is None and callable(_get_):
        return _get_()[0]
    return component_markup(view)


def component_node_ids(view) -> set:
    '''The `id`s of the elements the view creates.

    `ReactiveHTML` hands these back as the second element of `_get_template()` -- the
    nodes it will wire up as JS variables.  An ESM view creates its own elements, so the
    ids are read out of the module source instead; both `id: 'x'` (an attrs object) and
    `id="x"` (a markup string) are recognised, which covers either way a port writes it.

    Note this is a *set*, so `assertIn('gpucanvas', component_node_ids(v))` is exact
    membership rather than a substring match that could hit a comment.
    '''
    _esm_ = _esm_of(view)
    if _esm_ is None:
        _get_ = getattr(view, '_get_template', None)
        return set(_get_()[1]) if callable(_get_) else set()
    return set(_re.findall(r"""\bid\s*[:=]\s*['"]([A-Za-z_][\w-]*)['"]""", _esm_))


def component_source(view) -> str:
    '''Markup and JavaScript together -- "does this view carry X at all".'''
    _esm_ = _esm_of(view)
    if _esm_ is not None:
        return _esm_
    return component_markup(view) + '\n' + component_js(view)


def script_names(view) -> set:
    '''The named handlers/helpers the view defines.

    `ReactiveHTML` names them as `_scripts` keys; an ESM module names them as function
    declarations.  Lifecycle and param-change entries are not function declarations in
    the ESM form (they are `model.on(...)` callbacks), so a name that exists as a
    `_scripts` key may legitimately not appear here after a port -- `has_script()` is
    what most call sites actually want.
    '''
    _esm_ = _esm_of(view)
    if _esm_ is None:
        return set(_scripts_of(view))
    return {_n_ for _n_ in _iter_function_names(_esm_)}


def has_script(view, name: str) -> bool:
    '''Whether the view defines `name` -- as a `_scripts` key, an ESM function
    declaration, or an ESM `model.on('name', ...)` registration.'''
    _esm_ = _esm_of(view)
    if _esm_ is None:
        return name in _scripts_of(view)
    if name in script_names(view):
        return True
    return (f"model.on('{name}'" in _esm_) or (f'model.on("{name}"' in _esm_)


def named_scripts(view) -> dict:
    '''`{name: source}` covering **all** of the view's JavaScript, partitioned.

    For scanners that want to name the offending handler in their failure message.  The
    partition is exact and lossless under both APIs, which is what lets a scan over it
    claim to have seen everything:

    * `ReactiveHTML` -- one entry per `_scripts` key (list values joined).
    * ESM -- one entry per `function NAME(...)`, plus a `'<module>'` entry holding
      everything outside them (top-level setup and `model.on(...)` callbacks), so a
      binding written in an arrow callback is not invisible to the scan.
    '''
    _esm_ = _esm_of(view)
    if _esm_ is None:
        return {_k_: '\n'.join(_v_ if isinstance(_v_, list) else [_v_])
                for _k_, _v_ in _scripts_of(view).items()}
    _out_, _rest_ = {}, _esm_
    for _name_ in sorted(script_names(view)):
        _body_ = _extract_function(_esm_, _name_)
        if _body_:
            _out_[_name_] = _body_
            _rest_ = _rest_.replace(_body_, '\n', 1)
    _out_['<module>'] = _rest_
    return _out_


def component_script(view, name: str) -> str:
    '''The source of one named handler or helper.

    `ReactiveHTML`: the `_scripts[name]` body.  ESM: the text of `function name(...)
    {...}`, brace-matched.  Raises `KeyError` when the view does not define it, so a
    typo or a rename fails loudly instead of quietly asserting against an empty string.
    '''
    _esm_ = _esm_of(view)
    if _esm_ is None:
        _scripts_ = _scripts_of(view)
        if name not in _scripts_:
            raise KeyError(f'{_cls(view).__name__} has no _scripts[{name!r}] '
                           f'(has: {sorted(_scripts_)})')
        return str(_scripts_[name])
    _body_ = _extract_function(_esm_, name)
    if _body_ is None:
        raise KeyError(f'{_cls(view).__name__}._esm defines no function {name!r} '
                       f'(defines: {sorted(script_names(view))})')
    return _body_


# ---------------------------------------------------------------------------
# ESM source scanning
#
# The component JS has no regex literals, so blanking comments and string literals is
# enough to match braces reliably.  TestBraceMatcher in test_js_assets.py is what holds
# that.
# ---------------------------------------------------------------------------

def strip_noise(text: str) -> str:
    """Blank out comment and string-literal *content*, preserving length and newlines.

    Offsets into the result are offsets into the original, which is what lets the brace
    matcher below slice the real source, and what lets a scanner report a real line
    number.

    ONE left-to-right pass, and that is the whole point rather than an implementation
    detail.  A sequence of independent regexes -- strip `//` comments, then strip
    `'...'` -- gets `'http://www.w3.org/2000/svg'` wrong: the comment pass eats from the
    `//` to the end of the line, taking the closing quote and the statement's `;` with
    it, and everything after that is parsed in the wrong state.  test_js_assets.py had
    exactly that bug and it silently mis-read the declarations of any file with a URL in
    it.  A single pass cannot make that mistake, because a `//` inside a string is
    already inside a string when it is reached.
    """
    _out_ = list(text)
    _i_, _n_ = 0, len(text)

    def _blank_(_j_: int) -> None:
        if text[_j_] != '\n':
            _out_[_j_] = ' '

    while _i_ < _n_:
        _c_ = text[_i_]
        if _c_ == '/' and _i_ + 1 < _n_ and text[_i_ + 1] == '/':
            while _i_ < _n_ and text[_i_] != '\n':
                _out_[_i_] = ' '
                _i_ += 1
        elif _c_ == '/' and _i_ + 1 < _n_ and text[_i_ + 1] == '*':
            _out_[_i_] = _out_[_i_ + 1] = ' '
            _i_ += 2
            while _i_ < _n_:
                if text[_i_] == '*' and _i_ + 1 < _n_ and text[_i_ + 1] == '/':
                    _out_[_i_] = _out_[_i_ + 1] = ' '
                    _i_ += 2
                    break
                _blank_(_i_)
                _i_ += 1
        elif _c_ in ('"', "'", '`'):
            _quote_ = _c_
            _i_ += 1
            while _i_ < _n_:
                if text[_i_] == '\\':
                    _blank_(_i_)
                    if _i_ + 1 < _n_:
                        _blank_(_i_ + 1)
                    _i_ += 2
                    continue
                if text[_i_] == _quote_:
                    break
                # A ', " or newline inside a template literal is ordinary content; only
                # the matching backtick ends it.
                _blank_(_i_)
                _i_ += 1
            _i_ += 1
        else:
            _i_ += 1
    return ''.join(_out_)


#: The old private name, kept because the brace matcher below and its tests use it.
_strip_noise = strip_noise


def declaration_column(text: str, name: str) -> int:
    """The indentation of the line `function NAME(` is declared on, or -1.

    Nesting depth, cheaply and without a parser: a function declared at a deeper column
    than another is inside it.  That is what tells a *nested* function apart from one the
    brace matcher ran past the end into, which look identical from the extracted text
    alone.

    The LINE's indentation, not the offset of the `function` keyword: an entry module
    writes `export function render(...)`, which would otherwise measure as column 7 and
    read as more deeply nested than the handlers inside it.
    """
    _clean_ = strip_noise(text)
    _i_ = 0
    while True:
        _i_ = _clean_.find('function ', _i_)
        if _i_ < 0:
            return -1
        _rest_ = _clean_[_i_ + len('function '):]
        _open_ = _rest_.find('(')
        if _open_ > 0 and _rest_[:_open_].strip() == name:
            _bol_ = _clean_.rfind('\n', 0, _i_) + 1
            return len(_clean_[_bol_:_i_]) - len(_clean_[_bol_:_i_].lstrip())
        _i_ += len('function ')


def _iter_function_names(text: str):
    '''Names of top-level `function NAME(` declarations.'''
    _clean_ = _strip_noise(text)
    _i_ = 0
    while True:
        _i_ = _clean_.find('function ', _i_)
        if _i_ < 0:
            return
        _rest_ = _clean_[_i_ + len('function '):]
        _open_ = _rest_.find('(')
        if _open_ > 0:
            _name_ = _rest_[:_open_].strip()
            if _name_.isidentifier():
                yield _name_
        _i_ += len('function ')


def _extract_function(text: str, name: str):
    '''`function name(...) { ... }` including the header, or None.'''
    _clean_ = _strip_noise(text)
    _start_ = -1
    _i_ = 0
    while True:
        _i_ = _clean_.find('function ', _i_)
        if _i_ < 0:
            break
        _rest_ = _clean_[_i_ + len('function '):]
        _open_ = _rest_.find('(')
        if _open_ > 0 and _rest_[:_open_].strip() == name:
            _start_ = _i_
            break
        _i_ += len('function ')
    if _start_ < 0:
        return None
    # Step over the parameter list before looking for the body.  `render({ model, el })`
    # destructures, so the first `{` after the name is a *parameter* brace, not the body.
    _paren_ = _clean_.find('(', _start_)
    if _paren_ < 0:
        return None
    _depth_ = 0
    _after_params_ = -1
    for _j_ in range(_paren_, len(_clean_)):
        if _clean_[_j_] == '(':
            _depth_ += 1
        elif _clean_[_j_] == ')':
            _depth_ -= 1
            if _depth_ == 0:
                _after_params_ = _j_ + 1
                break
    if _after_params_ < 0:
        return None
    _brace_ = _clean_.find('{', _after_params_)
    if _brace_ < 0:
        return None
    _depth_ = 0
    for _j_ in range(_brace_, len(_clean_)):
        if _clean_[_j_] == '{':
            _depth_ += 1
        elif _clean_[_j_] == '}':
            _depth_ -= 1
            if _depth_ == 0:
                return text[_start_:_j_ + 1]
    return None
