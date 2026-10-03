#
# p2s_help - one parameter's documentation, sliced out of a component's docstring
#
#   p2s.help('xyp', 'x_distributions')      # just that parameter
#   p2s.help('xyp')                         # the whole docstring, as help(p2s.xyp) would
#
# The component docstrings are long (xyp's is ~18k characters) but regular, which is what
# makes a slicer enough (PLANNING.md §7 F16, DT 2026-10-03):
#
#   - '=== %< === %< ...' lines split a docstring into topic chunks;
#   - a parameter is introduced by a definition line, `name = form   # comment`, where name
#     may be an alternation (`x_order|y_order`), and further forms continue on lines that
#     start with '=' under it.
#
# A chunk that defines only a few parameters is about one topic -- the distributions chunk,
# the line chunk -- so the whole chunk is returned: its prose is the explanation.  A chunk
# that defines many is a list ('Render Options'), so only the parameter's own block is
# returned: its definition line and everything indented under it.
#
# Only the docstring is read.  chordp's class needs scipy, and help() must not.
#
import difflib
import inspect
import re

_SEPARATOR_ = re.compile(r'^\s*=== %<')
_DEFINE_    = re.compile(r'^(?P<indent>\s*)(?P<names>[A-Za-z_]\w*(?:\s*\|\s*[A-Za-z_]\w*)*)\s*=(?!=)')

# A chunk defining more than this many top-level parameters is a list, not a topic.
_TOPIC_MAX_PARAMETERS_ = 3


def _chunks_(doc: str) -> list[list[str]]:
    _out_: list[list[str]] = [[]]
    for _ln_ in inspect.cleandoc(doc).splitlines():
        if _SEPARATOR_.match(_ln_): _out_.append([])
        else:                       _out_[-1].append(_ln_)
    return [_c_ for _c_ in _out_ if any(_l_.strip() for _l_ in _c_)]


def _definitions_(chunk: list[str]) -> list[tuple[int, int, list[str]]]:
    """(line index, indent, names) for every definition line in the chunk."""
    _out_ = []
    for _i_, _ln_ in enumerate(chunk):
        _m_ = _DEFINE_.match(_ln_)
        if _m_:
            _out_.append((_i_, len(_m_.group('indent')), [_n_.strip() for _n_ in _m_.group('names').split('|')]))
    return _out_


def _indent_(line: str) -> int:
    return len(line) - len(line.lstrip())


def _block_(chunk: list[str], start: int, indent: int) -> list[str]:
    """The definition line at `start` and what belongs to it: '=' continuation lines and
    anything indented deeper, including blank lines inside that indented text."""
    _out_ = [chunk[start]]
    _i_   = start + 1
    while _i_ < len(chunk):
        _ln_ = chunk[_i_]
        if not _ln_.strip():
            _nxt_ = next((_l_ for _l_ in chunk[_i_ + 1:] if _l_.strip()), None)
            if _nxt_ is None or _indent_(_nxt_) <= indent: break
        elif _indent_(_ln_) <= indent and not _ln_.lstrip().startswith('='):
            break
        _out_.append(_ln_)
        _i_ += 1
    return _out_


def documentedParameters(doc: str) -> set[str]:
    """Every name a definition line in `doc` introduces, at any depth."""
    return {_n_ for _c_ in _chunks_(doc) for _, _, _names_ in _definitions_(_c_) for _n_ in _names_}


def parameterSection(doc: str, parameter: str) -> str | None:
    """The documentation of `parameter` in `doc`, or None when no definition line names it.
    Several sections (a parameter documented in two places) are joined by a blank line."""
    _found_: list[str] = []
    for _chunk_ in _chunks_(doc):
        _defs_ = _definitions_(_chunk_)
        _hits_ = [(_i_, _ind_) for _i_, _ind_, _names_ in _defs_ if parameter in _names_]
        if not _hits_: continue
        _top_    = min(_ind_ for _, _ind_, _ in _defs_)
        _topics_ = {tuple(_names_) for _, _ind_, _names_ in _defs_ if _ind_ == _top_}
        if len(_topics_) <= _TOPIC_MAX_PARAMETERS_:
            _text_ = '\n'.join(_chunk_).strip('\n')
        else:
            _text_ = '\n\n'.join('\n'.join(_block_(_chunk_, _i_, _ind_)) for _i_, _ind_ in _hits_)
        if _text_ not in _found_: _found_.append(_text_)
    return '\n\n'.join(_found_) if _found_ else None


def helpText(component: str, doc: str, parameter: str | None = None) -> str:
    """What p2s.help() prints.  Raises ValueError naming close matches for an unknown parameter."""
    if parameter is None: return inspect.cleandoc(doc)
    _section_ = parameterSection(doc, parameter)
    if _section_ is not None: return f'{component}({parameter}=...)\n\n{_section_}'
    _known_ = sorted(documentedParameters(doc))
    _close_ = difflib.get_close_matches(parameter, _known_, n=3, cutoff=0.6)
    _hint_  = (f' -- did you mean {", ".join(repr(_c_) for _c_ in _close_)}?' if _close_
               else f" -- help('{component}') shows the whole docstring")
    raise ValueError(f'{component}: no documentation for parameter {parameter!r}{_hint_}')
