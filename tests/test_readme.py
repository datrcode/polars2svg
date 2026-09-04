"""README.md is executable documentation, and this file is what makes that true.

The README is the project's highest-traffic artifact and, until this file existed,
the only one with no test behind it.  That combination had already produced five
defects at once (2026-09-04 audit): a signature claim that raised ``TypeError`` if
you followed it, an interactive variant that does not exist, a diagnostic function
that is absent in exactly the broken install it is recommended for, a maintainer
script naming a path that only exists in one of the two repos, and three hero
images returning 404 from the public repo.

Two properties of this project make that failure mode structural rather than
unlucky, and both shape the checks below:

1. **README.md is ported byte-identical from `_dev` to production, while `docs/`
   deliberately is not** (see ``tools/diff_packaging_prod.sh``).  So a path that
   resolves here can still 404 on PyPI, and a claim can be true in one repo and
   false in the other.  ``TestReadmeAssetsResolve`` therefore checks paths against
   *the repo it is running in*, which is the point: it has to fail in production
   for production's missing images to be visible at all.
2. **Only production runs CI.**  Anything these tests are meant to catch on the
   public side is caught after the port, not before, so the failure messages name
   the fix rather than just the mismatch.

The prose is checked as well as the code blocks.  Three of the five defects were
in sentences, not snippets -- claims about an API that no code block demonstrated,
which is precisely how a README ends up asserting something no one ever ran.
"""
import ast
import os
import re
import tempfile
import tomllib
import unittest
from typing import Any

import polars2svg
from polars2svg import Polars2SVG

_HERE_      = os.path.dirname(os.path.abspath(__file__))
_ROOT_      = os.path.dirname(_HERE_)
_README_    = os.path.join(_ROOT_, 'README.md')
_PYPROJECT_ = os.path.join(_ROOT_, 'pyproject.toml')

# Names the README documents as conditionally exported -- they live behind an
# optional extra in polars2svg/__init__.py's `try: ... except ImportError: pass`
# guards, so they are legitimately absent from a bare install.  They are still
# checked, just conditionally: present-if-importable, rather than unconditionally
# present.  Adding a name here is a claim that the README explains the guard.
_GUARDED_EXPORTS_ = {
    'gpu_backend': 'polars2svg.tfdp_layout',
    'TFDPLayout':  'polars2svg.tfdp_layout',
}

# Names the README's `p2s.tile(...)` block uses illustratively -- it shows call
# shapes rather than being a runnable script, so these are seeded rather than
# defined on the page.  Keep this list as short as the README allows: every entry
# is a line of documentation the reader cannot copy verbatim.
_ILLUSTRATIVE_NAMES_ = ('chart_a', 'chart_b', 'charts')


def _read_readme() -> str:
    with open(_README_, encoding='utf-8') as _f_:
        return _f_.read()


def _code_blocks(md: str, lang: str) -> list[tuple[int, str]]:
    """Every fenced block of `lang`, as (1-based opening-fence line, body)."""
    _blocks_: list[tuple[int, str]] = []
    _cur_:    list[str] | None      = None
    _start_:  int                   = 0
    for _i_, _line_ in enumerate(md.split('\n'), start=1):
        if _cur_ is None:
            _m_ = re.match(r'^```(\w*)\s*$', _line_)
            if _m_ is not None and _m_.group(1) == lang:
                _cur_, _start_ = [], _i_
        elif _line_.strip() == '```':
            _blocks_.append((_start_, '\n'.join(_cur_)))
            _cur_ = None
        else:
            _cur_.append(_line_)
    return _blocks_


def _is_counter_example(body: str) -> bool:
    """A block the README presents as *not* working.

    Marked by the same '# x' comment the rendered page already shows the reader,
    so the marker is visible documentation rather than invisible test metadata.
    """
    return body.lstrip().startswith('# ✗')


class TestReadmeSnippetsExecute(unittest.TestCase):
    """Every ```python block on the page runs, in document order, sharing state.

    Sharing one namespace is deliberate: the README's later blocks reuse `df`,
    `p2s` and `chart` from earlier ones, and a reader working top-to-bottom gets
    exactly this. Running them independently would pass while the page as read
    was broken.
    """

    def test_all_python_blocks_execute(self) -> None:
        _md_     = _read_readme()
        _blocks_ = _code_blocks(_md_, 'python')
        self.assertGreater(len(_blocks_), 0, 'no ```python blocks found in README.md')

        _ns_: dict[str, Any] = {'__name__': '__readme__'}
        # Seeded so the tile block's call shapes are still type/arity checked.
        _p2s_ = Polars2SVG()
        import polars as pl
        _seed_ = _p2s_.xyp(pl.DataFrame({'x': [1, 2], 'y': [3, 4]}), 'x', 'y', wxh=(120, 90))
        for _n_ in _ILLUSTRATIVE_NAMES_:
            _ns_[_n_] = [_seed_, _seed_] if _n_ == 'charts' else _seed_

        _ran_, _skipped_ = 0, []
        _cwd_ = os.getcwd()
        with tempfile.TemporaryDirectory() as _tmp_:
            # The 'save it' block writes scatter.svg into the working directory.
            os.chdir(_tmp_)
            try:
                for _line_, _body_ in _blocks_:
                    if _is_counter_example(_body_):
                        continue    # asserted separately, and asserted to *fail*
                    try:
                        exec(compile(_body_, f'README.md:{_line_}', 'exec'), _ns_)
                        _ran_ += 1
                    except ImportError as _e_:
                        # The README promises: "Calling a component that needs an
                        # extra you haven't installed raises a clear ImportError
                        # naming the extra".  Honour that here so a bare install
                        # skips the block instead of failing it -- but only for
                        # that exact shape.  Any other ImportError is a real bug.
                        if 'polars2svg[' not in str(_e_):
                            raise
                        _skipped_.append(f'README.md:{_line_} ({_e_})')
                    except Exception as _e_:
                        self.fail(
                            f'README.md:{_line_} -- a documented snippet raised '
                            f'{type(_e_).__name__}: {_e_}\n\n'
                            f'The block was:\n{_body_}'
                        )
            finally:
                os.chdir(_cwd_)

        self.assertGreater(_ran_, 0, f'every python block was skipped: {_skipped_}')


class TestReadmeCounterExampleStillFails(unittest.TestCase):
    """The anti-hallucination section is an invariant, not a claim.

    README.md carries a section rebutting a `from polars2svg import display_svg`
    snippet that circulates in search results and AI-generated answers.  That
    section asserts the name has never existed "in any released version" -- so if
    anyone ever adds it, the README silently becomes the wrong answer to a
    question people are actively arriving with.  These tests fail first.
    """

    def test_readme_marks_at_least_one_counter_example(self) -> None:
        _blocks_ = _code_blocks(_read_readme(), 'python')
        self.assertTrue(
            any(_is_counter_example(_b_) for _, _b_ in _blocks_),
            "no counter-example block found -- if the 'snippet that doesn't work' "
            'section was removed, remove these tests with it; if its marker '
            'changed, update _is_counter_example()',
        )

    def test_display_svg_is_not_importable(self) -> None:
        with self.assertRaises(ImportError):
            # The ignore is the assertion restated for the type checker: mypy
            # flagging this name as absent is the property under test.
            from polars2svg import display_svg  # type: ignore[attr-defined] # noqa: F401

    def test_display_svg_is_not_on_the_package(self) -> None:
        self.assertFalse(
            hasattr(polars2svg, 'display_svg'),
            'README.md states there is no display_svg in polars2svg, and there '
            'now is one -- the counter-example section is misleading readers',
        )


class TestReadmeApiNamesResolve(unittest.TestCase):
    """Every `p2s.<name>` and `polars2svg.<name>` the page mentions is real.

    This is the check that catches the hallucination class directly, and it reads
    prose as well as code: `p2s.spreadlinespi` was implied by a sentence, not by
    any runnable block.
    """

    def _mentions(self, pattern: str) -> set[str]:
        _md_ = _read_readme()
        # Drop counter-example blocks: they cite names that are *supposed* to be absent.
        for _, _body_ in _code_blocks(_md_, 'python'):
            if _is_counter_example(_body_):
                _md_ = _md_.replace(_body_, '')
        return set(re.findall(pattern, _md_))

    def test_instance_methods_and_enums_resolve(self) -> None:
        _p2s_     = Polars2SVG()
        _missing_ = sorted(
            _n_ for _n_ in self._mentions(r'\bp2s\.([A-Za-z_][A-Za-z0-9_]*)')
            if not hasattr(_p2s_, _n_)
        )
        self.assertEqual(
            _missing_, [],
            'README.md documents p2s.<name> that does not exist on a Polars2SVG '
            f'instance: {_missing_}',
        )

    def test_package_level_names_resolve(self) -> None:
        _unconditional_: list[str] = []
        _guarded_:       list[str] = []
        for _n_ in sorted(self._mentions(r'\bpolars2svg\.([A-Za-z_][A-Za-z0-9_]*)')):
            if hasattr(polars2svg, _n_):
                continue
            (_guarded_ if _n_ in _GUARDED_EXPORTS_ else _unconditional_).append(_n_)

        self.assertEqual(
            _unconditional_, [],
            'README.md documents polars2svg.<name> that is not exported: '
            f'{_unconditional_}. If the name is behind an optional extra, add it '
            'to _GUARDED_EXPORTS_ *and* say so in the README -- an unexplained '
            'guard is how gpu_backend() came to be recommended for diagnosing '
            'the one install where it is missing.',
        )

        # A guarded name absent here is fine (this environment lacks its extra);
        # a guarded name whose module *is* importable must be on the package.
        for _n_ in _guarded_:
            _mod_ = _GUARDED_EXPORTS_[_n_]
            try:
                __import__(_mod_)
            except ImportError:
                continue
            self.fail(f'{_mod_} imports but polars2svg.{_n_} is not exported from it')


class TestReadmeAssetsResolve(unittest.TestCase):
    """Every path the page points at exists *in this repo*.

    Deliberately repo-relative.  README.md is ported byte-identical to production
    while docs/ is not, so this failing in one repo and passing in the other is
    the signal, not a false alarm.
    """

    def _links(self) -> list[str]:
        return re.findall(r'\]\(([^)]+)\)', _read_readme())

    def test_relative_links_exist(self) -> None:
        _missing_ = sorted({
            _p_ for _p_ in self._links()
            if not _p_.startswith(('http://', 'https://', '#', 'mailto:'))
            and not os.path.exists(os.path.join(_ROOT_, _p_.split('#')[0]))
        })
        self.assertEqual(_missing_, [], f'README.md links to missing paths: {_missing_}')

    def test_anchor_links_have_headings(self) -> None:
        _md_ = _read_readme()
        _slugs_ = {
            re.sub(r'[^a-z0-9 -]', '', _h_.lower()).replace(' ', '-')
            for _h_ in re.findall(r'^#{1,6}\s+(.*?)\s*$', _md_, flags=re.MULTILINE)
        }
        _missing_ = sorted({
            _p_ for _p_ in self._links()
            if _p_.startswith('#') and _p_[1:] not in _slugs_
        })
        self.assertEqual(_missing_, [], f'README.md has dead anchor links: {_missing_}')

    def test_hosted_images_exist_in_this_repo(self) -> None:
        """The gallery <img> tags resolve to files this repo actually ships.

        They are absolute raw.githubusercontent.com URLs pointing into this
        project's own default branch, so the file has to be committed *here* for
        the tag to render on GitHub and on the PyPI project page.  It was not:
        docs/images/ has never existed in the public repo, and all three hero
        images returned 404 from PyPI.
        """
        _refs_ = re.findall(
            r'https://raw\.githubusercontent\.com/[^/]+/[^/]+/[^/]+/([^"\'\s>]+)',
            _read_readme(),
        )
        self.assertGreater(len(_refs_), 0, 'no hosted image references found')
        _missing_ = sorted({_p_ for _p_ in _refs_ if not os.path.exists(os.path.join(_ROOT_, _p_))})
        self.assertEqual(
            _missing_, [],
            f'README.md embeds images this repo does not ship: {_missing_}. These '
            'render as broken images on GitHub and on the PyPI project page. '
            'docs/ is excluded from the dev->prod port (tools/diff_packaging_prod.sh), '
            'so these files must be committed to production explicitly.',
        )


class TestReadmeExtrasExist(unittest.TestCase):
    """Every `pip install polars2svg[x]` on the page names a real extra.

    The install block is the most-copied part of any README, and its extras are
    the project's most-churned metadata (mlx-cpu / mlx-cuda / mlx-cuda13 all
    arrived after the section was first written; mlx-cuda13 had gone missing
    from one of the two install blocks).
    """

    def test_documented_extras_are_declared(self) -> None:
        with open(_PYPROJECT_, 'rb') as _f_:
            _declared_ = set(tomllib.load(_f_)['project'].get('optional-dependencies', {}))

        _documented_: set[str] = set()
        for _spec_ in re.findall(r'polars2svg\[([a-zA-Z0-9,_\-]+)\]', _read_readme()):
            _documented_.update(_e_.strip() for _e_ in _spec_.split(','))

        self.assertGreater(len(_documented_), 0, 'no extras documented in README.md')
        _unknown_ = sorted(_documented_ - _declared_)
        self.assertEqual(
            _unknown_, [],
            'README.md documents extras absent from pyproject.toml '
            f'[project.optional-dependencies]: {_unknown_}',
        )


class TestReadmeIsSyntacticallyClaimable(unittest.TestCase):
    """Blocks the executor skips are still parsed, so a counter-example cannot rot
    into something that is merely malformed rather than meaningfully wrong."""

    def test_every_python_block_parses(self) -> None:
        for _line_, _body_ in _code_blocks(_read_readme(), 'python'):
            with self.subTest(block=f'README.md:{_line_}'):
                try:
                    ast.parse(_body_)
                except SyntaxError as _e_:
                    self.fail(f'README.md:{_line_} is not valid Python: {_e_}')


if __name__ == '__main__':
    unittest.main()
