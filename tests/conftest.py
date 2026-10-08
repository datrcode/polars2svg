"""Shared pytest fixtures for the polars2svg test suite.

Color state is per-instance, so there is nothing here to reset any more.

There used to be an autouse fixture clearing ``color_overrides_lu`` before and after
every test: ``Polars2SVG()`` returned a process-wide singleton and the override map was
created once and reused, so ``setColorOverrides({'A': '#ff0000'})`` in one test poisoned
every later test that colored by a category named ``A`` -- an order-dependent failure
that only showed up in certain selections. Instances are now independent (each
``__init__`` builds its own ``to_color_lu`` / ``color_overrides_lu``), and every test
that sets overrides builds its own instance in ``setUp``, so the isolation is structural
rather than something a fixture has to keep restoring.
"""
import os
from collections.abc import Generator
from typing import Any

import pytest

# Under pytest-xdist each worker is its own process with its own Polars thread pool, and
# Polars sizes that pool to every core: 18 workers x 18 threads stalled the suite for
# 27+ minutes on polars 2.0, where two threads each run it in about 32 s (PLANNING.md
# V12).  Polars reads POLARS_MAX_THREADS once, at import, so this must run before
# anything imports polars -- xdist exports PYTEST_XDIST_WORKER before it loads this
# file.  A value already in the environment wins.  A serial run keeps every thread, as
# users do.  The check turns a pool that escaped the cap into an error at startup
# rather than a stall.
if os.environ.get('PYTEST_XDIST_WORKER'):
    os.environ.setdefault('POLARS_MAX_THREADS', '2')
    import polars as _pl_
    if _pl_.thread_pool_size() != int(os.environ['POLARS_MAX_THREADS']):
        raise RuntimeError(f'polars was imported before tests/conftest.py could cap its thread pool '
                           f'({_pl_.thread_pool_size()} threads, not {os.environ["POLARS_MAX_THREADS"]}); '
                           'an uncapped -n auto run stalls (PLANNING.md V12)')


def pytest_addoption(parser):
    """--interaction: run the browser-driven suite under tests/interaction/.

    Declared here rather than in tests/interaction/conftest.py because pytest only
    honours pytest_addoption from *initial* conftests -- those on the path from the
    rootdir down to the arguments.  Defined one level down, `pytest tests/
    --interaction` dies with "unrecognized arguments" and only `pytest
    tests/interaction --interaction` works, which is a trap rather than a gate.
    The option is consumed by tests/interaction/conftest.py, which owns the skip
    logic and the fixtures.
    """
    parser.addoption('--interaction', action='store_true', default=False,
                     help='run the browser-driven interaction tests (require chromium)')


_WIRE_SAFE_ = (str, int, float, bool, type(None))


@pytest.hookimpl(wrapper=True)
def pytest_report_to_serializable(report: pytest.TestReport) -> Generator[None, dict[str, Any] | None, dict[str, Any] | None]:
    """Let subtest reports cross the pytest-xdist worker boundary.

    xdist ships every report from worker to controller through execnet, which can only
    serialize builtin types -- and pytest's subtest report carries the ``subTest(...)``
    kwargs verbatim.  A subtest keyed by an enum member, a polars dtype or an arbitrary
    object therefore crashes the worker's report with ``DumpError: can't serialize``
    under ``-n``, while passing serially.  The kwargs are only ever used to label the
    subtest, so anything that is not exactly a wire-safe scalar is replaced by its repr
    (``type(...) in`` rather than isinstance: an IntEnum *is* an int but still fails).
    """
    _data_ = yield
    _ctx_ = (_data_ or {}).get('_subtest.context')
    if _ctx_ and _ctx_.get('kwargs'):
        _ctx_['kwargs'] = {_k_: _v_ if type(_v_) in _WIRE_SAFE_ else repr(_v_)
                           for _k_, _v_ in _ctx_['kwargs'].items()}
    return _data_
