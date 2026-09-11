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
