"""Shared pytest fixtures for the polars2svg test suite.

Test isolation for process-global Polars2SVG color state.

``color_overrides_lu`` is shared across *every* Polars2SVG instance (see
``p2s_colors_mixin.py`` — the map is created once and reused, so two freshly
constructed ``Polars2SVG()`` objects reference the same dict). That means
``setColorOverrides()`` in one test leaks the override into every later test:
e.g. ``test_color_overrides`` maps ``'A' -> '#ff0000'`` and, with no cleanup,
poisons the categorical timep goldens (whose categories are ``A``/``B``) when
they run afterwards. The leak is order-dependent, so it only surfaces in certain
selections — exactly the kind of flakiness this fixture removes.

The autouse fixture below clears the override map before and after every test so
no test can inherit (or leak) an override. Only ``color_overrides_lu`` is reset;
``to_color_lu`` holds deterministic base hash colors and is intentionally
persistent (it never stores override results), so it is left alone.
"""
import pytest
from polars2svg import Polars2SVG

# All Polars2SVG instances share one color_overrides_lu dict; hold a reference to
# it so we can clear in place without constructing an instance per test.
_SHARED_COLOR_OVERRIDES = Polars2SVG().color_overrides_lu


@pytest.fixture(autouse=True)
def _reset_global_color_overrides():
    _SHARED_COLOR_OVERRIDES.clear()
    yield
    _SHARED_COLOR_OVERRIDES.clear()
