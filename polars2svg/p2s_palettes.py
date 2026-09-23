from typing import Any

#
# p2s_palettes - the named palettes, and the one function that resolves a palette
# selection into the three pieces of color state a Polars2SVG instance carries.
#
# This module is deliberately dependency-free (stdlib typing only).  In particular it
# does NOT import from p2s_colors_mixin, which imports *this* module -- a hex check
# here would need isHexColor() and close that cycle.  The split is therefore:
#
#   - resolvePalette() validates STRUCTURE: the palette name, and that an override
#     names a slot that actually exists.
#   - P2SColorsMixin.setPalette() validates the hex VALUES of user overrides, with
#     the same isHexColor() check setColorOverrides() already uses.
#   - tests/test_palettes.py validates the hex values of the built-in tables, which
#     are static data and so are checked once rather than on every construction.
#
# Slot-key parity between palettes is load-bearing, not cosmetic.  colorTyped() is a
# bare dict subscript, so a slot missing from one palette is a KeyError raised deep in
# a component at render time -- not an error at construction, and not necessarily in a
# component the change's author was looking at.  _assertSlotParity_() below turns that
# into an import-time failure instead.
#

#
# The light palette -- the framework's default, and what every render looked like
# before palettes existed.  Values are unchanged from when this table lived inline in
# p2s_colors_mixin.__p2s_colors_mixin_init__().
#
_LIGHT_: dict[tuple[str, str], str] = {
    ('background',    'default'):     '#ffffff',
    ('data',          'default'):     '#3939ff',
    ('axis',          'default'):     '#a0a0a0',
    ('axis',          'label'):       '#404040',
    ('axis',          'min'):         '#0015D2',
    ('axis',          'max'):         '#A10000',
    ('axis',          'inner'):       '#a0a0a0',
    ('axis',          'origin'):      '#404040',
    ('error',         'default'):     '#ff0000',
    ('label',         'defaultfg'):   '#000000',
    ('label',         'inner'):       '#a0a0a0',
    ('distributions', 'default'):     '#000000',
    ('distributions', 'fill'):        '#f0f0f0',
    ('indicator',     'more_rows'):   '#cc3333',
    ('indicator',     'available'):   '#5f9e5f', # faded green
    ('indicator',     'unavailable'): '#cccccc', # gray, barely perceptible on the light background
    ('selection',     'default'):     '#ff0000',
    ('overlay',       'hint'):        '#0000cc', # keyboard-help text drawn on the canvas
    ('setop',         'replace'):     '#000000', # drag-band: no modifier -> replace selection
    ('setop',         'add'):         '#00ff00', # ctrl        -> add
    ('setop',         'subtract'):    '#ff0000', # shift       -> subtract
    ('setop',         'intersect'):   '#0000ff', # shift+ctrl  -> intersect
    ('direction',     'new'):         '#ff0000', # entity appears (left-pointing marker)
    ('direction',     'ending'):      '#0000ff', # entity departs (right-pointing marker)
    ('direction',     'new_muted'):   '#d3494e', # the same pair on a collapsed cloud,
    ('direction',     'ending_muted'):'#658cbb', # muted so the cloud stays readable
    ('multiset',      'str'):         '#7f8367', # derived from polarsOperation behavior / not used
    ('multiset',      'int'):         '#19d084', # derived from polarsOperation behavior / not used
    ('multiset',      'float'):       '#e3e294', # derived from polarsOperation behavior / not used
}

#
# The dark palette.
#
# Values were not picked by eye.  Each was solved so that its WCAG contrast ratio
# against the dark background reproduces the light palette's own ratio against white,
# which is what carries the visual hierarchy across -- a muted gridline stays as muted
# relative to its canvas, an axis label stays as prominent.  The three greys are exact
# solutions (axis/default 2.59 against the light table's 2.61; axis/label 10.41 against
# 10.37).
#
# Two deliberate departures from that rule:
#
#   - label/defaultfg is #e6e6e6 (15.0), not white.  The light table's 21.0 is simply
#     unreachable on #121212 -- pure white tops out at 18.7 -- and text at that end of
#     the range haloes against a dark canvas, so the softer value is the better render
#     as well as the achievable one.
#   - error/default and selection/default land brighter than their light counterparts
#     (5.7 vs 4.0).  Both are attention colors; matching the ratio exactly would have
#     made them quieter than the labels beside them, which inverts their job.
#
# The multiset entries are NOT theme colors.  They record what the hash-derived
# colorizer happens to emit for three dtypes (see the derivation note in
# p2s_colors_mixin), and that hash is palette-independent, so they are identical in
# both tables on purpose -- PLANNING.md C3 covers whether they should exist at all.
#
_DARK_: dict[tuple[str, str], str] = {
    ('background',    'default'):     '#121212',
    ('data',          'default'):     '#6e6eff',
    ('axis',          'default'):     '#575757',
    ('axis',          'label'):       '#c1c1c1',
    ('axis',          'min'):         '#7f9cff',
    ('axis',          'max'):         '#ff7b7b',
    ('axis',          'inner'):       '#575757',
    ('axis',          'origin'):      '#c1c1c1',
    ('error',         'default'):     '#ff4d4d',
    ('label',         'defaultfg'):   '#e6e6e6',
    ('label',         'inner'):       '#575757',
    ('distributions', 'default'):     '#e6e6e6',
    ('distributions', 'fill'):        '#262626',
    ('indicator',     'more_rows'):   '#e06666',
    ('indicator',     'available'):   '#4a854a', # faded green, darkened to sit under more_rows
    ('indicator',     'unavailable'): '#3b3b3b', # gray, barely perceptible on the dark background
    ('selection',     'default'):     '#ff4d4d',
    ('overlay',       'hint'):        '#8fb0ff', # keyboard-help text drawn on the canvas
    ('setop',         'replace'):     '#e6e6e6', # drag-band: no modifier -> replace selection
    ('setop',         'add'):         '#5ce65c', # ctrl        -> add
    ('setop',         'subtract'):    '#ff6b6b', # shift       -> subtract
    ('setop',         'intersect'):   '#7f9cff', # shift+ctrl  -> intersect
    ('direction',     'new'):         '#ff6b6b', # entity appears (left-pointing marker)
    ('direction',     'ending'):      '#7f9cff', # entity departs (right-pointing marker)
    ('direction',     'new_muted'):   '#c96f73', # the same pair on a collapsed cloud,
    ('direction',     'ending_muted'):'#7fa3cc', # muted so the cloud stays readable
    ('multiset',      'str'):         '#7f8367', # hash-derived, palette-independent -- see above
    ('multiset',      'int'):         '#19d084', # hash-derived, palette-independent -- see above
    ('multiset',      'float'):       '#e3e294', # hash-derived, palette-independent -- see above
}

#
# Color Spectrum: the 10-class "Spectral" diverging scheme from ColorBrewer
# (colorbrewer2.org). Color specifications and designs (c) 2002 Cynthia Brewer,
# Mark Harrower, and The Pennsylvania State University; used under the
# Apache-style ColorBrewer license (attribution reproduced in NOTICE).
#
# M. Harrower and C. A. Brewer, "ColorBrewer.org: An Online Tool for
# Selecting Colour Schemes for Maps," The Cartographic Journal, vol. 40,
# no. 1, pp. 27-37, 2003, doi: 10.1179/000870403235002042.
#
# - for the colorSpectrumPolarsOperations() to work, there needs to be at least three colors
# - both palettes use it unchanged.  Its legibility does invert on a dark canvas (the
#   pale yellows in the middle become the most prominent steps rather than the least),
#   which is a real cost; re-tuning it is PLANNING.md F5's remaining work, and needs its
#   own provenance note since the NOTICE attribution covers Spectral as published.
#
_SPECTRAL_: list[str] = ['#9e0142','#d53e4f','#f46d43','#fdae61','#fee08b',
                         '#e6f598','#abdda4','#66c2a5','#3288bd','#5e4fa2']

#
# grayscale_ramp - (lo, hi) endpoints for grayscaleSpectrumPolarsOperations(), which
# shades the distribution strips.  The ramp runs lo at a normalized 0.0 to hi at 1.0.
#
# Light is (0.8, 0.0): 0.0 -> #cccccc, barely perceptible on white, and 1.0 -> black.
# Dark has to invert, not merely lighten -- keeping the light ramp on a dark canvas
# would make "near zero" the single most prominent thing in the plot.
#
_RAMP_LIGHT_: tuple[float, float] = (0.8, 0.0)
_RAMP_DARK_:  tuple[float, float] = (0.2, 1.0)

PALETTES: dict[str, dict[str, Any]] = {
    'light': {'color_type_lu': _LIGHT_, 'spectrum_palette': _SPECTRAL_, 'grayscale_ramp': _RAMP_LIGHT_},
    'dark':  {'color_type_lu': _DARK_,  'spectrum_palette': _SPECTRAL_, 'grayscale_ramp': _RAMP_DARK_},
}

#
# _assertSlotParity_() - every palette defines exactly the same slots as the light one.
# Runs at import so a table edited on one side only fails loudly and immediately,
# rather than as a KeyError from colorTyped() inside whichever component happens to
# ask for the missing slot first.
#
def _assertSlotParity_() -> None:
    _reference_ = set(_LIGHT_)
    for _name_, _entry_ in PALETTES.items():
        _slots_ = set(_entry_['color_type_lu'])
        if _slots_ != _reference_:
            _missing_ = sorted(_reference_ - _slots_)
            _extra_   = sorted(_slots_ - _reference_)
            raise RuntimeError(f'p2s_palettes: palette {_name_!r} does not match the light '
                               f'palette\'s slots. missing={_missing_} unexpected={_extra_}')

_assertSlotParity_()

#
# paletteNames() - the selectable palette names, sorted for stable error messages
#
def paletteNames() -> list[str]:
    return sorted(PALETTES)

#
# resolvePalette() - turn a palette selection into fresh per-instance color state
#
# - palette   : a name from paletteNames()
# - overrides : optional {(type, subtype): hex} layered onto that palette.  Keys must
#               name existing slots; a typo would otherwise be silently inert, which
#               is the failure mode this check exists to prevent.  Values are checked
#               by the caller (see the module note above).
#
# Returns a dict of the three pieces of state, with the mutable ones COPIED.  The
# copies matter: color_type_lu is mutated in place by callers (tests do this, and
# setColorOverrides-style workflows may), and handing out the module-level table would
# let one instance's edit reach every instance built afterwards -- the exact defect
# that removing the Polars2SVG singleton was meant to end.
#
def resolvePalette(palette: str = 'light',
                   overrides: dict[tuple[str, str], str] | None = None) -> dict[str, Any]:
    if not isinstance(palette, str):
        raise TypeError(f'resolvePalette(): palette must be a string naming a palette, '
                        f'got {type(palette).__name__} {palette!r}')
    if palette not in PALETTES:
        raise ValueError(f'resolvePalette(): unknown palette {palette!r}. '
                         f'Valid palettes: {paletteNames()}')
    _entry_  = PALETTES[palette]
    _colors_ = dict(_entry_['color_type_lu'])
    if overrides:
        if not isinstance(overrides, dict):
            raise TypeError(f'resolvePalette(): overrides must be a dict keyed by '
                            f'(type, subtype), got {type(overrides).__name__}')
        _unknown_ = sorted(set(overrides) - set(_colors_))
        if _unknown_:
            raise ValueError(f'resolvePalette(): override key(s) do not name a palette slot: '
                             f'{_unknown_}. Valid slots: {sorted(_colors_)}')
        _colors_.update(overrides)
    return {
        'name':             palette,
        'color_type_lu':    _colors_,
        'spectrum_palette': list(_entry_['spectrum_palette']),
        'grayscale_ramp':   tuple(_entry_['grayscale_ramp']),
    }
