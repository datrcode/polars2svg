'''Shared injection corpus: hostile strings, and every render that can carry one.

Used by tests/test_profile_conformance.py (Profile A) and available to any test
that wants to push adversarial text through a component.

Two halves, and both matter:

  PAYLOADS    - hostile strings, each naming the *escaping mistake* it detects
                rather than the attack it performs.  The point of the set is
                coverage of failure modes, not of exploits: `<script>` catches a
                site that escapes nothing, but a site that strips '<' and leaves
                '"' alone is only caught by an attribute-breakout payload, and a
                site that escapes on the way in and again on the way out is only
                caught by a pre-encoded one.

  renderMatrix() - every text-bearing surface across the eight components, as
                (case_name, render_callable) pairs.  The payload goes in as row
                data or as a label-map value -- i.e. through the paths SECURITY.md
                classes as *untrusted*.  Component parameters (color=, wxh=) are
                trusted configuration and are deliberately not fuzzed here.

The corpus is deliberately not self-asserting.  What counts as a pass is the
caller's business: test_profile_conformance.py runs each render through the
Profile A output contract, which is a structural check.

Every case carries a `reach`, and that field is the whole defence against a
vacuous suite.  A render that silently drops the label satisfies any structural
check trivially, so `reach` states where the string is supposed to end up and the
conformance test proves it got there, using a benign probe so that sanitizers and
croppers do not confuse "absorbed safely" with "never rendered".  Five of the
first seventeen cases written here were vacuous until this was added -- link
labels needed a label field in the relationship, spreadlinesp does not draw node
labels at all, and piep crops a long label -- so treat a NOWHERE result as a
broken case, never as a passing one.

  reach='text'      the string is drawn as text content; a PREFIX must appear
                    (components crop labels to fit, and cropping an escaped
                    string is itself a hazard -- see the tspan note in linkp.py)
  reach='attr'      it reaches an attribute value, usually after id sanitising
  reach='absorbed'  it must reach NEITHER, by design; the value selects something
                    (a color) rather than being displayed.  Stated per case with
                    the reason, so "absorbed" is a claim under test rather than an
                    excuse for a case that quietly does nothing.
'''
import datetime
from typing import NamedTuple

import polars as pl


class Case(NamedTuple):
    '''One render that can carry a hostile string.

    `render` is a thunk so a case that raises does so inside the test, where it is
    reported as a finding rather than breaking collection.  `reach` is documented
    in the module docstring; `why` is required for reach='absorbed' and explains
    what absorbs the value.
    '''
    name:   str
    render: object
    reach:  str = 'text'
    why:    str = ''


# ---------------------------------------------------------------------------
# The payloads.
#
# Keyed by the mistake each one detects.  A component that escapes correctly
# turns every one of these into inert text; a component that does not produces
# either malformed XML or a document carrying an element/attribute outside the
# Profile A vocabulary, and the contract check catches both.
# ---------------------------------------------------------------------------
PAYLOADS = {
    # Escapes nothing at all.  The historical payload -- kept because it is the
    # one the old single-payload test used, so a regression is comparable.
    'script_element':      '<script>alert(1)</script>',

    # Escapes '<' but relies on a blocklist of tag NAMES.  No <script> anywhere.
    'img_onerror':         '<img src=x onerror="alert(1)">',
    'svg_onload':          '<svg onload="alert(1)"></svg>',

    # Escapes element syntax but not quotes: breaks out of an attribute VALUE
    # rather than out of an element.  Both quote flavours, because a site that
    # builds attributes with single quotes is only caught by the second.
    'attr_break_double':   '" onload="alert(1)',
    'attr_break_single':   "' onload='alert(1)",
    'attr_break_bare':     'x" onmouseover=alert(1) y="',

    # Closes the element it lands in, then opens a new one.  Catches a site that
    # escapes attribute values but writes text content raw.
    'text_break_out':      '</text><script>alert(1)</script><text>',
    'tspan_break_out':     '</tspan></text><script>alert(1)</script>',

    # Already-encoded input.  Catches double-escaping (which is a display bug,
    # not a hole) and, more importantly, an unescape-then-embed path.
    'pre_encoded_lt':      '&lt;script&gt;alert(1)&lt;/script&gt;',
    'numeric_entity':      '&#60;script&#62;alert(1)&#60;/script&#62;',
    'hex_entity':          '&#x3c;script&#x3e;alert(1)&#x3c;/script&#x3e;',

    # Bare '&' is not a valid entity start; a site that escapes '<' and '>' but
    # forgets '&' produces a document that will not parse.
    'bare_ampersand':      'A & B & C',
    'malformed_entity':    '&notanentity; &#; &#x;',

    # Structural escapes out of the surrounding syntax.
    'cdata_break':         ']]><script>alert(1)</script><![CDATA[',
    'comment_break':       '--><script>alert(1)</script><!--',
    'doctype':             '<!DOCTYPE x [<!ENTITY e SYSTEM "file:///etc/passwd">]>',

    # A URL scheme in a position that might become a reference.
    'javascript_url':      'javascript:alert(1)',
    'data_html_url':       'data:text/html;base64,PHN2Zz48L3N2Zz4=',
    'css_url_import':      '@import url("http://example.invalid/x.css");',
    'css_expression':      'expression(alert(1))',

    # Identifier-position payloads: these target id="" / url(#...) construction
    # rather than text, which is a different code path (xyp's per-segment
    # linearGradient ids are built from a data column).
    'id_break':            'a" x="0',
    'id_url_break':        'a) url(http://example.invalid/x) (',
    'id_fragment':         'a#b c(d)e"f',

    # Not injection -- display and parser robustness.  A bidi override can make a
    # label read as something else entirely; NUL and lone surrogates break
    # serializers.
    'rtl_override':        'moc.elpmaxe‮/gro.dab//:ptth',
    'zero_width':          'ad​min',
    'newlines_tabs':       'a\nb\tc\rd',
    'long_label':          'A' * 4096,
}


# ---------------------------------------------------------------------------
# Frames.  Each builder puts `payload` in the position named by the case.
# ---------------------------------------------------------------------------
def _histoFrame_(payload: str):
    return pl.DataFrame({'cat': [payload, 'other', payload, 'third'],
                         'v':   [1, 2, 3, 4]})


def _tupleBinFrame_(payload: str):
    return pl.DataFrame({'a': [payload, 'y'], 'b': ['1', '2'], 'v': [1, 2]})


def _xyCategoricalFrame_(payload: str):
    return pl.DataFrame({'x': [payload, 'c', 'e'], 'y': [1.0, 2.0, 3.0]})


def _xyGradientFrame_(payload: str):
    _a_ = pl.DataFrame({'t': list(range(6)), 'v': [2, 3, 4, 3, 5, 1], 'grp': [payload] * 6})
    _b_ = pl.DataFrame({'t': list(range(6)), 'v': [8, 7, 5, 6, 5, 2], 'grp': ['other'] * 6})
    return pl.concat([_a_, _b_])


def _edgeFrame_(payload: str):
    return pl.DataFrame({'fm': [payload, 'C', 'E'],
                         'to': ['C', 'E', payload],
                         'p':  ['calls', 'sends', 'calls']})


def _labelledEdgeFrame_(payload: str):
    # The payload has to be in the LABEL column for draw_link_labels= to draw it;
    # with it only in fm/to the label column is benign and the case is vacuous.
    return pl.DataFrame({'fm': ['A', 'C', 'E'],
                         'to': ['C', 'E', 'A'],
                         'p':  [payload, 'sends', payload]})


def _timeEdgeFrame_(payload: str):
    return pl.DataFrame({'fm':   [payload, payload, 'C'],
                         'to':   ['C', 'E', 'E'],
                         'time': [datetime.datetime(2024, 1, _d_) for _d_ in (1, 2, 3)]})


# spreadlinesp keys anno= by the *string* form of a bin's timestamp (see the anno
# block in spreadlinesp.py), not by a datetime -- a datetime key silently matches
# nothing, which is how this case was vacuous.
_ANNO_TS_ = '2024-01-01 00:00:00.000000'


def _timeFrame_(payload: str):
    return pl.DataFrame({'dt': [datetime.datetime(2024, 1, _d_) for _d_ in (1, 2, 3, 4)],
                         'c':  [payload, 'b', payload, 'c']})


#
# renderMatrix() - every untrusted-text surface across the eight components
# - returns Case records; `render` is a thunk, so a case that raises does so
#   inside the test and is reported rather than breaking collection
# - tile() is absent on purpose: it embeds foreign SVG verbatim and is outside
#   Profile A by construction (PLANNING.md A3, SECURITY.md)
#
def renderMatrix(p2s, payload: str) -> list:
    _rels_       = [('fm', 'to')]
    _rels_label_ = [('fm', 'to', 'p')]
    return [
        Case('histop.bin_label',
             lambda: p2s.histop(_histoFrame_(payload), 'cat', draw_labels=True)),
        Case('histop.tuple_bin_label',
             lambda: p2s.histop(_tupleBinFrame_(payload), ('a', 'b'), draw_labels=True)),
        Case('histop.color_column',
             lambda: p2s.histop(_histoFrame_(payload), 'cat', color='cat', draw_labels=True)),

        # Legends draw the category values themselves, which is a text surface
        # distinct from the axis/label paths above and easy to miss.
        Case('xyp.legend_entry',
             lambda: p2s.xyp(_xyCategoricalFrame_(payload), 'x', 'y',
                             color='x', legend=True)),
        Case('histop.legend_entry',
             lambda: p2s.histop(_histoFrame_(payload), 'cat',
                                color='cat', legend='top')),

        Case('piep.slice_label',
             lambda: p2s.piep(_histoFrame_(payload), 'cat', draw_labels=True)),

        Case('xyp.categorical_x',
             lambda: p2s.xyp(_xyCategoricalFrame_(payload), 'x', 'y')),
        Case('xyp.categorical_color',
             lambda: p2s.xyp(_xyCategoricalFrame_(payload), 'x', 'y', color='x')),
        # The per-segment linearGradient id is built FROM a data column, so this
        # is the one case whose payload lands in an attribute rather than in text.
        Case('xyp.gradient_line_id',
             lambda: p2s.xyp(_xyGradientFrame_(payload), 't', 'v', color='v', dot_size='v',
                             opacity='v', line=('grp', p2s.LINEOPACITY_FIELD_VARIABLE),
                             draw_context=False),
             reach='attr'),

        # color= maps a value to a color and never displays it.  Kept as a case
        # because "it is absorbed" is exactly the kind of claim that stops being
        # true when someone adds a legend or a CSS class keyed by category.
        Case('timep.color_column',
             lambda: p2s.timep(_timeFrame_(payload), 'dt', color='c'),
             reach='absorbed',
             why='color= selects a color; the raw value is never drawn or used as an id'),

        Case('linkp.node_label_from_data',
             lambda: p2s.linkp(_edgeFrame_(payload), relationships=_rels_,
                               pos={payload: [0, 0], 'C': [1, 1], 'E': [0, 1]},
                               draw_node_labels=True)),
        Case('linkp.node_labels_map_value',
             lambda: p2s.linkp(_edgeFrame_(payload), relationships=_rels_,
                               pos={payload: [0, 0], 'C': [1, 1], 'E': [0, 1]},
                               node_labels={payload: payload}, draw_node_labels=True)),
        # draw_link_labels= needs a label field in the relationship; without one
        # linkp warns and draws nothing, which is how this case was vacuous.
        Case('linkp.link_label',
             lambda: p2s.linkp(_labelledEdgeFrame_(payload), relationships=_rels_label_,
                               pos={'A': [0, 0], 'C': [1, 1], 'E': [0, 1]},
                               draw_link_labels=True)),
        # The curved variant renders labels through <textPath>, a different path.
        Case('linkp.link_label_curve',
             lambda: p2s.linkp(_labelledEdgeFrame_(payload), relationships=_rels_label_,
                               pos={'A': [0, 0], 'C': [1, 1], 'E': [0, 1]},
                               link_shape='curve', draw_link_labels=True)),

        Case('chordp.label_radial',
             lambda: p2s.chordp(df=_edgeFrame_(payload), relationships=_rels_,
                                draw_labels=True, label_style='radial')),
        Case('chordp.label_circular',
             lambda: p2s.chordp(df=_edgeFrame_(payload), relationships=_rels_,
                                draw_labels=True, label_style='circular')),
        Case('chordp.node_labels_map_value',
             lambda: p2s.chordp(df=_edgeFrame_(payload), relationships=_rels_,
                                draw_labels=True, node_labels={payload: payload})),

        Case('smallp.category_label',
             lambda: p2s.smallp(_histoFrame_(payload),
                                p2s.histop(_histoFrame_(payload), 'cat', wxh=(96, 96)),
                                'cat', draw_labels=True)),

        # spreadlinesp draws no entity labels at all (draw_labels=True raises
        # NotImplementedError), so its untrusted-text surface is anno=, the
        # {time: label} event annotation map -- not the node names.
        Case('spreadlinesp.annotation_label',
             lambda: p2s.spreadlinesp(_timeEdgeFrame_(payload), _rels_,
                                      ego='C', time='time',
                                      anno={_ANNO_TS_: payload})),
        Case('spreadlinesp.node_name',
             lambda: p2s.spreadlinesp(_timeEdgeFrame_(payload), _rels_,
                                      ego=payload, time='time'),
             reach='absorbed',
             why='spreadlinesp draws no entity labels; node names position geometry only'),
    ]
