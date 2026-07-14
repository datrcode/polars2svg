# Gallery

Every chart below is a real polars2svg render, generated from seeded random
data by [`docs/gen_examples.py`](https://github.com/datrcode/polars2svg/blob/main/docs/gen_examples.py).
Click through for the code that produced each one.

## [xyp — scatter](xyp.md)

Each row becomes a dot at (x, y); axes may be numeric or categorical.

[![xyp scatter](../examples/xyp_scatter.svg)](xyp.md)

## [histop — histogram](histop.md)

One horizontal bar per category/bin; bar length and bar color are independent
encodings.

[![histop bars](../examples/histop_bars.svg)](histop.md)

## [timep — temporal bars](timep.md)

Time binned along the x-axis — chronological, or folded into a repeating cycle
(day-of-week below).

[![timep periodic](../examples/timep_periodic.svg)](timep.md)

## [piep — pie / donut](piep.md)

Bins become slices; mirrors histop's parameters.

[![piep donut](../examples/piep_donut.svg)](piep.md)

## [linkp — network](linkp.md)

Node-link graph from an edge list, with pluggable layouts.

[![linkp network](../examples/linkp_network.svg)](linkp.md)

## [chordp — chord diagram](chordp.md)

Weighted flows as ribbons around a circle.

[![chordp flows](../examples/chordp_flows.svg)](chordp.md)

## [spreadlinesp — spread lines](spreadlinesp.md)

Egocentric influence over time: senders above the ego line, receivers below.

[![spreadlinesp ego](../examples/spreadlinesp_ego.svg)](spreadlinesp.md)

## [smallp — small multiples](smallp.md)

A grid of one template component, faceted by a field with shared axes.

[![smallp facets](../examples/smallp_facets.svg)](smallp.md)
