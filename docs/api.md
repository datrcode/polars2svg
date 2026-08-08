# API reference

Generated from the source by [mkdocstrings](https://mkdocstrings.github.io/). The
hand-written [component pages](components/index.md) are the place to start; this page is the
exhaustive signature reference.

## Polars2SVG

The entry point. Every component is a method on this class — `xyp()`, `histop()`, `timep()`,
`piep()`, `linkp()`, `chordp()`, `spreadlinesp()` and `smallp()` — each returning an SVG
string.

::: polars2svg.Polars2SVG

## Exceptions

::: polars2svg.Polars2SVGError
::: polars2svg.InvalidSpecError
::: polars2svg.DataError

## Legends

::: polars2svg.LegendInfo

## Layouts

!!! note "These require the `layouts` extra"

    `import polars2svg` succeeds without them — the names are simply absent when their
    dependencies (networkx, and scipy/scikit-learn for the MDS pair) aren't installed. This
    page documents them regardless, because it is generated from the source rather than from
    an installed package. Install with `pip install polars2svg[layouts]`.

::: polars2svg.LayoutAlgorithm
::: polars2svg.TFDPLayout
::: polars2svg.NCPLayout
::: polars2svg.laguerre_voronoi
