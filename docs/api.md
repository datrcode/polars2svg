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

## Backgrounds

`linkp`'s and `xyp`'s `background=` takes a `{name: shape}` dict drawn beneath the plot in
world coordinates. A bare shape inherits the component's `background_*` styling; a record
built with `Polars2SVG.bgShape()` carries its own, and the two forms mix freely in one
dict. `FlowFieldBackground` is a producer of such a dict for an existing node layout — it
describes a layout rather than producing one, and is deliberately not a `LayoutAlgorithm`.
Unlike the layouts below it needs no optional extra: NumPy and Polars are core
dependencies.

::: polars2svg.FlowFieldBackground

## Output contract

The Profile A enforcement point — see [Output contract](guides/output-contract.md) for
when you need it. `checkOutputContract()` returns the violations it found;
`assertOutputContract()` raises `OutputContractError` on the first non-empty result. Both
**report rather than rewrite**: a violation means the render is not fit to serve, not that
it has been made fit. Neither is a sanitizer.

`ALLOWED_ELEMENTS` and `ALLOWED_ATTRIBUTES` are the allow-list the check is built from —
the elements and attributes the components actually emit. They are exported so you can
assert against them directly if you are composing renders yourself.

::: polars2svg.checkOutputContract
::: polars2svg.assertOutputContract
::: polars2svg.Violation
::: polars2svg.OutputContractError

## Component keyword TypedDicts

Every component factory is typed `**kwargs: Unpack[<Component>Kwargs]`, so a type checker
flags a misspelled parameter at the call site and editors complete the parameter set. They
are exported so you can annotate a kwargs dict of your own and pass it with `**`:

```python
opts: p2s.XYpKwargs = {'dot_size': 6, 'wxh': (400, 300)}
p2s.xyp(df, 'x', 'y', **opts)
```

!!! note "`ChPKwargs` needs the `layouts` extra"

    It sits behind the same guard as `ChP` itself — chordp's node ordering needs scipy, so
    importing it eagerly would make a bare `import polars2svg` require the extra. The other
    eight are always available.

::: polars2svg.XYpKwargs
::: polars2svg.HistopKwargs
::: polars2svg.TimepKwargs
::: polars2svg.PiepKwargs
::: polars2svg.LinkPKwargs
::: polars2svg.ChPKwargs
::: polars2svg.SpreadLinesPKwargs
::: polars2svg.SmallpKwargs
::: polars2svg.TileKwargs

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
