from typing import Protocol, runtime_checkable


@runtime_checkable
class LayoutAlgorithm(Protocol):
    """
    Structural protocol satisfied by every layout algorithm class in polars2svg.

    Any class that implements a ``results()`` method returning a dict of
    ``{node: (x, y)}`` positions satisfies this protocol automatically —
    no explicit subclassing is required.

    Usage::

        assert isinstance(my_layout, LayoutAlgorithm)
        pos = my_layout.results()
    """

    def results(self) -> dict:
        """Return a dict mapping each node to its (x, y) position."""
        ...


@runtime_checkable
class SketchRepresentable(Protocol):
    """
    Structural protocol for components that need a *static* representation in
    non-interactive contexts (chiefly ``panelizeSketch()``).

    Static plot components (xyp, histop, ...) satisfy the sketch path through
    their ``_repr_svg_()`` already and need not implement this. It exists for
    **interactive-only** widgets (e.g. ``stack_controli``) that are normally
    only meaningful inside ``panelize()`` but may still appear in a layout that
    is sketched.

    A widget implements ``sketchHtml()`` to return a self-contained HTML/SVG
    snapshot of its current state. Returning ``None`` (or not implementing the
    method at all) lets the sketch builder fall back to ``_repr_svg_()`` and,
    failing that, a generic labeled placeholder box.

    Usage::

        if isinstance(widget, SketchRepresentable):
            html = widget.sketchHtml(use_webgpu)
    """

    def sketchHtml(self, use_webgpu: bool = False) -> str | None:
        """Return a static HTML/SVG snapshot, or ``None`` to defer."""
        ...
