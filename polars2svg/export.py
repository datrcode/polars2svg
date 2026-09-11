from pathlib import Path
#
# export - save rendered components to disk (SVG always; PNG via optional deps)
#
# Every rendered component (XYp, Histop, Timep, Piep, LinkP, ChP, Smallp,
# SpreadLinesP) exposes a standalone SVG document through _repr_svg_().  The
# ExportMixin turns that into a file:
#
#   plot.save('chart.svg')     # write the SVG document verbatim
#   plot.save('chart.png')     # rasterize to PNG (extension-dispatched)
#   plot.savePNG('chart.png')  # rasterize to PNG explicitly
#
# SVG export is dependency-free.  PNG export rasterizes with svglib + reportlab,
# which are an optional install (`pip install polars2svg[export]`); savePNG /
# svgToPNGBytes raise a clear ImportError naming that extra when they are absent.
#
from typing import Any
import io
import os
import re


#
# _fixSVGForRasterize_() - work around a svglib/tinycss2 crash
# - svglib raises AttributeError on a CSS <style> rule with no declaration block
#   (e.g. "<style> .foo </style>").  Such rules are visually inert, so blank out
#   any <style> element whose content has no '{'.
#
def _fixSVGForRasterize_(svg: str) -> str:
    def _blank_empty_style_(m: Any) -> str:
        return '<style></style>' if '{' not in m.group(1) else m.group(0)
    return re.sub(r'<style>(.*?)</style>', _blank_empty_style_, svg, flags=re.DOTALL)


#
# svgToPNGBytes() - rasterize an SVG document string to PNG bytes
# - uses svglib (SVG -> reportlab drawing) + reportlab renderPM (drawing -> PNG)
# - raises a clear ImportError pointing at the [export] extra when either the
#   svglib or reportlab dependency is missing
#
# Note on the svglib<2 cap in pyproject's [export] extra: svglib 1.x carries the
# SVG's px width/height straight through as reportlab points (200px -> 200pt), so
# renderPM's default 72dpi rasterizes one SVG user unit to one PNG pixel.  svglib
# 2.x instead applies the correct CSS px->pt conversion (200px -> 150pt), which at
# 72dpi silently shrinks every export to 0.75 scale and mis-scales nested <svg>
# tiles (smallp/spreadlinesp) outright.  reportlab 5 itself is fine here — it is
# svglib 2 that changes the geometry — so the cap is on svglib, not reportlab.
#
# The io.StringIO wrapper below is LOAD-BEARING — do not "simplify" it into a
# temp-file path.  svglib resolves xlink:href/href on <image> and <use> against
# the *source path* of the document it parsed, so rasterizing from a file path
# turns any external reference in the SVG into a filesystem read whose contents
# land in the output PNG.  tile() embeds foreign SVG verbatim (PLANNING.md A3,
# SECURITY.md), so that reference can come from markup polars2svg did not
# produce.  Handing svglib a stream instead leaves its source_path a non-str and
# SvgRenderer.xlink_href_target() then refuses to resolve a path at all — true of
# every version the [export] extra admits (svglib 1.5.0 through 1.6.0), but a
# property of svglib rather than of us, which is why
# tests/test_export_save.py::TestRasterizeDoesNotReadLocalFiles pins the
# behaviour end-to-end instead of trusting this comment.
#
def svgToPNGBytes(svg: str) -> bytes:
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
    except ImportError as _e_:
        raise ImportError(
            "PNG export requires the optional 'export' dependencies "
            "(svglib, reportlab). Install them with:\n"
            "    pip install polars2svg[export]"
        ) from _e_
    _drawing_ = svg2rlg(io.StringIO(_fixSVGForRasterize_(svg)))
    return renderPM.drawToString(_drawing_, fmt='PNG')


class ExportMixin:
    #
    # ExportMixin - save()/savePNG() for any component with a _repr_svg_()
    #
    # Mixed into every rendered component.  Relies only on _repr_svg_() returning
    # a standalone SVG document, so it works uniformly across the eight components.
    #

    # ---------------------------------------------------------------------
    # Host-class attributes this mixin reads off `self`.
    #
    # A mixin is never instantiated on its own -- these are provided by whatever
    # class mixes it in (every component).  Declared here so a checker
    # can follow the mixin's own methods; bare annotations, so nothing exists at
    # runtime and nothing is shadowed.
    #
    # Typed `Any` on purpose: the mixin genuinely does not know the concrete type,
    # and several hosts declare the same name with a narrower type of their own.
    # ---------------------------------------------------------------------
    _repr_svg_: Any

    #
    # save() - write this rendering to disk, dispatching on the file extension
    # - a '.png' path rasterizes (see savePNG); every other path writes the SVG
    #   document verbatim (UTF-8).  Returns the path written.
    #
    def save(self, path: str | Path) -> str:
        _path_ = os.fspath(path)
        if _path_.lower().endswith('.png'):
            return self.savePNG(_path_)
        with open(_path_, 'w', encoding='utf-8') as _f_:
            _f_.write(self._repr_svg_())
        return _path_

    #
    # savePNG() - rasterize this rendering to a PNG file
    # - requires the optional [export] extra (svglib + reportlab); raises a clear
    #   ImportError naming it when absent.  Returns the path written.
    #
    def savePNG(self, path: str | Path) -> str:
        _path_      = os.fspath(path)
        _png_bytes_ = svgToPNGBytes(self._repr_svg_())
        with open(_path_, 'wb') as _f_:
            _f_.write(_png_bytes_)
        return _path_
