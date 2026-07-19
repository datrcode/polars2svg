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
import io
import os
import re


#
# _fixSVGForRasterize_() - work around a svglib/tinycss2 crash
# - svglib raises AttributeError on a CSS <style> rule with no declaration block
#   (e.g. "<style> .foo </style>").  Such rules are visually inert, so blank out
#   any <style> element whose content has no '{'.
#
def _fixSVGForRasterize_(svg):
    def _blank_empty_style_(m):
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
def svgToPNGBytes(svg):
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

    #
    # save() - write this rendering to disk, dispatching on the file extension
    # - a '.png' path rasterizes (see savePNG); every other path writes the SVG
    #   document verbatim (UTF-8).  Returns the path written.
    #
    def save(self, path):
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
    def savePNG(self, path):
        _path_      = os.fspath(path)
        _png_bytes_ = svgToPNGBytes(self._repr_svg_())
        with open(_path_, 'wb') as _f_:
            _f_.write(_png_bytes_)
        return _path_
