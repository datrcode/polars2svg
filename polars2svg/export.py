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
# SVG export is dependency-free.  PNG export rasterizes with cairosvg, an optional
# install (`pip install polars2svg[export]`) that loads the cairo C library at run
# time; savePNG / svgToPNGBytes raise a clear ImportError naming what is missing.
#
from typing import Any
import ctypes
import ctypes.util
import os
import sys


#
# _registerBundledFont_() - make the bundled Noto Sans subset available to cairo
# - every component lays its text out with that face's metrics (p2s_font_metrics.py) and
#   asks for "'Noto Sans', sans-serif", but few machines have Noto Sans installed: macOS
#   cairo then draws Helvetica and fontconfig picks whatever 'sans-serif' means locally
#   (Verdana here), so PNG labels came out narrower or wider than the space laid out
# - registers the TTF for this process only: CoreText on macOS, where cairo's fonts come
#   from, and fontconfig wherever it is loadable.  Neither touches the user's fonts
# - runs once per process; a failure leaves the system fonts in place, never an error
#
_FONT_REGISTERED_: bool | None = None   # None until the first attempt

def _registerBundledFont_() -> bool:
    global _FONT_REGISTERED_
    if _FONT_REGISTERED_ is not None: return _FONT_REGISTERED_
    _FONT_REGISTERED_ = False
    _ttf_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'NotoSans-Regular-subset.ttf').encode()
    if sys.platform == 'darwin':
        try:
            _cf_ = ctypes.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
            _ct_ = ctypes.CDLL('/System/Library/Frameworks/CoreText.framework/CoreText')
            _cf_.CFURLCreateFromFileSystemRepresentation.restype  = ctypes.c_void_p
            _cf_.CFURLCreateFromFileSystemRepresentation.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_bool]
            _cf_.CFRelease.argtypes                               = [ctypes.c_void_p]
            _ct_.CTFontManagerRegisterFontsForURL.restype         = ctypes.c_bool
            _ct_.CTFontManagerRegisterFontsForURL.argtypes        = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
            _url_ = _cf_.CFURLCreateFromFileSystemRepresentation(None, _ttf_, len(_ttf_), False)
            if _url_:
                # 1 is kCTFontManagerScopeProcess
                _FONT_REGISTERED_ = bool(_ct_.CTFontManagerRegisterFontsForURL(_url_, 1, None))
                _cf_.CFRelease(_url_)
        except (OSError, AttributeError):
            pass
    _fc_path_ = ctypes.util.find_library('fontconfig')
    if _fc_path_:
        try:
            _fc_ = ctypes.CDLL(_fc_path_)
            _fc_.FcConfigAppFontAddFile.restype  = ctypes.c_int
            _fc_.FcConfigAppFontAddFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            # a NULL config is the current one, the one cairo's fontconfig backend asks
            _FONT_REGISTERED_ = bool(_fc_.FcConfigAppFontAddFile(None, _ttf_)) or _FONT_REGISTERED_
        except (OSError, AttributeError):
            pass
    return _FONT_REGISTERED_


#
# _fetchDataURLsOnly_() - cairosvg's url_fetcher: inline data: resources, nothing else
# - tile() embeds foreign SVG verbatim (PLANNING.md A3, SECURITY.md), so a document
#   arriving here can carry an <image>/<use> href, or a CSS url(), that polars2svg never
#   wrote.  Fetched, a file path is a local file read whose contents land in the PNG and
#   an http URL is a request from whatever machine rasterizes it.  Every one of them gets
#   the same empty document cairosvg's own safe mode returns
# - passed explicitly rather than relying on that default, because the default depends on
#   the unsafe= flag and the library version; tests/test_export_save.py pins the result
#
_EMPTY_SVG_ = b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>'

def _fetchDataURLsOnly_(url: str, resource_type: str) -> bytes:
    if url.startswith('data:'):
        from cairosvg.url import fetch
        return bytes(fetch(url, resource_type))
    return _EMPTY_SVG_


#
# svgToPNGBytes() - rasterize an SVG document string to PNG bytes
# - cairosvg, from the document's own bytes (no base URL, so nothing relative resolves),
#   with unsafe=False (no XML entities, no external DTDs) and _fetchDataURLsOnly_()
# - one SVG pixel is one PNG pixel (cairosvg's 96 dpi is the CSS px)
# - an area the SVG does not paint stays transparent, as a browser shows it; every
#   component paints its own canvas, so that is only spreadlinesp's letterbox margins
# - raises ImportError naming the [export] extra when cairosvg is absent, and naming the
#   cairo library when cairosvg is installed but cannot load it
#
def svgToPNGBytes(svg: str) -> bytes:
    try:
        import cairosvg.surface
    except ImportError as _e_:
        raise ImportError(
            "PNG export requires the optional 'export' dependency (cairosvg). "
            "Install it with:\n"
            "    pip install polars2svg[export]"
        ) from _e_
    except OSError as _e_:
        raise ImportError(
            "PNG export needs the cairo graphics library, which cairosvg loads at run time "
            f"and could not find ({_e_}). Install cairo: `brew install cairo` on macOS, "
            "`apt install libcairo2` on Debian/Ubuntu; Windows needs a cairo runtime such as "
            "GTK's (see the README)."
        ) from _e_
    _registerBundledFont_()
    return bytes(cairosvg.surface.PNGSurface.convert(
        bytestring=svg.encode('utf-8'), unsafe=False, url_fetcher=_fetchDataURLsOnly_))


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
    # - requires the optional [export] extra (cairosvg); raises a clear
    #   ImportError naming it when absent.  Returns the path written.
    #
    def savePNG(self, path: str | Path) -> str:
        _path_      = os.fspath(path)
        _png_bytes_ = svgToPNGBytes(self._repr_svg_())
        with open(_path_, 'wb') as _f_:
            _f_.write(_png_bytes_)
        return _path_
