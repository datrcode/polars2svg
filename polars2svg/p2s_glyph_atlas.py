#
# GlyphAtlas - GPU text rendering with SVG-identical layout
#
# Glyph bitmaps are rendered once at ATLAS_PX from the bundled NotoSans subset into
# an R8 shelf-packed atlas.  Glyph *positions* never come from scaled atlas metrics:
# pen advances use the same per-integer-size PIL fonts as P2STextMixin.textLength(),
# so GPU text layout matches the SVG text layout decisions (cropText, svgAxisLabels,
# axis label fitting) exactly.
#
import base64
import math
import os

__name__ = 'p2s_glyph_atlas'

ATLAS_PX  = 48
_PAD_     = 2
_INITIAL_CHARSET_ = ''.join(chr(c) for c in range(32, 127))

class GlyphAtlas:
    def __init__(self, font_path=None):
        if font_path is None:
            font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     'fonts', 'NotoSans-Regular-subset.ttf')
        self.font_path  = font_path
        self.version    = 0
        self._charset_  = set()
        self._glyphs_   = {}    # ch -> {'u0','v0','u1','v1','bx','by','w','h'} (px metrics at ATLAS_PX)
        self._size_fonts_ = {}  # int size -> PIL font (advance metrics, mirrors textLength caching)
        self._img_      = None
        self._png_b64_  = None
        from PIL import ImageFont
        self._font48_   = ImageFont.truetype(self.font_path, size=ATLAS_PX)
        self._ascent48_, self._descent48_ = self._font48_.getmetrics()
        self._build_(_INITIAL_CHARSET_)

    def _sizeFont_(self, size):
        if size not in self._size_fonts_:
            from PIL import ImageFont
            self._size_fonts_[size] = ImageFont.truetype(self.font_path, size=size)
        return self._size_fonts_[size]

    #
    # _build_() - (re)render and shelf-pack the atlas for the union of the current charset and new_chars
    #
    def _build_(self, new_chars):
        from PIL import Image, ImageDraw
        self._charset_ |= set(new_chars)
        _renderable_ = []
        for ch in sorted(self._charset_):
            _bbox_ = self._font48_.getbbox(ch)
            if _bbox_ is None: continue
            _w_, _h_ = _bbox_[2] - _bbox_[0], _bbox_[3] - _bbox_[1]
            if _w_ <= 0 or _h_ <= 0: continue   # whitespace -- advance only, no quad
            _renderable_.append((ch, _bbox_, _w_, _h_))
        # Shelf pack into a fixed-width atlas
        _atlas_w_ = 1024
        _x_, _y_, _shelf_h_ = _PAD_, _PAD_, 0
        _placements_ = []
        for ch, _bbox_, _w_, _h_ in _renderable_:
            if _x_ + _w_ + _PAD_ > _atlas_w_:
                _x_  = _PAD_
                _y_ += _shelf_h_ + _PAD_
                _shelf_h_ = 0
            _placements_.append((ch, _bbox_, _x_, _y_, _w_, _h_))
            _x_ += _w_ + _PAD_
            _shelf_h_ = max(_shelf_h_, _h_)
        _atlas_h_ = 1 << max(1, math.ceil(math.log2(max(1, _y_ + _shelf_h_ + _PAD_))))
        _img_  = Image.new('L', (_atlas_w_, _atlas_h_), 0)
        _draw_ = ImageDraw.Draw(_img_)
        self._glyphs_ = {}
        for ch, _bbox_, _gx_, _gy_, _w_, _h_ in _placements_:
            _draw_.text((_gx_ - _bbox_[0], _gy_ - _bbox_[1]), ch, fill=255, font=self._font48_)
            self._glyphs_[ch] = {
                'u0': _gx_ / _atlas_w_,         'v0': _gy_ / _atlas_h_,
                'u1': (_gx_ + _w_) / _atlas_w_, 'v1': (_gy_ + _h_) / _atlas_h_,
                'bx': _bbox_[0],                'by': _bbox_[1] - self._ascent48_,  # relative to baseline
                'w':  _w_,                      'h':  _h_,
            }
        self._img_     = _img_
        self._png_b64_ = None
        self.version  += 1

    #
    # layoutText() - per-glyph quad placement matching svgText()/textLength() semantics
    # - returns a list of 12-float tuples: (ox, oy, dx, dy, w, h, cos, sin, u0, v0, u1, v1)
    # - (ox, oy) is the SVG text anchor point; rotation (degrees) rotates around it,
    #   matching transform="rotate(deg, x, y)"
    #
    def layoutText(self, txt, x, y, txt_h, anchor='start', rotation=None, dy=0.0):
        _size_ = int(round(txt_h))
        if _size_ <= 0 or not txt: return []
        _missing_ = set(txt) - self._charset_
        if _missing_: self._build_(_missing_)
        _font_  = self._sizeFont_(_size_)
        _scale_ = _size_ / float(ATLAS_PX)
        _total_ = _font_.getlength(txt)
        if   anchor == 'middle': _anchor_dx_ = -_total_ / 2.0
        elif anchor == 'end':    _anchor_dx_ = -_total_
        else:                    _anchor_dx_ = 0.0
        if rotation is not None:
            _rad_ = math.radians(float(rotation))
            _cos_, _sin_ = math.cos(_rad_), math.sin(_rad_)
        else:
            _cos_, _sin_ = 1.0, 0.0
        _out_ = []
        for i, ch in enumerate(txt):
            _g_ = self._glyphs_.get(ch)
            if _g_ is None: continue   # whitespace / unrenderable -- advance handled by prefix length
            _pen_x_ = _font_.getlength(txt[:i])
            _out_.append((float(x), float(y),
                          _anchor_dx_ + _pen_x_ + _g_['bx'] * _scale_,
                          _g_['by'] * _scale_ + dy,
                          _g_['w'] * _scale_, _g_['h'] * _scale_,
                          _cos_, _sin_,
                          _g_['u0'], _g_['v0'], _g_['u1'], _g_['v1']))
        return _out_

    #
    # payload() - JSON-safe atlas description for the JS runtime (texture upload)
    #
    def payload(self):
        if self._png_b64_ is None:
            import io
            _buf_ = io.BytesIO()
            self._img_.save(_buf_, format='PNG')
            self._png_b64_ = base64.b64encode(_buf_.getvalue()).decode()
        return {
            'png_b64': self._png_b64_,
            'w':       self._img_.width,
            'h':       self._img_.height,
            'version': self.version,
        }
