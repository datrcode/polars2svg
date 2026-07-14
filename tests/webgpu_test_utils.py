#
# Shared helpers for WebGPU payload tests: decode payload buffers back to numpy
# and parse primitive attributes out of the SVG representation for parity checks.
#
import base64
import re

import numpy as np

from polars2svg.p2s_displaylist import FLOATS_PER_INSTANCE


def decode_buffer(payload, kind):
    '''Decode a payload instance buffer into an (n, floats_per_instance) float32 array.'''
    if kind not in payload['buffers']:
        return np.zeros((0, FLOATS_PER_INSTANCE[kind]), dtype=np.float32)
    raw = base64.b64decode(payload['buffers'][kind])
    arr = np.frombuffer(raw, dtype='<f4')
    return arr.reshape(-1, FLOATS_PER_INSTANCE[kind])


def manifest_count(payload, kind):
    '''Total instance count for a kind across all manifest entries.'''
    return sum(m['count'] for m in payload['manifest'] if m['kind'] == kind)


def parse_svg_fill_rects(svg):
    '''Parse (x, y, w, h, fill) for every fill-painted <rect> in document order.'''
    out = []
    for m in re.finditer(r'<rect ([^/>]*)/?>', svg):
        attrs = dict(re.findall(r'([\w-]+)="([^"]*)"', m.group(1)))
        if attrs.get('fill', 'none') == 'none':
            continue
        out.append((float(attrs['x']), float(attrs['y']),
                    float(attrs['width']), float(attrs['height']), attrs['fill']))
    return out


def parse_svg_circles(svg):
    '''Parse (cx, cy, fill_or_None) for every <circle> in document order.'''
    out = []
    for m in re.finditer(r'<circle ([^/>]*)/?>', svg):
        attrs = dict(re.findall(r'([\w-]+)="([^"]*)"', m.group(1)))
        out.append((float(attrs['cx']), float(attrs['cy']), attrs.get('fill')))
    return out


def hex_to_rgb01(hexcolor):
    return (int(hexcolor[1:3], 16) / 255.0,
            int(hexcolor[3:5], 16) / 255.0,
            int(hexcolor[5:7], 16) / 255.0)
