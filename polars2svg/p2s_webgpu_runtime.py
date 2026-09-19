#
# p2s_webgpu_runtime - the single shared JS/WGSL runtime for all components
#
# One shader module, five pipelines (rect / circle / line / tri / glyph) that render
# the DisplayList payload produced by p2s_displaylist.DisplayList.webgpu_payload().
# Coordinates in the instance buffers are SVG's y-down pixels; the shared uniform
# vec4(2/w, -2/h, -1, 1) maps them to NDC, so buffers are used as-is.
#
# The runtime installs itself once per page as window.__P2S_GPU__ (device/adapter and
# pipeline caches are shared across every view on the page).
#
import json

from .p2s_esm import js_text

#: The shared JS/WGSL runtime, read from polars2svg/js/fragments/p2s_gpu_runtime.js.
#: It used to be a raw string literal here.  It is a real .js file now so that an editor
#: and a linter can see it, and so that exactly one copy feeds both the Panel views
#: (via the *_GPU _esm compositions) and standalone_html() below.
#:
#: It is a FRAGMENT, not an ES module: no import, no export.  standalone_html() drops it
#: into a classic <script>, and the ESM component modules concatenate it -- see the note
#: at the top of p2s_esm.py for why sibling imports are not an option here.
P2S_GPU_JS = js_text('fragments/p2s_gpu_runtime.js')

#
# standalone_html() - self-contained HTML (canvas + runtime + payload) for notebook
# display and manual verification; no Panel required
#
def standalone_html(payload: dict, border: str = '1px solid #ccc') -> str:
    import random
    _canvas_id_ = f'p2s_gpu_{random.randint(100000, 999999)}'  # nosec B311 - non-cryptographic DOM id scoping, see SECURITY.md
    w, h = payload['wxh']
    _payload_json_ = json.dumps(payload)
    return f"""\
<div style="display:inline-block;border:{border};">
<canvas id="{_canvas_id_}" width="{w}" height="{h}" style="display:block;"></canvas>
</div>
<script>
(async () => {{
{P2S_GPU_JS}
const canvas  = document.getElementById('{_canvas_id_}');
const payload = {_payload_json_};
if (!window.__P2S_GPU__.supported()) {{
  canvas.parentElement.innerHTML =
    '<p style="color:#a00;padding:12px;font-family:monospace">WebGPU not available in this browser.</p>';
  return;
}}
await window.__P2S_GPU__.render(canvas, payload);
}})();
</script>
"""
