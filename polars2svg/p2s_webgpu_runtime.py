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

P2S_GPU_JS = r"""
if (!window.__P2S_GPU__) {
window.__P2S_GPU__ = {
  supported: function() { return !!navigator.gpu; },

  _b64ToBuffer: function(b64) {
    const bin = atob(b64);
    const buf = new ArrayBuffer(bin.length);
    const view = new Uint8Array(buf);
    for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
    return buf;
  },

  ensure: function() {
    // Cache the in-flight promise, not just the resolved device: panelize() mounts
    // every view concurrently and each render() awaits ensure() in the same tick. If
    // we only guarded on this._device, all N callers would sail past the guard before
    // any requestDevice() resolved and each would mint its OWN GPUDevice. The shared
    // pipeline/atlas caches (keyed by format, not device) would then bind one device's
    // pipelines to another device's command encoder -> validation error -> blank canvas.
    if (this._devicePromise) return this._devicePromise;
    this._devicePromise = (async () => {
      if (!navigator.gpu) throw new Error('WebGPU not available');
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) throw new Error('No WebGPU adapter');
      this._device = await adapter.requestDevice();
      this._fmt    = navigator.gpu.getPreferredCanvasFormat();
      return this._device;
    })();
    return this._devicePromise;
  },

  _shaderSrc: `
struct Uni { res : vec4<f32> };
@group(0) @binding(0) var<uniform> u : Uni;

fn to_ndc(p : vec2<f32>) -> vec4<f32> {
  return vec4<f32>(p.x * u.res.x + u.res.z, p.y * u.res.y + u.res.w, 0.0, 1.0);
}

const corners = array<vec2<f32>, 6>(
  vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
  vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0));

// ---- rect (rounded-rect SDF) ----
struct RectIn {
  @location(0) xywh : vec4<f32>,
  @location(1) rx   : f32,
  @location(2) col  : vec4<f32>,
};
struct RectOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) col   : vec4<f32>,
  @location(1) local : vec2<f32>,
  @location(2) half  : vec2<f32>,
  @location(3) rx    : f32,
};
@vertex fn rect_vs(inst : RectIn, @builtin(vertex_index) vi : u32) -> RectOut {
  var o : RectOut;
  let c = corners[vi];
  let p = inst.xywh.xy + c * inst.xywh.zw;
  o.pos   = to_ndc(p);
  o.col   = inst.col;
  o.half  = inst.xywh.zw * 0.5;
  o.local = (c - vec2<f32>(0.5, 0.5)) * inst.xywh.zw;
  o.rx    = inst.rx;
  return o;
}
@fragment fn rect_fs(v : RectOut) -> @location(0) vec4<f32> {
  if (v.rx > 0.0) {
    let q = abs(v.local) - (v.half - vec2<f32>(v.rx, v.rx));
    let d = length(max(q, vec2<f32>(0.0, 0.0))) - v.rx;
    if (d > 0.0) { discard; }
  }
  return v.col;
}

// ---- circle (quad + SDF; fill + optional stroke) ----
struct CircIn {
  @location(0) geo    : vec4<f32>,  // cx, cy, radius, stroke_w
  @location(1) fill   : vec4<f32>,
  @location(2) stroke : vec4<f32>,
};
struct CircOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) fill   : vec4<f32>,
  @location(1) stroke : vec4<f32>,
  @location(2) local  : vec2<f32>,  // px from center
  @location(3) rsw    : vec2<f32>,  // radius, stroke_w
};
@vertex fn circle_vs(inst : CircIn, @builtin(vertex_index) vi : u32) -> CircOut {
  var o : CircOut;
  let ext = inst.geo.z + inst.geo.w * 0.5 + 1.0;
  let c   = corners[vi] * 2.0 - vec2<f32>(1.0, 1.0);
  o.pos    = to_ndc(inst.geo.xy + c * ext);
  o.fill   = inst.fill;
  o.stroke = inst.stroke;
  o.local  = c * ext;
  o.rsw    = vec2<f32>(inst.geo.z, inst.geo.w);
  return o;
}
@fragment fn circle_fs(v : CircOut) -> @location(0) vec4<f32> {
  let d  = length(v.local);
  let r  = v.rsw.x;
  let sw = v.rsw.y;
  let aa = 0.75;
  if (sw > 0.0 && v.stroke.a > 0.0) {
    let outer = r + sw * 0.5;
    let inner = r - sw * 0.5;
    if (d <= inner) { return v.fill; }
    let edge = 1.0 - smoothstep(outer - aa, outer + aa, d);
    if (edge <= 0.0) { discard; }
    var c = v.stroke;
    c.a = c.a * edge;
    return c;
  }
  let edge = 1.0 - smoothstep(r - aa, r + aa, d);
  if (edge <= 0.0) { discard; }
  var c = v.fill;
  c.a = c.a * edge;
  return c;
}

// ---- line (screen-aligned quad + dash) ----
struct LineIn {
  @location(0) pts  : vec4<f32>,  // x0, y0, x1, y1
  @location(1) wd   : f32,
  @location(2) col  : vec4<f32>,
  @location(3) dash : vec2<f32>,  // on, off (0 = solid)
};
struct LineOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) col   : vec4<f32>,
  @location(1) along : f32,
  @location(2) dash  : vec2<f32>,
};
@vertex fn line_vs(inst : LineIn, @builtin(vertex_index) vi : u32) -> LineOut {
  var o : LineOut;
  let p0  = inst.pts.xy;
  let p1  = inst.pts.zw;
  var dir = p1 - p0;
  let len = max(length(dir), 1e-6);
  dir = dir / len;
  let n   = vec2<f32>(-dir.y, dir.x) * max(inst.wd, 0.5) * 0.5;
  let c   = corners[vi];
  let p   = mix(p0, p1, c.x) + n * (c.y * 2.0 - 1.0);
  o.pos   = to_ndc(p);
  o.col   = inst.col;
  o.along = c.x * len;
  o.dash  = inst.dash;
  return o;
}
@fragment fn line_fs(v : LineOut) -> @location(0) vec4<f32> {
  if (v.dash.x > 0.0) {
    let period = v.dash.x + v.dash.y;
    if ((v.along % period) > v.dash.x) { discard; }
  }
  return v.col;
}

// ---- tri (per-vertex colored triangles) ----
struct TriIn {
  @location(0) xy  : vec2<f32>,
  @location(1) col : vec4<f32>,
};
struct TriOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) col : vec4<f32>,
};
@vertex fn tri_vs(v : TriIn) -> TriOut {
  var o : TriOut;
  o.pos = to_ndc(v.xy);
  o.col = v.col;
  return o;
}
@fragment fn tri_fs(v : TriOut) -> @location(0) vec4<f32> { return v.col; }

// ---- glyph (atlas-textured quads, per-glyph rotation about the text anchor) ----
struct GlyphIn {
  @location(0) od   : vec4<f32>,  // ox, oy, dx, dy
  @location(1) whcs : vec4<f32>,  // w, h, cos, sin
  @location(2) uv   : vec4<f32>,  // u0, v0, u1, v1
  @location(3) col  : vec4<f32>,
};
struct GlyphOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv  : vec2<f32>,
  @location(1) col : vec4<f32>,
};
@group(0) @binding(1) var atlas_samp : sampler;
@group(0) @binding(2) var atlas_tex  : texture_2d<f32>;
@vertex fn glyph_vs(inst : GlyphIn, @builtin(vertex_index) vi : u32) -> GlyphOut {
  var o : GlyphOut;
  let c     = corners[vi];
  let local = inst.od.zw + c * inst.whcs.xy;
  let rot   = vec2<f32>(local.x * inst.whcs.z - local.y * inst.whcs.w,
                        local.x * inst.whcs.w + local.y * inst.whcs.z);
  o.pos = to_ndc(inst.od.xy + rot);
  o.uv  = mix(inst.uv.xy, inst.uv.zw, c);
  o.col = inst.col;
  return o;
}
@fragment fn glyph_fs(v : GlyphOut) -> @location(0) vec4<f32> {
  let a = textureSample(atlas_tex, atlas_samp, v.uv).r;
  if (a <= 0.004) { discard; }
  var c = v.col;
  c.a = c.a * a;
  return c;
}
`,

  _pipelines: function(device, fmt) {
    if (this._pipeCache && this._pipeCacheKey === fmt) return this._pipeCache;
    const module = device.createShaderModule({ code: this._shaderSrc });
    const blend = {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one',       dstFactor: 'one-minus-src-alpha', operation: 'add' },
    };
    const target = { format: fmt, blend: blend };
    const mk = (vs, fs, buffers, stepMode) => device.createRenderPipeline({
      layout: 'auto',
      vertex:   { module, entryPoint: vs, buffers: [{ arrayStride: buffers.stride, stepMode: stepMode,
                  attributes: buffers.attrs }] },
      fragment: { module, entryPoint: fs, targets: [target] },
      primitive: { topology: 'triangle-list' },
      multisample: { count: 4 },
    });
    const f4 = 'float32x4', f2 = 'float32x2', f1 = 'float32';
    this._pipeCache = {
      rect: mk('rect_vs', 'rect_fs', { stride: 36, attrs: [
        { shaderLocation: 0, offset: 0,  format: f4 },
        { shaderLocation: 1, offset: 16, format: f1 },
        { shaderLocation: 2, offset: 20, format: f4 }] }, 'instance'),
      circle: mk('circle_vs', 'circle_fs', { stride: 48, attrs: [
        { shaderLocation: 0, offset: 0,  format: f4 },
        { shaderLocation: 1, offset: 16, format: f4 },
        { shaderLocation: 2, offset: 32, format: f4 }] }, 'instance'),
      line: mk('line_vs', 'line_fs', { stride: 44, attrs: [
        { shaderLocation: 0, offset: 0,  format: f4 },
        { shaderLocation: 1, offset: 16, format: f1 },
        { shaderLocation: 2, offset: 20, format: f4 },
        { shaderLocation: 3, offset: 36, format: f2 }] }, 'instance'),
      tri: mk('tri_vs', 'tri_fs', { stride: 24, attrs: [
        { shaderLocation: 0, offset: 0, format: f2 },
        { shaderLocation: 1, offset: 8, format: f4 }] }, 'vertex'),
      glyph: mk('glyph_vs', 'glyph_fs', { stride: 64, attrs: [
        { shaderLocation: 0, offset: 0,  format: f4 },
        { shaderLocation: 1, offset: 16, format: f4 },
        { shaderLocation: 2, offset: 32, format: f4 },
        { shaderLocation: 3, offset: 48, format: f4 }] }, 'instance'),
    };
    this._pipeCacheKey = fmt;
    return this._pipeCache;
  },

  _atlasTexture: async function(device, atlas) {
    if (this._atlasTex && this._atlasVersion === atlas.version) return this._atlasTex;
    const bytes  = new Uint8Array(this._b64ToBuffer(atlas.png_b64));
    const blob   = new Blob([bytes], { type: 'image/png' });
    const bitmap = await createImageBitmap(blob);
    const tex = device.createTexture({
      size: [atlas.w, atlas.h], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: tex }, [atlas.w, atlas.h]);
    this._atlasTex     = tex;
    this._atlasVersion = atlas.version;
    return tex;
  },

  _hexToLinearPremul: function(hex) {
    const srgbToLin = (c) => (c <= 0.04045) ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    const r = parseInt(hex.slice(1, 3), 16) / 255.0;
    const g = parseInt(hex.slice(3, 5), 16) / 255.0;
    const b = parseInt(hex.slice(5, 7), 16) / 255.0;
    return { r: r, g: g, b: b, a: 1.0 };
  },

  render: async function(canvas, payload) {
    const device = await this.ensure();
    const fmt    = this._fmt;
    const pipes  = this._pipelines(device, fmt);
    const w = payload.wxh[0], h = payload.wxh[1];
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('webgpu');
    ctx.configure({ device, format: fmt, alphaMode: 'premultiplied' });

    // shared uniform: pixel -> NDC
    const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(uni, 0, new Float32Array([2.0 / w, -2.0 / h, -1.0, 1.0]));

    // instance / vertex / index buffers
    const bufs = {};
    for (const kind of ['rect', 'circle', 'line', 'glyph', 'tri_v']) {
      if (payload.buffers[kind]) {
        const data = this._b64ToBuffer(payload.buffers[kind]);
        bufs[kind] = device.createBuffer({ size: Math.ceil(data.byteLength / 4) * 4,
                                           usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufs[kind], 0, data);
      }
    }
    if (payload.buffers.tri_i) {
      const data = this._b64ToBuffer(payload.buffers.tri_i);
      bufs.tri_i = device.createBuffer({ size: Math.ceil(data.byteLength / 4) * 4,
                                         usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(bufs.tri_i, 0, data);
    }

    // bind groups (glyph pipeline carries the atlas)
    const groups = {};
    for (const kind of ['rect', 'circle', 'line', 'tri']) {
      groups[kind] = device.createBindGroup({ layout: pipes[kind].getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: uni } }] });
    }
    if (payload.atlas) {
      const tex  = await this._atlasTexture(device, payload.atlas);
      const samp = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
      groups.glyph = device.createBindGroup({ layout: pipes.glyph.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: uni } },
                  { binding: 1, resource: samp },
                  { binding: 2, resource: tex.createView() }] });
    }

    // 4x MSAA color target resolved into the canvas
    const msaa = device.createTexture({ size: [w, h], sampleCount: 4, format: fmt,
                                        usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const bg  = this._hexToLinearPremul(payload.bg || '#ffffff');
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{
      view: msaa.createView(), resolveTarget: ctx.getCurrentTexture().createView(),
      clearValue: bg, loadOp: 'clear', storeOp: 'store' }] });

    for (const entry of payload.manifest) {
      const kind = entry.kind;
      if (kind === 'tri') {
        if (!bufs.tri_v || !groups.tri) continue;
        pass.setPipeline(pipes.tri);
        pass.setBindGroup(0, groups.tri);
        pass.setVertexBuffer(0, bufs.tri_v);
        pass.setIndexBuffer(bufs.tri_i, 'uint32');
        if (entry.scissor) pass.setScissorRect(entry.scissor[0], entry.scissor[1], entry.scissor[2], entry.scissor[3]);
        else               pass.setScissorRect(0, 0, w, h);
        pass.drawIndexed(entry.count, 1, entry.first, 0, 0);
      } else {
        if (!bufs[kind] || !groups[kind]) continue;
        pass.setPipeline(pipes[kind]);
        pass.setBindGroup(0, groups[kind]);
        pass.setVertexBuffer(0, bufs[kind]);
        if (entry.scissor) pass.setScissorRect(entry.scissor[0], entry.scissor[1], entry.scissor[2], entry.scissor[3]);
        else               pass.setScissorRect(0, 0, w, h);
        pass.draw(6, entry.count, 0, entry.first);
      }
    }
    pass.end();
    device.queue.submit([enc.finish()]);
  },
};
}
"""

#
# standalone_html() - self-contained HTML (canvas + runtime + payload) for notebook
# display and manual verification; no Panel required
#
def standalone_html(payload, border='1px solid #ccc'):
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
