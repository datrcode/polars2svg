//
// p2s_gpu_mount - the WebGPU canvas a *_GPU view draws into
//
// Replaces two things from the ReactiveHTML build: the jinja head/tail that wrapped
// each template in <div id="gpuwrap"><canvas id="gpucanvas">, and the two script
// fragments the old `_withGpuScripts_` appended (the runtime onto `render`, and a
// `gpu_payload` param-change entry).  Same DOM, same ids, same behaviour.
//
// Only the *_GPU compositions include this file, which is what keeps the ~14 KB
// runtime off the SVG views -- the reason the subclass split exists at all.  An
// entry module therefore calls p2sGpuWrap() defensively:
//
//     return (typeof p2sGpuWrap === 'function') ? p2sGpuWrap(model, root) : root;
//
// This is a FRAGMENT: no import, no export.  It also assumes p2s_gpu_runtime.js has
// been concatenated ahead of it, which is what installs window.__P2S_GPU__.
//

function p2sGpuDraw(model, canvas) {
  if (model.use_webgpu && window.__P2S_GPU__ && window.__P2S_GPU__.supported() && !model.gpu_error) {
    window.__P2S_GPU__.render(canvas, model.gpu_payload)
      .catch(function(e) {
        console.warn('p2s webgpu:', e);
        model.gpu_error = (e && e.message) ? e.message : String(e);
      });
  }
}

// Wrap `rootSvg` in the canvas scaffold and start rendering; returns the element the
// component should hand back to Panel.  A non-GPU view passes straight through, so an
// entry module needs no branch of its own.
function p2sGpuWrap(model, rootSvg) {
  if (!model.use_webgpu) { return rootSvg; }

  const wrap = htmlEl('div', {
    id: 'gpuwrap',
    style: 'position:relative;width:' + model.svg_w + 'px;height:' + model.svg_h + 'px;',
  });
  const canvas = htmlEl('canvas', {
    id: 'gpucanvas',
    width: model.svg_w,
    height: model.svg_h,
    style: 'position:absolute;left:0;top:0;',
  }, wrap);
  wrap.appendChild(rootSvg);

  // The SVG overlays the canvas.  Under ReactiveHTML this came from a
  // `{% if use_webgpu %}` *appended inside* the root tag's existing style attribute, so
  // this appends too rather than replacing: LINKPI's root carries `user-select:none`,
  // and overwriting the attribute would silently drop it on the GPU variant only.  The
  // other contracts set no style on their root, so appending is identical for them.
  const prevStyle = rootSvg.getAttribute('style') || '';
  rootSvg.setAttribute('style', prevStyle + 'position:absolute;left:0;top:0;');

  if (!window.__P2S_GPU__.supported()) {
    model.gpu_error = 'WebGPU is not available in this browser.';
  } else {
    p2sGpuDraw(model, canvas);
  }
  model.on('gpu_payload', function() { p2sGpuDraw(model, canvas); });

  return wrap;
}
