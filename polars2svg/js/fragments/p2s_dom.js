//
// p2s_dom - element construction for the ESM component modules
//
// A ReactiveHTML view declared its DOM in an HTML template and Panel handed each
// id'd node back as a JS variable.  An ESM component builds its own DOM, so this is
// what replaces the template: one helper, used once per element the template had.
//
// Ids are kept exactly as the templates wrote them.  Nothing depends on them any
// more -- the render closure holds every element directly -- but the Playwright
// harness addresses elements by id, and each component sits in its own shadow root,
// so they stay both free and useful.
//
// This is a FRAGMENT: no import, no export.  See polars2svg/p2s_esm.py.
//

function svgEl(tag, attrs, parent) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  if (attrs) {
    for (const k in attrs) {
      const v = attrs[k];
      if (v !== null && v !== undefined) { el.setAttribute(k, v); }
    }
  }
  if (parent) { parent.appendChild(el); }
  return el;
}

function htmlEl(tag, attrs, parent) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const k in attrs) {
      const v = attrs[k];
      if (v !== null && v !== undefined) { el.setAttribute(k, v); }
    }
  }
  if (parent) { parent.appendChild(el); }
  return el;
}

//
// p2sInk() - a color from the Python-side palette, by name
//
// model.palette is Polars2SVG.interactivePalette() -- {ink, bg, hint, selection,
// setop:{replace,add,subtract,intersect}}.  Overlays used to hardcode '#000000',
// which is 1.12:1 against the dark palette's #121212: the drag band and the status
// line were not merely off-theme, they were invisible.
//
// The fallback is the LIGHT value for each key, so a view that somehow renders before
// the param arrives looks exactly as it did before palettes existed -- and a key added
// to the Python dict but not here degrades to a visible color rather than 'undefined'.
//
const P2S_INK_FALLBACK = {
  ink: '#000000', bg: '#ffffff', hint: '#0000cc', selection: '#ff0000',
  replace: '#000000', add: '#00ff00', subtract: '#ff0000', intersect: '#0000ff',
};

function p2sInk(model, key) {
  const pal = (model && model.palette) || {};
  const val = (pal.setop && pal.setop[key] !== undefined) ? pal.setop[key] : pal[key];
  return val || P2S_INK_FALLBACK[key] || '#000000';
}
