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
