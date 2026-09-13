from typing import Any

"""
Shared ReactiveHTML base for polars2svg's Panel views.

Every interactive view used to be built per call with
``type('XYZ', (ReactiveHTML,), {...})``.  That was not, as the comments around
it suggested, because a ReactiveHTML subclass fixes its width and height at
class-compile time -- it does not: ``_template`` is re-rendered per instance by
``ReactiveHTML._get_template()``, which runs it through jinja2 with every param
of the *instance* in the context, so ``{{ svg_w }}`` resolves per view.

The real reason was narrower, and it is what this base restores.

**The first-paint problem.**  A ``${param}`` written *between* tags (rather than
inside one) makes that param a ReactiveHTML *child*.  ``mod_inner`` is bound
that way, and children take a different path in and out of the browser:

  * ``ReactiveHTML._init_params()`` drops every child from the data model it
    builds, so ``data.mod_inner`` starts at the bokeh field's default -- which
    ``construct_data_model()`` took from the param's **class** default;
  * ``ReactiveHTML._update_model()`` does send a child's new value to
    ``data.mod_inner`` on every later write, and sends it **raw**.

So the render script's ``mod.innerHTML = data.mod_inner`` sees the class default
on first paint and instance values ever after.  Baking the plot into the class
default via ``type()`` made those agree; a static class cannot, and the view
paints blank until something writes ``mod_inner`` again -- the "extra step" the
racetrack_svg_framework version needed.

The class default cannot simply be replaced by passing the value through
``data`` either: ``_init_params()`` runs every *string* param through Panel's
HTML sanitizer, which strips an SVG to nothing (3663 bytes -> 104 in a
measured xyp render).  The child path is what bypasses it.

``_init_params()`` below therefore puts each child's instance value back into
the freshly built data model, raw -- matching exactly what the update path
already sends, and making first paint identical to what the dynamic class
produced.
"""

from panel.reactive import ReactiveHTML


class P2SReactiveHTML(ReactiveHTML):
    """ReactiveHTML with per-instance initial values for child params."""

    __abstract = True

    def _init_params(self) -> dict[str, Any]:
        _params_ = super()._init_params()
        _data_   = _params_.get('data')
        if _data_ is None:
            return _params_
        _fields_ = _data_.properties()
        for _child_ in self._parser.children.values():
            # A child whose param type has no data-model field (List/Dict/...)
            # is carried by the children mechanism alone -- nothing to seed.
            if _child_ in _fields_:
                setattr(_data_, _child_, getattr(self, _child_))
        return _params_
