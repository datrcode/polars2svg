from typing import Any

_INTERACTIVE_DEPS_MSG_ = (
    "interactive components require the optional 'interactive' dependencies "
    "(panel, jupyter_bokeh, param). Install them with:\n"
    "    pip install polars2svg[interactive]"
)


# interactive_controller.py / stack_control.py pull in panel (+ jupyter_bokeh,
# param), an optional dependency — every method below imports lazily so the
# static render path never pays for it, and importing raises a clear
# ImportError naming the extra rather than a bare "no module named panel".
def _importInteractiveController_():
    try:
        from . import interactive_controller as _mod_
    except ImportError as _e_:
        raise ImportError(_INTERACTIVE_DEPS_MSG_) from _e_
    return _mod_


def _importStackControl_():
    try:
        from . import stack_control as _mod_
    except ImportError as _e_:
        raise ImportError(_INTERACTIVE_DEPS_MSG_) from _e_
    return _mod_


class P2SInteractiveMixin:
    def __init__(self):
        pass

    def __p2s_interactive_mixin_init__(self):
        pass

    def interactiveController(self):
        '''Create a fresh InteractionController — the shared model/view/controller
        that coordinates selection and drill-down across interactive panels. Normally
        created for you by ``panelize()``; construct one directly only when wiring
        interactive components by hand.'''
        return _importInteractiveController_().InteractionController()

    def panelize(self, layout: Any, stack: str = 'default', use_webgpu: bool = False) -> Any:
        '''
        Compose interactive components into a single cross-linked dashboard.

        ``layout`` is a nested list describing the panel grid — each inner list is a
        row of components (interactive variants from ``xypi``/``histopi``/``linkpi``/…,
        or plain components which are wrapped automatically). Returns a widget whose
        ``.mvc`` drives shared selection/drill-down across panels.

        use_webgpu=True renders supported components (xyp, histop) on a WebGPU
        canvas; unsupported components keep their SVG wrappers.

        Example::

            panel = p2s.panelize([[p2s.xypi(chart_a), p2s.histopi(chart_b)]])
            # await panel.mvc.replaceStack('default', df_new)   # swap the backing data
        '''
        return _importInteractiveController_().panelize(layout, stack, use_webgpu=use_webgpu)

    def panelizeSketch(self, layout):
        return _importInteractiveController_().panelizeSketch(layout)

    def xypi(self, _xyp_, **kwargs):
        '''Wrap a static ``xyp`` component as an interactive, cross-linkable panel
        (brushing/selection). Pass the result to ``panelize()``.'''
        return _importInteractiveController_().xypi(_xyp_, **kwargs)

    def histopi(self, _histop_, **kwargs):
        '''Wrap a static ``histop`` component as an interactive, cross-linkable panel
        (bar selection). Pass the result to ``panelize()``.'''
        return _importInteractiveController_().histopi(_histop_, **kwargs)

    def timepi(self, _timep_, **kwargs):
        '''Wrap a static ``timep`` component as an interactive, cross-linkable panel
        (time-range selection). Pass the result to ``panelize()``.'''
        return _importInteractiveController_().timepi(_timep_, **kwargs)

    def linkpi(self, _linkp_, mvc=None, **kwargs):
        '''Wrap a static ``linkp`` graph as an interactive, cross-linkable panel
        (node/edge selection). Pass the result to ``panelize()``.'''
        return _importInteractiveController_().linkpi(_linkp_, mvc=mvc, **kwargs)

    def smallpi(self, _smallp_, **kwargs):
        '''Wrap a static ``smallp`` small-multiples view as an interactive,
        cross-linkable panel. Pass the result to ``panelize()``.'''
        return _importInteractiveController_().smallpi(_smallp_, **kwargs)

    def stack_controli(self, component, stack_name='default', **kwargs):
        return _importStackControl_().stack_controli(component, stack_name, **kwargs)
