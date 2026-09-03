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
def _importInteractiveController_() -> Any:
    try:
        from . import interactive_controller as _mod_
    except ImportError as _e_:
        raise ImportError(_INTERACTIVE_DEPS_MSG_) from _e_
    return _mod_


def _importStackControl_() -> Any:
    try:
        from . import stack_control as _mod_
    except ImportError as _e_:
        raise ImportError(_INTERACTIVE_DEPS_MSG_) from _e_
    return _mod_


def _importSpreadLinePI_() -> Any:
    try:
        from . import spreadlinepi as _mod_
    except ImportError as _e_:
        raise ImportError(_INTERACTIVE_DEPS_MSG_) from _e_
    return _mod_


class P2SInteractiveMixin:
    def __init__(self) -> None:
        pass

    def __p2s_interactive_mixin_init__(self) -> None:
        pass

    def interactiveController(self) -> Any:
        '''Create a fresh InteractionController — the shared model/view/controller
        that coordinates selection and drill-down across interactive panels. Normally
        created for you by ``panelize()``; construct one directly only when wiring
        interactive components by hand.'''
        return _importInteractiveController_().InteractionController()

    def panelize(self, layout: Any, stack: str = 'default', use_webgpu: bool = False,
                 websocket_max_message_size: int | None = None) -> Any:
        '''
        Compose interactive components into a single cross-linked dashboard.

        ``layout`` is a nested list describing the panel grid — each inner list is a
        row of components (interactive variants from ``xypi``/``histopi``/``linkpi``/…,
        or plain components which are wrapped automatically). Returns a widget whose
        ``.mvc`` drives shared selection/drill-down across panels.

        use_webgpu=True renders every component with a ``webgpu()`` representation —
        all eight of them — on a WebGPU canvas; anything else keeps its SVG wrapper.
        The interactive overlay layer (selection rectangles, menus) stays SVG on top of
        the GPU canvas either way.

        ``websocket_max_message_size`` is the Bokeh WebSocket message limit (in bytes)
        you intend to serve this dashboard under. It is used only to decide whether to
        warn: if the composed SVG document would exceed it, panelize() logs a warning
        naming the measured size. A large linkp — e.g. timing marks over a
        netflow-sized frame — can push the document past Bokeh's 20 MB default and make
        the browser fail with "Unexpected end of JSON input". Pass the same value to
        your serve call to actually raise the limit (and to silence the warning)::

            panel = p2s.panelize(layout, websocket_max_message_size=200*1024*1024)
            panel.show(websocket_max_message_size=200*1024*1024)

        Example::

            panel = p2s.panelize([[p2s.xypi(chart_a), p2s.histopi(chart_b)]])
            # await panel.mvc.replaceStack('default', df_new)   # swap the backing data
        '''
        return _importInteractiveController_().panelize(
            layout, stack, use_webgpu=use_webgpu,
            websocket_max_message_size=websocket_max_message_size)

    def panelizeSketch(self, layout: list, use_webgpu: bool = False) -> Any:
        '''Static (non-server) HTML sketch of a ``panelize()`` layout.

        ``use_webgpu`` has the same meaning as in ``panelize()``.'''
        return _importInteractiveController_().panelizeSketch(layout, use_webgpu=use_webgpu)

    def xypi(self, _xyp_: Any, **kwargs: Any) -> Any:
        '''Wrap a static ``xyp`` component as an interactive, cross-linkable panel
        (brushing/selection). Pass the result to ``panelize()``.'''
        return _importInteractiveController_().xypi(_xyp_, **kwargs)

    def histopi(self, _histop_: Any, **kwargs: Any) -> Any:
        '''Wrap a static ``histop`` component as an interactive, cross-linkable panel
        (bar selection). Pass the result to ``panelize()``.'''
        return _importInteractiveController_().histopi(_histop_, **kwargs)

    def timepi(self, _timep_: Any, **kwargs: Any) -> Any:
        '''Wrap a static ``timep`` component as an interactive, cross-linkable panel
        (time-range selection). Pass the result to ``panelize()``.'''
        return _importInteractiveController_().timepi(_timep_, **kwargs)

    def linkpi(self, _linkp_: Any, mvc: Any = None, **kwargs: Any) -> Any:
        '''Wrap a static ``linkp`` graph as an interactive, cross-linkable panel
        (node/edge selection). Pass the result to ``panelize()``.'''
        return _importInteractiveController_().linkpi(_linkp_, mvc=mvc, **kwargs)

    def smallpi(self, _smallp_: Any, **kwargs: Any) -> Any:
        '''Wrap a static ``smallp`` small-multiples view as an interactive,
        cross-linkable panel. Pass the result to ``panelize()``.'''
        return _importInteractiveController_().smallpi(_smallp_, **kwargs)

    def chordpi(self, _chordp_: Any, **kwargs: Any) -> Any:
        '''Wrap a static ``chordp`` diagram as an interactive, cross-linkable panel
        (arc/ribbon selection). Pass the result to ``panelize()``.'''
        return _importInteractiveController_().chordpi(_chordp_, **kwargs)

    def piepi(self, _piep_: Any, **kwargs: Any) -> Any:
        '''Wrap a static ``piep`` chart as an interactive, cross-linkable panel
        (wedge selection). Pass the result to ``panelize()``.'''
        return _importInteractiveController_().piepi(_piep_, **kwargs)

    def spreadlinepi(self, _spread_: Any, **kwargs: Any) -> Any:
        '''Wrap a static ``spreadlinesp`` view as an interactive, cross-linkable panel
        (node/cloud selection). Pass the result to ``panelize()``.'''
        # Imported from its own module rather than off interactive_controller: that
        # module only re-exports the name when it wins the circular-import race.
        return _importSpreadLinePI_().spreadlinepi(_spread_, **kwargs)

    def stack_controli(self, component: Any, stack_name: str = 'default', **kwargs: Any) -> Any:
        return _importStackControl_().stack_controli(component, stack_name, **kwargs)
