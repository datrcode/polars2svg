#
# Treatment declarations for the interactive wrappers' dispatch registries.
#
# Every entry in linkpi's layout and background registries carries one of these.  A
# Treatment records what CAN be done about an operation that runs long -- deliberately
# not how long it will run.
#
# That distinction is the load-bearing design decision.  Measurements taken 2026-08-26/27
# found the same algorithm on the same graph family yielding per-doubling exponents of
# 3.1, then 1.7, then 2.8; and the flow-map layout varying 12x in runtime at a *fixed*
# flow count, depending only on how clustered the flows were.  Nothing fitted to numbers
# that unstable would classify reliably, and a cost model that is wrong by 100x still
# mis-classifies.  So the registry declares properties of the *code* -- whether there is
# a loop to interrupt, whether the handler is pure -- which are stable, readable and
# checkable, and the actual bounding of runaway work is left to a deadline on the
# operations that can take one and to an explicit confirmation on the ones that cannot.
#

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple


@dataclass(frozen=True)
class Treatment:
    '''What can be done about an operation that runs longer than a user will wait.

    truncatable
        The operation has a step loop *we* own, so it can be given a deadline and asked
        for its best-so-far result.  False for anything imported: networkx and scipy
        expose no per-iteration hook.

    killable
        The handler is a pure function of its arguments, so it can be moved to a separate
        process and killed by a watchdog.  Defaults to False -- the conservative value,
        because the failure mode of a wrong True (state silently stashed on the
        controller and lost across the process boundary) is invisible, while the failure
        mode of a wrong False is only a missed optimisation.

    levers
        Parameters that trade output quality for time, most-preferred first.  Empty when
        the operation has no knob short of not running at all.

    confirm_above
        Node count above which the operation should ask before running, or None when it
        never needs to.  This is a declared observation written down from a measurement
        by a human, not a value fitted or extrapolated at runtime -- see the module note
        above for why that distinction matters here.
    '''
    truncatable:   bool = False
    killable:      bool = False
    levers:        Tuple[str, ...] = ()
    confirm_above: Optional[int] = None


@dataclass(frozen=True)
class RegistryEntry:
    '''A dispatch-registry value: the handler, plus what can be done about it.

    Deliberately *not* callable.  Making it callable would let every existing
    ``registry[key](...)`` keep working untouched, which is exactly the problem: a call
    site that never has to mention the treatment is a call site that will silently
    bypass it.  Callers unwrap ``.handler``, so consulting the declaration is a visible
    choice rather than an easy omission.
    '''
    handler:   Callable
    treatment: Treatment = field(default_factory=Treatment)


# Shared declarations, so the registries read as a table rather than a wall of kwargs.

# Measured well inside interactive latency at n=4000 (2026-08-27): hyper tree 0.04 s,
# connected components 0.02 s, circle pack 0.01 s, landmark mds 0.10 s, pivot mds 0.06 s,
# neighborhood (graph) 0.64 s.  Nothing to bound, nothing to ask about.
CHEAP = Treatment(killable=True)

# Not measured, but structurally trivial -- no iteration over the graph beyond a single
# pass.  Declared separately from CHEAP so that a later measurement has somewhere
# specific to land.
ASSUMED_CHEAP = Treatment(killable=True)

# Louvain community detection (networkx).  Measured 0.02 / 0.08 / 0.18 / 0.49 s at
# n=500/1000/2000/4000 (2026-08-27) -- comfortably interactive, and imported, so there is
# no loop to interrupt.  Declared anyway: it is reachable from a keystroke like everything
# else, and an operation nothing can see is an operation nothing can protect.
COMMUNITY_DETECTION = Treatment(truncatable=False, killable=True)

# link_shape='flowmap' (ODFlowLayout).  The one genuinely unbounded operation still
# reachable from a keystroke.  Measured 1.90 / 29.41 / 284.61 s at 100 / 200 / 400 flows
# on Metal, with per-doubling exponents of 3.3-4.0 -- and 12x variation at a *fixed* flow
# count depending only on how clustered the flows are, which is why the threshold below is
# a place to ask a question rather than a predicted duration.  It is ours and loop-based,
# so a deadline can bound it properly later; until then asking is the only protection.
# 200 is the count linkp already warns at (see __flowmapControlPoints__).
FLOWMAP = Treatment(truncatable=True, killable=True,
                    levers=('samples_per_flow', 'iterations', 'top_k_flows'),
                    confirm_above=200)


# ---------------------------------------------------------------------------------------
# The layout-operation declarations, by menu label.
#
# Module level rather than built inside the registry, because two things need them and one
# of them has no controller instance to ask: the picker menus are serialised into the
# browser at construction time, and an operation that will stop to ask a question should
# say so in the menu rather than only when the user presses the key.
#
# Anything absent from this map is CHEAP.  That is the deliberate default -- a new layout
# is assumed harmless until measured -- and the registry test asserts every entry declares
# something, so the assumption is visible rather than silent.
# ---------------------------------------------------------------------------------------

# networkx exposes no per-iteration hook, so spring_layout cannot be given a deadline --
# but it is a pure function of its arguments, so a watchdog could kill it in a subprocess,
# and it takes an iterations= knob.  Measured 0.19 / 0.78 / 3.15 / 10.39 s at
# n=500/1000/2000/4000 (2026-08-27); growth ~1.7x per doubling puts it near half a minute
# around n=8000, which is where asking first starts to beat simply running it.
SPRING_NX = Treatment(truncatable=False, killable=True,
                      levers=('iterations',), confirm_above=8000)

# Ours, with real iteration loops (_stage2_optimize / _stage3_refine), so it takes a
# deadline.  Measured 2.46 / 3.02 / 10.65 / 18.10 s at n=500/1000/2000/4000 (2026-08-27) --
# the most expensive operation left in the registry.
NCP_PACK = Treatment(truncatable=True, killable=True,
                     levers=('force_iterations', 'power_iterations'), confirm_above=6000)

# Ours, with a max_iter loop, so it could take a deadline -- but it does not need one.
# Measured 0.19 / 0.13 / 0.30 / 1.08 s at n=500/1000/2000/4000 (2026-08-27, Apple-silicon
# Metal): the cheapest force layout here by a wide margin, because it is the only one whose
# kernels run on the GPU.  It is registered only when the mlx extra is installed, which is
# also what makes it fast, so the measurement covers the case in which the entry exists at
# all.  No confirmation: asking before a one-second operation is worse than running it.
TFDP = Treatment(truncatable=True, killable=True, levers=('max_iter',), confirm_above=None)

LAYOUT_TREATMENTS = {
    'spring nx': SPRING_NX,
    'ncp pack':  NCP_PACK,
    't-fdp':     TFDP,
}


def treatment_for(label: str) -> Any:
    '''The declaration for a layout-operation menu label; CHEAP when none is recorded.'''
    return LAYOUT_TREATMENTS.get(label, CHEAP)


def menu_annotation(treatment: Any, unit: str = 'nodes') -> str | None:
    '''The suffix a picker shows for an operation that will stop to ask, or None.

    Static, from the declaration rather than from the current graph: the menus are
    serialised into the browser once at construction, while the graph size changes as the
    user walks the stack.  "asks above N" is true whatever is on screen; a baked-in
    "4,166 flows" would start lying the moment anything was pushed or popped.
    '''
    if treatment is None or treatment.confirm_above is None:
        return None
    return f'asks >{treatment.confirm_above:,} {unit}'
