#
# Budget - a wall-clock limit and/or a caller-supplied stop signal for an iterative layout.
#
# The iterative layouts in this package (ODFlowLayout, NCPLayout, TFDPLayout) all run a
# fixed iteration count chosen for output quality, not for time.  On a large enough input
# that count is a wait nobody sits through, and until now the only way to shorten it was to
# ask for fewer iterations up front -- a guess made before seeing how long an iteration
# actually takes.
#
# A Budget lets the caller say "stop when this long has passed" or "stop when I say so"
# instead.  Force layouts truncate gracefully: fewer iterations is a less-relaxed layout,
# not a broken one, so returning best-so-far is a real answer rather than a failure.
#
# Deliberately opt-in.  With no Budget every layout behaves exactly as before, which is
# what keeps their determinism tests meaningful -- a wall-clock limit is by nature not
# reproducible, so it must never be the default.
#

import time
from typing import Callable, Optional


class Budget:
    '''Stop condition for an iterative layout, checked once per outer iteration.

    Parameters
    ----------
    time_budget : float, optional
        Seconds of wall clock, measured from ``start()``. None means no time limit.
    should_stop : callable, optional
        Called with no arguments once per iteration; a truthy return stops the loop.
        This is how an interactive cancel reaches a layout already in flight -- there is
        no way to interrupt the thread itself, so the loop has to be asked to look.

    After the run, ``stopped_at`` is the iteration the loop broke on (None if it finished
    the full count) and ``note`` is a sentence for the caller to surface, in the manner of
    FlowFieldBackground's ``budget_note``.
    '''

    def __init__(self, time_budget: Optional[float] = None,
                 should_stop: Optional[Callable[[], bool]] = None) -> None:
        if time_budget is not None and float(time_budget) < 0:
            raise ValueError('time_budget must be >= 0')
        self.time_budget = None if time_budget is None else float(time_budget)
        self.should_stop = should_stop
        self.stopped_at: Optional[int]   = None
        self.note:       Optional[str]   = None
        self._t0_:       Optional[float] = None
        self._total_:    Optional[int]   = None
        self._done_:     int             = 0

    def start(self, total_iterations: int) -> 'Budget':
        '''Begin timing.  Call once, immediately before the loop.'''
        self._t0_        = time.monotonic()
        self._total_     = int(total_iterations)
        self.stopped_at  = None
        self.note        = None
        self._done_      = 0
        return self

    def expired(self) -> bool:
        '''True when the loop should stop now.  Call once at the top of each iteration.

        Counts completed iterations itself rather than taking an index, so a layout that
        spends one budget across several loops (NCPLayout's two stages) reports a total
        rather than whichever stage happened to stop -- an index of 0 from a second stage
        reads as "nothing ran", which is the opposite of the truth.

        Checks the cancel signal before the clock so an explicit cancel is reported as a
        cancel rather than as a timeout that happened to coincide with it.
        '''
        if self.should_stop is not None and self.should_stop():
            return self._stop_('cancelled')
        if self.time_budget is not None and self._t0_ is not None \
                and (time.monotonic() - self._t0_) >= self.time_budget:
            return self._stop_('time budget reached')
        self._done_ += 1
        return False

    def _stop_(self, why: str) -> bool:
        self.stopped_at = self._done_
        self.note = (f'{why} after {self._done_} of {self._total_} iterations'
                     if self._total_ is not None else f'{why} after {self._done_} iterations')
        return True

    #
    # Convenience for the callers, so each layout does not re-derive it.
    #
    @staticmethod
    def note_of(budget: Optional['Budget']) -> Optional[str]:
        return None if budget is None else budget.note
