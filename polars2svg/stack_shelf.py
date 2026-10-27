"""The stack control's shelf of named frames, and the labels stack frames carry
(PLANNING.md §7 F18; the design is ``20261026_stack_control_interactions.md``, cited by
Q-number below).

The stack is linear undo: a component's push cuts every frame above the displayed one.
The shelf is the safety net beside it -- named frames that push, pop, collapse, rebase
and ``replaceStack`` never touch.  One shelf per stack (Q17), held on the controller
and *not* inside ``stacks[name]``, because ``replaceStack`` rebuilds that dict.

Labels: a stack frame's label is set when the frame is created from the shelf (or by a
set operation) or when it is shelved, and is kept after that until the frame is
shelved again (Q1).  It is keyed by the frame object, so it follows the frame through
collapse, rebase, deletion and linkpi's per-level rebuild with no bookkeeping in any of
them.  ``id()`` alone is not a safe key -- a dropped frame's id can be reused -- so every
read is identity-guarded, the way the stack control's tile cache is.

Frames pushed from here -- a shelf entry, a set-operation result -- go on top of the
whole stack and never cut, unlike a component's push: the trashcan does the cleanup
now, so nothing above the displayed frame has to be thrown away to keep the stack tidy.

The widget's operations return None when done and a short reason when refused; the
stack control shows that reason on its status row (Q6).
"""
import re
import unicodedata
import asyncio
import inspect
import logging
from collections.abc import Callable
from typing import Any

import polars as pl


# Default shelf capacity (Q17): per stack, 0 means no shelf.
_MAX_SHELF_ = 4

# Columns a component adds to the frames it pushes (chordp's and histop's row index).
# They are not the user's, so a column check ignores them -- selectedDataFrame() strips
# the same ones.
_INTERNAL_COLUMNS_ = ('__p2s_index__',)


# A name typed into the widget's naming dialog is capped at this; one from Python is not
# (it is also the key Python looks the frame up by).
_MAX_SHELF_NAME_ = 64

# Set operations by modifier-click (A the displayed frame, B the clicked one): the op
# name the widget sends, and the symbol that starts the result's label (Q3, Q4).
SET_OPERATORS = {'union': '\u222a', 'subtract': '\u2212', 'intersect': '\u2229'}


def cleanShelfName(raw: object) -> str | None:
    """A name from the browser, made safe to store and draw: control and format
    characters removed, whitespace runs collapsed, capped at ``_MAX_SHELF_NAME_``.
    None when nothing is left."""
    if not isinstance(raw, str):
        return None
    _s_ = ''.join(' ' if _ch_.isspace() else _ch_ for _ch_ in raw
                  if _ch_.isspace() or unicodedata.category(_ch_)[0] != 'C')
    _s_ = re.sub(r'\s+', ' ', _s_).strip()[:_MAX_SHELF_NAME_].strip()
    return _s_ or None


def setOperation(op: str, a: pl.DataFrame, b: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """A op B, matched on ``columns`` (the base's), nulls equal.

    Every result is built from A, so A's own duplicate rows survive (Q5): union is A
    followed by B's rows not in A, intersect is A semi-joined to B, subtract is A
    anti-joined to B (today's ``q``).  So a result is always a superset or a subset of
    A, and comparing lengths says whether anything changed.  Internal columns are
    dropped from the result.
    """
    _a_ = a.drop([_c_ for _c_ in _INTERNAL_COLUMNS_ if _c_ in a.columns])
    _b_ = b.select(columns)
    if op == 'intersect':
        return _a_.join(_b_, on=columns, how='semi', nulls_equal=True)
    if op == 'subtract':
        return _a_.join(_b_, on=columns, how='anti', nulls_equal=True)
    if op == 'union':
        _extra_ = b.drop([_c_ for _c_ in _INTERNAL_COLUMNS_ if _c_ in b.columns]) \
                   .join(_a_.select(columns), on=columns, how='anti', nulls_equal=True)
        return pl.concat([_a_, _extra_], how='diagonal_relaxed') if len(_extra_) else _a_
    raise ValueError(f'setOperation(): unknown op {op!r}; expected one of {sorted(SET_OPERATORS)}')


def baseColumns(base: pl.DataFrame) -> list[str]:
    """The base frame's columns, without the internal ones."""
    return [_c_ for _c_ in base.columns if _c_ not in _INTERNAL_COLUMNS_]


def missingColumns(base: pl.DataFrame, df: pl.DataFrame) -> list[str]:
    """The base's columns that ``df`` lacks: a frame missing any of them cannot be drawn
    by every view on the stack (Q15)."""
    _have_ = set(df.columns)
    return [_c_ for _c_ in baseColumns(base) if _c_ not in _have_]


class StackShelfMixin:
    """Shelf and frame-label state for :class:`InteractionController`.

    The host supplies ``stacks``, ``view_stack``, ``view_refs`` and ``__stackName__()``.
    """

    stacks:          Any
    view_stack:      Any
    view_refs:       Any
    max_stack_depth: Any

    def __stackName__(self, stack: str | None, who: str = 'selectedDataFrame') -> str:
        raise NotImplementedError     # pragma: no cover -- the host's

    async def __notifySelection__(self, stack: str) -> None:
        raise NotImplementedError     # pragma: no cover -- the host's

    def _initShelf_(self) -> None:
        self.max_shelf: int = _MAX_SHELF_
        # {stack_name: [(name, df), ...]} in the order shelved -- the order the widget
        # draws, so a new entry takes the first free row and moves no existing one.
        self.shelves: dict[str, list[tuple[str, pl.DataFrame]]] = {}
        # {stack_name: {id(df): (df, label)}}
        self.frame_labels: dict[str, dict[int, tuple[pl.DataFrame, str]]] = {}
        self.shelve_callbacks: list[tuple[str | None, Callable[..., Any]]] = []

    # ── labels ───────────────────────────────────────────────────────────────

    def frameLabel(self, stack: str, df: pl.DataFrame) -> str | None:
        """The label of stack frame ``df``, or None when it has none."""
        _entry_ = self.frame_labels.get(stack, {}).get(id(df))
        return _entry_[1] if _entry_ is not None and _entry_[0] is df else None

    def _setFrameLabel_(self, stack: str, df: pl.DataFrame, label: str) -> None:
        self.frame_labels.setdefault(stack, {})[id(df)] = (df, label)
        self._pruneFrameLabels_(stack)

    def _pruneFrameLabels_(self, stack: str) -> None:
        # A label holds its frame alive -- which is also why its id cannot be reused while
        # it is held -- so drop the ones whose frame has left the stack.
        _labels_ = self.frame_labels.get(stack)
        if not _labels_:
            return
        _s_   = self.stacks.get(stack)
        _ids_ = {id(_d_) for _d_ in _s_['dfs']} if _s_ is not None else set()
        for _id_ in [_k_ for _k_ in _labels_ if _k_ not in _ids_]:
            del _labels_[_id_]

    # ── reading the shelf ────────────────────────────────────────────────────

    def shelfEntries(self, stack: str) -> list[tuple[str, pl.DataFrame]]:
        """``[(name, df), ...]`` in the order shelved.  The list itself: do not mutate it."""
        return self.shelves.get(stack, [])

    def shelf(self, stack: str | None = None) -> dict[str, pl.DataFrame]:
        """The shelf of ``stack`` (the only stack when None), as ``{name: df}`` in the
        order shelved."""
        return dict(self.shelfEntries(self.__stackName__(stack, 'shelf')))

    def stackIndexOf(self, stack: str, df: pl.DataFrame) -> int | None:
        """Where ``df`` -- the same object -- is on the stack, or None."""
        _s_ = self.stacks.get(stack)
        if _s_ is None:
            return None
        return next((_i_ for _i_, _d_ in enumerate(_s_['dfs']) if _d_ is df), None)

    # ── shelving ─────────────────────────────────────────────────────────────

    def shelveRefusal(self, stack: str, name: str, df: pl.DataFrame) -> str | None:
        """Why ``df`` cannot be shelved as ``name`` (a short message for the widget's
        status row), or None when it can."""
        if self.max_shelf <= 0:
            return 'shelf is off'
        _entries_ = self.shelfEntries(stack)
        if len(_entries_) >= self.max_shelf:
            return f'shelf is full ({self.max_shelf})'
        for _nm_, _d_ in _entries_:
            if _d_ is df:
                return f'already shelved as "{_nm_}"'
            if _nm_ == name:
                return f'"{name}" is already on the shelf'
        _s_ = self.stacks.get(stack)
        if _s_ is not None and _s_['dfs']:
            _missing_ = missingColumns(_s_['dfs'][0], df)
            if _missing_:
                return 'missing column' + ('s' if len(_missing_) > 1 else '') + ': ' + ', '.join(_missing_)
        return None

    def entryRefusal(self, stack: str, df: pl.DataFrame) -> str | None:
        """Why a shelf entry cannot be used against the stack as it is now (Q18): a
        ``replaceStack`` may since have brought a base with other columns."""
        _s_ = self.stacks.get(stack)
        if _s_ is None or not _s_['dfs']:
            return 'no stack'
        _missing_ = missingColumns(_s_['dfs'][0], df)
        return ('columns no longer match: ' + ', '.join(_missing_)) if _missing_ else None

    def _shelve_(self, stack: str, name: str, df: pl.DataFrame) -> None:
        # Unchecked: callers have already asked shelveRefusal().  Shelving a frame that is
        # on the stack labels its row with the name (diagram 5).
        self.shelves.setdefault(stack, []).append((name, df))
        if self.stackIndexOf(stack, df) is not None:
            self._setFrameLabel_(stack, df, name)

    def pinFrame(self, name: str, df: pl.DataFrame | None = None, stack: str | None = None) -> pl.DataFrame:
        """Put ``df`` on the shelf as ``name``; ``df=None`` shelves the displayed frame.

        Raises ``ValueError`` when the shelf is off or full, the name is taken, the frame
        is already shelved, or it lacks the base's columns.  Returns the shelved frame.
        """
        _sn_ = self.__stackName__(stack, 'pinFrame')
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f'pinFrame(): name must be a non-empty string, got {name!r}')
        if df is None:
            _s_ = self.stacks[_sn_]
            df  = _s_['dfs'][_s_['index']]
        elif not isinstance(df, pl.DataFrame):
            raise TypeError(f'pinFrame(): df must be a polars DataFrame, got {type(df).__name__}')
        _why_ = self.shelveRefusal(_sn_, name, df)
        if _why_ is not None:
            raise ValueError(f'pinFrame(): cannot shelve {name!r} -- {_why_}')
        self._shelve_(_sn_, name, df)
        self._refreshShelfViews_(_sn_)
        self._runShelveCallbacks_(_sn_, name, df)
        return df

    def unpinFrame(self, name: str, stack: str | None = None) -> pl.DataFrame:
        """Take ``name`` off the shelf and return its frame.  A stack frame pushed from
        it keeps the name as its label (diagram 2)."""
        _sn_ = self.__stackName__(stack, 'unpinFrame')
        _df_ = self._unshelve_(_sn_, name)
        if _df_ is None:
            raise ValueError(f'unpinFrame(): no shelf entry named {name!r}; have {list(self.shelf(_sn_))}')
        self._refreshShelfViews_(_sn_)
        return _df_

    def _unshelve_(self, stack: str, name: str) -> pl.DataFrame | None:
        _entries_ = self.shelves.get(stack, [])
        for _i_, (_nm_, _d_) in enumerate(_entries_):
            if _nm_ == name:
                del _entries_[_i_]
                return _d_
        return None

    # ── notification ─────────────────────────────────────────────────────────

    def onShelve(self, callback: Any, stack: str | None = None) -> Any:
        """``callback(name, df)`` after every frame shelved, from the widget or from
        ``pinFrame()``.  A plain function or a coroutine function; one that raises is
        logged.  ``stack=`` limits it to one stack.  Returns the callback, so it also
        works as a decorator."""
        if not callable(callback):
            raise TypeError(f'onShelve(): expected a callable, got {type(callback).__name__}')
        if stack is not None: self.__stackName__(stack, 'onShelve')
        self.shelve_callbacks.append((stack, callback))
        return callback

    def _shelveCallbacks_(self, stack: str) -> list[Callable[..., Any]]:
        return [_cb_ for _sn_, _cb_ in self.shelve_callbacks if _sn_ is None or _sn_ == stack]

    async def _notifyShelve_(self, stack: str, name: str, df: pl.DataFrame) -> None:
        for _cb_ in self._shelveCallbacks_(stack):
            try:
                _r_ = _cb_(name, df)
                if inspect.isawaitable(_r_):
                    await _r_
            except (MemoryError, KeyboardInterrupt):
                raise
            except Exception:
                logging.getLogger('polars2svg_logger').exception(
                    'onShelve(): callback %r raised; the views carry on', _cb_)

    def _runShelveCallbacks_(self, stack: str, name: str, df: pl.DataFrame) -> None:
        # pinFrame() is synchronous -- it is called from a notebook cell -- so a coroutine
        # callback is scheduled on the running loop when there is one (Jupyter's), and
        # run to completion when there is not.
        if not self._shelveCallbacks_(stack):
            return
        _co_ = self._notifyShelve_(stack, name, df)
        try:
            _loop_ = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(_co_)
        else:
            _loop_.create_task(_co_)

    def _refreshShelfViews_(self, stack: str) -> None:
        # Only the stack control draws the shelf; every other view ignores it.
        for _vid_, _vsn_ in self.view_stack.items():
            _fn_ = getattr(self.view_refs.get(_vid_), 'refreshShelf', None)
            if _vsn_ == stack and _fn_ is not None:
                _fn_()

    # ── the widget's operations ──────────────────────────────────────────────
    #
    # Each returns None when done and a short reason when refused (Q6).  A reason is
    # never empty, so the `... or None` on a delegated result changes nothing; it is the
    # None path TestAnnotationSanity looks for.

    async def _broadcastStack_(self, stack: str, notify: bool) -> None:
        # Every view on the stack, the way replaceStack() reaches them: these operations
        # start in the stack control, which has no link targets of its own.
        _s_ = self.stacks[stack]
        self._pruneFrameLabels_(stack)
        for _vid_, _vsn_ in list(self.view_stack.items()):
            _view_ = self.view_refs.get(_vid_)
            if _vsn_ == stack and _view_ is not None:
                await _view_.display(_s_['dfs'][_s_['index']], _s_['dfs'], _s_['index'])
        if notify:
            await self.__notifySelection__(stack)

    async def jumpTo(self, stack: str, index: int) -> str | None:
        """Display stack frame ``index`` (a plain click)."""
        _s_ = self.stacks[stack]
        if not 0 <= index < len(_s_['dfs']):
            return 'no such frame'
        _moved_ = index != _s_['index']
        _s_['index'] = index
        await self._broadcastStack_(stack, notify=_moved_)
        return None

    async def pushOnTop(self, stack: str, df: pl.DataFrame, label: str | None = None) -> str | None:
        """Push ``df`` on top of the whole stack, cutting nothing, and display it."""
        _s_ = self.stacks[stack]
        if len(_s_['dfs']) >= self.max_stack_depth:
            return f'stack is at its {self.max_stack_depth}-frame limit'
        _s_['dfs'] = _s_['dfs'] + [df]
        _s_['index'] = len(_s_['dfs']) - 1
        if label is not None:
            self._setFrameLabel_(stack, df, label)
        await self._broadcastStack_(stack, notify=True)
        return None

    async def deleteFrame(self, stack: str, index: int) -> str | None:
        """Delete stack frame ``index``.  The displayed frame cannot be; the frames above
        a deleted one stay, and deleting the base makes the next frame the base."""
        _s_ = self.stacks[stack]
        if not 0 <= index < len(_s_['dfs']):
            return 'no such frame'
        if index == _s_['index']:
            return 'the displayed frame cannot be deleted'
        _s_['dfs'] = _s_['dfs'][:index] + _s_['dfs'][index + 1:]
        if index < _s_['index']:
            _s_['index'] -= 1
        # The displayed frame is unchanged, so the selection is too: no notification.
        await self._broadcastStack_(stack, notify=False)
        return None

    async def combine(self, stack: str, op: str, b: pl.DataFrame, b_label: str | None) -> str | None:
        """Displayed frame ``op`` B, pushed on top and labelled with the operator and B's
        label.  A result the length of A, or empty, changes nothing (diagram 8)."""
        if op not in SET_OPERATORS:
            return f'unknown operation {op!r}'
        _s_ = self.stacks[stack]
        _a_ = _s_['dfs'][_s_['index']]
        if b is _a_:
            return 'no change'
        _missing_ = missingColumns(_s_['dfs'][0], b)
        if _missing_:
            return 'columns do not match: ' + ', '.join(_missing_)
        try:
            _r_ = setOperation(op, _a_, b, baseColumns(_s_['dfs'][0]))
        except (pl.exceptions.PolarsError, ValueError) as _e_:
            logging.getLogger('polars2svg_logger').warning('stack control %s: %s', op, _e_)
            return f'cannot {op} these frames'
        if len(_r_) == 0:
            return 'empty result'
        if len(_r_) == len(_a_):
            return 'no change'
        _sym_ = SET_OPERATORS[op]
        return await self.pushOnTop(stack, _r_, f'{_sym_} {b_label}' if b_label else _sym_) or None

    async def useStackFrame(self, stack: str, index: int, op: str | None = None) -> str | None:
        """A click on stack frame ``index``: plain jumps to it, a modifier combines it."""
        _s_ = self.stacks[stack]
        if not 0 <= index < len(_s_['dfs']):
            return 'no such frame'
        if op is None:
            return await self.jumpTo(stack, index)
        _b_ = _s_['dfs'][index]
        return await self.combine(stack, op, _b_, self.frameLabel(stack, _b_)) or None

    async def useShelfEntry(self, stack: str, name: str, op: str | None = None) -> str | None:
        """A click on shelf entry ``name``.  Plain pushes it on top -- or jumps to it when
        that same frame is already on the stack (Q9) -- labelled with its name; a
        modifier combines it with the displayed frame."""
        _df_ = dict(self.shelfEntries(stack)).get(name)
        if _df_ is None:
            return 'no such shelf entry'
        _why_ = self.entryRefusal(stack, _df_)
        if _why_ is not None:
            return _why_
        if op is not None:
            return await self.combine(stack, op, _df_, name)
        _at_ = self.stackIndexOf(stack, _df_)
        if _at_ is not None:
            return await self.jumpTo(stack, _at_)
        return await self.pushOnTop(stack, _df_, name) or None

    async def deleteShelfEntry(self, stack: str, name: str) -> str | None:
        """Take ``name`` off the shelf; a stack frame pushed from it keeps the name."""
        if self._unshelve_(stack, name) is None:
            return 'no such shelf entry'
        self._refreshShelfViews_(stack)
        return None

    async def shelveDisplayed(self, stack: str, raw_name: object) -> str | None:
        """Shelve the displayed frame under a name typed into the widget."""
        _name_ = cleanShelfName(raw_name)
        if _name_ is None:
            return 'a shelf name cannot be empty'
        _s_  = self.stacks[stack]
        _df_ = _s_['dfs'][_s_['index']]
        _why_ = self.shelveRefusal(stack, _name_, _df_)
        if _why_ is not None:
            return _why_
        self._shelve_(stack, _name_, _df_)
        self._refreshShelfViews_(stack)
        await self._notifyShelve_(stack, _name_, _df_)
        return None
