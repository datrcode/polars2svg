#!/usr/bin/env bash
#
# preflight.sh - run CI's fast checks locally, before pushing
#
#   ./tools/preflight.sh
#
# These mirror the four checks .github/workflows/ci.yml runs, in its two
# non-container jobs (`mypy (public surface)` and `bandit + pip-audit + ruff`).
# They are plain CLI invocations with no GitHub-specific context, so they
# reproduce natively on macOS in ~12s.  Keep this file in sync with ci.yml --
# it is a convenience copy, not the source of truth.
#
# The mypy step was rewritten 2026-09-13 (PLANNING.md Q1, audit F1).  It used to
# be `uvx mypy polars2svg`, which resolves into an isolated env with no polars,
# numpy or PIL and so checked nothing.  ci.yml's type-check job carries the same
# rewrite, so the two still mirror each other -- but the ci.yml side installs the
# project with `uv sync` and this side uses whatever .venv holds, which is the
# same thing only while .venv is lock-aligned.  See _MYPY_CEILING_ below.
#
# Caveats worth knowing before you trust a green run:
#
#   - mypy is a CEILING, not a clean bill of health -- see _MYPY_CEILING_ below.
#     104 real errors stand behind it.
#   - ruff checks the whole tree as of Q3 (2026-09-14), not just polars2svg/.
#     tests/ had never been linted and held 106 E9/F findings including an F821;
#     those are fixed and the scope now matches ci.yml's.
#   - bandit and mypy are pure functions of the repo contents and genuinely
#     predict CI.  pip-audit is not.  Its result depends on pypi.org being
#     reachable and on the vulnerability database's contents at the moment CI
#     runs, both of which can differ minutes later -- so a green pip-audit here
#     is weak evidence, not a guarantee.
#   - ruff and mypy are now pinned in [dependency-groups].dev and run from
#     .venv, so they no longer drift version-to-version (Q1).  bandit and
#     pip-audit are still bare `uvx` and still resolve unpinned, so CI can run
#     a newer one than you just did.
#
# The third CI job (`Linux clean-room wheel install + tests`) is deliberately
# NOT here: it builds the wheel and runs the suite inside a stock
# python:3.13-slim container on linux/amd64, and the whole point of it is being
# a different platform than this machine.  Let CI own that one.
#
# The test suite is also deliberately not here.  The golden-image tests render
# against this machine's fonts and fail locally while passing on CI, so folding
# them in would leave preflight permanently red -- and a gate that is always red
# is a gate you learn to ignore.  Run tests as their own deliberate step:
#
#   .venv/bin/python -m pytest tests/
#
# Unlike CI, this runs all four checks even after one fails, so a single pass
# shows you everything that needs fixing.  Exits non-zero if any check failed.
#
set -uo pipefail

_HERE_="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT_="$(dirname "$_HERE_")"
cd "$_ROOT_" || exit 1

# pip-audit reads an exported requirements file rather than the venv: CI audits
# the locked resolution, and a local .venv has the dev group and every extra
# installed on top of that.
#
# The three --extra flags and --no-deps mirror ci.yml exactly; keep them that way.
# Without the extras this audits six packages and says nothing about panel, bokeh,
# lxml, requests, urllib3 or reportlab. Without --no-deps, pip-audit resolves by
# INSTALLING into a throwaway venv, and `export` pulls rlPyCairo -> pycairo, which
# has no Linux wheel and needs cairo's headers -- green on a Mac with cairo around,
# a build failure on CI's runner.  --no-deps alone is not enough: it skips resolution
# but still invokes pip.  --disable-pip is what keeps pip out, and it is only accepted
# alongside --no-deps.
_REQS_="$(mktemp -t p2s-preflight-reqs)"
trap 'rm -f "$_REQS_"' EXIT

_FAILED_=()

# bandit logs one WARNING per word of every `# nosec <code> - <reason>` comment in
# the tree: it treats everything after `nosec` as a list of test ids, so the prose
# half of the repo's annotation convention becomes ~200 lines of
# "Test in comment: cryptographic is not a test name or id, ignoring".  Harmless,
# but it buries the actual finding.  Drop those and the startup INFO banner; leave
# every other line alone so a real bandit error still surfaces.
_denoise_() {
    grep -v -E '^\[[a-z_]+\][[:space:]]+WARNING[[:space:]]+Test in comment:|^\[main\][[:space:]]+INFO'
}

# _step_ <label> <command...> -- run a check, print its output only on failure.
_step_() {
    local _label_="$1"; shift
    printf '  %-34s' "$_label_"
    local _out_
    if _out_="$("$@" 2>&1)"; then
        printf 'ok\n'
    else
        printf 'FAIL\n'
        printf '%s\n' "$_out_" | _denoise_ | sed 's/^/      /'
        printf '\n'
        _FAILED_+=("$_label_")
    fi
}

printf '\npreflight (mirrors ci.yml fast jobs)\n\n'

# mypy runs from .venv rather than uvx, and that changes what it means.  `uvx mypy`
# resolves into an isolated env with no polars, numpy or PIL, so every operation on
# a DataFrame/Series/ndarray is Any and type-checks trivially: it printed "Success:
# no issues found in 47 source files" on 2026-09-13 while the same config resolved
# against .venv found 730 errors in 17 files.  See PLANNING.md Q1 and audit F1.
#
# It cannot be a plain pass/fail step yet.  104 is the standing count, and a gate
# that is always red is a gate you learn to ignore -- the same reason the test suite
# is not in this script.  So it is a CEILING: green at or below the number, red
# above it.  That is enough to catch the drift that motivated Q1 (the count moved
# 725 -> 729 over three commits in a single day, with CI green throughout).
#
# The number is a function of the DEPENDENCY VERSIONS as well as the code, and
# that bit an earlier draft of this file.  A .venv built ad-hoc with
# `uv pip install -e .` drifts ahead of uv.lock (it had polars 1.44.2 against the
# lock's 1.41.2), and the two report different counts -- 729 vs 730, the odd one
# in ncp_layout.py.  So the ceiling only means something against a stated
# environment.  Measured 2026-09-13 against uv.lock: polars 1.41.2, numpy 2.4.6,
# mypy 2.3.1, extras layouts+interactive+export, no mlx.  Keep .venv aligned with
# the lock (`uv sync --extra layouts --extra interactive --extra export --group
# dev`) and this agrees with CI exactly; a polars bump legitimately moves it and
# is a deliberate re-baseline, not a regression.
#
# Lower _MYPY_CEILING_ as Q4 burns the count down; never raise it except as such
# a re-baseline, with the new dependency version named.  Q4 step 1 replaces this
# with the per-module ratchet in tests/test_typing_surface.py, at which point
# this goes back to a plain _step_ invocation.
_MYPY_CEILING_=104

_step_mypy_() {
    printf '  %-34s' "mypy (resolved, ceiling $_MYPY_CEILING_)"
    local _out_ _n_ _rc_
    _out_="$(.venv/bin/python -m mypy polars2svg 2>&1)"; _rc_=$?
    # mypy exits 0 clean, 1 with findings, >=2 on a crash or bad config (127 if
    # the interpreter or the module is missing).  Without this guard a missing
    # mypy would produce no ': error: ' lines, count 0, and report a false green
    # -- which is the exact failure mode this whole step exists to correct.
    if [ "$_rc_" -gt 1 ]; then
        printf 'FAIL (mypy did not run, exit %d)\n' "$_rc_"
        printf '%s\n' "$_out_" | tail -5 | sed 's/^/      /'
        printf '      is the dev group installed?  VIRTUAL_ENV="$PWD/.venv" uv pip install -e %s --group dev\n\n' "'.[layouts,interactive,export]'" 
        _FAILED_+=('mypy')
        return
    fi
    _n_="$(printf '%s\n' "$_out_" | grep -c ': error: ')"
    if [ "$_n_" -lt "$_MYPY_CEILING_" ]; then
        printf 'ok (%d -- lower _MYPY_CEILING_ to %d)\n' "$_n_" "$_n_"
    elif [ "$_n_" -eq "$_MYPY_CEILING_" ]; then
        printf 'ok (%d)\n' "$_n_"
    else
        printf 'FAIL (%d > %d)\n' "$_n_" "$_MYPY_CEILING_"
        printf '%s\n' "$_out_" | grep ': error: ' | tail -20 | sed 's/^/      /'
        printf '\n'
        _FAILED_+=('mypy')
    fi
}

_step_mypy_
_step_ 'bandit (security scan)'  uvx bandit -r polars2svg/
_step_ 'ruff'                    .venv/bin/python -m ruff check .

# Two commands, so it needs a subshell rather than a bare _step_ invocation.
#
# Retried for the same reason ci.yml retries it (see the comment on that step):
# pip-audit hits pypi.org's JSON API once per dependency and turns a stalled
# connection into a hard failure, because upstream catches ConnectTimeout but
# not ReadTimeout and mounts no urllib3 Retry.  The backoff here is shorter than
# CI's 10s/20s -- this script is meant to be interactive and fast, and a dev who
# hits three failures in a row can just run it again.
_step_ 'pip-audit (dependencies)' bash -c \
    "uv export --no-hashes --no-dev --no-emit-project \
         --extra interactive --extra layouts --extra export -o '$_REQS_' >/dev/null || exit 1
     for _attempt_ in 1 2 3; do
         uvx pip-audit --no-deps --disable-pip --timeout 30 -r '$_REQS_' && exit 0
         [ \"\$_attempt_\" -lt 3 ] && sleep \$((_attempt_ * 3))
     done
     exit 1"

printf '\n'
if [ ${#_FAILED_[@]} -eq 0 ]; then
    printf 'preflight green -- ci.yml fast jobs should pass\n'
    printf 'reminder: the test suite is separate (.venv/bin/python -m pytest tests/)\n\n'
    exit 0
fi

printf 'preflight RED -- %d check(s) failed: %s\n\n' "${#_FAILED_[@]}" "${_FAILED_[*]}"
exit 1
