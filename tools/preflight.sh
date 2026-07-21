#!/usr/bin/env bash
#
# preflight.sh - run CI's fast checks locally, before pushing
#
#   ./tools/preflight.sh
#
# These are the same four commands .github/workflows/ci.yml runs, in its two
# non-container jobs (`mypy (public surface)` and `bandit + pip-audit + ruff`).
# They are plain CLI invocations with no GitHub-specific context, so they
# reproduce natively on macOS in ~12s.  Keep this file in sync with ci.yml --
# it is a convenience copy, not the source of truth.
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
# exactly what a bare `pip install polars2svg` resolves, and a local .venv has
# the dev group and every extra installed on top of that.
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

_step_ 'mypy (public surface)'   uvx mypy polars2svg
_step_ 'bandit (security scan)'  uvx bandit -r polars2svg/
_step_ 'ruff (E9, F)'            uvx ruff check polars2svg/

# Two commands, so it needs a subshell rather than a bare _step_ invocation.
_step_ 'pip-audit (dependencies)' bash -c \
    "uv export --no-hashes --no-dev --no-emit-project -o '$_REQS_' >/dev/null && uvx pip-audit -r '$_REQS_'"

printf '\n'
if [ ${#_FAILED_[@]} -eq 0 ]; then
    printf 'preflight green -- ci.yml fast jobs should pass\n'
    printf 'reminder: the test suite is separate (.venv/bin/python -m pytest tests/)\n\n'
    exit 0
fi

printf 'preflight RED -- %d check(s) failed: %s\n\n' "${#_FAILED_[@]}" "${_FAILED_[*]}"
exit 1
