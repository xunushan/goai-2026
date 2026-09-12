#!/usr/bin/env bash
# Run every codex_agent test suite.
#
#   bash tests/run_all.sh
#
# No Codex, no simulator, no network: the whole suite spends zero quota. The
# Codex allowance is reserved for the official evaluation run, so nothing here
# may call the real CLI. `fake_codex` stands in for it in test_bridge.py, and
# mock_bridge.py replaces the bridge itself in test_adapter.py.
#
# Override the interpreter with PYTHON=... (the default is the local anaconda
# python, which has numpy, scipy, yaml and PIL).
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

PYTHON="${PYTHON:-/opt/anaconda3/bin/python}"
if ! command -v "$PYTHON" >/dev/null 2>&1 && [ ! -x "$PYTHON" ]; then
    echo "python not found at $PYTHON; set PYTHON=..." >&2
    exit 1
fi

failures=0

run() {
    local name="$1"
    shift
    echo
    echo "=== $name ==="
    if "$@"; then
        return 0
    fi
    echo "--- $name FAILED" >&2
    failures=$((failures + 1))
}

# A stub that cannot run makes every bridge assertion meaningless, so check it
# first rather than reporting a wall of confusing failures.
if [ ! -x tests/fake_codex ]; then
    echo "tests/fake_codex is not executable; run: chmod +x tests/fake_codex" >&2
    exit 1
fi

run "motion (poses, interpolation, guardrail)" "$PYTHON" tests/test_motion.py
run "protocol (parsing the model's reply)"     "$PYTHON" tests/test_protocol.py
run "prompt (the information boundary)"        "$PYTHON" tests/test_prompt.py
run "bridge (real server, fake Codex CLI)"     "$PYTHON" tests/test_bridge.py
run "adapter (real Model, mock bridge)"        "$PYTHON" tests/test_adapter.py

echo
if [ "$failures" -ne 0 ]; then
    echo "FAILED: $failures suite(s)"
    exit 1
fi
echo "all suites passed"
