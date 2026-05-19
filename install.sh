#!/usr/bin/env bash
# install.sh - Semantic Compressor setup script for Linux / macOS.
#
# Creates a .venv, installs requirements.txt, and runs a quick sanity test
# (tests/test_profiler.py) to verify the environment is functional.
#
# Usage:
#   ./install.sh
#
# Exit codes:
#   0 - success
#   1 - Python 3.11+ not found
#   2 - venv creation failed
#   3 - pip install failed
#   4 - sanity test failed

set -euo pipefail

# Resolve the script's own directory so the script can be invoked from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

printf '=== Semantic Compressor - Unix installer ===\n'

# -----------------------------------------------------------------------------
# 1. Locate a Python 3.11+ interpreter (prefer 3.13)
# -----------------------------------------------------------------------------
printf '\n[1/5] Looking for Python 3.11+...\n'

python_ok() {
    # Returns 0 if "$1" is a Python interpreter with version >= 3.11.
    local exe="$1"
    if ! command -v "$exe" >/dev/null 2>&1; then
        return 1
    fi
    "$exe" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1
}

PYTHON_EXE=""
for candidate in python3.13 python3.12 python3.11 python3 python; do
    if python_ok "$candidate"; then
        PYTHON_EXE="$candidate"
        version="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
        printf '  Found Python %s at %s\n' "$version" "$candidate"
        break
    fi
done

if [[ -z "$PYTHON_EXE" ]]; then
    printf 'ERROR: No Python 3.11+ interpreter found on PATH.\n' >&2
    printf 'Install Python 3.11, 3.12, or 3.13 and re-run.\n' >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# 2. Create venv if absent
# -----------------------------------------------------------------------------
printf '\n[2/5] Preparing .venv...\n'

VENV_DIR="$SCRIPT_DIR/.venv"
if [[ -d "$VENV_DIR" ]]; then
    printf '  .venv already exists, reusing it.\n'
else
    printf '  Creating .venv with %s...\n' "$PYTHON_EXE"
    if ! "$PYTHON_EXE" -m venv "$VENV_DIR"; then
        printf 'ERROR: Failed to create virtual environment at %s.\n' "$VENV_DIR" >&2
        exit 2
    fi
    printf '  Created %s\n' "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    printf 'ERROR: venv python not found at %s.\n' "$VENV_PYTHON" >&2
    exit 2
fi

# -----------------------------------------------------------------------------
# 3. Upgrade pip
# -----------------------------------------------------------------------------
printf '\n[3/5] Upgrading pip...\n'
if ! "$VENV_PYTHON" -m pip install --upgrade pip --quiet; then
    printf 'ERROR: pip upgrade failed.\n' >&2
    exit 3
fi
printf '  pip is up to date.\n'

# -----------------------------------------------------------------------------
# 4. Install requirements
# -----------------------------------------------------------------------------
printf '\n[4/5] Installing requirements.txt...\n'
REQ_PATH="$SCRIPT_DIR/requirements.txt"
if [[ ! -f "$REQ_PATH" ]]; then
    printf 'ERROR: requirements.txt not found at %s.\n' "$REQ_PATH" >&2
    exit 3
fi
if ! "$VENV_PYTHON" -m pip install -r "$REQ_PATH"; then
    printf 'ERROR: pip install -r requirements.txt failed.\n' >&2
    exit 3
fi
printf '  Requirements installed.\n'

# -----------------------------------------------------------------------------
# 5. Quick sanity test
# -----------------------------------------------------------------------------
printf '\n[5/5] Running sanity test (tests/test_profiler.py)...\n'
if ! "$VENV_PYTHON" -m pytest tests/test_profiler.py -q; then
    printf 'ERROR: Sanity test failed. The environment is installed but tests do not pass.\n' >&2
    exit 4
fi
printf '  Sanity test passed.\n'

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
printf '\n=== Install complete ===\n'
printf '\nNext steps:\n'
printf '  1. Activate the venv:        source .venv/bin/activate\n'
printf '  2. Run the end-to-end POC:   python examples/run_poc.py\n'
printf '  3. Run the full test suite:  python -m pytest\n\n'

exit 0
