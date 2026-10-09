#!/usr/bin/env bash
set -euo pipefail
TASK_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_VENV_PATH="${HERCULANEUM_VENV:-/workspace/.venvs/herculaneum}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
mkdir -p "$UV_CACHE_DIR"
if [[ ! -x "$TASK_VENV_PATH/bin/python" ]]; then
  uv venv --python 3.12 "$TASK_VENV_PATH"
fi
"$TASK_VENV_PATH/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12)'
uv pip sync --python "$TASK_VENV_PATH/bin/python" --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --require-hashes "$TASK_REPO_ROOT/requirements-cpu.lock"
uv pip install --python "$TASK_VENV_PATH/bin/python" --no-deps --no-build-isolation --editable "$TASK_REPO_ROOT"
uv pip check --python "$TASK_VENV_PATH/bin/python"
cd "$TASK_REPO_ROOT"
"$TASK_VENV_PATH/bin/python" scripts/verify_sources.py
"$TASK_VENV_PATH/bin/python" -m unittest discover -s tests -v
