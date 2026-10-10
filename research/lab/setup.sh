#!/usr/bin/env bash
set -euo pipefail
cd /workspace/herculaneum-lab
export UV_CACHE_DIR=/workspace/.cache/uv
mkdir -p "$UV_CACHE_DIR" /workspace/.venvs
if [[ ! -x /workspace/.venvs/herculaneum/bin/python ]]; then
  uv venv --python python3 /workspace/.venvs/herculaneum
fi
source ./activate.sh
python -c 'import sys; assert sys.version_info[:2] == (3, 12), "The reviewed lock targets Python 3.12"'
uv pip sync --python "$VIRTUAL_ENV/bin/python" --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --require-hashes requirements.lock
uv pip check --python "$VIRTUAL_ENV/bin/python"
python fetch_references.py
python verify_lab.py
