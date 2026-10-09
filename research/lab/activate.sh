#!/usr/bin/env bash
# Source this file before using the lab; keep caches in writable workspace paths.
export UV_CACHE_DIR=/workspace/.cache/uv
export XDG_CACHE_HOME=/workspace/.cache/herculaneum
export MPLCONFIGDIR=/workspace/herculaneum-lab/config/matplotlib
export JUPYTER_CONFIG_DIR=/workspace/herculaneum-lab/config/jupyter
export JUPYTER_DATA_DIR=/workspace/herculaneum-lab/config/jupyter-data
export JUPYTER_RUNTIME_DIR=/workspace/herculaneum-lab/config/jupyter-runtime
export IPYTHONDIR=/workspace/herculaneum-lab/config/ipython
mkdir -p "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR" "$IPYTHONDIR"
source /workspace/.venvs/herculaneum/bin/activate
