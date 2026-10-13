#!/usr/bin/env bash
# Install a separate pinned CPU runtime without touching the running cloud job.
set -euo pipefail
TASK_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK_PROFILE="${1:-pherc343}"
TASK_VENV="${2:-$TASK_REPO_ROOT/.venvs/$TASK_PROFILE}"
case "$TASK_PROFILE" in
  pherc343) TASK_PYTHON=3.14.7; TASK_LOCK="$TASK_REPO_ROOT/research/workspace/requirements-pherc343.lock" ;;
  history) TASK_PYTHON=3.12.14; TASK_LOCK="$TASK_REPO_ROOT/requirements-cpu.lock" ;;
  *) echo 'Usage: setup_portable_cpu.sh [pherc343|history] [venv-path]' >&2; exit 2 ;;
esac
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 2; }
TASK_ENV_PARENT="$(dirname -- "$TASK_VENV")"
mkdir -p "$TASK_ENV_PARENT"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$TASK_ENV_PARENT/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$TASK_ENV_PARENT/.python-managed}"
if [[ ! -x "$TASK_VENV/bin/python" ]]; then
  uv venv --python "$TASK_PYTHON" "$TASK_VENV"
fi
"$TASK_VENV/bin/python" -c "import platform; assert platform.python_version() == '$TASK_PYTHON', platform.python_version()"
uv pip sync --python "$TASK_VENV/bin/python" --require-hashes --link-mode hardlink \
  --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match "$TASK_LOCK"
if [[ "$TASK_PROFILE" == history ]]; then
  uv pip install --python "$TASK_VENV/bin/python" --no-deps --no-build-isolation --editable "$TASK_REPO_ROOT"
fi
uv pip check --python "$TASK_VENV/bin/python"
"$TASK_VENV/bin/python" -B -c 'import sys,torch,numpy,zarr; print({"python":sys.version.split()[0],"torch":torch.__version__,"numpy":numpy.__version__,"zarr":zarr.__version__,"cuda_available":torch.cuda.is_available()})'
