# Try Herculaneum

[Overview](../README.md) · [Project tour](project-tour.md)

Choose the small offline demo to inspect the infrastructure, or the separate research replay to reproduce a saved comparison. The full CPU environment is available for development.

## Get the infrastructure

You need Git, Python 3.12 and access to this private repository. These commands target Linux with a POSIX filesystem; the cache and local artifact store rely on POSIX filesystem behaviour.

```bash
git clone --single-branch --branch main https://github.com/jl-drx/Herculaneum.git
cd Herculaneum
PYTHONPATH=src python3.12 -m herculaneum --help
```

The help command and local artifact demo use the Python standard library. They do not require installing the ML stack.

## Offline artifact demo

Run from the repository root. This creates one explicitly synthetic result in a temporary directory, publishes it through the real local store, verifies it, then deliberately alters the temporary stored copy to demonstrate corruption detection. All demo files are removed when the block exits.

```bash
PYTHONPATH=src python3.12 - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory

from herculaneum.artifacts import (
    ArtifactIntegrityError,
    LocalArtifactStore,
    publish_run,
    verify_run,
)

with TemporaryDirectory(prefix="herculaneum-demo-") as temporary:
    root = Path(temporary)
    source = root / "example.json"
    source.write_text('{"synthetic": true, "tiles": 1}\n', encoding="utf-8")
    store = LocalArtifactStore(root / "store")
    publish_run(
        store, "visitor-demo", {"example.json": source},
        metadata={"purpose": "synthetic infrastructure demonstration"},
    )
    manifest = verify_run(store, "visitor-demo")
    print(f"PASS: {manifest['state']}; {len(manifest['artifacts'])} artifact verified")

    stored = root / "store/runs/visitor-demo/artifacts/example.json"
    stored.write_text("deliberately altered demo bytes", encoding="utf-8")
    try:
        verify_run(store, "visitor-demo")
    except ArtifactIntegrityError:
        print("PASS: altered artifact rejected")
    else:
        raise RuntimeError("Corruption was not detected")
PY
```

Expected output:

```text
PASS: complete; 1 artifact verified
PASS: altered artifact rejected
```

The demo has no network calls, credentials, scans, checkpoints or paid compute. It demonstrates storage integrity, not a scientific result. The same publish/verify/corruption behaviour is covered through the command interface in [test_cli.py](../tests/test_cli.py).

## Replay a real experiment

Experiment 39 has a 12.3 MB capsule of retained predictions, labels, masks and comparison evidence. Replaying it needs Python 3.12 and the hash-pinned NumPy dependency, but no model weights, GPU or cloud credentials.

The research history is on a separate branch. From the infrastructure checkout, fetch the pinned snapshot and create a separate directory. Skipping LFS smudging avoids automatically downloading the large workspace bundles; the experiment-39 capsule is an ordinary Git blob at this snapshot.

```bash
git fetch --no-tags origin 9f6fe45ab20eb0961d5e06aba5c3038088941180
GIT_LFS_SKIP_SMUDGE=1 git worktree add --detach ../herculaneum-replay 9f6fe45ab20eb0961d5e06aba5c3038088941180
cd ../herculaneum-replay
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r research/requirements-replay.lock
REPLAY_PARENT="$(mktemp -d)"
.venv/bin/python research/tools/replay_phase39.py \
  --capsule research/replay/plinius-phase39-replay.tar.gz \
  --destination "$REPLAY_PARENT/result"
```

Use an unused worktree path if `../herculaneum-replay` already exists. The replay destination must not already exist; the command creates `result` inside the new temporary parent.

Success prints a JSON receipt with `"status": "PASS"` and `"all_reported_metrics_and_decisions_identical": true`. The output directory contains `REPLAY_RECEIPT.json` and the reconstructed comparison. A digest mismatch or metric difference stops the replay.

Read the [original result](https://github.com/jl-drx/Herculaneum/blob/9f6fe45ab20eb0961d5e06aba5c3038088941180/research/lab/reports/thirtyninth-experiment/RESULTS.md) alongside that receipt. The replay reproduces saved-output arithmetic and decisions; it does not retrain a model, rerun inference or establish accuracy on unseen data. Git fetch may transfer research history beyond the small capsule; the capsule size is not a promise about the entire checkout size.

## Full CPU development environment

For adapters and the complete infrastructure test suite, install `uv` and use the existing hash-pinned CPU setup from the infrastructure checkout:

```bash
HERCULANEUM_VENV="$PWD/.venv" UV_CACHE_DIR="$PWD/.cache/uv" bash scripts/setup_cpu.sh
source .venv/bin/activate
herculaneum --help
```

These explicit paths make the environment local to the checkout. Omitting them retains the script's existing `/workspace` defaults. Setup downloads Python dependencies, installs the package, verifies retained sources and runs the tests. It does not acquire scans or checkpoints, connect to R2, or start paid compute.

After setup:

```bash
python scripts/verify_sources.py
python -m unittest discover -s tests -v
```

The CPU lock is not a CUDA installer. Live cloud integration and GPU execution have separate prerequisites in the [cloud guide](cloud-connections.md) and [GPU policy](gpu-pilot.md). [Validation](validation.md) records the dates and limits of earlier checks.
