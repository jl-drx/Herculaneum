# Running and resuming the native learning comparison

This study has no paid resources, credentials, downloads or inference from
Hecate/other released models. Execution uses Python3.12.14 and the versions in
requirements.txt, four CPU threads, float32 CNN training. The original qualified
runtime is `/workspace/.venvs/herculaneum/bin/python`.

The source packets already live in the private repository. Follow the existing
`research/planning/pherc0139-learning-readiness-20261011/PORTABILITY.md` to fetch
only the two LFS archives in INPUT_ARCHIVES.json and restore them. The resulting
`native-reference/` directory is the `--source` argument. Its `ct/chunks/` must
contain the536 uncompressed source chunks. No full-scroll scan is required.

`GO.json` binds the data and code. It is created once, after qualification;
**do not re-freeze or edit the plan on resume**. Copy this study's frozen files
into a new run directory only for initial execution; for restart use the
checkpoint branch below. The worker verifies all source and code hashes first.

In the original workspace, the run directory is
`/workspace/herculaneum-lab/reports/pherc0139-native-learning-20261011`, and the
source is `/workspace/herculaneum-lab/reports/pherc0139-native-reference-20261011`.
Start/resume with the qualified interpreter:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python native_learning.py run \
  --run /path/to/native-learning-run \
  --source /path/to/native-reference
```

Use the `native_learning.py` inside the restored run, not a later edited copy.
It loads the latest committed receipt, verifies the checkpoint and preserves
model/optimizer/RNG/schedule state. A `.pt` without its matching `.json` receipt
is incomplete and ignored. No completed arm is rerun. The original environment
has21 synthetic qualification checks, including exact restart; numerical
equivalence on different CPU hardware is not promised merely by matching package
versions. Record and qualify any environment change before resuming research.

The training process should be launched detached with stdout/stderr to a local
log, then checked using process start ticks and committed receipts. A separate
backup process is required; running training alone does not publish checkpoints:

```bash
python backup.py --repo /path/to/Herculaneum \
  --run /path/to/native-learning-run --state /path/to/backup-state --watch
python status.py --run /path/to/native-learning-run \
  --backup-state /path/to/backup-state
```

The watcher waits in short increments, snapshots approximately every10 minutes
and attempts a final snapshot after the worker exits. It uses ordinary private
Git/LFS authentication, verifies repository privacy, uploads LFS bodies first,
then advances the dedicated ref without force. No GitHub Actions runner is used.
It never stages changes in the main checkout. Failures go to backup-state/ERROR;
the last successful receipt remains visible and the next interval retries.
Loss of the worker host also stops this watcher. Check backup freshness, not just
worker liveness. A checkpoint preserves committed work; it does not keep a host
alive or cause the platform to resume a process automatically.

## Restore after host loss

Fetch `codex/plinius-native0139-learning-checkpoints` into a separate temporary
checkout with LFS smudging disabled. Read INDEX.json, then fetch only its listed
generation archives with `git lfs fetch --include='generation-*.tar.gz' origin
codex/plinius-native0139-learning-checkpoints` and `git lfs checkout`. Materialize
those archives beside INDEX.json. They are private checkpoint files, not an
Actions artifact. Use:

```bash
python restore_backup.py --backup /path/to/checkpoint-checkout \
  --destination /path/to/restored-native-learning-run
```

Every archive/member checksum is verified, path escapes are refused, and existing
different bytes are never overwritten. Restore the two input packets separately
and use the resume command above. The latest and best checkpoints are saved for
each arm, together with baseline constants, normalization, all schedule/history
state and endpoint dependencies. Intermediate checkpoints not selected at a
snapshot may remain local only. Raw completed held-out maps are published too.

The9 endpoints are jointly sealed before any held-out prediction. STATUS reports
training only until then. A model that fits training labels is not yet evidence
of physical ink or transferable learning. See PROTOCOL.md for interpretation.
