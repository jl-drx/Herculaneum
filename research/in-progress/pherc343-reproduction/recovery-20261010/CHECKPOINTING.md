# Live checkpointing without restarting the current map

Hecate's upstream `predict` stores float32 numerator and denominator arrays in a temporary directory. Those arrays are not durable restart records. The old runtime's arrays were absent after restoration. The replacement worker's arrays are present and can be read without changing the worker.

[checkpoint_hecate.py](../../../tools/checkpoint_hecate.py) captures completed rows from those arrays. Tile origins are ordered by row. Once the logged completed batch reaches the next origin row `Y`, no later patch can modify any output row below `Y`. The observer saves those rows using the original division, clipping and uint8 rounding. It reads them twice to reject a changing capture, publishes immutable generations with SHA256 manifests, and atomically updates `LATEST.json` only after files and metadata have been flushed. It refuses ambiguous temporary-array identification, unexpected source/runner identities or inconsistent geometry/progress.

The observer runs separately, checks every30 seconds and writes a new generation only when more rows are complete. It sends no signals, reads no process memory and does not edit the active inference source. `/proc/<pid>/maps` was not readable; the implementation does not depend on it. It verifies the worker command and start identity and requires a unique pair of temporary arrays with the expected dimensions and creation-era modification times. Exactly one matching pair existed at startup.

Checkpoints are under:

```text
/workspace/herculaneum-lab/reports/fortysixth-experiment/work/hecate-checkpoints/
```

`BINDING.json` pins every file of the prepared padded input, model weights, upstream Hecate source, protocol, qualification, display statistics and composition code. `WATCHER.json` identifies the observer/worker and run; `WATCHER_STATUS.json` is its dated heartbeat. `LATEST.json` names the committed generation; incomplete `.partial-*` directories are never selected. Seven previously completed model maps remain untouched.

## Verified recovery semantics

The saved rows are final uint8 pixels, not an approximation of incomplete sums. Recovery fills those rows verbatim, zero-initializes the remaining accumulators, and replays the earliest patches whose support overlaps an unsaved row. The first replayed index is rounded down to the original global batch boundary, preserving batch membership and blank-tile selection. The patch/model calculation and accumulation order are retained from the hash-pinned upstream function. Only starting indices, the output normalization row range, and a post-batch checkpoint callback are added in a separately derived function.

Recovery redoes a small overlapping band and any work after the most recent completed row; it does not promise a fixed maximum number of seconds lost. The current worker is not restarted to gain checkpointing. A future recovery worker checkpoints its own completed rows after batches and, after writing the final map and receipt, invokes the existing receipt-verifying supervisor to perform the original composition.

[validate_hecate_checkpoint.py](../../../tools/validate_hecate_checkpoint.py) compares interrupted/resumed maps with the original uninterrupted function on deterministic fixtures with blank patches, irregular edges, different interruption positions and a fully complete checkpoint. It also includes a fixed 96×128 real Hecate input fixture. Missing completion manifests, changed bindings and altered prefix bytes must be rejected. `CHECKPOINT_VALIDATION.json` records the actual outcome and exact tested tool identities; this is a bounded runtime qualification, not a full-map replay or biological accuracy claim.

**Result: PASS.** All seven cases produced bitwise-identical output, and the invalid-checkpoint checks passed. The real fixture contained six tile origins; validation evaluated16 real tiles across reference, interrupted and resumed passes. Its saved32-row prefix required replay from origin0, so actual work-skipping was tested by the larger deterministic fixtures. No full-map replay or live-worker crash was injected. The original worker stayed running throughout qualification.

## Commands for a future recovery

Use the pinned `/workspace/.venvs/pherc343/bin/python` environment. Do not start another inference worker while the current one is alive. First read the status and exact PID command, check the observer heartbeat, and verify a checkpoint:

```bash
/workspace/.venvs/pherc343/bin/python -B \
  /workspace/Herculaneum/research/tools/checkpoint_hecate.py verify \
  --lab /workspace/herculaneum-lab \
  --root /workspace/herculaneum-lab/reports/fortysixth-experiment/work/hecate-checkpoints
```

If the worker has ended without a completed padded map, run the same command with `resume` in place of `verify`, under a detached session with a retained log. The existing pipeline lock prevents concurrent inference; the command refuses an existing final output or receipt. Inspect such partial publication before taking any action. During that recovery, `RESUME_STATUS.json` within the checkpoint directory gives PID, progress and update time; also verify the actual process. After the map completes, the original supervisor updates `PIPELINE_STATUS.json` for composition. A different machine/runtime still needs the established real-input numerical qualification before inference; hashes alone do not qualify a different numerical environment.

If only the observer disappears while the same original recovery worker remains alive, restart mode `watch`; its separate lock prevents duplicate observers. Do not run `resume` merely because an observer has stopped.

## Persistence and incident limits

These saves protect against losing a process and `/tmp` while the `/workspace` checkpoint files survive. File flushes do not establish a platform-wide durability guarantee. An archived checkpoint in GitHub is an additional dated backup; ongoing observer saves are local and are not automatically pushed. The archive manifest states precisely which generation was backed up. A complete workspace loss can require that older external backup and the existing Git LFS inputs.

`HECATE_CHECKPOINT_BACKUP.zip` contains one such backup; [CHECKPOINT_BACKUP.json](CHECKPOINT_BACKUP.json) records its file and member hashes, row coverage and timestamp. Extract it into a new checkpoint root, verify all archived bytes against the manifest and run mode `verify` before considering `resume`. It contains completed output rows and input/model fingerprints, not the CT inputs or model weights.

Checkpointing cannot prevent a platform restart or reveal its cause. See the separate [incident record](../../../planning/CLOUD_RESTART_20261010.md). There is no evidence sufficient to decide whether the user's message triggered the replacement or whether not messaging would have avoided it. No lifecycle claim or support response is invented.
