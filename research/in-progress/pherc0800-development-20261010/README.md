# PHerc0800 larger development field — running study51

The fixed16-map CPU comparison started **10 October2026,19:18 BST** after all
rendering, batch and restart gates passed. This page is a dated launch record;
check live receipts/process identity before reporting current status. No GPU,
Actions run or spare-VM task was launched.

The field is segment20251028222030, x512:2048/y512:2048, using original geometry,
66 planes, d0, all four models, both padding phases and both depth orders. The
common scored support is1,875,648 pixels (79.50%). It is previously inspected
development material, not held-out accuracy or authenticated ink. Unsupported
pixels remain masked. No response-based depth/model selection or ensemble fit.

[PROTOCOL.md](source/PROTOCOL.md) and [INFERENCE_GO.json](source/INFERENCE_GO.json)
freeze the scope and all worker/input identities. [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)
identifies the exact copied original files. This is an in-progress export, not a
sealed experiment completion or a replacement for the original lab.

All104 valid render rectangles and9 distributed32-square sites passed independent
sampling (1,043,328 samples,99.64134% exact, every difference<=1 uint8 count).
The source work used384 observed chunks:17 reused and367 newly fetched, within
the768MiB frozen bound. Four distinct tiles per model/order passed batching versus
singleton inference; maximum probability difference7.16e-7 against1e-5 tolerance.
Eight original-framework direct/cache comparisons and cached replays matched
exactly. Recovery after an injected stop with two committed batches passed;
six missing/corrupt/changed cache cases were rejected.

The run has3,454 planned forward batches /13,250 model tiles. Qualification
estimates3.88 model hours; allow additional replay, I/O and backup overhead,
and refine against actual Hecate progress. This is an estimate, not a deadline.

## Status and restart

On the cloud lab:

```bash
python3 -B research/tools/status_phase51.py
```

Read `/workspace/herculaneum-lab/reports/fiftyfirst-experiment/work/inference/PROGRESS.json`
and `work/inference-worker.log`. A map completes only when its COMPLETE.json is
written after exact saved-logit replay. The whole study completes only when all
16 maps and INFERENCE_RESULTS.json exist and pass the final review. A PID or
count alone is insufficient.

If the process has stopped, first check the recorded PID/start identity, verify
restored files and inspect the error. Use the existing qualified environment,
4 CPU threads and the original absolute lab path. The exact runner is:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TMPDIR=/workspace/scratch/phase51-tmp \
  /workspace/.venvs/pherc343/bin/python -B \
  /workspace/herculaneum-lab/reports/fiftyfirst-experiment/worker.py
```

Only start after confirming no worker is live. The worker also holds an exclusive
file lock, verifies all input/model/script pins, and rejects a completed study.
Preserve partial state. Committed raw logits are checked and replayed through
fresh accumulation; incomplete uncommitted batches can be recomputed. Do not
change the frozen worker or silently reuse another batch/input identity.

## Private remote checkpoints

[The separate checkpoint branch](https://github.com/jl-drx/Herculaneum/tree/codex/plinius-pherc0800-phase51-checkpoints)
contains append-only LFS generations and INDEX.json. Each generation includes
only previously unbacked-up immutable files; restore every listed generation in
order. Inputs are saved once; subsequent generations add committed batch logits
and completed maps. Earlier checkpoint weights and runtime source are separate
prerequisites listed in the index, not duplicated.

The backup writer checks every archive member, uploads the LFS object, then
updates its own Git ref with a non-forced push. It never stages the research
checkout. The intended interval is10 minutes while the worker lives. Only the
last **successfully published** index establishes durable coverage. Recent work
can be lost between uploads, and neither a process nor the installed environment
is preserved by these data backups. See REMOTE_BACKUP_VALIDATION.json for actual
upload/download/restore evidence and the latest checked scope.

To restore downloaded generations without inference:

```bash
python3 research/tools/restore_phase51_backup.py \
  --backup /path/to/downloaded-checkpoint-branch \
  --destination /workspace/herculaneum-lab
```

Fetch only the index's named LFS objects. All archives and extracted members are
checksum checked; an existing different file stops restoration. A temporary
alternate destination is useful for read-only testing, but the frozen worker's
absolute input pins require the original lab path. Restore prior prerequisites
and the qualified Python environment separately before starting any inference.

Routine short checks stay in this chat. Use the spare VM only when a longer local
run creates worthwhile independent parallel work; no duplicate audit is needed
for this launch. GPU spend remains£0; the first-pilot£10 cap is unchanged.
