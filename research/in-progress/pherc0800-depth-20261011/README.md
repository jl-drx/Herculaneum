# PHerc0800 study52 — qualified shifted-depth run

All32 shifted qualification cases and16 actual-worker d0 baseline replays passed.
The frozen CPU run has32 new maps /3,096 batches, d−15/+15, all four models,
both padding phases and both depth orders on x512:2048/y1024:1536. The16 saved
d0 maps remain the comparator. Read [execution notes](source/EXECUTION_NOTES.md),
[qualification summary](source/QUALIFICATION_SUMMARY.json) and the
[frozen protocol](source/PROTOCOL.md). No ink result or GPU spend is claimed.

The production worker started **11 October 2026, 00:48 BST**. It is saving
batches and has completed its first maps with exact cached replay. Allow roughly
**3–4 hours** for the fixed run; Hecate accounts for most compute, so early map
counts do not predict a linear completion time. [INITIAL_PROGRESS.json](INITIAL_PROGRESS.json)
is a dated observation; use the live status tool below for later checks.

The initial245,853,791-byte GitHub archive was freshly downloaded and all653
members restored. Two incremental generations were then downloaded and restored
as a chain:734files, including the first40 committed production batches.
See [initial restoration](LAUNCH_BACKUP_VALIDATION.json),
[production checkpoint restoration](PRODUCTION_BACKUP_VALIDATION.json) and
[launch record](LAUNCH_STATUS.json). This verifies data restoration on this host,
not independent model execution or the availability of later live batches.
The10-minute checkpoint watcher is active.

Live status in the cloud checkout:

```bash
python3 -B research/tools/status_phase52.py
```

The read-only status tool checks recorded PID/start identity and committed
receipts. Completion requires all32 maps, exact replay and later scientific
review. A PID, partial image or count alone is insufficient.

The separate private checkpoint ref is
[codex/plinius-pherc0800-phase52-checkpoints](https://github.com/jl-drx/Herculaneum/tree/codex/plinius-pherc0800-phase52-checkpoints).
Download only the named LFS generations in INDEX.json. Restore with:

```bash
python3 research/tools/restore_phase52_backup.py \
  --backup /path/to/downloaded-branch --destination /workspace/herculaneum-lab
```

Every archive/member checksum is checked; existing identical files are reused
and conflicting files are rejected. Earlier study51 inputs/logits and runtime
prerequisites are separate. The qualified environment and original absolute
paths must be restored independently; data archives do not preserve processes.

After an actual interruption, inspect logs/errors and verify retained files,
then deliberately invoke `python3 research/tools/launch_phase52.py` only when no
worker or previous watcher is live. It checks the qualification/GO/initial
remote-restore receipt, prevents duplicates, preserves earlier launch records,
and starts the checkpoint watcher. It rejects a completed study. Frozen scripts
or inputs must not be changed to force a resume.

Checkpoint publication is intended every10minutes and at worker exit. Only the
last successful remote index defines durable coverage. New batches since that
upload and live processes may be lost on platform replacement. Original-grid
cohorts and depth-bound raw logits make verified retained batches reusable.

All source copies are identified by [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json).
The cumulative archival seal remains at study50. This is an active development
comparison, not a held-out accuracy test. No Actions dispatch, spare-VM assignment
or paid compute is needed. GPU spend remains£0; first-pilot cap£10total unchanged.
