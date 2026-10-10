# PHerc0800 study52 — qualified shifted-depth run

All32 shifted qualification cases and16 actual-worker d0 baseline replays passed.
The frozen CPU run has32 new maps /3,096 batches, d−15/+15, all four models,
both padding phases and both depth orders on x512:2048/y1024:1536. The16 saved
d0 maps remain the comparator. Read [execution notes](source/EXECUTION_NOTES.md),
[qualification summary](source/QUALIFICATION_SUMMARY.json) and the
[frozen protocol](source/PROTOCOL.md). No ink result or GPU spend is claimed.

At this prelaunch record, remote backup restoration and production launch remain
pending. The dated launch/status receipt will supersede this paragraph once the
worker starts. The earlier design/feasibility packet is preserved unchanged.

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
