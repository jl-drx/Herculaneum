# Study52 completed-output recovery, 11 October 2026

The PHerc0800 shifted-depth inference completed before this environment was
restored. The private checkpoint ref at `02812de445f6ab0069174805bb14b6014bed159c`
contains all 32 final maps and `INFERENCE_RESULTS.json`; its last publication
was **11 October 2026, 03:44:59 BST**. No worker restart was needed or attempted.

All 21 named LFS archives (878,076,803 bytes) were selectively fetched. Every
archive and each of the 7,080 members was checked against `INDEX.json` by the
unchanged repository restore tool before restoring the original study path.
See `RECOVERY_PLAN.json` and `RESTORE_RESULT.json`.

The separate read-only checker `verify_completion.py` passed at **08:43 BST**:

- Complete Cartesian coverage: four models, two depth offsets, two padding
  phases and two depth orders, giving 32 final map receipts.
- All 3,096 committed raw-logit batches, their hashes, indices, original batch
  cohorts, depth positions and frozen GO binding.
- All 200 final-map output hashes; finite aligned probabilities, fixed display
  conversion and correct support-mask application.
- All 16 saved d0 crop hashes and dimensions. These plus the shifted outputs
  provide the full 48-map comparison without new inference.
- The 487 earlier prerequisites available in this snapshot also matched their
  pins. The 471 absent prerequisites belong to study51; they are unnecessary
  for reviewing completed outputs. A full input-dependent worker replay would
  additionally require those inputs and code restored.

The saved production receipts attest exact cached-logit replays for all maps.
This recovery did not repeat those accumulation replays or model execution.
It did not inspect prediction images or decide whether a candidate meets the
scientific review criterion. GPU spending remains £0.

`PROCESS_OBSERVATION.json` records that no worker or checkpoint watcher was
live. The ordinary `status_phase52.py` expects a live-run `RUN_STATE.json`, which
was intentionally not in the checkpoint archive. Do not fabricate that file or
launch a completed study merely to make that status command run.

Read the original frozen protocol at
`/workspace/herculaneum-lab/reports/fiftysecond-experiment/PROTOCOL.md` before
the scientific review. All 48 views must be reviewed with the fixed support and
0–255 display; no intermediate depth search or change to the completed design
is authorized by this recovery.
