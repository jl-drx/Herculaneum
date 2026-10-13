# Restore and replay this qualification

Private branch: `codex/plinius-pherc343-reproduction`. This packet extends the
earlier geometry intake without modifying it. It contains the original intake,
12 new fine coordinate rasters, source/metadata receipts, protocols, scripts,
geometry results and the unexecuted CT pilot request. It contains no CT image
bodies, teacher/annotation values, weights or secrets.

From an authorized checkout:

```bash
git lfs fetch --include='research/workspace/bundles/pherc0139-fine-geometry-20261011.tar.gz' --exclude='' origin codex/plinius-pherc343-reproduction
git lfs checkout research/workspace/bundles/pherc0139-fine-geometry-20261011.tar.gz
python research/planning/pherc0139-fine-geometry-20261011/restore_packet.py \
  --bundle research/workspace/bundles/pherc0139-fine-geometry-20261011.tar.gz \
  --destination /tmp/plinius-fine-geometry
```

The standard-library restorer verifies the archive and every member against
`BUNDLE.json`, rejects unsafe/duplicate paths and refuses an existing destination.
Use Python3.12, NumPy2.5.3 and tifffile2026.9.20 for offline replay. No Torch,
OpenCV, GPU, R2 credentials or remote CT download is needed.

From the restored directory:

```bash
python qualify.py --output /tmp/plinius-fine-geometry-replay
python verify.py
python verify_pilot.py
```

Compare the replay's `RESULTS.json`, `ADMISSION.csv` and `COMPUTE_BINDING.json`
against `qualification/`. Exact bytes should match. Verification writes receipts
only in the restored copy. The replay computes geometry with one numerical CPU
thread; it does not process or infer on CT data.

`acquire.py` and `prepare_ct_pilot.py` preserve the completed coordinate/metadata
acquisition procedure. They are not required for offline replay. The CT payload
plan is unexecuted and needs its separately frozen comparison protocol before use.

These are recovery instructions, not a spare-VM assignment. Keep short replay
and qualification work in the present chat. Original source code and public
coordinate data retain their upstream terms; see the repository's
`research/THIRD_PARTY_NOTICES.md`. No blanket relicensing is asserted.
