# Restore and replay the prospective intake

The private branch is `codex/plinius-pherc343-reproduction`. This packet is
separate from the old whole-workspace manifest and the running trainer backups.
Its single Git LFS archive contains the metadata, native coordinate rasters,
frozen selection inputs, source code, all rejected candidates and saved results.
It contains no CT, predictions, model weights or credentials.

From an authorized checkout, fetch only this archive:

```bash
git lfs fetch --include='research/workspace/bundles/pherc0139-paired-packet-20261011.tar.gz' --exclude='' origin codex/plinius-pherc343-reproduction
git lfs checkout research/workspace/bundles/pherc0139-paired-packet-20261011.tar.gz
python research/planning/pherc0139-paired-packet-20261011/restore_packet.py \
  --bundle research/workspace/bundles/pherc0139-paired-packet-20261011.tar.gz \
  --destination /tmp/plinius-paired-intake
```

The standard-library restorer verifies the archive and every file against
`BUNDLE.json`, rejects unsafe/duplicate paths and refuses an existing destination.
Use Python3.12 with NumPy2.5.3 and tifffile2026.9.20 for offline geometry replay
(`requirements-replay.txt`). Torch, GPU, R2 credentials and remote scans are not
needed. Run from the restored packet:

```bash
python select_regions.py --output /tmp/plinius-paired-selection-replay
python verify_packet.py
python source_contracts.py
```

Compare the six replay selection files against `selection/`; exact bytes should
match. `source_contracts.py` reproduces the frame/extent summary and next fine
coordinate plan. `verify_packet.py` checks saved inputs/results and geometry
fixtures. None of these three scripts makes network calls or learns from targets.
Verification writes `VERIFICATION.json` in the restored copy only.

`acquire_metadata.py` and `acquire_coordinates.py` document the completed public
acquisition; they are not required for offline replay. Do not execute future
chunk plans as if they were admitted training data. `NEXT_FINE_GEOMETRY_PLAN.json`
and `selection/NEXT_NATIVE_CT_PLAN.json` are explicitly unexecuted proposals.

This short replay stays in the present chat; these instructions provide recovery
and portability, not a request to occupy the spare VM. GPU engineering files
are ordinary Git files in the adjacent `gpu-qualification-20261011` directory.

Original upstream assets remain subject to their own terms. Nieuwlaar README
and benchmark source are pinned to217e7521670db49cde97872ea6a8bdfb80c449f1;
the renderer is from ScrollPrize/villa530584554c031c982ba3aa243e1ea39c4bef9818.
Source notices and context are in the repository's `research/THIRD_PARTY_NOTICES.md`.
No blanket relicensing or model/data redistribution permission is asserted.
