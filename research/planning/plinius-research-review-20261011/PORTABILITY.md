# Selective private download and CPU replay

This packet contains the new9CT chunks (18MiB), the36 PHerc343 calibration
figures, all new reports/code/metadata/receipts and source checksums. It contains
no model weights, credentials, full scroll scans or live process state. Existing
PHerc343 map regeneration still needs the earlier reproduction bundle; the
rendered review panels here can be inspected directly.

From the private research branch, fetch **only this LFS object**:

```bash
git lfs pull --include='research/workspace/bundles/plinius-research-review-20261011.tar.gz' --exclude=''
python3 research/planning/plinius-research-review-20261011/restore_packet.py \
  --bundle research/workspace/bundles/plinius-research-review-20261011.tar.gz \
  --destination /tmp/plinius-research-review-20261011
```

The helper requires Python3.11+ and the standard library. It verifies the archive
SHA256 and every member, rejects unknown paths and overwrites, and creates a new
destination. `BUNDLE.json` beside the helper identifies the exact covered files.
The research archive has its own manifest; it is not an extension of the formal
experiment50 cumulative seal or study52 live checkpoint chain.

For metadata and landmark replay in the restored directory:

```bash
cd /tmp/plinius-research-review-20261011
python3 metadata/analyse_metadata.py
python3 paired-data/analyse_landmarks.py \
  --input paired-data/LANDMARK_INPUT.json --output /tmp/landmark-replay.json
cmp paired-data/LANDMARK_RESULTS.json /tmp/landmark-replay.json
```

The first command is standard-library-only; the second needs NumPy. Neither
fetches network data. For a fresh numerical/figure replay of the reconstruction
comparison, use a **second disposable copy** of the restored directory so saved
results remain reviewable:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 reconstruction-control/analyse.py --selfcheck
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 reconstruction-control/analyse.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 reconstruction-control/verify_independently.py
```

Analysis uses NumPy and Matplotlib; semantic checks and the independent checker
also use SciPy. `RUNTIME.json` records the tested versions. Numerical replay was
tested in the existing Python3.12 environment on this cloud host; a fresh
machine/package installation is not claimed. Cross-version figures can differ
in encoding or fonts; compare scientific metrics, source hashes and full support,
not PNG file identity. The independent verifier reports numerical tolerances.
No Torch, CUDA, Runpod or R2 credentials are required.

Do not invoke the fetch scripts for ordinary replay: all their requested bodies
are already included. The PHerc343 calibration script rejects an existing output
folder and retains original absolute lab paths; reuse its existing36 figures or
restore the separately documented study46 inputs before regenerating them.
Some primary-source citation paths in agent reports likewise retain historical
cloud paths; pinned public URLs and exact hashes accompany those citations.

CT and derivative figures retain the catalogue'sCC BY-NC4.0 data terms and
Vesuvius Challenge attribution. Upstream excerpts retain source identities and
terms; see `SOURCE_NOTICES.md`. Everything remains in the private repository.
No Actions run or spare-VM job was requested by this export.
