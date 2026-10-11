# Download and replay the private w040 comparison

The archive contains all 75 CT chunks, all six coordinate TIFFs, fixed geometric
query coordinates and support masks, source metadata/code, frozen plans,
complete score landscapes, independent reviews and six review figures.
Large disposable copies of assembled CT arrays and intermediate Newton arrays
are omitted; `DERIVED_OMISSIONS.json` identifies them by checksum. These are
calculated outputs, not missing observations. The frozen query fields needed
for the CT comparison are included.

Fetch only this LFS object from `codex/plinius-pherc343-reproduction`:

```bash
git lfs pull --include='research/workspace/bundles/pherc0139-correspondence-20261011.tar.gz' --exclude=''
python3 research/planning/pherc0139-correspondence-20261011/restore_packet.py \
  --bundle research/workspace/bundles/pherc0139-correspondence-20261011.tar.gz \
  --destination /tmp/pherc0139-correspondence-20261011
```

The restore helper uses only Python 3.11+ standard library. It checks the archive
and every file against `BUNDLE.json`, rejects traversal, symlinks, unknown files
and overwrites, and creates a new destination.

For a complete offline numerical replay, use Python 3.12 and the recorded
NumPy/SciPy versions. In the restored directory:

```bash
python3 -m venv /tmp/w040-replay-venv
/tmp/w040-replay-venv/bin/pip install -r requirements-replay.txt
/tmp/w040-replay-venv/bin/python replay.py --output /tmp/w040-replay-results
```

Package installation needs network access; the replay itself does not. It checks
all archived inputs, reconstructs the arrays from the 75 retained chunks,
repeats the fixed smoothing, geometry-based sampling and finite translation
comparison, then independently recomputes all 52,728 NCC values using direct
spatial covariance. It compares the numerical results and every frozen decision.
Inputs and historical evidence remain unchanged. Choose a new output directory
for each replay. No Torch, GPU, Runpod or R2 credentials are needed.

Numerical tolerance across compatible machines is 1e-8 for saved scores and
summaries; array byte identity is reported separately. Geometry query bytes are
fixed inputs and must match exactly. The publication is a private research
snapshot, not an extension of the study 52 checkpoint chain or cumulative seal.

The original geometry solver and every intermediate-output checksum are also
included. To regenerate its omitted disposable arrays, install
`requirements-geometry-and-panels.txt` and run:

```bash
/tmp/w040-replay-venv/bin/python replay_geometry.py --output /tmp/w040-geometry-replay
```

This wrapper rebinds historical paths to the included, checksum-verified catalog
and TIFFs; it does not alter the frozen solver or physical parameters. It compares
all query coordinates and support masks with the saved reference and reports
array byte identity separately. It reads no CT or targets. The geometry replay
uses a 1e-4 fine-coordinate numerical tolerance while requiring the original
0.01 native-voxel residual bound and identical support decisions.

`PORTABILITY_VALIDATION.json` in the repository records the actually tested
restoration and CPU replay. A new machine's package installation is not claimed
to have been tested here. The compact text-only GitHub directory is for review;
restore the LFS archive to run the analysis. No Actions run or spare-VM task is
needed for this short replay.
