# Private download and offline replay

The existing private branch is `codex/plinius-pherc343-reproduction` in
`jl-drx/Herculaneum`. No new PR, Actions run or spare-VM assignment is needed.
The analysis archive and536 native CT chunks are separate so reviewing the
findings does not require downloading the235MiB of CT input. All source data
and code keep their original provenance and licensing; no blanket new license
is asserted.

From a clone made with `GIT_LFS_SKIP_SMUDGE=1` (or the existing checkout):

```bash
git lfs pull --include='research/workspace/bundles/pherc0139-learning-readiness-20261011.tar.gz' --exclude=''
python3 research/planning/pherc0139-learning-readiness-20261011/restore_packet.py \
  --bundle research/workspace/bundles/pherc0139-learning-readiness-20261011.tar.gz \
  --destination /workspace/review-learning
```

This standard-library restorer checks the archive and every member's length and
SHA256, rejects unsafe members and requires a new destination. The manifest is
BUNDLE.json beside the restorer. A minimal replay environment requires Python3.12
and the exact versions in requirements-replay.txt. Installed environments,
credentials, live workers and model weights are not part of these archives.

```bash
python3 -m venv /workspace/review-learning-venv
/workspace/review-learning-venv/bin/python -m pip install -r /workspace/review-learning/requirements-replay.txt
/workspace/review-learning-venv/bin/python /workspace/review-learning/soft-controls/review/compact_verify.py \
  --packet /workspace/review-learning/soft-controls --output /workspace/replay-soft-controls
/workspace/review-learning-venv/bin/python /workspace/review-learning/native-reference/replay.py \
  --output /workspace/replay-native-annotations
/workspace/review-learning-venv/bin/python /workspace/review-learning/native-reference/review/independent/recheck_portable.py \
  --packet /workspace/review-learning/native-reference --output /workspace/replay-native-independent
/workspace/review-learning-venv/bin/python /workspace/review-learning/native-reference/geometry/review/check_geometry.py \
  --root /workspace/review-learning/native-reference/geometry --output /workspace/replay-native-geometry.json
```

Choose fresh output paths. These commands never acquire data or run a model.
Compact soft-control replay independently recomputes teacher sampling and all
summaries from the retained crop and saved derived UV fields; it does not
rederive those UV fields or native validity from raw CT. The original stronger
full-geometry independent receipt is included. To repeat that stronger check,
restore the existing correspondence/supervision packets using their own
PORTABILITY.md and run soft-controls/review/independent_check.py with its
documented command-line arguments. Those older archives are not duplicated here.
Use a disposable copy of soft-controls for that archival stronger checker: it
writes its review outputs beneath the supplied `--root`. The compact checker
above writes to a separate new output directory.
The design inventory's old path-specific audit scripts are provenance; the
commands above are the portable replay entrypoints for this packet.

For actual CPU model preparation or full native CT-support replay, additionally:

```bash
git lfs pull --include='research/workspace/bundles/pherc0139-native-ct-20261011.tar.gz' --exclude=''
python3 research/planning/pherc0139-learning-readiness-20261011/restore_packet.py \
  --manifest research/planning/pherc0139-learning-readiness-20261011/CT_BUNDLE.json \
  --bundle research/workspace/bundles/pherc0139-native-ct-20261011.tar.gz \
  --destination /workspace/review-learning/native-reference/ct/chunks
/workspace/review-learning-venv/bin/python /workspace/review-learning/native-reference/ct/replay_ct.py \
  --output /workspace/replay-native-ct
```

The CT archive contains only source chunks relative to `ct/chunks/`; the analysis
archive retains their plan, acquisition receipts, code and validity results.
Both archives are required for native CT replay. Each28×128×128 chunk is stored
once; no full scan, duplicated training tensors, checkpoints or GPU runtime are
included. A9GiB-RAM Linux machine has ample memory for these sequential replay
checks. Training runtime/memory have not yet been benchmarked.

Private GitHub access on another machine uses the user's own normal Git
authentication. Do not copy cloud credentials. The saved portability validation
records the exact relocation/replay and fresh remote-download checks performed;
same-host relocation is not a claim that the spare VM has run them.
