# Restore additional input and the first case's qualification

**Availability: locally committed and restore-tested; publication blocked.**
The new Git push and follow-up ref read returned HTTP 503. The commands below
become usable on another machine after a successful push and fresh remote
verification. See [PERSISTENCE.json](PERSISTENCE.json). Earlier parent archives
remain the previously published, verified checkpoints.

This **378,709,958-byte (361.17 MiB)** archive contains 2,137 payload files and
applies 2,126 changed paths. It adds 1,009 source records, the continuation records
and the complete first-case qualification, including sparse TIFF metadata,
previous smoke fixtures, teacher/material patch arrays, code and verification.

Archive SHA256:
`f44ed05c21b2ee51dc0bd383ae71aef24847700726c1fe4284df327311843dc0`.
[BUNDLE.json](BUNDLE.json) binds the archive and manifest.

[LOCAL_RESTORE.json](LOCAL_RESTORE.json) records all file checks, a combined copy
matching the live cache, unchanged original qualifications and a byte-identical
independent qualification replay. The [fresh production replay](../native-added-qualification-20261012/PRODUCTION_REPLAY.json)
also reproduces all case results and both derived files byte for byte. These are
local restore/replay checks; publication is recorded separately.

First restore the [21:47 cumulative checkpoint](../native-coverage-retry-20261012-2147/PORTABILITY.md).
Its combined manifest must be
`665c69cb1e6fb890ce06685679910f0293799d52d63dbcb935b852b13ba92899`.
That base already includes the earlier 20:53 retry. Apply this update once, to
that exact unmodified restore, not directly to the completed-original base.

From the private repository checkout:

```bash
git lfs pull --include='research/workspace/bundles/native-coverage-retry-20261012-2318.tar.gz' --exclude=''
python research/in-progress/native-common-support-20261012/verify_bundle.py \
  --archive research/workspace/bundles/native-coverage-retry-20261012-2318.tar.gz \
  --expected-sha256 f44ed05c21b2ee51dc0bd383ae71aef24847700726c1fe4284df327311843dc0 \
  --output /tmp/plinius-retry-2318
python /tmp/plinius-retry-2318/code/checkpoint_delta.py apply \
  --base /tmp/plinius-current-after-retry-2147 --delta /tmp/plinius-retry-2318 \
  --output /tmp/plinius-current-after-retry-2318
python /tmp/plinius-current-after-retry-2318/code/continue_acquisition.py inspect \
  --base /tmp/plinius-current-after-retry-2318/base \
  --work /tmp/plinius-current-after-retry-2318/work
python /tmp/plinius-current-after-retry-2318/work/added-qualification-20261012/verify_case.py \
  --work /tmp/plinius-current-after-retry-2318/work \
  --qualification /tmp/plinius-current-after-retry-2318/work/added-qualification-20261012 \
  --output /tmp/plinius-added-independent-check.json
```

Expected cache: **3,515 present, zero confirmed missing, 14,055 pending**.
The independent verifier needs NumPy and no network/model inference. It compares
all 1,290 candidates against a full-raster CT reconstruction, including all
36,120 depth counts and the class/UV totals. Both older original-stream
qualifications remain unchanged in their dated directories.

For a fresh production decode/resize replay, with the versions in the
[qualification results](../native-added-qualification-20261012/RESULTS.json):

```bash
python research/in-progress/native-added-qualification-20261012/replay_production.py \
  --current /tmp/plinius-current-after-retry-2318 \
  --output /tmp/plinius-added-production-check.json
```

The helper builds a temporary reference view from retained bytes, reruns the
stored qualifier and requires identical case results and derived file hashes.
The new freeze timestamp is allowed to differ. It uses no downloaded input,
neural model or additional evaluation material. Keep all output paths new.

Allow roughly 3 GiB of extra disk for the compressed update, extracted delta and
new combined copy. Retained source data for further acquisition still needs
another 5.08 GB, plus outputs. The restored cache is partial; a single qualified
source does not admit the intended full training pair or authorize paid compute.
Resume only the same fixed requests after source access recovers. Keep private
and retain the original public-source attribution.
