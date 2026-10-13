# Restore the private partial intake

This checkpoint preserves the completed downloads and exact unresolved requests.
It is **not a complete or admitted training dataset**. No inference, paid job,
RunPod/R2 credential or GitHub Actions run is required for its offline review.

The archive is **514,725,982 bytes (490.9 MiB)**, SHA256
`15580fe689f66cd9dd2d0901d3de117d8c26c00fce931390d0027be80aa9377a`.
[BUNDLE.json](BUNDLE.json) records 2,887 payload files, approximately 590 MiB
uncompressed. It includes every completed CT/teacher object, original annotation
shard, metadata/receipt, partial qualification, source code and transport record.
Large request/dependency tables are indexed in [LARGE_RECORDS.json](LARGE_RECORDS.json)
and stored in the archive rather than duplicated in ordinary Git.

Use the private `jl-drx/Herculaneum` branch `codex/plinius-pherc343-reproduction`:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --single-branch --branch codex/plinius-pherc343-reproduction https://github.com/jl-drx/Herculaneum.git
cd Herculaneum
git lfs pull --include='research/workspace/bundles/native-coverage-admission-partial-20261012.tar.gz' --exclude=''
python research/in-progress/native-common-support-20261012/verify_bundle.py \
  --archive research/workspace/bundles/native-coverage-admission-partial-20261012.tar.gz \
  --expected-sha256 15580fe689f66cd9dd2d0901d3de117d8c26c00fce931390d0027be80aa9377a \
  --output /tmp/plinius-admission-review
```

For an existing checkout, update that branch and perform only the selective LFS
pull. Extraction requires a new directory and verifies the archive, every file,
safe paths and SHA256SUMS. Keep the repository, archive and extracted data private.

With NumPy installed in an isolated environment, verify every original native
candidate and class count independently:

```bash
python /tmp/plinius-admission-review/code/verify_original.py \
  --work /tmp/plinius-admission-review/work \
  --output /tmp/plinius-admission-verified.json
```

The original check used Python 3.14.7 and NumPy 2.4.4. It reconstructs full
known/centre/all21 input-support fields from the 400 saved CT chunks, uses
integral sums, and compares all 876 rows, including the 395 unknown-input rows.
It performs no network requests or model execution. Original annotation decoding
used Zarr 3.2.1 plus the retained explicit CRC32C/Blosc decoder; those source
shards and decoded targets/masks are included. The numerical verifier uses the
saved decoded annotations, while the manifest protects all source bytes.

Read-only cache inspection works with the standard library:

```bash
python /tmp/plinius-admission-review/code/continue_acquisition.py inspect \
  --base /tmp/plinius-admission-review/base --work /tmp/plinius-admission-review/work
python /tmp/plinius-admission-review/code/continue_acquisition.py inspect \
  --base /tmp/plinius-admission-review/base --work /tmp/plinius-admission-review/work/original
```

The original checkpoint's pending counts are 16,649 added requests and 381 original requests.
Their remaining nominal bodies total 6,248,165,613 bytes. Allow space for them
in addition to this checkpoint, receipts, later derived targets and the runtime;
do not download every historical LFS archive. No model weights are in this file.

## Apply the later cache addendum before resuming

A later retry retained 52 additional complete CT chunks before the same proxy
failure. The **19,943,009-byte (19.0 MiB)** addendum contains those chunks and
receipts, continuation history and diagnostics in 125 payload files. Its SHA256
is `9b6ed93ae3adf35b7fc3487db0a289ecb9464800da37365ed006aa7507f1e8a0`.
See [ADDENDUM_BUNDLE.json](ADDENDUM_BUNDLE.json). The first archive remains unchanged.

After the original restore and numerical replay above:

```bash
git lfs pull --include='research/workspace/bundles/native-coverage-admission-addendum-20261012.tar.gz' --exclude=''
python research/in-progress/native-common-support-20261012/verify_bundle.py \
  --archive research/workspace/bundles/native-coverage-admission-addendum-20261012.tar.gz \
  --expected-sha256 9b6ed93ae3adf35b7fc3487db0a289ecb9464800da37365ed006aa7507f1e8a0 \
  --output /tmp/plinius-admission-addendum
python /tmp/plinius-admission-addendum/code/checkpoint_delta.py apply \
  --base /tmp/plinius-admission-review --delta /tmp/plinius-admission-addendum \
  --output /tmp/plinius-admission-current
python /tmp/plinius-admission-current/code/continue_acquisition.py inspect \
  --base /tmp/plinius-admission-current/base --work /tmp/plinius-admission-current/work
python /tmp/plinius-admission-current/code/continue_acquisition.py inspect \
  --base /tmp/plinius-admission-current/base --work /tmp/plinius-admission-current/work/original
```

Application validates both manifests and the exact base identity before making
a **new copy**; it preserves the original restore and archives the three replaced
operational status files. Allow about another 613 MiB for the combined copy.
Its new manifest covers all files. The current cache inspection should report
949 added objects and 424 original objects present, with **16,621 added and
357 original requests pending** (6,224,310,509 nominal remaining bytes).

The combined copy retains the historical `qualification/` unchanged. Those
counts describe the first checkpoint. Use the untouched original restore for
the frozen `verify_original.py` replay; do not run that verifier against the
enlarged cache and expect an unchanged input-support inventory. The additional
bytes have not been qualified as new training examples. The partial 64 KiB
diagnostic response is not a source object or a usable CT chunk.

[The local addendum restore check](ADDENDUM_LOCAL_RESTORE.json) verifies all
125 files, applies 117 new or updated paths to a fresh copy, and checks all 1,373
complete source objects against the frozen request lists and receipts. A separate
comparison confirmed that the combined work tree matches the live partial cache,
the original 2,887-file restore still verifies, and qualification files are unchanged.
Both the [base archive](REMOTE_RESTORE.json) and [addendum](ADDENDUM_REMOTE_RESTORE.json)
were then downloaded separately from private GitHub into empty LFS caches and
matched their verified archive bytes and SHA256 values. This is a remote
availability check in the same workspace, not a second-machine scientific review.

## Resume the fixed acquisition

Only after public S3 transport recovers, resume the **same** request manifests:

```bash
python /tmp/plinius-admission-current/code/continue_acquisition.py acquire \
  --base /tmp/plinius-admission-current/base --work /tmp/plinius-admission-current/work \
  --attempt network-restored-01
python /tmp/plinius-admission-current/code/continue_acquisition.py acquire \
  --base /tmp/plinius-admission-current/base --work /tmp/plinius-admission-current/work/original \
  --attempt network-restored-01
```

Each invocation first verifies existing files, saves an attempt record and
requires a successful bounded source probe before four-reader bulk acquisition.
A new attempt identifier prevents overwriting its history. Existing successful
receipts are reused; source validators and scientific eligibility are unchanged.
A successful probe does not guarantee every subsequent request. The inspect
commands have been tested on a fresh local restore. Both later one-byte probes
passed but the actual bulk retry failed; a 64 KiB prefix passed and the next
ETag-pinned range failed too. Successful probes do not establish recovered bulk
access or a safe range workaround. Successful complete resumption remains untested.

Do not rerun write-once preparation or overwrite the existing `qualification/`
record. After complete acquisition, a separate complete-population qualification
must decode teacher support, report every failure/class stratum and admit the
intended cohort. Transport-unavailable inputs must never become blank targets,
and the readable subset must not replace the frozen design.
