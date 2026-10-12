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

The expected pending counts are 16,649 added requests and 381 original requests.
Their remaining nominal bodies total 6,248,165,613 bytes. Allow space for them
in addition to this checkpoint, receipts, later derived targets and the runtime;
do not download every historical LFS archive. No model weights are in this file.

Only after public S3 transport recovers, resume the **same** request manifests:

```bash
python /tmp/plinius-admission-review/code/continue_acquisition.py acquire \
  --base /tmp/plinius-admission-review/base --work /tmp/plinius-admission-review/work \
  --attempt network-restored-01
python /tmp/plinius-admission-review/code/continue_acquisition.py acquire \
  --base /tmp/plinius-admission-review/base --work /tmp/plinius-admission-review/work/original \
  --attempt network-restored-01
```

Each invocation first verifies existing files, saves an attempt record and
requires a successful bounded source probe before four-reader bulk acquisition.
A new attempt identifier prevents overwriting its history. Existing successful
receipts are reused; source validators and scientific eligibility are unchanged.
A successful probe does not guarantee every subsequent request. The inspect
commands have been tested on a fresh local restore; successful bulk resumption
after recovery remains untested. The earlier actual recovery attempt stopped
on the documented proxy error and is not described as successful resumption.

Do not rerun write-once preparation or overwrite the existing `qualification/`
record. After complete acquisition, a separate complete-population qualification
must decode teacher support, report every failure/class stratum and admit the
intended cohort. Transport-unavailable inputs must never become blank targets,
and the readable subset must not replace the frozen design.
