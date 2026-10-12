# Restore the completed original stream and current partial added cache

**Current availability: local only.** The source commit and archive are prepared
and locally verified, but the GitHub push failed with the same proxy HTTP 503.
[PERSISTENCE.json](PERSISTENCE.json) records the failed operation and the last
acknowledged remote commit. The new selective-download instructions below become
usable on another machine after a successful push and remote availability check;
they are not a claim that the new archive is already downloadable from GitHub.
The original 490.9 MiB base and earlier 19.0 MiB addendum were previously uploaded
and verified. Preserve the current local source cache and new archive meanwhile.

This is a **cumulative update against the original 490.9 MiB checkpoint**. It
includes the earlier 19.0 MiB addendum; do not apply both addenda to the same base.
It is not the complete matched training dataset. No inference, paid runtime,
GitHub Actions job, R2 credential or spare VM is required for numerical review.

The new archive is **344,986,026 bytes (329.0 MiB)** with 2,022 payload files.
Its SHA256 is
`4f70a9640ff62a48435398d7c13603009f077dec2436ced5092088abce02c4c0`.
[BUNDLE.json](BUNDLE.json) pins its manifest and the required base identity.
All new source objects, receipts, continuation records and complete original
qualification are included. Models and credentials are excluded. Keep private.

Use the private `jl-drx/Herculaneum` branch `codex/plinius-pherc343-reproduction`.
For a new checkout, set `GIT_LFS_SKIP_SMUDGE=1` when cloning so old archives are
not all downloaded. From that checkout, fetch only these two objects:

```bash
git lfs pull --include='research/workspace/bundles/native-coverage-admission-partial-20261012.tar.gz,research/workspace/bundles/native-coverage-original-complete-20261012.tar.gz' --exclude=''
python research/in-progress/native-common-support-20261012/verify_bundle.py \
  --archive research/workspace/bundles/native-coverage-admission-partial-20261012.tar.gz \
  --expected-sha256 15580fe689f66cd9dd2d0901d3de117d8c26c00fce931390d0027be80aa9377a \
  --output /tmp/plinius-original-base
python research/in-progress/native-common-support-20261012/verify_bundle.py \
  --archive research/workspace/bundles/native-coverage-original-complete-20261012.tar.gz \
  --expected-sha256 4f70a9640ff62a48435398d7c13603009f077dec2436ced5092088abce02c4c0 \
  --output /tmp/plinius-original-update
python /tmp/plinius-original-update/code/checkpoint_delta.py apply \
  --base /tmp/plinius-original-base --delta /tmp/plinius-original-update \
  --output /tmp/plinius-original-current
```

Every output directory must be new. The update checks the base and its own
manifest before making a new combined copy. The first checkpoint and its dated
481-candidate qualification stay unchanged. Three updated operational records
are archived. Allow roughly 3 GiB for the two compressed files, their extracted
trees and the combined copy, plus the runtime. Completing added acquisition
requires another approximately 5.83 GB of source bodies and space for derived
outputs. Do not download every historical research bundle.

With NumPy installed, replay all **876 completed original candidate decisions**:

```bash
python /tmp/plinius-original-update/code/verify_original_completion.py \
  --current /tmp/plinius-original-current --output /tmp/plinius-original-verified.json
python /tmp/plinius-original-update/code/analyse_original.py \
  --original /tmp/plinius-original-current/work/original \
  --support /tmp/plinius-original-current/work/completion-20261012/qualification/ORIGINAL_SUPPORT.csv \
  --verification /tmp/plinius-original-verified.json \
  --output /tmp/plinius-original-exposure.json
```

The wrapper changes only the path view in a temporary directory. It runs the
unchanged independent verifier, pinned by SHA256, against the new dated result.
It requires an exactly matching verification record and leaves both historical
qualifications intact. The check uses saved decoded annotations and byte-verified
CT; original annotation shards and their earlier official/explicit decode checks
are preserved. It does not run a network request or a neural model. The original
environment used Python 3.14.7 and NumPy 2.4.4.

[The local restore receipt](LOCAL_RESTORE.json) verifies all 2,022 update files,
applies 2,011 paths to a fresh combined copy, and reproduces both the complete
original verifier output and exposure report byte for byte. Both source-cache
inspectors pass, and the original checkpoint still verifies unchanged. This
establishes local replay; GitHub upload and remote retrieval are separate checks.

For the earlier **481-candidate** numerical replay, use the untouched original
base restore and its `code/verify_original.py`; the combined current cache has
additional inputs and must not be used as if it were that historical snapshot.

Read-only current cache inspection requires only the standard library:

```bash
python /tmp/plinius-original-current/code/continue_acquisition.py inspect \
  --base /tmp/plinius-original-current/base --work /tmp/plinius-original-current/work
python /tmp/plinius-original-current/code/continue_acquisition.py inspect \
  --base /tmp/plinius-original-current/base --work /tmp/plinius-original-current/work/original
```

Expect **1,522 added objects present / 16,048 pending**, and **781 original objects
present / zero pending**. Earlier failure files are historical; the original
completion record and verified receipts now account for every original request.
The added dependency table was made from 1,515 receipts before the seven later
pooled-transport successes; `work/completion-20261012/QUALIFICATION_INPUTS.json`
preserves that exact receipt set. No smaller added cohort has been admitted.

Only after a meaningful transport recovery, resume the unchanged added list:

```bash
python /tmp/plinius-original-current/code/continue_acquisition.py acquire \
  --base /tmp/plinius-original-current/base --work /tmp/plinius-original-current/work \
  --attempt transport-restored-02
```

Use a new attempt identifier. The one-byte preflight does not guarantee later
bulk success. Do not rerun the failed persistent-connection diagnostic as a known
fix, overwrite dated qualifications, zero-fill unavailable data or promote the
first completed segment into a replacement study. Complete population-wide
teacher decoding, support/class accounting and coverage checks before training.

The code/data attribution and public-source CC BY-NC 4.0 metadata from the
original checkpoint remain applicable. The frozen prospective evaluation reserve
was not opened. This packet adds no external assignment or paid-work permission.
