# Restore the cumulative source retry update

This **290,148,807-byte (276.71 MiB)** private archive preserves 1,989 payload
files. It contains all source progress since the completed-original checkpoint,
including the earlier 8.25 MiB addendum and this retry's 912 new records. It adds
no full-population supervision qualification or new model result.

Archive SHA256:
`4d16cece37b134da339775306d00dbd9c8bf1df2ea6d8e416561719714f72917`.
[BUNDLE.json](BUNDLE.json) also pins the manifest and packaging implementation.

[LOCAL_RESTORE.json](LOCAL_RESTORE.json) records successful extraction of all
1,989 payload files, application of 1,980 changed paths and exact agreement
between the combined restore and the live cache. Original sources and both
dated qualifications remain byte-identical. This operational check did not
rerun the unchanged numerical experiment or involve a second machine.

Start with the combined restore from the
[completed-original instructions](../native-coverage-completion-20261012/PORTABILITY.md):
the first 490.9 MiB archive plus its 329.0 MiB cumulative update. The required
combined manifest SHA256 is
`8e170377e63744a1fb2c6d331196ed41cc816166ec23845cf92f7c5745780b7c`.
**Do not also apply the earlier 20:53 addendum.** Do not apply this update to a
cache already modified by another acquisition; the base identity is exact.

From the private repository checkout on `codex/plinius-pherc343-reproduction`:

```bash
git lfs pull --include='research/workspace/bundles/native-coverage-retry-20261012-2147.tar.gz' --exclude=''
python research/in-progress/native-common-support-20261012/verify_bundle.py \
  --archive research/workspace/bundles/native-coverage-retry-20261012-2147.tar.gz \
  --expected-sha256 4d16cece37b134da339775306d00dbd9c8bf1df2ea6d8e416561719714f72917 \
  --output /tmp/plinius-retry-2147
python /tmp/plinius-retry-2147/code/checkpoint_delta.py apply \
  --base /tmp/plinius-original-current --delta /tmp/plinius-retry-2147 \
  --output /tmp/plinius-current-after-retry-2147
python /tmp/plinius-current-after-retry-2147/code/continue_acquisition.py inspect \
  --base /tmp/plinius-current-after-retry-2147/base \
  --work /tmp/plinius-current-after-retry-2147/work
```

All output directories must be new. The helper verifies the base and all update
bytes, makes a new combined copy and preserves changed operational records under
`retry-2147-history/`. Both input directories remain untouched. Allow roughly
2 GiB of additional space for the compressed update, extracted update and new
combined copy, beyond the already-restored base. Completing acquisition still
needs another 5.51 GB of source bodies and space for derived outputs.

Expected inspection: **2,506 present, zero confirmed missing, 15,064 pending**.
Original intake remains 781 present / zero pending. Existing dated qualifications
retain their original scope; do not use the enlarged cache as if it were the
earlier partial snapshot. Original numeric replay can use the unchanged wrapper
from the 329.0 MiB update. New teacher ranges are not yet population-qualified.

After a meaningful recovery, resume `continue_acquisition.py acquire` using
these combined `base`/`work` paths and a new attempt identifier. A successful
probe alone does not establish sustained access. Retain all public-source
attribution and keep the artifacts private. This adds no paid-work permission.
