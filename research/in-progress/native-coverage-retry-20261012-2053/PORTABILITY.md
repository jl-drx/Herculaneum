# Add the latest 72 source ranges

This optional **8,654,575-byte (8.25 MiB)** archive adds the latest complete teacher
ranges and receipts, and preserves the failed retry. It does not change either
dated original-stream qualification or admit the full added population.

**Availability: private GitHub, verified.** The addendum was published at
`62151d37f93b76cef67b07ee3be8aa3d841b6d0c`. A fresh selective download into
an empty LFS cache matched its length and SHA256; see
[REMOTE_RESTORE.json](REMOTE_RESTORE.json).

[LOCAL_RESTORE.json](LOCAL_RESTORE.json) records a fresh extraction of all 159
payload files and application of 151 changed paths to a new combined copy. Its
cache matches the current workspace, and the original sources and both dated
qualifications remain byte-identical. This is an operational restore check;
the unchanged numerical experiment was not rerun.

Start with the combined current restore produced by the
[completed-original instructions](../native-coverage-completion-20261012/PORTABILITY.md):
the original 490.9 MiB archive plus the 329.0 MiB cumulative update. Do not apply
this small update to the original partial restore alone, or to a cache already
modified by another acquisition. Its base manifest is pinned exactly.

From the private repository checkout:

```bash
git lfs pull --include='research/workspace/bundles/native-coverage-retry-20261012-2053.tar.gz' --exclude=''
python research/in-progress/native-common-support-20261012/verify_bundle.py \
  --archive research/workspace/bundles/native-coverage-retry-20261012-2053.tar.gz \
  --expected-sha256 c92d28da57d3042dedf26dc32d5a38cab20f5d90a6f80a3122bad41c93e795e7 \
  --output /tmp/plinius-retry-2053
python /tmp/plinius-retry-2053/code/checkpoint_delta.py apply \
  --base /tmp/plinius-original-current --delta /tmp/plinius-retry-2053 \
  --output /tmp/plinius-current-after-retry
python /tmp/plinius-current-after-retry/code/continue_acquisition.py inspect \
  --base /tmp/plinius-current-after-retry/base --work /tmp/plinius-current-after-retry/work
```

The helper verifies all base and update bytes, makes a **new combined copy**,
archives the two changed operational records under `retry-2053-history/`, and
creates a fresh full manifest. Both input directories remain untouched. Allow
roughly another 1 GiB for the combined copy. Expected inspection: 1,594 present,
zero confirmed missing and 15,976 pending requests. Original intake remains
781 present / zero pending.

All scientific results still belong to their original dated inputs. The 876-row
completed-original verifier can use this new combined directory through the
wrapper from the 329.0 MiB cumulative update; original sources are unchanged.
The 481-row historical replay still requires the first untouched partial restore.
Do not interpret additional teacher bytes as qualified or independently correct
ink labels. Keep the source attribution and all artifacts private.

After source transport recovers, use `continue_acquisition.py acquire` with the
new combined `base`/`work` paths and a **new attempt identifier**. The request
manifest, original intake protocol and cache identities stay unchanged. A
successful probe alone does not establish that the subsequent bulk request will
succeed. This packet authorizes no paid resource or replacement cohort.
