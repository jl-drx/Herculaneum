# Completed PHerc343 maps and review

The unchanged eight-map CPU run and four-model composition finished on10 October
2026 at **14:51:20 BST**. Output integrity and all-region review passed. The result
qualifies the published-image positive control for planning PHerc0800 transfer;
it does not establish ink AP, raw-author-map equality or authenticated letters.

[Full assessment of Sam's follow-up and all18 regions](../../../reviews/sam-pherc343-followup-20261010/README.md)
includes the replay, independent distance check, Hecate phase sensitivity and
proposed bounded grid-origin diagnostic. No new model inference was run in the
completion review. GPU spend remains£0.

The original displayed panel has RGB MAE0.0180978 on the0–255 scale and yellow-mask
IoU99.8283%. All18 previously inspected boxes reproduce closely (MAE0.01587–0.07309;
maximum difference5). These are display comparisons on an exposed reference.
The global difference166 lies on a cyan row annotation; the original metric is
retained unchanged. All60 saved checkpoint prefixes match the final Hecate map.

## Download just the completed results

On branch `codex/plinius-pherc343-reproduction`, with Git LFS installed:

```bash
git lfs pull --include="research/workspace/bundles/pherc343-completed-20261010-001.tar.gz" --exclude=""
python3 research/in-progress/pherc343-reproduction/completion-20261010/restore_results.py \
  --bundle research/workspace/bundles/pherc343-completed-20261010-001.tar.gz \
  --destination /tmp/pherc343-completed
```

Use a new destination. The standard-library restore checks the archive and each
member before accepting the result. `RESTORE_VALIDATION.json` records a successful
local relocation/readback of all116 files. The supplied checker then passed all
eight relocated maps and all four merges (`RESTORED_MAP_VERIFICATION.json`). This
is not a separate-machine test. `REMOTE_VERIFICATION.json` also confirms an
actual GitHub LFS download through a new empty object store, with matching archive
length/SHA256 and all40 submitted review files byte-identical in Git.

- Compressed download: **90,304,265 bytes** (about86.1MiB).
- Expanded files: **140,762,172 bytes** (about134.2MiB).
- Archive SHA256: `a581f5cd8ab38b8e31e12e93d2b699305fefdd7372498278f1e2ac4c64a60e27`.
- `BUNDLE.json` contains every path, size and SHA256; the old workspace snapshot
  and all its existing bundle objects remain unchanged.

The bundle includes all eight raw maps and receipts, four ownership merges,
the final annotated PNG, CT and ink component PNGs, sidecars/CSV, reference PNG
and sidecar, ownership source, `valid.npy`, frozen recipes, completion receipts,
pipeline logs and checkpoint metadata. It excludes CT volumes, model weights,
installed environments and redundant prefix arrays. Full inputs remain in the
earlier selectable workspace snapshot. No credentials are included.

The final map is restored to:

`reports/fortysixth-experiment/work/image/concat_w047-w048_R5B2_z9500-11000.png`

The small bundle suffices for the supplied eight-map/merge checker and saved-map
review. The independent CT-layer audit additionally needs the original rendered
volume. Historical absolute paths in receipts remain evidence, not new-machine
paths; pass relocated directories explicitly to the read-only review tools.

## Record scope

This is an append-only completion supplement. The original phase46 working
snapshot manifest and phase45 cumulative seal/register remain historical records.
`PIPELINE_STATUS.json` is the final producer receipt; its name
`COMPUTE_COMPLETE_AWAITING_REVIEW` predates the separately completed review and is
preserved unchanged. `integrity-check/VERIFICATION.json` likewise records its
then-current scope; the linked later review resolves the all-region review item.
The cumulative research-register update remains separate from this completed,
verified output package.

No paid GitHub Actions run, GPU resource, R2 upload or public publication was made.
The archive is retained through private repository LFS, using the existing
authorized research-backup workflow.
