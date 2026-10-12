# Strong native baselines: inputs prepared, inference unrun

This page records the completed input-preparation phase. The subsequent
[qualified comparison](../native-common-comparison-20261012/README.md) started
on 12 October; its execution and results are recorded separately. The input
packet and the failed original panel remain unchanged.

12 October 2026. This packet prepares the fixed Reader v2 / dense-native /
Hecate / seed42 / seed43 development comparison. The native0139 worker has
finished; no training or inference worker is running for this packet. No GPU,
paid compute, Actions workflow or spare-VM job was started.

## Main finding and design correction

The planned union-mask selection failed admission: all 18 PHerc0841 cores had
zero positive reference pixels, across 169,462 native-supported pixels. Its
validation area is much broader than its supervision area, so spatial spreading
over their union missed the annotated text. This is a reference-support design
failure, not an ink-model result. The original panel, exact coordinates and
receipts remain unchanged in [original-union-panel](original-union-panel/).

Before any inference, a separately frozen [supervision panel](SUPERVISION_PANEL.md)
used the same four segments, fixed grid, minimum coverage and deterministic
spreading rule, with supervision alone supplying selection and scoring support.
Its selector did not consume ink values, CT intensities or model predictions.
The change was informed by the first panel's class counts: this is an adaptive
development correction, not untouched evaluation. PHerc0009B's six coordinates
remain identical; all 24 resulting cores are retained.

| Segment | Scoring pixels | Reference positives | Cores with both classes |
|---|---:|---:|---:|
| PHerc0841 / 20260220213127 | 68,067 | 465 | 2 / 6 |
| PHerc0841 / 20260220214732 | 47,555 | 2,699 | 3 / 6 |
| PHerc0841 / 20260221022814 | 61,167 | 7,386 | 2 / 6 |
| PHerc0009B / 20250919125754 | 55,461 | 9,587 | 3 / 6 |
| **Total** | **232,250** | **20,137** | **10 / 24** |

All four segments support AP/AUC calculations; 14 individual cores are one-class
and must have undefined ranking metrics. Positive prevalence varies from 0.68%
to 17.29%, so report individual segments and equal-parent aggregates. This small
panel does not establish a universal model winner. It compares agreement with
transferred/model-reviewed references, not independently authenticated ink.
Physical registration and upstream model exposure remain unresolved. The fresh
PHerc0175 A/B and PHerc1203 families remain unopened.

## What passed

- All twelve annotation shards were hash/length/ETag checked. CRC32C/index/codec
  validation and independent installed-Zarr decoding agreed; S/V encodings are
  binary 0/255. The two published union orders differ on zero native pixels.
- The initial raw CT 404 was investigated before accepting fill. Complete source
  inventories and pinned Zarr metadata establish 38 canonical-fill chunks in the
  original panel. The corrected panel has 216 retrieved source chunks and no
  omitted chunks; 56 were reused from verified cache. Across both panels, new CT
  response bodies total **157,712,384 bytes**. No fine CT or full scan was fetched.
- Six full-canvas synthetic interpolation checks preserve input/output support
  exactly, and both grid schedulers match their pinned upstream functions.
  Hecate uses linear XY interpolation at 9.366um and area interpolation at 8.64um;
  neither receives a locally reset grid. Plane-wise preparation matches the
  pinned vectorized algorithm byte-for-byte in both synthetic spacing controls.
- [Read-only verification](supervision-panel/VERIFIED_INPUTS.json) checked all
  24 score masks against native CT, every source receipt and **1,050 patch files**.
  All 28/31 source planes remain available; the model recipes select their own
  declared windows. No physical-surface alignment claim follows from these checks.

The initial 404 and one premature inputs invocation are recorded. The latter
stopped before opening ink because CT completion was absent; it was rerun after
completion. No failed attempt was silently treated as a successful download.
See [CT fill amendment](CT_FILL_AMENDMENT.md) and
[network accounting](NETWORK_ACCOUNTING.json). `CT_COMPLETE.downloaded_body_bytes`
counts associated retrieved objects including cache; the accounting report uses
the durable ledgers for actual new network bytes.

## Next execution

The fixed workload is **1,698 nonempty forwards**: 216 each for the four hybrid
models, plus 834 for Hecate. Retained CPU measurements give an arithmetic estimate
of **47 minutes**; use **40–60 minutes** for planning, not a runtime promise.
The Reader timing proxy is deliberately conservative and comes from a different
Python/PyTorch environment. See [COMPUTE_PLAN.json](COMPUTE_PLAN.json).

Run this bounded inference comparison here on CPU after adapting and qualifying
the existing resumable runner for the new multi-segment patch packet. Before the
first forward, pin that runner and its actual runtime/checkpoints, verify its
sparse accumulation against the original pipelines with synthetic logits, and
freeze the execution receipt. Retain all model/core outputs and tile-level
checkpoints. This packet does **not** claim that runner integration, real-model
output parity, or the five-model comparison has already executed.

No GPU is needed for this small baseline comparison. This choice does not impose
a CPU-improvement gate on adequately powered training. A matched broader-coverage
GPU pair remains a separate proposal requiring current pricing, pinned CUDA,
secure artifact delivery, independent shutdown and explicit user approval within
the standing £10 total first-pilot cap. No private credentials are in this packet.

## Portable private input bundle

The private LFS bundle is
`research/workspace/bundles/native-common-support-inputs-20261012.tar.gz`.
[BUNDLE.json](BUNDLE.json) binds its SHA256, size and manifest. It includes both
panels, raw acquired chunks, canonical-fill receipts, annotation shards, global
model patches, code and metadata. Duplicate bodies use internal tar hardlinks.
It excludes model weights, previous unrelated experiments and inference results.
[MODEL_PINS.json](MODEL_PINS.json) retains exact checkpoint identities; its local
paths describe the current cache, not portable download locations.

[Local restoration](LOCAL_RESTORE.json) verified all 2,369 files and reproduced
the input report byte-for-byte. A [fresh private GitHub LFS fetch](REMOTE_RESTORE.json)
returned the identical archive. These publication receipts were added after the
immutable input archive was built.

In a clone of this private branch, fetch only this LFS object:

```bash
git lfs pull --include=research/workspace/bundles/native-common-support-inputs-20261012.tar.gz
python research/in-progress/native-common-support-20261012/verify_bundle.py \
  --archive research/workspace/bundles/native-common-support-inputs-20261012.tar.gz \
  --expected-sha256 SHA_FROM_BUNDLE_JSON --output /path/to/new/extraction
```

The bundle verifier uses the Python standard library (Python 3.12+). For the
read-only numerical replay, install the pinned `requirements-inputs.txt` in an
isolated environment, then run the extracted verifier:

```bash
python /path/to/new/extraction/supervision-pilot/implementation/verify_inputs.py \
  --work /path/to/new/extraction/supervision-pilot --output /path/to/new/replay.json
```

This replay makes no network request and runs no model. It needs a few GB of RAM;
the other 9GB VM is adequate. No delegation or workflow has been requested for it.
