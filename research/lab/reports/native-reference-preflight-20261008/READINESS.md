# Readiness and next research step — 8 October 2026

**Proceed with free CPU research and bounded public-source validation. The paid GPU pilot is not launch-ready.**

| Capability | Evidence and limit |
| --- | --- |
| Existing 9 µm CPU pipeline | Previously validated on genuine bounded source data, with strict checkpoint matching and source-linked outputs. No new model inference occurred during this source audit. |
| Private R2 artifact persistence | Ten original local publication checks passed at `e9e7db9`; six scoped temporary-credential checks followed. Live read-only verification of the retained two-artifact receipt now also passed using the updated constructor at `49fabf3e890d5b880f391b27d581f1e7a1ebdbcc`. Credentials remain local; this cloud process cannot sign R2 requests. |
| Runpod account access | Local OAuth and a read-only Pods listing succeeded. No provider tools are connected in this cloud process. |
| Future worker secret delivery | Bucket-only temporary triplet scope and local backend access verified. Runpod import is NONE; console account/team and boot-time delivery remain unverified. Local browser automation could not confidently establish its Chrome URL. No worker exists. |
| Native PHerc1667 ablation | Pinned configuration/model source identity and local interpolation behavior verified. Original `l_2`/`l_5` inputs and historical preprocessing/runtime lineage remain unresolved. |
| Paid launch | Existing operational validator returned `ready=false`, exit 2, with unresolved identities, checkpoint payloads, image digest, quote and independently verified shutdown. `launch_performed=false`. |

## Concrete scientific progress

Both pinned configuration sources are 1,742 bytes and match the expected Git blob and SHA-256. This closes source verification and static review, not the full runtime gate. The wrapper applies no input normalization itself.

Paper-pinned training code with `Normalize(mean=0,std=1)` supports a `/255` candidate; a separate optimized inference pipeline explicitly uses `/200` and different blending. The exact original ablation invocation has not been linked to those paths. Current example metadata's `l2_0_crop` belongs to PHerc139, so it cannot establish PHerc1667's `l_2` identity. Do not pick a preprocessing path because its predictions look more like writing.

The existing CPU environment lacks Transformers and Albumentations. The source audit did not alter it, install packages, download weights/scans/labels, train a model, or produce a new ink result. See `PREFLIGHT.md`, `status.json` and the retained immutable source manifests for evidence.

The separate public-source audit inspected 19 current native surfaces. Their chunk shape is `[109,128,128]`; a chunk-aligned 1024-square crop requires 64 complete chunks and 109 MiB decoded, even if only 62 depth planes are used. Eighteen sources are uncompressed; w013's compressed wire size still needs verification. This establishes a useful metadata-derived staging bound, not the original paper's input identity or an acquired crop. See `source-preflight.md` and its immutable metadata manifest. The two audits consumed at most 785,656 response-body bytes combined, with no scan chunk or weight acquisition.

**Citation correction:** the pseudo-labeling ablation is Extended Data Figure **7**, not Figure 6. The original frozen repository specification still contains the older citation; record this correction in its next reviewed version rather than treating the frozen plan as already updated.

## Next work

The next bounded CPU milestone is an **offline geometry-selection and physical-overlap audit** for one new 384×384 PHerc0800 screening crop, using the five retained XYZ maps and validity masks. Exclude previously acquired coverage, compare physical CT support, and freeze coordinates before acquiring intensities or reading predictions. This is a proposed audit, not a completed selection or new inference result. If no acceptable crop exists, retain that outcome rather than relaxing the criteria after viewing predictions.

For a chunk-aligned 384-square crop under the retained `[31,128,128]` uncompressed uint8 source contract, acquisition would require at most nine payloads (4,571,136 bytes, approximately 4.36 MiB) plus metadata. The existing stride-64, 128-square inference schedule would require 25 tiles per checkpoint, 50 total. Prior measurements suggest roughly 8–10 seconds of inference plus startup with four CPU threads; this is an estimate, not a new benchmark. Source hashes and coverage/physical-overlap checks must be completed first. Independent physical coverage is not yet established, and no target labels are available. A small crop is a screening step; it cannot by itself establish coherent text or target accuracy.

Preserve the blocked native ablation specification until original source aliases, render/depth correspondence and preprocessing lineage can be established. A current public native render is not automatically the original paper input. No substitute region, layer offset or transform should silently convert the plan into a purported reproduction.

This does not prevent continued work with the already validated eligible PHerc0800 9 µm pipeline. Any further search must use newly frozen geometry/source-coverage selections, retain source coordinates and masks, separate observed source from declared fill, and keep novel output private for review. Current target results still support no authenticated letterforms, transcription or translation.

Use the local task for authenticated account operations; `../runpod-setup/COORDINATION.md` records the exact task identities, current one-way send capability and pending worker-secret request. No budget or local setup permission needs repeating.

The £10 TOTAL first-pilot cap remains aggregate across failed attempts and attributable costs. The chosen pilot's scientific/runtime blockers, a pinned CUDA environment, a current conservative quote and independent shutdown must be settled before any paid launch. Preparation or a local OAuth result is not authorization to skip those prerequisites.
