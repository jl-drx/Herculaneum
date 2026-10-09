# Phase 3: bounded larger surface methods and budget

Recommend a **1536 × 1536 native surface context**, selected and frozen using geometry/data coverage before inspecting its detector maps. At nominal 8.64 µm planar sampling this spans about 13.27 mm per side. Retain all 31 depth layers and the source/mesh provenance, Boolean geometry mask and immutable input hashes.

The prepared standalone adapter is `/workspace/herculaneum-lab/run_ink_large.py`. The existing `run_ink_baseline.py` remains unchanged, SHA-256 `2cbdb280777e19794fc78857969f9c1d43965da17772f6a1659264e6f5a48a59`. The new source is SHA-256 `54170b298c716c4b8cb56da539bb74812a20cb8666888d02ddaf2984ddea3fbf`; [the static source audit](large-adapter-source-audit.json) records its relation to the original.

## Changes and preservation

The separate adapter raises the maximum ROI dimension from 1024 to 2048 and adds an explicit scheduled-tile budget. Default `--max-total-tiles 1200` permits two checkpoints over a 1536-square input: 1058 scheduled checkpoint tiles. The hard budget cap is 1922; two checkpoints over a 2048-square input require an explicit `--max-total-tiles 1922`. Budget guards count every scheduled tile before mask/empty-source skips, so sparse inputs do not evade the declared compute limit.

Only dimension/budget guards, help text and related metadata changed. AST comparisons confirm **identical** source functions for checkpoint hashing/loading/strict matching, upstream-source validation, model construction, Hann weighting, normalization/inference, mask/empty-tile handling and score summaries. The scheduler's range and final-edge alignment body is identical after its enlarged dimension guard. No imports, model forward passes, scientific inference, dependency changes or expensive dummy tests were run in this preparation.

The adapter still reconstructs each checkpoint's exact embedded config, verifies pinned checkpoint bytes, loads with `weights_only=True`, requires a strict state match, and uses CPU FP32, four threads, one 128-pixel patch per batch, the unmodified upstream normalization and unquantized float outputs. Existing source files, results and the upstream modules remain untouched. The inherited metadata's central 128-pixel review region is only a small diagnostic window; the phase 3 common analysis domain must be computed separately as specified below.

## Compute and memory estimate

The earlier 980-pixel experiment scored **173 of 225 scheduled tiles per seed**, taking 30.93 seconds for seed 42 and 27.77 seconds for seed 43. Small fully scored contexts observed approximately 0.16–0.20 seconds per tile. Dividing that runtime by 225 rather than 173 scored tiles would underestimate the cost of an all-valid larger ROI. These are observed local timings at four threads; projections are estimates, not runtime guarantees.

| ROI | Tiles per checkpoint | Native uint8 input | Per-seed inference at 0.16–0.20 s/tile | Two seeds |
| --- | ---: | ---: | ---: | ---: |
| 1024² control | 225 | 31 MiB | 36–45 s | 72–90 s |
| **1536² primary** | **529** | **69.75 MiB** | **85–106 s** | **170–212 s** |
| 2048² alternative | 961 | 124 MiB | 154–192 s | 308–384 s |

Allow additional process/model-load and output time. The environment's CPU quota is four cores, so run four-thread inference processes sequentially. The current memory limit is 32 GiB with ample available memory. Model parameters alone are about 132 MiB; checkpoint deserialization and construction temporarily retain another parameter copy. Activation memory stays bounded by the unchanged single 17 × 128 × 128 patch. Per-map float32 storage is 9 MiB for 1536² and 16 MiB for 2048²; two score maps, mean and difference occupy 36/64 MiB, plus masks and accumulation arrays. Reserve roughly 2 GiB for one process as a conservative operational budget; actual peak process memory has not been measured.

## Frozen controls

1. Run both verified final seed checkpoints over the complete 1536 context, forward depth window `7:24`.
2. Before scores are inspected, freeze the **central 1024 × 1024 control crop**, offset `(256,256)` from the primary origin. Preserve all 31 layers, exact crop provenance and the corresponding mask.
3. Run seed 42 on that fixed crop at `5:22`, `9:26`, and reverse `7:24`. These are predefined depth/orientation diagnostics, not three additional independent validators. The unchanged original runner supports these 1024-square commands.

This totals 1733 scheduled checkpoint tiles: 1058 primary plus 675 controls. At the observed rate, inference is approximately 4.6–5.8 minutes, with startup/output overhead bringing the operational estimate to roughly 5–6 minutes. This bounded diagnostic budget is justified before moving to 2048 or changing the model. No score-dependent crop, threshold or favorable control should replace the frozen plan.

The primary source origin should align to the native 64-pixel tile grid. Adding 256 retains that phase. The primary 1536 dimensions and central 1024 dimensions also land exactly on stride 64, avoiding special final-edge phase changes. Score comparisons use an area at least 64 pixels from the central crop edge: an 896 × 896 core, source offset `(320,320)` from the 1536 origin. In this area the control's forward-window patch context matches the primary run; unrelated context/normalization changes should not be mistaken for depth effects.

## Analysis domain and support limits

Use a common domain across compared conditions: geometry-valid pixels, evaluated masks from every compared map, the declared selected-depth observed-data convention, a 64-pixel crop-edge margin, and a 64-pixel distance from invalid geometry computed on the **original global geometry mask**. For comparisons across depth windows, require observed support under every relevant window and document its definition. Nonzero intensity is an operational data-present test; it is not an ink label or an independent physical authenticity test.

The margin does **not** guarantee that every contributing 128-pixel patch contains only valid/observed material: a patch can reach farther than 64 pixels from an output pixel. The joint analysis will therefore report a stricter support domain that excludes pixels affected by any contributing scored tile containing invalid geometry or missing selected-depth support. Keep both primary and strict-support domains explicit; do not silently alter normalization or scheduling to manufacture agreement.

Retain separate raw seed maps, seed disagreement, all fixed controls and their common-domain summaries. Any 0.5 midpoint count remains a behavior diagnostic without target labels. A larger response is not a transcription: require stable source-supported stroke geometry and independent review before proposing letters or translation.

Example primary command after the frozen input/mask/provenance exist:

```bash
python /workspace/herculaneum-lab/run_ink_large.py SURFACE.npy \
  --provenance PROVENANCE.json --mask VALID_MASK.npy \
  --output-dir FRESH_OUTPUT_DIRECTORY \
  --seed42-checkpoint /workspace/herculaneum-lab/data/models/ink9um-seed42-step075000.pth \
  --seed43-checkpoint /workspace/herculaneum-lab/data/models/ink9um-seed43-step075000.pth \
  --layer-start 7 --direction forward --threads 4 --max-total-tiles 1200
```
