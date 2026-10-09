# Proposed zero-training CPU parity check

This is a source-audit recommendation, not an execution GO. Root must freeze its actual fixture formulas, call budget, tolerances, source pins and implementation after independent review. Do not load a checkpoint or execute historical code on the strength of this document alone.

## Prerequisites

Require the exact full safetensors byte length and advertised SHA256; unchanged phase21 header; 508 exact named F32 tensors and complete interval accounting; finite tensor values; all 244 duplicate-name pairs byte-equal; authored config pin and current/historical source hashes. Fail before model binding on any conflict. Use ordinary CPU FP32 parameters and `load_state_dict(..., strict=True)` with all keys; verify every loaded state tensor equals the corresponding input and validate the complete 264-group alias partition. A safetensors file has no config, step or semantic-state provenance; record these as separately bound source inputs.

Use the nine exact active modules in `HISTORICAL_FACTORY_CLOSURE.json` in a fresh namespace. Do not run upstream package initializers or import the historical inference entrypoint. Its optional inactive ResNet, Primus and DINO imports may be satisfied only by named fail-closed stubs under guarded fixed configuration, with zero runtime use. Historical activations must be the actual fetched old source because active block constructors compare classes to SwiGLU/GLU. Factory config normalization can be a reviewed exact extraction of historical `normalize_training_config_for_inference`, with the flat-only input guard asserted before its invocation.

Separate historical and current factories into subprocesses to avoid module namespace collision and different global state. Pin Torch2.12.0+cpu, NumPy2.5.3, CPU threads, deterministic settings and no autocast/compile. Preserve original model bytes and discard random construction state only after strict binding. Freeze no-optimizer/no-backward/no-GPU counters.

## Input arithmetic without models

A small source-bound function harness should exercise the following synthetic-only cases. Each construction must be explicit in the final protocol; no real image, pixel label or inference map is needed.

1. A 28×128×128 depth/XY-coded ramp. Assert exact selected source indices for default `[6:23)` and four published native CLI intervals `[2:19)`, `[5:22)`, `[7:24)`, `[9:26)`. Check aligned21 default `[2:19)` and all five 17-plane windows. Include reverse-after-selection as a diagnostic only, without selecting direction from output.
2. An independently allocated FP32 copy in historical HWC→ZYX strides and an ordinary C-contiguous ZYX copy. Apply old/new robust normalization and compare normalized C-order bytes. Check input mutation separately: old may alias, new must not. Percentile/median/MAD arithmetic should use the same NumPy runtime and exact source. Compare fresh copies so the first path cannot change the second path's raw input.
3. Constant nonzero and all-zero arrays exercise scale fallback and nonempty scheduling separately. An asymmetric sparse crop exercises MAD≈0/preclip-standard-deviation fallback. A small spatial crop and a less-than-17-depth crop exercise zero insertion *before* inference normalization, while a training-style valid-slices reference leaves padding zero. Do not expect training/inference equality at padded boundaries.
4. A synthetic two-tile probability/logit example checks FP32 sigmoid then probability-space Hann blending. Verify nonperiodic128 Hann weights, maximum normalization and .001 floor. Test byte-output quantization and the author's `clip((uint8−64)/128,0,1)` decode separately from raw probabilities. Include near-threshold values that visibly create ties after quantization; do not test identity of the two score scales.

Windows, masks, positions, alias identities and serialized source pins require exact equality. FP32 normalized input bytes should be exact for the source-defined same-layout path; any failure must be recorded before choosing whether a new investigation is warranted. The final model comparison tolerance must be root-frozen before outputs; this audit proposes comparing raw logits directly under identical contiguous input and same CPU kernels, and recording both bitwise identity and maximum error rather than silently relaxing on failure.

## Minimal model calls

Two fixed synthetic model inputs are sufficient for an initial narrow smoke comparison: one normalized constant/zero input and one deterministic depth/XY-asymmetric nonconstant input, each FP32 `[1,1,17,128,128]`. Run each once through each factory, two forwards per factory / four total. Match complete strict-loaded states and resolved architecture first. Retain output logits and hashes, output shapes, finiteness, model state hash before/after, mode, counts and stub-use receipts. Compare full logits at the root's predeclared tolerance. This is not a model-quality test; neither path should be treated as a teacher.

If root wants to test historical reader memory layout effects, that is a separately counted additional same-input forward for each factory. Do not hide extra calls inside a fixture or expand to native pixels/regions. Similarly, full-frame occupancy, Zarr fetching, TIFF writing, ensemble inference and historical CUDA2.4.1 reproduction are beyond this minimal parity check.

## Interpretation

A pass supports source-bound compatibility of these two fixed factory paths and fixtures under the current CPU runtime. It does not validate native physical plane coordinates, historical training provenance, teacher independence, transfer accuracy, readable ink or a new Plinius model. A failure must stop the parity stage; retain the failed fixed test before investigating a revised protocol. Keep seed42 and previous development conclusions unchanged.
