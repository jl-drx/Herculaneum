# Ninth smoothing control: independent completed-output review

PASS. The single offline numerical/checkpoint review agrees with the analyzer and verifies the intended historical matching. Both mitigation screens and both baseline-retention margins fail. Smoothing improves mean ranking agreement in both pairs, but its operating-point effects vary by arm and region; the predeclared shared-mitigation flag is false.

The executed control plan SHA256 is `9d72d3177d5aa68cd5d59db6f316ed1a5c335f442b6c5c2439f45bfdaebe29cd`. The reviewed analyzer metrics SHA256 is `5b70146072474d00998bd82dc164c0cf0e6cedf4a570418dedaeb5066f11cad0`. The [independent receipt](result/independent-verification.json) records all 252 input/output identities checked before and after the review. Every size and SHA256 stayed unchanged. No benchmark metric code was imported; ranking uses the pinned sixth independent pair-concordance/tied-group AP implementation.

## Matching and actual updates

There are **64 actual new updates**, 32 each for A_smooth and B_smooth. The completed eighth A/B records contribute 64 historical paired updates by literal reuse; they were not executed again. Historical frozen/A/B checkpoints, score maps, evaluated masks, result records and threshold locks retain their original identities. The eighth C arm is excluded from the ninth optimization comparison.

All five compared roles share the actual seed42 initializer semantic hash `21c186000b9b3093835e185342668acd0d86147a719c37513e937924b77ea228`. The new plan preserves the exact eighth inputs, teacher maps, participation masks, schedule, calibration regions, normalization, optimizer and runtime settings. Every historical and new per-step patch/D4 choice and class-exposure receipt matches the fixed schedule. Each arm sees repeated manual exposures of 230,340 positives and 387,868 negatives, and teacher-support exposures of 40,620 positives and 389,748 negatives. The teacher term has coefficient zero in manual-only arms. These counts contain repeated cells rather than independent examples.

Both new first-step fingerprints equal `571d10c58008cb6f9e581a717bbb067846dae6a2ee2d29bdf9eb141ca9a02ec3`, independently reconstructed from the pinned raw source, unchanged NumPy normalizer, D4 transforms, hard masks and targets including ignored-target sentinels. Initializer, first-step schedule and unsmoothed U/P scalar receipts agree with the historical counterparts within the fixed 1e-7 tolerance. The recorded smoothed softplus check agrees with the saved first-step losses within 1e-6. It is evaluated on the same actual first forward before its existing backward/update, without an extra model pass or optimizer step.

The saved coefficient arithmetic is exact: A_smooth=U_s and B_smooth=U_s+0.25P_s, with epsilon 0.5 and participating targets 0.25/0.75. All 64 new update probes change, unknown-output gradients are recorded as exactly zero, and the masks retain hard membership. Independent artificial-logit fixtures on the final frozen loss source establish the direct softplus value/gradient formula, exact epsilon-zero equivalence and ignored-gradient zeros. Raw actual model logits were not retained, so the completed-output review does not independently recompute full numerical BCE losses; it checks matching scalar/analytic receipts, source implementation and coefficient arithmetic.

## Saved state verification

The initializer and two new final states were deserialized serially with CPU `weights_only=True`, after fixed serialized/expanded ZIP bounds and SHA256/size checks. One unchanged CPU factory construction supplied authoritative canonical parameter and buffer names without a forward call. All final states preserve the exact 508 tensor keys/shapes/dtypes, 264 unique storages, 244 alias groups, strides, offsets and allocation sizes. All values are finite and aliases agree bitwise. Unique state storage is 138,185,992 bytes beneath the fixed 160 MiB cap. All 264 canonical parameter tensors change in each new final state; this counts tensors containing changes, not changed individual elements. Semantic state and parameter hashes agree with the worker receipts.

Historical final tensors are supported by their unchanged raw checkpoint hashes and the immutable eighth independent PASS proof. They were not re-deserialized. The independent review establishes strict schema compatibility; the workers' recorded persisted roundtrip and strict reload are their own verified operations.

## Numerical result

The two reused supplied-supervision diagnostic domains contain 49,033 cells with 19,545 positives, and 21,328 cells with 4,767 positives. Common masks, prevalence, raw score quantiles, AUC/AP with tied-score handling, midpoint counts, primary/secondary confusion counts, transfer flags, equal-context summaries, factorial contrasts and protocol screens are reconstructed from source masks and raw scores. The maximum difference from the analyzer is `2.220446049250313e-16`; saved loss-coefficient arithmetic has zero difference.

Every local calibration threshold is reconstructed from the negative-score crossing boundary and next float32 value, excluding the entire boundary tie. New thresholds follow the same maximum-of-three primary rule and w029-only secondary rule, and their receipts are written before diagnostic inference. Historical rules/maps are reused unchanged. The conservative development guarantee does not imply the same operating behavior on another region.

| Role | Mean AUC | Mean AP | Primary sensitivity | Primary specificity |
|---|---:|---:|---:|---:|
| Frozen seed42 | 0.999608 | 0.998510 | 0.903171 | 0.999668 |
| A, epsilon 0 | 0.989429 | 0.976560 | 0.688869 | 0.998946 |
| B, epsilon 0 | 0.987941 | 0.979877 | 0.771325 | 0.997219 |
| A_smooth, epsilon 0.5 | 0.994717 | 0.988662 | 0.736300 | 0.991793 |
| B_smooth, epsilon 0.5 | 0.995598 | 0.990528 | 0.690585 | 0.997592 |

A_smooth minus A mean AUC/AP is +0.005287/+0.012101. Mean primary sensitivity improves by 0.047431, but the per-region changes are +0.147096 and −0.052234. The rank14 sensitivity loss exceeds the allowed 0.02, so its mitigation screen fails despite the favorable average.

B_smooth minus B mean AUC/AP is +0.007657/+0.010652. Mean primary sensitivity falls by 0.080740; its per-region changes are −0.020721 and −0.140759. Its mitigation screen fails. Both smoothed arms meet the specificity target in each diagnostic region but fail the specified frozen-baseline retention margins.

The primary sensitivity B-minus-A effect is +0.082456 without smoothing and −0.045716 with smoothing. Their difference is −0.128172. This descriptive interaction and the individual region failures agree with the saved analyzer. Both smoothed models still have lower mean AUC, AP and primary sensitivity than the frozen reference. Ranking improvement alone does not satisfy the predeclared operating-point screens.

## Interpretation and integrity

These are previously examined development retention diagnostics, with one initialization, two adaptation contexts and two diagnostic contexts from the same PHerc0814 specimen. The fixed baseline/teacher configurations include the source representation; exact historical pixel and teacher exposure remain unresolved. Matching the declared BCE smoothing convention does not recreate the original combined Dice/BCE objective or complete trainer. No fresh holdout, independent ink accuracy, calibrated ink confidence, statistical significance, physical independence, authentic reading or novelty is established.

The result supplies mixed smoothing/operating-point evidence under this fixed continuation recipe. It does not pass the shared-mitigation criterion or support promotion of either adapted model. It cannot establish a general causal mechanism from one seed or choose a new sampling, optimizer, replay or threshold recipe.

The verifier executed once and exited zero. Only additive ninth-review files were created; all original and frozen inputs, previous studies and result records remain unchanged. No real model forward/backward, training, inference or network call occurred during completed-output verification. The pre-run fixture's permitted autograd used artificial logits only. No further edits or verification runs are planned.
