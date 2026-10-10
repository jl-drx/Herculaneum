# Retained objective and method evidence — 9 October 2026

**Nonmanual prediction KL has already been optimized; a clean preservation-coverage comparison has not been identified in the retained studies.** Phases 10 and 13 used broader KL in arm B, but also added hard pseudo-label BCE. Their full-backbone continuation cannot isolate the effect of KL coverage in phase 20's feature adapter. This is a factual source synthesis, not a protocol approval or scientific decision.

The [machine-readable matrix](EVIDENCE_MATRIX.json) links every claim to source IDs. [SOURCE_PINS.json](SOURCE_PINS.json) supplies exact absolute paths, SHA-256, byte lengths, primary URLs, retained retrieval dates and line references. This audit used retained text, source code and JSON only: zero network requests, scientific-array reads, checkpoint reads or model calls. External claims refer to those pinned snapshots, rather than a live refresh.

## What changed in the earlier studies

Let **M** be supplied supervision excluding supplied validation, intersected with acquisition-observed support. **P** is acquired, finite, evaluated seed43-map support outside M, including deliberately retired validation. **U/P** denote separate manual/pseudo BCE means; **D_M/D_P** denote separate Bernoulli KL means from seed42 initializer outputs to student outputs. Seed43 supplies hard pseudo targets; seed42 supplies preservation targets. Neither target establishes authentic ink.

| Phase | Actual optimized objective or operation | Held / changed | Coverage inference |
|---|---|---|---|
| 8 | A: U; B: U + 0.25 P; C also balances manual region/classes | Same seed42 initializer and 32-batch PHerc0139 schedule; unsmoothed targets | No KL optimized. |
| 9 | A/B BCE targets change from 0/1 to 0.25/0.75 | Same inputs, masks, seed43 targets, schedule and optimizer | Smoothing control; not the complete original Dice/BCE recipe. |
| 10 | A: U + D_M; B: U + 0.25 P + D_M + 0.25 D_P | Adds seed42 KL to each matched phase 9 arm | B protects nonmanual outputs, but A-versus-B also changes pseudo BCE and weighting. |
| 11 | Reuses existing states on Frag4, forward/reverse | No training | Acquisition/evaluation diagnostic. |
| 12 | Reconstructs phase 10 trajectory and independent same-batch first-step rates | Same starting batch/gradient; rate changes | No preservation-support change. |
| 13 | Exact phase 10 A/B objectives | Only training hyperparameter change: 1e-4 to 1e-6; adds fixed-risk measurements | Broader B KL remains coupled to pseudo BCE. |
| 14–17 | Calibration, source/inference and optical/geometry diagnostics | No new coverage-training contrast | No missing support ablation hidden here. |
| 18 | U + D_M on both 801-parameter output-residual branches | Raw-neighbor versus explicit-depth feature bundle | Manual-only KL, applied to outputs. |
| 19–20 | Fusion readiness, then U + D_M on both 5,296-parameter feature branches | Branch enters 32-channel fused features before frozen 2D decoder | Manual-only output KL; no feature-preservation loss. Against phase 18, insertion site and capacity both change. |

Primary code: [M/P construction](/workspace/herculaneum-lab/reports/eighth-experiment/implementation/experiment_common.py:75), [separate KL means](/workspace/herculaneum-lab/reports/tenth-experiment/implementation/preservation.py:4), [phase 10 contract](/workspace/herculaneum-lab/reports/tenth-experiment/implementation/prepare_plan.py:25), [phase 13 coefficients](/workspace/herculaneum-lab/reports/thirteenth-experiment/implementation/run_worker.py:85), and [phase 20 manual-only objective](/workspace/herculaneum-lab/reports/twentieth-experiment/implementation/depth_adapter.py:6).

Phase 20's frozen backbone computes `network(depth_fusion(X) + adapter(features(X)))`. Its loss compares final ink **outputs** to initializer outputs on M; it does not match intermediate features. Frozen backbone parameters do not make predictions immutable when decoder inputs change. The [actual insertion code](/workspace/herculaneum-lab/reports/twentieth-experiment/implementation/fusion_adapter.py:99) establishes this mechanism. The result establishes regional losses under that recipe; it does not establish that missing KL coverage, geometry or domain transfer caused them. The retained report says 54,701 unique scheduled manual pixels, but repeated batch exposures remain the empirical loss units.

## What the primary method sources establish

| Source | Relevant documented experiment | Limit for this proposed contrast |
|---|---|---|
| PHerc1667 [paper v1](https://arxiv.org/html/2606.29085v1) | Fragment-aligned infrared supervision, iterative sealed-scroll pseudo-label expansion, separate labeling/validation columns; Dice plus strongly smoothed BCE. | It changes supervision across rounds. Its 2.4 µm ResNet3D / 62×256×256 ablation differs from the current 17×128×128 ink9um factory. The inspected excerpts do not test initialization KL coverage. They do not identify current curated w029 with historical l_2/l_5. |
| Russo [code dd5bc8f](https://github.com/DomRusso2/ink9um-dense/tree/dd5bc8fb300c810c6ec042cb7e068be78829d086) | Canonical high-resolution teacher predictions densify seven pooled segments; manual labels override; validation is excluded. Matched label-root ablation and released-region scoring. | This is dense supervised-target training, not a preservation-KL ablation. Actual `render_valid` is center-plane intensity > 0. Outside manual strips, the author explicitly lacks truth on most 1667 segments. The shipped ex016 w029 AUC is below its matched control despite a balanced-accuracy gain. |
| Nieuwlaar [code 217e752](https://github.com/Nieuwlaar/ink9um-dense-native/tree/217e7521670db49cde97872ea6a8bdfb80c449f1) | Adds native 113 keV dense targets while retaining a mixed corpus; reports sparse single-scroll fine-tuning degradation. Native reference classes derive from 78 keV model maps, with a second 59 keV model-reference check. | Teacher-family agreement, cloning ratio and reported AUC do not authenticate ink. Native 28-plane 9.362 µm inputs, pooled inputs, quotas and thresholds differ. These sources do not test our feature-fusion KL support or prove transfer to Paris/PHerc0800. |

Exact retained references: PH paper text lines **1347–1364**, card lines **17–23 / 39–51**, pinned Villa `e583fb67468f483fadd73d52e06f0ab0fe5ba813` training model lines **1148–1149 / 1588–1595**; Russo README **46–58 / 75–86 / 112–145**, label builder **127–157**; Nieuwlaar README **100–130 / 238–331**, native label builder **61–99**. File hashes and full URLs are in SOURCE_PINS.json.

Prediction averaging, same-trajectory weight soup and learned feature fusion are separate operations. Nieuwlaar's README lines **220–236** distinguish successful prediction averaging from failed cross-seed weight averaging; they supply no evidence that our intermediate-feature branch preserves baseline outputs. Its corrected darkness veto, lines **400–430**, also cautions against declaring validity or ink from a fixed CT polarity.

## Additive correction to the frozen earlier audit

The earlier frontier `RULES_AND_TRAINING.md` line 42 incorrectly said native manual target replacement does not add manual pixels to supervision. The exact pinned [builder](https://github.com/Nieuwlaar/ink9um-dense-native/blob/217e7521670db49cde97872ea6a8bdfb80c449f1/training/scripts/build_native_dense_labels.py#L68) forms `sup = valid & (t > 0)` at line 69, replaces targets at line 78, then explicitly sets **`sup = sup | m_sup` at line 79**. Retained code SHA-256 is **421fb3ff94f21620cb2902d055f5c34be83cee23f57fa1cf23e88dbde4e131ff**. This correction supersedes that sentence only. The old file is unchanged; the code does not prove every historical run used that exact revision.

## Concrete unresolved prerequisites

1. **Qualify broader support.** Both PHerc0139 context receipts say `geometry_validity: not_acquired`, with no mask file. HTTP200 acquisition observation and nonzero intensity occupancy are different from mapped papyrus or valid CT interpolation support. The official label attributes describe relaxed mapping ≥128, proximity to strict supervision and cleanup; that is annotation-publication policy, not an acquired full-validity raster.
2. **Use the stronger retained registration lead.** [PHERC0139_RENDER_LEAD.json](PHERC0139_RENDER_LEAD.json) literally extracts the matching transformed mesh and render record from official full metadata: curated w016's public segment `20250108000004-w029_2025010827`, volume `20260102150214`, renderer **650bd56663efb7225fdba73fe267ba86b8e29368**, scale/group 1/0, flip −1, normal flip true, rotation 0, 109 slices and zero crop parameters. The record names the exact transformed tifxyz and masked native Zarr. This strengthens provenance beyond minimal metadata; no mesh coordinates, raster support or exact renderer implementation were validated by this synthesis.
3. **Specify reductions before running.** Phase 10 B's `D_M + 0.25 D_P` is not a single mean over M∪P. A mean over larger support changes individual pixel weights and manual-versus-unlabeled weighting. Freeze the proposed support, reduction/stratification, coefficient and common diagnostics; retain manual BCE and do not interpret unannotated zeros as negatives.
4. **Keep teacher and transfer claims bounded.** Broader KL can preserve wrong teacher outputs. The audit has not measured teacher errors outside M. Existing full teacher logits can supply targets, not new ground truth. All six Paris regions remain reused DEVELOPMENT; neither this source audit nor repeating them establishes a fresh holdout or parent-level independence.

Root owns whether these prerequisites justify a new protocol. This audit makes no recommendation to change the frozen phase 20 artifacts, no efficacy or causal attribution, and no model promotion, reading, translation or publication claim.
