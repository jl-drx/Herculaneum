# Ninth study: smoothing does not consistently preserve retention

The two fixed CPU continuations completed successfully, but both failed the predeclared mitigation and baseline-retention screens. Restoring the published BCE target convention improved equal-context mean AUC/AP for both comparisons; sensitivity transferred inconsistently. Keep the unchanged seed42 baseline. These are descriptive supplied-annotation retention diagnostics on two reused PHerc0814 contexts, not independent ink accuracy or a reading.

Manual-only smoothing changed mean primary sensitivity from **68.89% to 73.63%** (+4.74 percentage points), but reduced it by 5.22 points in rank14. With pseudo-labels, smoothing reduced mean sensitivity from **77.13% to 69.06%** (−8.07 points). Frozen seed42 remains **90.32%**. All models meet the 95% specificity floor in both contexts; the floor is a constraint, not equality of their achieved operating points.

## Fixed comparison

Only A0.5 and B0.5 were trained: 32 updates each, **64 new updates total**, from the same seed42 state. The completed eighth A0/B0 runs and frozen baseline were reused by their exact artifact identities. The failed balanced-anchor C arm was not continued. A uses mean manual BCE; B adds 0.25 times mean fixed hard-teacher BCE. Targets inside the same masks change from 0/1 to 0.25/0.75. Input bytes, teacher, ordered patch/D4 schedule, optimizer, learning rate, masks, update count and calibration rules remain fixed.

The [protocol](PROTOCOL.md), [lead review](ROOT_REVIEW.md), [independent code review](review/CODE_REVIEW.md) and [ready plan](implementation/final-plan/control-plan.json) were fixed before new execution. Plan SHA-256: `9d72d3177d5aa68cd5d59db6f316ed1a5c335f442b6c5c2439f45bfdaebe29cd`. Source audit confirms the BCE formula, while the published original objective also includes Dice. This control restores only the declared BCE convention, not the full historical training recipe.

## Measured outcomes

Equal-context means give each context equal weight. They are two contexts from one specimen, not independent replicates. AUC/AP concern agreement with supplied annotations. Primary thresholds are selected separately for each model using only the same three calibration regions; no diagnostic labels determine thresholds.

| Model | Mean AUC | Mean AP | Primary sensitivity | Primary specificity |
|---|---:|---:|---:|---:|
| Frozen seed42 | 0.999608 | 0.998510 | 90.32% | 99.967% |
| A0 manual | 0.989429 | 0.976560 | 68.89% | 99.895% |
| B0 manual + pseudo | 0.987941 | 0.979877 | 77.13% | 99.722% |
| A0.5 manual | 0.994717 | 0.988662 | 73.63% | 99.179% |
| B0.5 manual + pseudo | 0.995598 | 0.990528 | 69.06% | 99.759% |

![Retention diagnostic comparison](visualization/render/retention-comparison.png)

### pherc0814-46527-fresh-rank01

Source bbox xywh `[1152, 1152, 384, 384]`; 49,033 participating pixels, 19,545 supplied positives and 29,488 supplied negatives; prevalence 0.398609. All five models share the complete fixed primary domain. The legacy filename retains `fresh`; its current role is **reused development retention diagnostic**.

| Model | AUC | AP | Primary sensitivity | Specificity | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen seed42 | 0.999882 | 0.999824 | 0.919621 | 1.000000 | 17974 | 0 | 29488 | 1571 |
| A0 manual | 0.994313 | 0.992855 | 0.749041 | 0.999220 | 14640 | 23 | 29465 | 4905 |
| B0 manual + pseudo | 0.996734 | 0.995759 | 0.905142 | 0.994438 | 17691 | 164 | 29324 | 1854 |
| A0.5 manual | 0.993594 | 0.990130 | 0.896137 | 0.983587 | 17515 | 484 | 29004 | 2030 |
| B0.5 manual + pseudo | 0.996075 | 0.994427 | 0.884421 | 0.995184 | 17286 | 142 | 29346 | 2259 |
### pherc0814-46527-fresh-rank14

Source bbox xywh `[128, 1152, 384, 384]`; 21,328 participating pixels, 4,767 supplied positives and 16,561 supplied negatives; prevalence 0.223509. All five models share the complete fixed primary domain. The legacy filename retains `fresh`; its current role is **reused development retention diagnostic**.

| Model | AUC | AP | Primary sensitivity | Specificity | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen seed42 | 0.999334 | 0.997196 | 0.886721 | 0.999336 | 4227 | 11 | 16550 | 540 |
| A0 manual | 0.984546 | 0.960266 | 0.628697 | 0.998672 | 2997 | 22 | 16539 | 1770 |
| B0 manual + pseudo | 0.979147 | 0.963994 | 0.637508 | 1.000000 | 3039 | 0 | 16561 | 1728 |
| A0.5 manual | 0.995839 | 0.987193 | 0.576463 | 1.000000 | 2748 | 0 | 16561 | 2019 |
| B0.5 manual + pseudo | 0.995120 | 0.986630 | 0.496748 | 1.000000 | 2368 | 0 | 16561 | 2399 |

## Calibration and secondary rule

Primary is the maximum of three per-region empirical 95%-specificity cutoffs with conservative ties. Secondary uses PHerc1667/w029 alone. New threshold files were written before diagnostic inference. Literal historical thresholds and maps are reused. Raw sigmoid scores are not calibrated ink probabilities.

| Model | Primary cutoff | Secondary cutoff | Primary-determining region |
|---|---:|---:|---|
| Frozen seed42 | 0.755894720554 | 0.755894720554 | pherc1667-w029 |
| A0 manual | 0.829143524170 | 0.829143524170 | pherc1667-w029 |
| B0 manual + pseudo | 0.625097572803 | 0.625097572803 | pherc1667-w029 |
| A0.5 manual | 0.563550233841 | 0.563550233841 | pherc1667-w029 |
| B0.5 manual + pseudo | 0.627626538277 | 0.625621020794 | pherc0814-46527-rank01 |

The secondary rule is identical to primary for frozen/A0/B0/A0.5. Only B0.5 differs:

| Context | Secondary sensitivity | Specificity | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|
| pherc0814-46527-fresh-rank01 | 0.887746 | 0.994947 | 17351 | 149 | 29339 | 2194 |
| pherc0814-46527-fresh-rank14 | 0.503461 | 1.000000 | 2400 | 0 | 16561 | 2367 |

B0.5 mean sensitivity becomes 69.56% under the secondary rule, still below B0's 77.13% and frozen's 90.32%. Thus the small change in which region determines the maximum cutoff does not reverse the result. Ranking improvements with weaker sensitivity indicate a score-scale/threshold-transfer issue as well as incomplete retention; they do not by themselves isolate its cause.

## Paired effects and fixed screens

The table reports equal-context continuous effects. Sensitivity/specificity differences are percentage points; AUC/AP differences are raw metric units.

| Contrast | ΔAUC | ΔAP | Δprimary sensitivity (pp) | Δprimary specificity (pp) |
|---|---:|---:|---:|---:|
| interaction difference of differences | +0.002370 | -0.001450 | -12.817 | +0.753 |
| pseudo at epsilon 0 | -0.001489 | +0.003316 | +8.246 | -0.173 |
| pseudo at epsilon 05 | +0.000881 | +0.001867 | -4.572 | +0.580 |
| smoothing with manual only | +0.005287 | +0.012101 | +4.743 | -0.715 |
| smoothing with pseudo | +0.007657 | +0.010652 | -8.074 | +0.037 |

Both mitigation screens fail. A0.5 passes the mean-gain requirement but loses 5.223 percentage points of sensitivity in rank14, beyond the predeclared 2-point allowance. B0.5 loses 2.072 and 14.076 points in the two contexts and fails the mean-gain requirement. Mean AUC/AP improvements are driven by rank14; both ranking measures decline slightly in rank01 for both smoothed pairs.

Both baseline-retention screens also fail. A0.5 sensitivity is 2.348/31.026 points below frozen; B0.5 is 3.520/38.997 points below. A0.5 also exceeds the rank01 AUC-loss allowance and the rank14 AP-loss allowance; B0.5 exceeds the rank14 AP-loss allowance. Comparisons use full precision: A0.5's rank14 AP delta is −0.01000266036, just beyond −0.01, but the large sensitivity loss independently fails retention. No margin was relaxed after inspection.

The sensitivity difference of differences is −12.817 percentage points. This describes an interaction between our smoothing and pseudo-label factors under this fixed continuation/calibration design. One seed and two reused contexts cannot establish a general causal mechanism. The hypothesis that smoothing alone consistently mitigates the earlier retention loss is unsupported by this screen.

## Execution and independent verification

The first actual forward reproduced historical hard-target U=0.31660833954811096 and P=0.3718268871307373 exactly in both arms. Their independently reconstructed normalized input/mask/target hash matches `571d10c58008cb6f9e581a717bbb067846dae6a2ee2d29bdf9eb141ca9a02ec3`. Double-precision softplus checks confirm the smoothed U/P within the fixed tolerance. No extra forward/update was added for this sentinel. Artificial-logit fixtures independently checked smoothing derivatives, zero-smoothing equivalence and zero gradients outside participation.

Independent completed-output verification passed: 252 stable artifact receipts, maximum numeric metric difference 2.22×10⁻¹⁶, exact objective coefficients, ordered exposures and 64 actual new update receipts. Both saved states strictly match schema/aliases, load with `weights_only=True`, and have all 264 canonical parameter tensors changed from the initializer. This audits receipts, tensors and metrics without replaying training or performing extra model inference. See [independent verification](review/result/independent-verification.json).

Workers used 25.47/25.67 seconds internally, 52.47 seconds combined controller wall time, and at most 1,371.22 MiB peak RSS. Run plus analysis outputs total **289,969,467 bytes**, below 400 MiB. Two final checkpoints are 138,351,655 bytes each; no intermediate checkpoint selection or retry occurred. New source verification used two public-code reads totaling 21,098 response-body bytes; no new scan/model download occurred. Model work was offline.

CPU runtime is Python 3.12.14, torch 2.12.0+cpu, NumPy 2.5.3, four threads. Recorded historical torch/thread settings match; full historical package/hardware equality is not established. Real first-step agreement and pinned artifacts strengthen the comparison without making a cross-environment bitwise claim.

## Decision and limits

Retain frozen seed42; do not promote either new checkpoint or scale these recipes onto a GPU. Keep the complete negative result and mixed ranking effects. The next research step is to separate continuation stability from calibration transfer and strengthen reference lineage/physical reservation before further model claims. A post-result [next-study draft](NEXT_STUDY_DRAFT.md) describes that work; it is not a frozen or executed tenth experiment.

The inspected contexts are in a known historical model-development corpus, have unresolved original pixel/teacher exposure and supplied-annotation lineage, and use a pooled-fine representation. They establish no independent ink accuracy, generalization, authenticated letters, transcription, translation or novelty. Research is primary; licensing/access/source-attribution obligations still apply, and prize eligibility has not been asserted.

Paid GPU spend remains **£0**; the aggregate first-pilot limit remains **£10** including failures/retries and attributable overhead. No R2 upload, publication, prize submission or credential import occurred. Local integrity and saved startup guidance are separate from backup/publication/restoration. Full precision metrics, confusion counts, score quantiles, raw maps and curves remain in [the analysis](implementation/analysis/metrics.json).
