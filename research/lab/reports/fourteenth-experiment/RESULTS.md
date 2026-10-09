# Plinius: calibration restores one domain, with Frag4 transfer unresolved

**Development-only calibration largely restores PHerc0814 sensitivity but does not recover the Frag4 baseline.** Both candidates now meet the 95% specificity floor in every diagnostic context. Neither passes the complete frozen recovery screen. Keep the original seed42 baseline and both thirteenth states as research controls; no model is promoted and no GPU rental follows.

The scientific execution and independent numerical verification both passed on their first attempt. This means the experiment was implemented and measured correctly; it does not mean the candidates passed their scientific criteria.

## What changed

Only the operating cutoff changed. The two final 1e-6 continuation states, all four diagnostic score maps per state, reference labels and masks remain unchanged. The original three calibration regions supply negatives to the original conservative 95%-specificity selector. The primary cutoff is the maximum of the three regional cutoffs; the fixed secondary rule uses PHerc1667 alone. Both new locks were saved before diagnostic-array analysis. Frag4, diagnostic PHerc0814 labels and adaptation labels contributed no fitted cutoff.

| State | Common baseline cutoff | Development-selected primary cutoff |
|---|---:|---:|
| Frozen seed42 | 0.7558947205543518 | 0.7558947205543518 |
| A: manual + preservation | 0.7558947205543518 | 0.7450202703475952 |
| B: manual + pseudo + preservation | 0.7558947205543518 | 0.7422624826431274 |

PHerc1667 determines the maximum in both new states. Primary and secondary cutoffs therefore coincide; this is an observed consequence of the fixed rules, not a choice made from diagnostic performance. The original baseline lock was reconstructed exactly from its retained calibration maps without running its model.

## Operating results

Values below are equal-context means within each specimen, in percent. They are descriptive results from two reused regions per specimen.

| Specimen / state | Sensitivity at common cutoff | Sensitivity after calibration | Specificity after calibration |
|---|---:|---:|---:|
| PHerc0814 / baseline | 90.317 | 90.317 | 99.967 |
| PHerc0814 / A | 88.931 | 90.309 | 99.967 |
| PHerc0814 / B | 88.915 | 90.534 | 99.967 |
| Frag4 / baseline | 3.025 | 3.025 | 95.161 |
| Frag4 / A | 0.354 | 1.454 | 95.985 |
| Frag4 / B | 0.183 | 1.205 | 96.245 |

On PHerc0814, B passes the frozen recovery screen. A misses its exact mean-sensitivity requirement by **0.00845 percentage points**, despite passing specificity and per-context loss constraints. Preserve that formal failure, while recognizing its tiny descriptive magnitude; it is not evidence of a material or statistically established regression.

Both Frag4 candidates fail mean sensitivity recovery and the context1 maximum-loss allowance. Context1 sensitivity is 6.050% for baseline, 2.907% for A and 2.410% for B. **All three detect zero of 15,252 supplied positives in context2** at their primary cutoffs. A and B nevertheless improve context2 specificity to 95.401% and 95.816%; the baseline's 94.054% misses the floor. That baseline failure remains visible rather than being hidden by the specimen mean.

| Frozen primary screen | A | B |
|---|---|---|
| PHerc0814 | Fail: mean sensitivity just below baseline | Pass |
| Frag4 | Fail: mean and context1 sensitivity | Fail: mean and context1 sensitivity |
| Both specimens | Fail | Fail |

The screen requires at least 95% specificity in each context, mean sensitivity at least baseline, and no individual sensitivity loss greater than 2 percentage points. Integer counts and exact rational arithmetic determine its flags. No criterion was changed after results. All new states meet the specificity floor, but their achieved operating points differ: Frag4 sensitivity comparisons are not comparisons at exactly matched specificity.

The transferred calibration rule addresses much of the PHerc0814 cutoff shift. Its performance on Frag4 remains inadequate. This leaves attainable performance under a separately validated domain-specific protocol unresolved; the experiment does not identify acquisition, representation, alignment or annotation history as the sole cause. AUC/AP remain exactly unchanged in the author reconstruction, as expected when the same scores are reused. The thirteenth ranking gains are not new evidence from this study.

![Per-context calibration transfer](reporting/render/calibration-transfer.png)

See the [complete metrics](implementation/analysis/metrics.json), [frozen design](RESEARCH_DESIGN.md), [calibration contract audit](CONTRACT_AUDIT.md) and [detailed reporting package](reporting/RESULTS.md).

## Verification and cost

Independent verification passed **511 scalar comparisons** with maximum discrepancy 1.1102e-16. It reconstructed canonical masks, all nine regional selections, all three primary/secondary locks, exact float32 cutoff bits and ties, confusion counts, rational screens and macros. Twelve literal-map AUC/AP comparisons agree. All 118 normalized input hashes match independent reconstruction. Six maps reconstructed from 118 raw tile logits have exact coverage and maximum absolute score difference **2.9802e-7**, below the predeclared 2e-6 tolerance. Cross-library sigmoid/Hann results are not claimed to be bitwise identical. All 315 checked nonweight input/source pins were unchanged. [Independent receipt](numeric-review/completed-review/independent-verification.json).

The independent checker used no Torch, models, checkpoint reads or network. Model state preservation and absent gradients are source-reviewed worker checks; numerical reconstruction does not independently replay inference. Historical baseline normalized-tile hashes are unavailable, so baseline comparison is bounded by its retained source/depth/normalization/adapter contracts and tile placement.

Execution used **118 CPU tile forwards**, no baseline/diagnostic forwards, no backwards or updates, and no new checkpoints. Controller wall time was 31.651 seconds; maximum worker RSS was 598,765,568 bytes. Run artifacts total 12,820,496 bytes and analysis 58,954 bytes. There were no scientific retries or new downloads. A pre-execution metadata-schema lookup was corrected before code freeze; its review is retained.

Root visually inspected the exported four-panel figure and tables for readable labels, correct operating points and explicit baseline specificity failure. PNG, SVG, CSV and pinned reporting receipts are retained.

## Decision and next direction

Together, the thirteenth and fourteenth controls establish stable small-step continuation and quantify the separate calibration-transfer problem. Further loss, rate or cutoff sweeps on these same diagnostics would provide a weak foundation for a frontier claim. Preserve the controls and now strengthen the benchmark across acquisition domains and additional physical specimens, with distinct calibration and evaluation support selected before predictions. [Next-study draft](NEXT_STUDY_DRAFT.md).

PHerc0814 is historical development evidence. Frag4's optical-derived labels have unmeasured registration residual and incomplete historical exposure information; it remains seen diagnostic evidence and excluded from calibration/training across aliases. Neither specimen supplies prospective accuracy here. There is no authenticated ink, letter, transcription, translation or novel-architecture claim. Plinius remains the working research-series name for credited adaptations of `scrollprize/ink_9um`.

GPU spend remains **GBP 0**, with the **GBP 10 TOTAL first-pilot cap** unchanged. Repository tracked files and prior evidence remain unchanged. No upload, publication, prize submission, outreach, credential operation or new terms acceptance occurred. Local artifact retention and saving startup configuration are separate from external backup or verified fresh-task restoration.
