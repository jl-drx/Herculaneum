# First real-data continuation pilot: retain the unchanged baseline

All three planned 32-step continuations completed successfully, producing three verified final checkpoints. **The balanced-anchor candidate failed the predeclared improvement screen.** Its equal-context sensitivity was 55.99%, versus 77.13% for ordinary hard-pseudo continuation and 90.32% for unchanged seed42. All three continuations also reduced AUC and average precision relative to the unchanged model on both fresh regions. Retain seed42; do not promote or scale this candidate on the strength of this pilot.

![Frozen comparison](results-summary-v2.png)

This is a negative result for one small continuation recipe. It does not reverse the seventh study's evidence that the released dense ex016 model can improve ranking over its matched control. Those checkpoints represent a different, much larger training history.

## Fixed experiment

Each arm began from identical strict seed42 weights in the unchanged 34,546,498-parameter factory. Each received the same 32 batches of two 17×128×128 patches, one patch from each of two retained PHerc0139 regions, with identical seeded XY augmentations. AdamW settings were fixed at learning rate 1e-4, betas (.9,.999), epsilon 1e-8, zero weight decay, no scheduler and gradient clipping at 1.0. No intermediate checkpoint selection, automatic retry or post-result tuning occurred.

Manual participation was supplied supervision excluding historical public validation. Historical validation was openly retired as held-out support in these adaptation regions, permitting fixed seed43 pseudo targets there. Historical validation ink labels did not enter training or calibration. The public references have incomplete human/teacher ancestry; “manual” denotes the operational arm, not verified human-only truth.

Let U be the pooled manual BCE, P the pooled teacher BCE, and R the equal mean of the four region×class manual BCE terms:

| Arm | Fixed objective |
| --- | --- |
| Frozen seed42 | No update; calibration comparator |
| A: manual-only | U |
| B: ordinary hard pseudo | U + 0.25 P |
| C: balanced anchors | 0.5 U + 0.5 R + 0.25 P |

Every trained arm saw 230,340 manual-positive and 387,868 manual-negative pixel visits. B and C additionally used the same 40,620 teacher-positive and 389,748 teacher-negative visits. These counts include repeated/overlapping patches; only 6,782 unique teacher-positive pixels were reachable on the selected tile grid. A computed teacher-loss diagnostics but did not optimize them. All arms used unsmoothed 0/1 targets, whereas the published initializer used BCE smoothing 0.5. A-versus-frozen therefore includes this shared loss-target change as well as continuation; B-versus-C is matched for it.

All PHerc0814 inputs were excluded from gradient updates. Calibration used the old PHerc1667/w029 primary region and two previously used PHerc0814 regions. The primary rule takes the maximum of three independently selected empirical 95%-specificity cutoffs. Each threshold receipt was written before fresh inference. The secondary rule uses PHerc1667 alone. Both rules were fixed before the run and applied identically to all four models.

## Fresh-region results

The two fresh PHerc0814 contexts were selected from a frozen metadata ranking and annotation class gates, with at least a 128-pixel source-XY gap from the old full contexts and from each other. Evaluation participation is explicitly **supplied supervision**, not the public validation mask; historical validation is zero in both contexts. The primary mask is the fixed 256×256 interior intersected with acquired source and common finite evaluated support. Both entire source contexts were observed and had nonzero selected-depth profiles.

| Fresh context | Source XYWH | Primary positive | Primary negative |
| --- | --- | ---: | ---: |
| Region 1 / fresh-rank01 | [1152,1152,384,384] | 19,545 | 29,488 |
| Region 2 / fresh-rank14 | [128,1152,384,384] | 4,767 | 16,561 |

All 70,361 primary pixels remained in the common evaluation masks. The following sensitivity/specificity values use the **development-locked primary cutoff**. They are not sensitivities at forced equal specificity on these fresh regions.

| Region | Model | AUC | AP | Sensitivity | Achieved specificity |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Frozen seed42 | .999882 | .999824 | 91.96% | 100.00% |
| 1 | A: manual | .994313 | .992855 | 74.90% | 99.92% |
| 1 | B: hard pseudo | .996734 | .995759 | 90.51% | 99.44% |
| 1 | C: balanced anchors | .996670 | .994933 | 62.83% | 99.94% |
| 2 | Frozen seed42 | .999334 | .997196 | 88.67% | 99.93% |
| 2 | A: manual | .984546 | .960266 | 62.87% | 99.87% |
| 2 | B: hard pseudo | .979147 | .963994 | 63.75% | 100.00% |
| 2 | C: balanced anchors | .972712 | .937042 | 49.15% | 100.00% |

Equal-context averages are descriptive; both regions belong to one specimen:

| Model | AUC | AP | Primary sensitivity | Primary specificity |
| --- | ---: | ---: | ---: | ---: |
| Frozen seed42 | .999608 | .998510 | 90.32% | 99.97% |
| A | .989429 | .976560 | 68.89% | 99.89% |
| B | .987941 | .979877 | 77.13% | 99.72% |
| C | .984691 | .965987 | 55.99% | 99.97% |

C loses 21.14 percentage points of mean sensitivity relative to B. Its second-region AP is lower by .02695, exceeding the prespecified .02 tolerance. The C-versus-B screen fails, and the corresponding comparisons against A and frozen seed42 also fail. All achieved specificities exceed 95%; specificity alone would conceal the loss of positive detection.

Calibration explains part, but not all, of the result. For frozen seed42, A and B, the maximum cutoff equals the PHerc1667 cutoff: .755895, .829144 and .625098 respectively. C needs .859406 because of the old PHerc0814 rank01 development region; its PHerc1667-only cutoff is .812803. Under the predeclared secondary rule, C's fresh sensitivities are 77.01% and 66.73%, averaging 71.87%. That is still below B's 77.13%, and changing a cutoff cannot repair C's second-region ranking loss. No fresh threshold was fitted to rescue the outcome.

## What follows from the result

The hypothesis that this particular region/class-balanced manual term would improve transfer is unsupported. Ordinary teacher supervision retains more sensitivity than manual-only continuation here, but it does not beat the initializer. Small-data full-network continuation, the shared loss-smoothing change, the limited teacher support, class weighting and the conservative calibration rule are not isolated by this three-arm pilot. Calling the result a proven domain mechanism or attributing it solely to forgetting would exceed the evidence.

The near-perfect frozen-model AUC is **not independent accuracy evidence**. These fresh crops were excluded from our current updates and cutoff fitting, but both model configurations already include this specimen/representation in their historical training and online-validation corpus. Supplied-supervision annotations also have incomplete teacher ancestry. This is principally a retention and transfer diagnostic within that reference corpus. The changed annotation-participation domain differs from the seventh study's public-validation panel; the absolute metrics should not be compared as if they sampled the same task distribution.

The two new contexts have source-XY separation, not verified complete 3D sampling independence. Pixel visits, pixels and neighboring crops are not independent experimental replicates. No confidence interval or significance claim is reported. There is no authenticated new letter, transcription or translation.

**Lead decision:** keep the frozen baseline, preserve all three negative-result checkpoints, and do not spend the GPU pilot budget scaling C. Before another candidate loss, isolate continuation stability: preserve the original target-smoothing convention, measure retention under a fixed diagnostic panel, and assess a separately controlled initialization-preservation/replay term only if needed. These are familiar stabilization controls, not a novelty claim. The current fresh regions are now development evidence for that next hypothesis; any later generalization claim needs newly reserved physical support and stronger annotation/exposure evidence. See [next-study notes](NEXT_STUDY_DRAFT.md).

## Execution evidence and limits

All four workers completed exactly once; A/B/C each performed 32 updates, for 96 real updates in total. Every arm's initial semantic state hash was identical. Each saved final state passed safe tensor-only reload, exact-value and shape checks, the unchanged storage/alias contract, semantic hashing and strict reload before evaluation. Final weights remained finite and differed from the initializer.

Summed worker wall time was **83.10 seconds**, including initialization, training, checkpoint verification and evaluation. The three training/checkpoint stages together took 43.45 seconds. Maximum measured peak RSS was **1,345.09 MiB**, below the 4 GiB acceptance bound. Run plus analysis outputs used **432,712,847 bytes**, below 700 MiB. This establishes feasibility for this tiny CPU workload; it is not a full-training throughput or CUDA benchmark.

Fresh acquisition used 18 scan chunks plus metadata: **32,152,754 measured response-body bytes / 22 requests**. Annotation selection used **101,459 measured body bytes / 186 requests**. Three earlier eligible-gap candidates failed the predeclared positive-class gate; all decisions and receipts are retained. No acquisition failed or retried. These are body-byte figures, excluding headers/TLS, with the inherited partial-read failure-accounting limitation documented.

The [independent data review](data-review/DATA_REVIEW.md) reconstructed all 6,193,152 pooled voxels and selected annotation/mask bytes; 178 receipts stayed unchanged. [Independent code review](design-audit/IMPLEMENTATION_REVIEW.md) and synthetic analytic loss/gradient/schedule/cutoff fixtures passed.

The [independent post-run verification](metric-review/result/independent-verification.json) also **passed**. All 223 input/output receipts stayed unchanged. The verifier reconstructed evaluation masks, tied-score AUC/AP, calibration cutoffs, confusion counts, aggregation and all failed decision gates; maximum metric discrepancy was 1.11×10⁻¹⁶. All 96 update receipts matched the fixed schedule and independently reconstructed pixel exposures, with exact saved-loss coefficient arithmetic. Each final checkpoint contained 264 actually changed canonical parameter tensors and preserved the 508-key, 264-storage, 244-alias contract. This checks retained training receipts and saved states; it does not independently replay the optimization or recompute BCE from training logits, which were not retained.

No paid GPU, R2 upload, publication or prize submission occurred. **Paid GPU spend remains £0; the £10 total cap is intact.** Artifacts remain local to this workspace. Saved startup guidance is not an R2 backup or proof of restoration in a new environment.

Reproduction/evidence: [protocol](PROTOCOL.md), [implementation decisions](IMPLEMENTATION_DECISIONS.md), [frozen training plan](implementation/final-plan/training-plan.json), [execution](implementation/run/execution.json), [full metrics](implementation/analysis/metrics.json), [runner instructions](implementation/README.md). Final checkpoints: [A](implementation/run/A/final-state.pth), [B](implementation/run/B/final-state.pth), [C](implementation/run/C/final-state.pth). None is adopted as the research baseline.
