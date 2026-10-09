# Plinius: stable continuation, with calibration still unresolved

**The conservative control works as an optimization test.** Both 32-update runs improve their fixed training objectives and pass the predeclared reference-ranking screen. Their final scores preserve the original ranking much better than the earlier 1e-4 continuations. However, sensitivity falls at the unchanged baseline detection cutoff, especially on Frag4. Keep the original seed42 baseline; retain both new states as research candidates, with no automatic promotion.

**Plinius** is the working research-series name suggested by the user. These controls adapt the credited published `scrollprize/ink_9um` seed42 model. A new name does not establish a new architecture, first-in-class technique or authenticated reading.

## Controlled training result

The only training hyperparameter change from the tenth preservation study is learning rate **1e-4 → 1e-6**. Initialization, 32-update schedule, two PHerc0139 input regions, masks, targets, augmentation, normalization, AdamW settings and preservation objective are unchanged. A uses smoothed manual BCE plus initializer-preservation KL; B adds the fixed 0.25 pseudo-target and pseudo-preservation terms. Every participating hard target is smoothed to 0.25/0.75. All 32 regenerated input/target/mask hashes match both historical arms before their cached teacher logits are reused.

The initializer risk is derived from that verified teacher cache. At updates 1, 8, 16 and 32, each current model is measured on the same full 32-batch schedule. These are equal-batch empirical losses, preserving repeated/overlapping exposures; they are not population estimates. The gate uses `math.fsum` of actual logged per-batch scalar objectives divided by 32, followed by final-minus-initial subtraction. Base BCE retains the original FP32 computation and KL the original FP64 computation.

| Final fixed-set result | A: manual + preservation | B: manual + pseudo + preservation |
|---|---:|---:|
| Initial base/full loss | 0.597077 | 0.745955 |
| Final base loss | 0.586598 | 0.732896 |
| Final full loss | 0.590246 | 0.737095 |
| Base change | −0.010479 | −0.013059 |
| Full change | −0.006831 | −0.008860 |
| Own-batch base-loss increases | 0/32 | 1/32 |
| Own-batch full-loss increases | 0/32 | 0/32 |

Both arms improve both losses at every measured checkpoint, with progressively lower fixed risk. Both exceed the predeclared final base/full improvement requirement of 0.0001. The first post-update logits and full parameter state match the twelfth 1e-6 probe exactly. Fixed-risk measurements preserve model, optimizer, gradients, modes and RNG according to the reviewed worker checks. The one B base-loss increase is retained; its full optimized objective still decreases.

![Fixed training-risk changes](visualization/render/fixed-risk-change.png)

Absolute A and B loss values are not comparable measures of model quality because B includes an additional weighted objective. The meaningful comparisons are each arm against its own initializer and the matched historical recipe. The earlier study lacks final fixed-risk measurements, so its last online window is not substituted for a final-risk comparator. This study demonstrates stable continuation on the defined data, not an optimal learning rate or inevitable long-run behavior.

See [training report](training-report/RESULTS.md), [frozen scientific design](RESEARCH_DESIGN.md), [training plan](implementation/final-plan/training-plan.json) and [gate](implementation/analysis/retention-gate.json).

## Conditional reference comparison

After independent training verification, both final states qualified for the already specified retention stage. Each was evaluated once on the same two PHerc0814 development-diagnostic regions and two Frag4/PHercParis1Fr39 regions. Inputs, forward depth convention, fixed masks and blending remain unchanged. Baseline and historical 1e-4 predictions were reused literally. No threshold, orientation, crop, checkpoint or model was selected from these results.

The descriptive ranking screen requires each specimen/domain's mean AUC and AP to remain within 0.01 of baseline, and each context within 0.02. **Both new states pass every component.** These are predeclared tolerance checks on reused evidence, not statistical significance or prospective accuracy.

| Equal-context mean | Frozen baseline | New A, 1e-6 | New B, 1e-6 | Old A, 1e-4 | Old B, 1e-4 |
|---|---:|---:|---:|---:|---:|
| PHerc0814 AUC | 0.999608 | 0.999621 | 0.999619 | 0.996026 | 0.995775 |
| PHerc0814 AP | 0.998510 | 0.998647 | 0.998631 | 0.991256 | 0.989587 |
| Frag4 AUC | 0.734809 | 0.740985 | 0.743636 | 0.564788 | 0.626298 |
| Frag4 AP | 0.186118 | 0.188433 | 0.190900 | 0.129121 | 0.134257 |

Frag4's new B ranking improves in both contexts. New A is nearly unchanged/slightly worse in context1 (AUC −0.000062, AP −0.000449), with context2 driving its mean gain. Neither arm is selected as the winner. B's small advantage on Frag4 does not establish a general benefit from pseudo targets, especially with one initializer and two contexts from one specimen. PHerc0814's near-perfect supplied-reference agreement remains affected by its known historical development-corpus role.

The unchanged **common baseline cutoff 0.7558947205543518** exposes a separate problem:

| Mean operating metric | Frozen baseline | New A | New B |
|---|---:|---:|---:|
| PHerc0814 sensitivity | 90.317% | 88.931% | 88.915% |
| PHerc0814 specificity | 99.967% | 99.973% | 99.973% |
| Frag4 sensitivity | 3.025% | 0.354% | 0.183% |
| Frag4 specificity | 95.161% | 96.893% | 97.210% |

The models rank the supplied labels at least as well on average while responding less often above the old cutoff. This supports separating ranking retention from operating-threshold transfer. It does not prove that recalibration will recover the desired detection performance. No cutoff was refitted here. Historical 1e-4 operating metrics in the machine-readable report also use this common baseline cutoff, so they differ from earlier reports that used each model's separately calibrated cutoff.

![Frag4 ranking and response at the unchanged cutoff](visualization/retention-render/frag4-ranking-and-response.png)

See [complete context metrics](retention/analysis/metrics.json) and [independent retention review](retention-review/completed-review/REVIEW.md). All 239,656 fixed primary pixels are covered with finite predictions across all five compared states. Masks were not shrunk to make coverage pass.

## Verification and resources

Independent NumPy training reconstruction passed 6,164 scalar comparisons across 64 pre/post pairs, 256 fixed-batch evaluations and 64 cache-derived baseline batches. It verifies fingerprints, teacher bits, first-step sentinels, reductions and exact gate decisions. Maximum FP32 scalar discrepancy is 1.5920e-7 and KL/drift discrepancy 6.9389e-18. Model/optimizer/gradient/RNG isolation and checkpoint reload are source-reviewed worker checks, not independently replayed training. [Training verification](numeric-review/completed-review/independent-verification.json).

An independent scorer confirms canonical masks/counts, tie-aware AUC/AP, exact fixed-cutoff confusion counts, deltas, macros and both screens. A separate NumPy reconstruction of all eight new score maps from 200 raw tile logits reproduces coverage exactly and scores within 2.3842e-7 absolute error, below its predeclared 2e-6 tolerance. Scores are not bitwise identical across NumPy/Torch transcendental kernels; no such claim is made. [Blend verification](review/retention-map-verification/REVIEW.md).

Training performed 384 batch-two forwards, 64 backwards and 64 updates, with zero teacher forwards. Summed controller worker time was 218.736 seconds; maximum worker RSS 1,530,109,952 bytes. Retention used 200 additional single-tile forwards, taking 40.149 seconds of controller wall time, with maximum RSS 598,425,600 bytes. There were no scientific retries, baseline reruns, new data downloads or additional reviewer model calls.

Two final raw state dictionaries of 138,351,655 bytes each are retained as candidates. Training run plus analysis uses 328,411,206 bytes; retention run plus analysis 39,579,591 bytes. No intermediate checkpoint or optimizer state was selected or published. Source/code/input plans and all previous studies remain frozen; the current phase is covered by an additive final manifest and read-only integrity verifier. Root visually inspected both exported figures for readable labels and agreement with the retained results.

## Decision

This establishes a sounder continuation control for Plinius research. **It does not establish a ready replacement model:** fixed-cutoff sensitivity is worse, Frag4 ranking remains modest, and both reference specimens are already seen development evidence. Keep Frag4 out of gradients, teacher targets and calibration across aliases. Preserve its unresolved optical-registration residual and historical-exposure qualifications. There is no independent ink accuracy, novel method, letter, transcription or translation claim.

Next, isolate calibration using only the original designated development regions and the existing frozen selector, without fitting Frag4 or changing model weights. Preserve these candidates and the original baseline. A separate source/annotation contract and reserved physical support remain necessary for stronger generalization claims. Details are in [NEXT_STUDY_DRAFT.md](NEXT_STUDY_DRAFT.md). Additional learning-rate or loss sweeps are not justified by this result.

GPU rental is still unnecessary for the next discriminating control. Spend remains **£0**, and the **£10 TOTAL first-pilot cap** is unchanged. No R2 upload, publication, prize submission, outreach, credential import or terms acceptance occurred. The repository remains unchanged at the reviewed commit; artifacts are local. Environment-draft persistence alone does not establish backup or fresh-task restoration.
