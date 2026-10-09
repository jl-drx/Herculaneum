# Ninth study: target-smoothing control on reused retention diagnostics

Scientific choices fixed before new training, 8 October 2026. The user authorized proceeding after the eighth negative pilot. The lead owns scientific review; independent agents check source interpretation, implementation and numerical evidence. All earlier files, checkpoints and failed gates remain immutable.

## Question and comparison

Does the shared change from the initializer's declared BCE smoothing convention to unsmoothed targets explain part of the retention loss? Run only two new continuations: manual-only with epsilon0.5 and manual-plus-pseudo with epsilon0.5. Reuse the completed eighth A/B epsilon0 arms and unchanged seed42 artifacts exactly. This forms a small 2×2 comparison of smoothing and inclusion of the fixed teacher term. Do not rerun historical training or pick new hyperparameters from the outcomes.

The explicit convention under test is y_s=(1−epsilon)y+epsilon/2, hence targets0.25/0.75 at epsilon0.5. Confirm this convention against pinned source before executing. The checkpoint config declares0.5, but this experiment restores only that BCE convention, not the complete original trainer: the retained configuration includes a combined Dice/BCE objective that the pilot does not reproduce. Historical trainer revision and full exposure remain unresolved.

Let U_e be mean BCE over the unchanged supplied manual participation and P_e the mean over the unchanged hard teacher-label participation, after applying epsilon to participating targets only. Compare:

- A0 (historical): U_0.
- B0 (historical): U_0+0.25P_0.
- A05 (new): U_0.5.
- B05 (new): U_0.5+0.25P_0.5.

Frozen seed42 is the retention reference. The failed balanced-anchor arm is not continued or added as another optimization choice. Masks and class membership retain hard-label semantics; smoothing must never turn an unknown target into background or alter participation.

## Everything else stays fixed

Use the exact eighth seed42 initializer/factory, fixed seed43 teacher maps and0.5 hard teacher threshold, two PHerc0139 adaptation regions, manual S-minus-historicalV mask and pseudo support including openly retired historicalV. No PHerc0814 gradient updates. HistoricalV ink values do not enter training or calibration.

Reuse the exact eighth32-step schedule, SHA25687bb53a0a80ba2268b1b2b020342058c2facd44f343c53b8a5ac478da06e40e1: batch two, one eligible patch per region, unchanged17×128×128 inputs, robust-MAD normalization and shared XY augmentation. Keep AdamW lr1e-4, betas(.9,.999), epsilon1e-8, zero weight decay, no scheduler and global gradient clipping1.0. FP32 CPU, four threads, deterministic algorithms, same runtime package versions. Each new arm loads the identical initializer with a new optimizer; exactly32 updates and only the final checkpoint are retained.

The first actual forward, before its backward/update, must reproduce the historical first-step unsmoothed U/P within absolute1e-7 and have identical initialization and schedule identity. Independently check the smoothed first-step BCE using softplus within1e-6. This sentinel adds no extra forward or optimization step. Analytic artificial-logit fixtures must demonstrate epsilon-zero equivalence, the0.25/0.75 target mapping and zero output-logit gradients outside participation. Preserve full checkpoint schema, shared-storage aliases, finite values and safe strict reload.

## Reused evaluation is development evidence

No new scans, labels or reference regions are acquired. The eighth study's two formerly fresh PHerc0814 supplied-supervision regions are now explicitly **reused development retention diagnostics**. The same three old development regions determine calibration. No ninth output is described as a new held-out result, independent ink accuracy, external generalization or an authentic reading.

Keep both eighth calibration rules: primary is the maximum of three per-region conservative empirical95%-specificity cutoffs; secondary uses PHerc1667 alone. Recompute each new model's thresholds only from those fixed calibration negatives, write the cutoff receipt before diagnostic inference, and retain old models' actual saved cutoffs/maps. Never fit a diagnostic-region cutoff to improve a result.

Report every model's per-region AUC, AP, prevalence, primary/secondary sensitivity, achieved specificity and confusion counts; retain raw maps and curves. Report A05−A0 and B05−B0 separately, B−A at both smoothing levels, and their difference as a descriptive interaction. Aggregate the two diagnostic contexts equally. They belong to one specimen and are not independent replicates; no pixel confidence intervals or significance claims.

Predeclared descriptive screens, applied separately to A05 and B05:

1. **Consistent mitigation relative to its unsmoothed counterpart:** primary specificity≥0.95 in each context, mean primary sensitivity gain≥0.02, no per-context sensitivity loss>0.02, AUC loss>0.005 or AP loss>0.01.
2. **Within the specified baseline-retention margins:** primary specificity≥0.95 in each context, and every context within0.02 sensitivity,0.005 AUC and0.01 AP below frozen seed42.

A shared smoothing explanation receives support only if both pairs pass the mitigation screen; mixed directions imply an interaction or unresolved variation. Passing a retention margin does not demonstrate superiority or justify model promotion. Preserve continuous differences and all failures. No post-result retry, new epsilon, extra steps or checkpoint selection is allowed within this study.

## Resources and decision

There are64 new updates total. Each worker has120 seconds wall,360 CPU seconds,4 GiB peak-RSS acceptance and12 GiB hard address space; each checkpoint file is capped at200 MiB. Total new run/analysis outputs must fit400 MiB. No automatic retries, new dependencies, GPU launch, R2 write, publication or outreach. Source-code verification alone may use the separately bounded128 KiB/18-request public-code audit; training and inference are offline.

If smoothing consistently mitigates loss, carry that convention into a separately frozen next design with stronger reference support. If not, retain the initializer and study continuation stability/knowledge preservation as a separate factor. Do not add replay, change sampling or alter learning rate during this control. Neither outcome establishes novelty. The £10 aggregate first-GPU cap remains untouched; this control requires no paid resources or new credentials.
