# Tenth study: fixed initial-prediction preservation during continuation

Scientific contract fixed before new model calls, 8 October 2026. Root leads strategy and executes all real model work; independent agents audit implementation, numerical evidence and reference provenance. This is a conventional distillation/preservation control, not a proposed novel method. Preserve all prior studies and the ninth final manifest unchanged.

## Question and arms

Does penalising prediction drift from the initializer on the same adaptation patches mitigate the loss of supplied-annotation retention, without changing data support, sampling, optimizer or calibration? Reuse the exact ninth A_smooth and B_smooth results plus frozen seed42. Train exactly two new arms, A_preserve and B_preserve, for 32 updates each. No eighth hard-target or C arm is retrained, no coefficient/seed sweep, and no intermediate checkpoint is selected.

Let z be the FP32 student's preupdate logits and t the detached FP32 logits of a separate frozen seed42 model given the **identical normalized and augmented patch tensor**. Temperature is 1. The teacher target is q=sigmoid(t); it is neither thresholded nor smoothed. Manual and original seed43 hard-pseudo targets retain ninth epsilon-0.5 smoothing, y_s=0.25+0.5y, with unchanged hard masks and class identities.

For each existing participating mask M or P, define

`D(z,t) = mean_M_or_P[logaddexp(z64,0) - logaddexp(t64,0) - sigmoid(t64)*(z64-t64)]`.

This is KL(Bernoulli(q)||Bernoulli(sigmoid(z))) in float64 arithmetic from the actual FP32 logits. Teacher logits are detached. Models, optimizer and saved logits remain FP32. Derivative with respect to z is sigmoid(z)-sigmoid(t), mapped back to the student's FP32 gradient. Do not clamp small numerical negatives; require each participating mean >= -1e-6 and finite. Use lambda=1 exactly, chosen as equal weight to each original supported target term, without outcome-based tuning.

- Historical A_smooth: U_s. New A_preserve: U_s + D_M.
- Historical B_smooth: U_s + 0.25 P_s. New B_preserve: U_s + 0.25 P_s + D_M + 0.25 D_P.

The original terms' coefficients are unchanged. This adds a preservation objective; no normalization by two or learning-rate adjustment is introduced. D_P is recorded but has zero coefficient for A_preserve. The initializer supplies a model-derived preservation target, not an additional manual annotation or independent source of truth. No formerly unknown pixel outside the original active mask gains a loss contribution: A uses M; B uses M union P. No PHerc0814 patch participates in gradients or preservation/replay.

## Matching and execution

Use the exact ninth/eighth initializer, fixed seed43 hard-teacher maps, two PHerc0139 adaptation regions, manual S-minus-historicalV and same pseudo support. HistoricalV ink values never become targets or calibration data. Exact 32-step schedule SHA256 `87bb53a0a80ba2268b1b2b020342058c2facd44f343c53b8a5ac478da06e40e1`; batch two, 17×128×128 inputs, robust-MAD normalization, same XY D4 choices. AdamW lr1e-4, betas(0.9,0.999), epsilon1e-8, zero weight decay, no scheduler, global gradient clipping1.0. Seed20261008; CPU four intra-op/one inter-op threads and deterministic algorithms. Current lab Python3.12.14, torch2.12.0+cpu, NumPy2.5.3; no dependency change.

Construct and strictly load the frozen teacher separately. Its parameters/storage must not alias the student's, all teacher parameters have requires_grad=False, and it remains eval/no_grad. Preserve the student's CPU RNG around teacher construction and verify teacher forwards do not change it. Before use, inspect the actual module configuration for nonzero dropout/stochastic depth, BatchNorm or tracked running-stat normalization and stop if train/eval equivalence is not supported. No preliminary real forward is allowed to tune the design. The first actual student/teacher forward pair is the sentinel. Teacher semantic state must equal the initializer initially and remain unchanged at completion, with no gradients.

At the first actual training step, before its single backward/update: initializer/schedule and normalized-input/target/mask fingerprint must equal ninth; old hard U/P match within absolute1e-7, and smoothed U/P match the corresponding ninth step within1e-7. Student/teacher logit maximum difference <=1e-7; preservation element-value maximum magnitude <=1e-7 and analytically computed preservation logit-gradient maximum <=1e-7. First combined parameter gradient norm must agree with the historical arm within absolute1e-6. These checks add no extra student forward or optimizer update. Stop on mismatch without changing the thresholds or retrying scientific execution.

Retain the preupdate student logits and frozen-teacher logits for every step as FP32 arrays of shape(32,2,1,128,128), plus per-step normalized-input/target/mask hashes, mask/class exposures, original U/P/R, D_M/D_P, total loss, gradient norm, inactive-output gradient check, parameter-update probe and resource use. Teacher predictions must match across both arms because inputs/teacher are identical. Save only the final raw student state dictionary; preserve finite values, all schema/storage aliases and safe strict persisted reload before evaluation. Do not save intermediate weights or optimizer state.

There are64 new student backward/update steps and64 extra frozen-teacher batch forwards in total, plus the usual final evaluation for each new student. Historical arms/maps are reused, not regenerated. Root runs the sequential controller once after protocol/code/environment/source freeze and independent pre-run review. Artificial-logit loss/derivative checks may run during preparation; reviewers may reconstruct losses and read safe checkpoints without real model forward/backward.

## Evaluation and predeclared interpretation

Use the same three fixed calibration regions and same two **reused development retention diagnostics** as ninth. No new region or label array is acquired for this study. The separate reference audit may inspect bounded official metadata and reserve candidates for a later study; those candidates never enter this run. Historical pixel/teacher exposure and annotation lineage remain unresolved; no fresh holdout, generalization, independent ink accuracy, reading, significance or novelty is claimed.

New thresholds use only the fixed calibration negatives and are written before diagnostic inference. Primary remains max of three per-region conservative empirical95%-specificity cutoffs; secondary remains PHerc1667/w029-only. Reuse literal historical raw maps/thresholds. Report per-context AUC/AP, prevalence, primary/secondary sensitivity and achieved specificity, confusion counts, raw score distributions and equal-context means. No per-diagnostic cutoff fitting, threshold shopping, pixel confidence intervals or significance testing. Report A_preserve−A_smooth and B_preserve−B_smooth, the B−A effects at both preservation levels, and their difference.

Apply the same descriptive screens to each new arm, at full precision:

1. **Mitigation versus its matched ninth comparator:** primary specificity>=0.95 in both contexts; mean primary sensitivity gain>=0.02; per-context sensitivity delta>=-0.02, AUC delta>=-0.005 and AP delta>=-0.01.
2. **Baseline-retention margins:** primary specificity>=0.95 in both contexts and per-context sensitivity/AUC/AP no more than0.02/0.005/0.01 below frozen seed42. No positive mean gain is required for this margin screen.

Shared mitigation is supported descriptively only if both matched pairs pass. No passing screen promotes a model or establishes an independent improvement. If either pair fails, retain the mixed/failing evidence and do not silently choose only the favorable arm.

Report the adaptation cost: mean U_s and weighted original objective over the fixed last eight preupdate batches (steps25–32), compared with the corresponding historical ninth scalar receipts and with U_s/P_s evaluated on the frozen teacher's saved logits for those same inputs. Report KL drift and mean absolute probability drift on participating masks, plus per-step curves. These are matched online training diagnostics from evolving models, **not final-model accuracy**. Positive loss reduction versus the initializer alone cannot establish useful transfer; preservation with no adaptation should not be called successful adaptation. These continuous diagnostics do not introduce an additional model-selection threshold.

## Bounds and stop rules

Each worker:120 seconds wall,360 CPU seconds,4 GiB peak-RSS acceptance and12 GiB hard address space. Each output file<=200 MiB; combined new run/analysis<=400 MiB. Strict offline model work, no automatic retry, new seed, coefficient change, extra student evaluation or dependency installation. The frozen teacher roughly doubles training forward work, not backward work; measured use must pass these bounds. On a genuine code defect, preserve the failed run and diagnose before any separately documented corrective execution; no optimization change may be disguised as a fix.

No GPU, R2 write, publication, submission, outreach or credential import. Paid GPU spend remains£0 and total first-pilot cap£10 including failures/retries and attributable overhead. The outcome determines whether a broader preservation design merits a separately reserved reference study; it does not authorize scaling a negative recipe or claim frontier performance.
