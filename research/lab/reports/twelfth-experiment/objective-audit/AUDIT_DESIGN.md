# Offline objective audit — preparation before numeric execution

Root owns scientific review. This stage reads retained code and JSON receipts, then prepares a fixed NumPy-only reconstruction. No real logit/mask arrays have been evaluated during preparation; no network, checkpoint deserialization, model construction, forward/backward or optimizer call is authorized here. Prior artifacts remain unchanged.

The static source finding is precise: tenth's `original_objective` is already the **epsilon 0.5 smoothed BCE base objective**, exactly the base term included in its optimized loss. The same-input initializer reference uses the same smoothed targets. It is not a comparison of unsmoothed BCE against a smoothed optimized loss. “Original” here means the ninth base objective before adding preservation; it does not mean the eighth hard-target objective or the upstream combined Dice/BCE recipe.

## Fixed objectives and roles

Let `s = 0.25 + 0.5*y`, `p = sigmoid(z)`, `q = sigmoid(t)`, and `bce(z,s) = logaddexp(0,z) - s*z`. Manual hard targets come only from supplied ink values on `M = observed & S & ~V`. Hard pseudo targets come from the **retained seed43 blended score map >= 0.5**, on `P = observed & seed43_evaluated & finite(seed43_scores) & ~M`. Retired validation is deliberately eligible for P; its ink label bits do not enter losses. Preservation's `t` is a different source: the frozen seed42 initializer's logits on the exact current normalized and augmented patch, not the seed43 pseudo map.

`U` and `P_loss` are separate means over their own participation across both batch regions. Their sample counts do not change the B coefficient. A optimizes `F_A = U + D_M`; B optimizes `F_B = U + 0.25*P_loss + D_M + 0.25*D_P`. The base terms are `G_A=U` and `G_B=U+0.25*P_loss`. `D` is the separately normalized teacher-to-student Bernoulli KL on its support. The four equal-region/class manual means form R, which is diagnostic and has coefficient zero in A/B. Unknown output locations are omitted, while image context outside the loss masks may still influence supported outputs.

Ninth's actual optimized losses are `G_A/G_B` with the same smoothed targets and no preservation. Ninth retains online scalar receipts, not all-step raw logits. Its losses will therefore be identified as receipt-based historical evidence; no unavailable ninth logits or KL will be invented.

## Exact matched decomposition

For each support independently:

`D = mean[softplus(z)-softplus(t)-q*(z-t)]`

`L = mean[(q-s)*(z-t)]`

`G(z)-G(t) = D + L`

`F(z)-F(t) = 2*D + L` at lambda 1.

B combines the M and P decompositions with the same 1 and 0.25 coefficients. At the initializer, preservation is zero. Thus, if reconstruction confirms a positive smoothed base delta against the same-input initializer, nonnegative preservation makes the full-objective delta positive as well. This is an output-space identity, not a claim about the sign of an individual parameter update or its cause.

For each participating output, the unconstrained full-loss optimum is `p*=(s+q)/2`, with `z*=log(p*/(1-p*))`. Compute its value as a mathematical batch diagnostic. It is not evidence the shared network can reach those independent logits, nor a generalization bound. Hard BCE on targets 0/1 is a separately labeled secondary diagnostic; its direction need not match smoothed BCE.

## Frozen comparison and summaries

Reconstruct all 32 preupdate steps in both A_preserve and B_preserve from their saved FP32 student/teacher logits. Use independent float64 NumPy algebra, exact retained masks, the frozen six-tile schedule and the recorded D4 augmentations. Do not import or execute the historical PyTorch loss or model modules. Verify every mask count/exposure against the step receipts and all reconstructed smoothed U/P/R, teacher U/P, KL, base and full losses against their recorded scalars.

Predefine windows 1–8, 9–16, 17–24, 25–32 and all 1–32. Window means are equal-weight means of per-step masked means, matching the prior late-eight report. Report same-batch teacher baselines and paired deltas; comparing unpaired early/late raw means alone can confound composition. Retain each step's M/P counts, hard positive/negative counts, regional mass and all region/class strata; pseudo-positive strata may be empty and remain explicitly undefined. Per-region/class summaries are diagnostic and do not retroactively change normalization or loss weights.

Output `audit.json` will contain `arms[arm].steps`, predefined `windows`, source/exposure verification, reconstruction errors, decomposition residuals, hard/smoothed/optimized objectives, free-logit optimum values and receipt-only ninth comparisons. A compact `steps.csv` will expose the main paired scalar traces. Record source identities before and after, root's final plan identity, runtime and resource usage. No saved model weights or new research predictions are produced.

Proposed fixed tolerances match the retained tenth independent review: 2e-6 absolute for reconstructed FP32 BCE/scalar objectives, 1e-12 for float64 KL/decomposition identities, and the original mean-KL floor -1e-6. Exact array encodings, shapes, source hashes, schedule/augmentation and exposure counts are hard gates. First-step student/teacher equality uses the original 1e-7 absolute guard. Stop on any failure without tolerance relaxation or an alternate recipe.

## Limits of the inference

The logs are preupdate predictions from evolving models on a changing schedule; step 32 describes the state before update 32. They do not give final-model loss on every training batch or a within-step before/after comparison. Adam momentum, clipping, shared parameters and changing batches can make online costs nonmonotonic. This audit cannot isolate learning-rate, representation, update-sign or data-quality causes from one scalar curve. It can establish which objective was measured, whether its algebra/support agree with the optimized recipe, and how curvature and signed initial-gradient drift account for the saved displacement. New model work needs a separate root-reviewed design.
