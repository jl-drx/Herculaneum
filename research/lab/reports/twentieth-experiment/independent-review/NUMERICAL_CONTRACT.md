# Phase 20 independent numerical contract

Frozen before any phase 20 scientific output reads. The governing recommendation is phase 19 `NEXT_STUDY_DESIGN.md`; the exact root phase 20 protocol, configuration, execution plan and root execution authorization will be pinned before actual output review. This contract carries forward phase 18 numerical tolerances without relaxation.

The independent reviewer uses separately implemented NumPy mathematics and indexing. It imports no producer numerical code or Torch/model implementation, decodes no model checkpoint and does not replay training. Scientific output reads require exact protocol and completed plan SHA256 pins, plus applicable root authorization. All reviewed input/output/source pins are checked before and after review. Every attempted check and unfavorable or undefined outcome is retained.

| Comparison | Frozen acceptance rule |
| --- | --- |
| Input and feature arrays/fingerprints; initial saved logits versus cached seed42 teacher; diagnostic output difference (student minus teacher); same current-batch immediate/fixed outputs | Exact dtype/shape and float32 bit equality, or exact declared SHA256 |
| Independently reconstructed FP32 smoothed BCE, base, hard BCE and fixed risk | Absolute error ≤ 2e-7 |
| FP64 KL and probability drift | Absolute error ≤ 1e-12 |
| Full objective from independent BCE plus KL | Absolute error ≤ 2e-7; producer component addition also checked at 1e-12 |
| FP32 preclip global gradient norm | Absolute error ≤ 1e-6 + 2e-7 × absolute expected norm |
| Reported FP64 gradient-group and parameter-update L2 norms | Absolute error ≤ 1e-12 |
| Global clipping coefficient | Absolute error ≤ 2e-7 |
| Supported manual logit derivative | Absolute error ≤ 2e-10 per scalar; every nonmanual derivative exactly zero |
| AdamW parameter/moment arithmetic, each scalar | Absolute error ≤ 2e-8 + 8 × largest float32 ULP spacing among corresponding before/after/expected values |
| Adjacent parameter/moment trace boundaries and initial parameters across arms | Float32 bit equality |
| Development sigmoid/Hann-blended score maps | Absolute error ≤ 2e-6 on exact declared support |
| AUROC and average precision | Absolute error ≤ 1e-12 |
| Counts, placements, supports, membership, shapes, steps and call accounting | Exact |
| Published gate arithmetic | Exact recomputation from producer's declared scalar metrics; no threshold tolerance |

The independent reviewer computes every decision component from separately reconstructed losses, maps and ranks. Any disagreement in the pass/fail classification of declared and independently reconstructed values is an explicit numerical margin ambiguity. Report distances to all boundaries. Do not promote, loosen a boundary, choose a favorable arithmetic path or conceal ambiguity.

The manual derivative is `[2 * sigmoid(student) - sigmoid(teacher) - (0.25 + 0.5 * hard_target)] / manual_count`, with exactly zero derivative on unknown/pseudo-only outputs. The objective is the equally weighted mean of per-batch manual FP32 smoothed BCE plus FP64 Bernoulli KL(seed42 || student), not a unique-pixel mean.

Both adapters have 5,296 ordered parameters: stem.weight (8,3,3,3,3), stem.bias (8), projection.weight (32,16,3,3), projection.bias (32). Every optimizer vector must contain all 5,296 entries for every one of the 32 updates in each arm. AdamW is learning rate 0.001, betas (0.9,0.999), epsilon 1e-8, zero weight decay, foreach=False and fused=False. Global L2 clipping has maximum norm 1 and clip epsilon 1e-6.

Unlike phase 18, this adapter changes the 32-channel fused representation before the frozen U-Net. New student logits must not be required to equal teacher plus a purported output-head residual. Saved logit differences are only the diagnostic arithmetic `student - teacher`; this does not prove the mechanism. Fused-feature cache fingerprints, state/mode hashes and residual summaries are checked for consistent lineage, shape and finite values, but no hidden feature array or convolution is independently replayed. New backbone loads and forwards occur in the producer and must be counted honestly. The reviewer performs none.

Producer autograd supplies parameter gradients. Independent checks verify the output-level derivative and masking, clipping, AdamW arithmetic, all ordered state transitions and published loss/rank arithmetic. These checks do not independently prove convolutional autograd, hidden-feature-to-logit execution or execution history. Pinned source review, retained execution receipts and original-state/mode checks are complementary evidence, not substitutes for fresh scientific validation.

Synthetic fixtures precede scientific reads and include all eight D4 transforms, centered and full one-sided depth stencils, signed nominal coordinates, clamped neighboring slices, feature/D4 commutation, robust-normalization fallback and immutability, zero identity, extreme logits, unsupported-output exclusion, equal-batch reduction, tied ranks, inclusive cutoff, changed-pin rejection, checkpoint-read rejection, all 5,296 optimizer coordinates, clipping branches, 32 scalar-oracle optimizer recurrences and deliberate corruptions. No actual scientific result will alter tolerances.

All six Paris1/Paris2 contexts remain reused development evidence. No independent confirmation, model promotion, paid-resource authorization, network access, upload or publication follows from this check.
