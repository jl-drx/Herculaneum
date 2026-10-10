# Phase18 independent numerical contract

Frozen before phase18 scientific outputs. Root approved these tolerances in-session on2026-10-08. The governing protocol is `../RESEARCH_DESIGN.md`, SHA256 `00a0e9a0860848153a7b1faa857a5210c8ff4abdb463a1ff691f40e3a7925f61`.

The independent reviewer uses separately implemented NumPy mathematics and indexing. It imports no author numerical code or Torch/model implementation, decodes no model checkpoint, and does not replay training. Scientific reads require the protocol and completed plan SHA256 pins plus the applicable root authorization. All input/output/source pins are checked before and after review; each attempt and unfavorable/undefined outcome is preserved.

| Comparison | Frozen acceptance rule |
| --- | --- |
| Input and feature arrays/fingerprints; zero-initialization identity; cached-reference plus residual composition; same current-batch immediate/fixed outputs | Exact dtype/shape and float32 bit equality, or exact declared SHA256 |
| Independently reconstructed FP32 BCE, base, hard-BCE, and fixed-risk values | Absolute error <=2e-7 |
| FP64 KL and probability drift | Absolute error <=1e-12 |
| Full objective reconstructed from independent BCE plus KL | Absolute error <=2e-7; producer full arithmetic from its saved components additionally checked at1e-12 |
| FP32 preclip global gradient norm | Absolute error <=1e-6 +2e-7*abs(independent expected norm) |
| Global clipping coefficient | Absolute error <=2e-7 |
| Supported manual logit derivative | Absolute error <=2e-10 per scalar; every nonmanual output derivative exactly zero |
| AdamW ordered parameter/moment arithmetic, each scalar | Absolute error <=2e-8 +8*max(float32 ULP spacing of the compared input/output values) |
| Repeated adjacent parameter/moment trace boundaries and initial parameters across arms | Float32 bit equality |
| Development sigmoid/blended score maps | Absolute error <=2e-6 on the exact declared evaluated support |
| AUROC and AP | Absolute error <=1e-12 |
| Counts, placements, supports, membership, shapes, steps/call accounting | Exact |
| Published gate arithmetic | Exact recomputation from the producer's declared scalar metrics, with no threshold tolerance |

Float32 ULP spacing means absolute `np.spacing(np.float32(value))`. The per-scalar AdamW comparison uses the largest spacing among the corresponding saved pre/post value and independently reconstructed value. This bounded tolerance permits independently rounded FP32 recurrences, not a refit of optimizer hyperparameters.

The reviewer also computes every decision component from independently reconstructed losses, score maps, and ranks. If numerical acceptance permits a producer result to lie across a declared gate boundary from its independent reconstruction, report that margin ambiguity explicitly; do not silently promote, loosen the boundary, choose a favorable arithmetic path, or hide the alternative outcome. Report distances to all gate boundaries.

The gradient arrays are author-produced autograd outputs. Independent reconstruction verifies their support-level derivative/masking receipts where retained, global clipping and AdamW arithmetic, trace continuity, and resulting published loss/rank arithmetic. It does not independently prove the convolutional autograd calculation or execution history. Source review and pinned execution receipts provide the complementary evidence. Cached seed42 logits establish no new backbone execution: new backbone loads/forwards/updates are all zero.

The retained manual logit derivative is independently reconstructed as `[2*sigmoid(student)-sigmoid(teacher)-(0.25+0.5*hard_target)]/manual_count`. The2e-10 absolute bound was added before any outputs when the producer's explicit logit-gradient retention became concrete; it accommodates the separately rounded FP32 BCE and FP64 KL branches. It does not permit nonzero unknown/pseudo-only derivatives.

Synthetic fixtures precede scientific reads and include all eight D4 transforms, depth endpoint stencils, signed nominal coordinates, raw neighboring-slice clamps, feature/D4 commutation, robust-normalization fallback and immutability, zero-reference identity, extreme logits, unsupported-output exclusion, equal-batch reduction, tie-aware rank metrics, inclusive cutoff, pin rejection, and AdamW boundary/recurrence cases. No scientific result will be used to alter these tolerances.
