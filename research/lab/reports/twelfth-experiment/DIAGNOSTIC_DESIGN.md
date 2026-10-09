# Twelfth study: diagnose the objective and input domain

Recorded before the new numeric audits or real model execution on 8 October 2026. Prior first–eleventh evidence is immutable. This study does not train a replacement model, retune Frag4 cutoffs or seek a better validation score.

## Questions and fixed evidence

The tenth run's `original_objective` already means the epsilon0.5 smoothed base BCE: A uses U; B uses U + 0.25P. It excludes the newly added prediction-preservation KL, but it does not use hard-label BCE. The same-input initializer comparison uses the same smoothing. A hard-versus-soft target mix-up therefore cannot explain the reported increase.

The saved 32 pre-update student and teacher batches for each preservation arm permit an exact offline loss audit. They do not show how a particular update affected that same batch afterward, and they do not measure the final model on a fixed full training set. Ninth's historical losses remain receipt-based because its all-step logits were not retained. No missing predictions will be invented or silently recomputed.

Reconstruct every saved batch, both arms, and all manual/pseudo region/class groups. Report all 32 steps, fixed windows1–8,9–16,17–24,25–32, and the full sequence. Use the actual epsilon0.5 targets, weights and participation. Hard BCE is a named secondary diagnostic. Compare both the smoothed base and complete preservation objective with the frozen teacher on identical inputs.

For teacher probability q, smoothed target s and student/teacher logit displacement dz, use the exact per-support decomposition:

`base_loss_change = KL(q || p) + mean[(q − s) * dz]`

At preservation coefficient1, `full_loss_change = 2 * KL(q || p) + that linear term`. Apply the same0.25 coefficient and separate participant means to pseudo support in B. This separates a signed initial output-gradient term from convex curvature, without identifying a parameter-update cause. The unconstrained per-output optimum `p* = (s + q) / 2` supplies an oracle gap only; it is not a realizable network or a generalization result. Unknown pixels never become negatives or enter any loss.

## Separate input-domain audit

Use exactly seven retained contexts: the two Frag4 regions, both PHerc0139 adaptation regions, and the three existing calibration contexts. No new downloads, regions, ink labels, model scores or checkpoint reads are needed. Enumerate all159 existing stride64 input tiles; keep overlapping tiles and contexts explicitly non-independent.

Measure fixed raw, depth-profile, adjacent-sample and normalization statistics. The main domain is the full17-plane patch seen by the existing unmasked robust-MAD normalizer. A fixed64-pixel interior and differently named source/support masks are descriptive secondary domains, not replacements for prior evaluation masks. Preserve zero occupancy, clipping ties, MAD and fallback behavior separately.

Frag4 is54keV, native3.24µm uint16, pooled3×3×3 to float32 with no quantization and nominal9.72µm spacing. Earlier inputs are78keV, native2.399µm, delivered as level2 uint8 with four-plane depth pooling and nearest-even quantization at9.596µm. These are real contract differences, not measured causes of score differences. Test `normalize(x)` versus `normalize(257*x)` on all50 Frag4 tiles, giving209 actual normalizer calls in total, to check the proposed scalar-intensity explanation directly. Do not introduce or fit a new preprocessing rule. Similar marginal statistics cannot rule out a semantic domain shift.

## Conditional minimal step probe

After the retained-evidence audits and mathematical review, decide whether an immediate-update measurement is still missing. If needed, freeze a separate executable plan for the exact historical first batch, initializer, augmentation, masks, normalizer, fresh AdamW configuration and global gradient clipping. The finite diagnostic compares two objective families A/B at three predeclared learning rates:1e−6,1e−5,1e−4. Each candidate starts from the identical initializer. No sequence of updates, optimizer carryover, holdout inference, checkpoint publication or choice of a winning training recipe is permitted.

At initialization the teacher/student KL and its gradient are zero, so the first update is shared by the matched smoothed and preservation variants. The initial detached logits can serve as the teacher; no redundant teacher forward is needed. Retain same-batch pre/post logits, the actual smoothed/full objective, a clearly secondary hard BCE, input/state fingerprints, gradient clipping and update norms. Check the analytical first AdamW step and the archived first-batch sentinels. A sufficiently small first-order step must descend locally for a nonzero gradient, but the historical finite step need not.

This probe can diagnose first-step overshoot on that batch. It cannot establish why all32 steps transferred poorly, whether later minibatches interfere, or which learning rate generalizes. An absent same-batch measurement is a scientific evidence gap, not a reason to ask the user to repeat ordinary research authorization. Root alone executes real model work after the concrete plan and code review.

## Scope and interpretation

Begin with local NumPy/math and source checks; root may execute the bounded step probe only if the audits justify it. No paid GPU, network acquisition, R2 upload, publication, submission, outreach, terms acceptance or new credential request. GPU spend stays£0 and the£10 total pilot cap remains unchanged. Frag4 is seen diagnostic evidence and remains excluded from training, teacher targets and calibration. Report supported explanations, ruled-out explanations and remaining uncertainty separately; do not relabel an inconclusive mechanism as a demonstrated bug.
