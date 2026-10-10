# Thirteenth technical review: acceptance checks

Prepared before implementation review or scientific execution. Source-only review; no real arrays, checkpoints or model calls by this reviewer. Root owns scientific decisions. Frozen earlier evidence remains unchanged.

Bound scientific design: `../RESEARCH_DESIGN.md`, SHA256 `59bee53d3f1a662b37aec43fde31698ce0a9946ea0dd0de7027c01c3cf254f36`, independently checked after root sealed it. This checklist does not alter its conditional retention criteria or authorize extra baseline evaluation.

## Matched continuation

- Exactly A_preserve and B_preserve, 32 updates each, constant learning rate 1e-6. Initializer, parameter order, AdamW settings, clipping, smoothing, support masks, schedule, augmentation, normalization and loss code match the retained tenth run.
- Objective is smoothed manual BCE plus forward Bernoulli KL from initializer to student; B adds 0.25 times the separately normalized pseudo BCE and KL. Teacher must be detached. Full preservation gradients are required after the first step; the first-step zero-KL equivalence does not extend to later steps.
- Each regenerated batch must match its archived input/target fingerprint in both historical arms. Cached teacher logits must retain their exact step/batch ordering, float32 shape and values; corresponding teacher arrays from the two arms must match bitwise. Cache reuse is justified by exact input matching and the already audited deterministic model contract, not by patch coordinates alone.
- First preupdate logits and the first postupdate state/logits must match the twelfth 1e-6 branch under the explicitly frozen equality rule. No tolerances may be expanded after observing a mismatch.

## Measurement isolation and meaning

- Each immediate pre/post pair uses the same batch, masks, targets and teacher. Postupdate measurement must not modify parameters, buffers, gradients, optimizer moments, RNG or subsequent batch construction.
- Fixed risk evaluates the complete retained 32-batch schedule at updates 1, 8, 16 and 32. Preserve repeated and overlapping exposures. Average the 32 independently support-normalized batch risks equally; do not pool all pixels or deduplicate repeats.
- Initial fixed risk comes from the same cached initializer logits and same masks/targets, with initializer KL exactly zero. Report manual/pseudo, base/full and drift separately. The final gate compares each arm with its own identical fixed baseline, requiring both base and full changes <= -1e-4.
- Fixed-risk forwards must preserve full model state, mode, RNG, optimizer state and outstanding gradients. They must not create autograd graphs or trigger additional updates.
- Retained 1e-4 preupdate online losses may be compared with corresponding new preupdate online losses. They are not a retained final fixed-risk measurement. New fixed risk is compared with its initializer. Historical final score arrays may support the separately frozen retention comparison.
- Adaptation risk and post-hoc seen development contexts are not independent validation. Gate selection is on adaptation risk only; no rate, update count, threshold, support or orientation selection from retention outcomes.

## Accounting and fail-closed behavior

- Per arm: 32 training preupdate forwards + 32 immediate postupdate forwards + 4 x 32 fixed-risk forwards = 192 forwards, 32 backwards and 32 updates. Total: 384 forwards, 64 backwards and 64 updates. Cached baseline requires no teacher forward.
- Freeze source, plan, historical receipts and resource limits before execution. Enforce 180 seconds per worker, 420 seconds combined, 4 GiB peak RSS acceptance, 4 CPU threads and 450 MiB combined output according to the final plan; distinguish sampled/reported peak RSS from a kernel-enforced memory limit.
- Fail on nonfinite losses, changed inputs/state/RNG, missing support, fingerprint/sentinel mismatch, timeout or accounting overrun. Preserve failure artifacts and do not retry automatically.
- Retain enough pre/post and fixed-risk logits and per-step records for independent loss reconstruction without model execution. Clearly separate independent numeric reconstruction from worker-reported full parameter/optimizer checks.

This checklist is not execution approval. A code-bound final review follows implementation inspection.
