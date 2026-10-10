# Phase 21 preservation-allocation numerical contract

Frozen before new scientific outputs. Governing root protocol SHA256: `498bf89ca0944ca26f596bdc8041951a82365667ccddf4eb88a8a1f0b33998b1`. Both 5,296-parameter adapters receive identical explicit-depth features and differ only in the declared allocation objective: A (`manual_double`) is B_M + 2 K_M; B (`manual_retired`) is B_M + K_M + K_V. M and explicitly retired supplied-reference V are disjoint and nonempty. Their sealed support/source hashes must match the final plan and independent reconstruction. They are not authenticated geometry-valid masks.

The independent reviewer uses NumPy mathematics and indexing; no producer numerical imports, Torch/model execution, checkpoint decoding, training replay or network calls. Scientific reads require a pinned final plan, root GO and retained-output authorization. All inputs and checker sources are pinned and rechecked; every failed attempt and unfavorable result is retained.

| Comparison | Frozen acceptance rule |
| --- | --- |
| Inputs, explicit-depth features, M/V/source-index fingerprints, initial original/zero-adapter logits versus teacher, student-minus-teacher diagnostic arrays, fixed/current-batch identity | Exact dtype/shape and float32 bit equality or exact declared SHA256 |
| Independently reconstructed manual smoothed BCE, hard BCE, own full and common objective_A/objective_B | Absolute error ≤ 2e-7 |
| FP64 K_M, K_V and probability drift on each stratum | Absolute error ≤ 1e-12 |
| Producer objective_A = base + 2 K_M; objective_B = base + K_M + K_V; own full equals the correct common formula | Component arithmetic ≤ 1e-12; exact own/common field equality |
| FP32 preclip global gradient norm | Absolute error ≤ 1e-6 + 2e-7 × absolute expected norm |
| Global clipping coefficient | Absolute error ≤ 2e-7 |
| Supported output derivative | Absolute error ≤ 2e-10 per scalar; unsupported output derivatives exactly zero |
| Reported FP64 gradient-group and parameter-update L2 norms | Absolute error ≤ 1e-12 |
| AdamW parameter/moment scalar arithmetic | Absolute error ≤ 2e-8 + 8 × largest corresponding float32 ULP spacing |
| Adjacent parameter/moment trace boundaries and initial tensors across arms | Float32 bit equality |
| Development sigmoid/Hann score reconstruction | Absolute error ≤ 2e-6 on exact declared support |
| AUROC and AP | Absolute error ≤ 1e-12 |
| Counts, support membership, placements, steps and call accounting | Exact |
| Declared mechanism/practical gate arithmetic | Exact recomputation from producer scalar values; no boundary tolerance |

These acceptance bounds carry forward phase 20. For the new derivative, p = sigmoid(student), q = sigmoid(teacher) and y_s = 0.25 + 0.5 y. A has `(p−y_s + 2(p−q))/|M|` on M and zero elsewhere. B has `(p−y_s + p−q)/|M|` on M, `(p−q)/|V|` on V and zero elsewhere. Retired hard-label values do not enter any objective, diagnostic label score or derivative. Unknown support is excluded. Historical full-array bytes may still be read solely to verify unchanged historical input fingerprints and must be reported honestly.

Every per-step group mean uses the participating outputs across that batch, not an equal mean of region means. All-schedule diagnostics use math.fsum across 32 batch records divided by 32, preserving repeated exposures. Record base, K_M, K_V, own full, both common objectives, manual hard BCE and both probability drifts. Unlike objectives must not be directly compared as a shared scalar endpoint.

All 5,296 ordered parameters (stem.weight/bias, projection.weight/bias), pre/post gradients, parameters and Adam moments are required at every update. AdamW remains rate 0.001, betas (0.9,0.999), epsilon 1e-8, zero decay, foreach=False, fused=False and global clip norm one with epsilon 1e-6. Independently check both arms' updates. Report first-step inter-arm parameter, output-logit and output-gradient maximum absolute differences, changed-scalar counts and bitwise equality as **diagnostics only**. There is no additional inter-arm acceptance threshold and no tolerance chosen from actual outcomes. Floating-point KL cancellation can prevent an algebraic zero from being an exactly zero derivative; matching initial state is still exact.

The feature residual is inserted before the frozen U-Net. Saved output-logit differences are only student minus teacher diagnostics and do not establish the mechanism. Hidden cache, residual and decoder-input-gradient hashes/norms are checked as execution receipts. This reviewer does not reconstruct hidden convolutions, replay autograd or independently prove execution history. Original-model state/mode and checkpoint reload receipts supplement source review; checkpoint bytes are not decoded here.

Five mechanism components and fourteen practical components are computed both from independently reconstructed values and from producer scalars. Report all margins and any pass/fail ambiguity; never select the favorable arithmetic route. Final development must retain both arms and all six reused contexts even if mechanism criteria fail. No fresh confirmation, baseline promotion or GPU authorization follows from either screen.

Before scientific reads, synthetic fixtures exercise both derivative formulas, ignored retired/unknown targets, nonempty/disjoint support, exact unsupported zeros, teacher direction and extreme logits, separate-group versus union reductions, D4 and explicit-depth features, objective arithmetic, all optimizer coordinates, adversarial trace corruption, tied ranks, inclusive cutoff, pin refusal and diagnostic-only inter-arm differences. Algebraic float64 synthetic identities use absolute 1e-12; moderate-logit central finite differences use step 1e-5 and absolute 2e-9. These are fixture checks, not relaxed scientific-output bounds.
