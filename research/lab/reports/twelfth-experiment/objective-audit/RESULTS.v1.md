# Matched-input objective audit

The audit confirms that the previously reported “original objective” was already the same **smoothed BCE base objective** used in training, with target `s = 0.25 + 0.5y`. It excluded the added preservation penalty. A hard-versus-smoothed target mismatch does not explain the reported increase: both the smoothed base and the full optimized objective are higher than the matched-input initializer on average in every fixed eight-step window for both preservation arms.

This is a reconstruction of saved **preupdate** logits from changing models and batches. It establishes the observed online loss differences, not the effect of an individual optimizer update or the final model’s loss.

## Objectives and exact comparison

For A, the base is `G = U` and the optimized total is `F = U + D_M`. For B, `G = U + 0.25P` and `F = U + 0.25P + D_M + 0.25D_P`. `U` and `P` each use their own combined-batch participation mean. Region/class anchor loss was diagnostic only, with coefficient zero.

Manual participation is actually acquired `S` outside `V`. Pseudo participation uses finite, evaluated retained seed43 scores outside manual participation, including the openly retired `V` membership. The pseudo hard target is the retained seed43 blended score thresholded at `>=0.5`; it is not the seed42 preservation probability `q` on the current patch. No retired `V` ink value contributes to either loss.

Let `D` be the appropriately weighted `KL(q || sigmoid(z))`, and `L` the appropriately weighted mean of `(q-s)*(z-t)`. The reconstruction verifies

`ΔG = D + L` and `ΔF = 2D + L`.

All differences below are student minus the same-input initializer; lower is better. Means give each step equal weight. B includes the fixed 0.25 pseudo coefficient in `D` and `L`.

| Arm | Steps | Δ smoothed base | Weighted KL D | Signed term L | Δ optimized total |
| --- | --- | ---: | ---: | ---: | ---: |
| A_preserve | 1–8 | +0.024329 | +0.043057 | −0.018728 | +0.067385 |
| A_preserve | 9–16 | +0.022458 | +0.056087 | −0.033629 | +0.078544 |
| A_preserve | 17–24 | +0.023824 | +0.050140 | −0.026316 | +0.073964 |
| A_preserve | 25–32 | +0.005913 | +0.046585 | −0.040672 | +0.052498 |
| B_preserve | 1–8 | +0.030079 | +0.058058 | −0.027979 | +0.088136 |
| B_preserve | 9–16 | +0.009568 | +0.051389 | −0.041821 | +0.060957 |
| B_preserve | 17–24 | +0.011761 | +0.044774 | −0.033013 | +0.056535 |
| B_preserve | 25–32 | +0.003659 | +0.053870 | −0.050212 | +0.057529 |

![Window-average decomposition](figures/objective-decomposition.png)

Across all 32 steps, A has mean `ΔG=+0.019131` and `ΔF=+0.068098`; B has `ΔG=+0.013767` and `ΔF=+0.065789`. The first snapshot has exactly zero differences. Base differences are positive on 26/32 A snapshots and 23/32 B snapshots; full differences are positive on 31/32 and 30/32 respectively. The negative mean signed term in every window is offset by the positive KL term. That algebra does not identify a parameter-step direction or an optimizer cause.

In B’s last window, manual smoothed BCE rises by `+0.004201`, while pseudo smoothed BCE falls by `−0.002168`. After the 0.25 pseudo weight, this still leaves the base higher by `+0.003659`. Hard-target BCE also rises, but it is a secondary diagnostic, not the objective optimized in these runs.

## Participation and normalization

Both arms use identical schedules and support counts. Totals below are **repeated participating pixel exposures**, not unique pixels or independent observations.

| Steps | Manual positive / negative | Pseudo positive / negative |
| --- | ---: | ---: |
| 1–8 | 60,316 / 100,788 | 11,519 / 89,521 |
| 9–16 | 53,636 / 98,580 | 7,286 / 102,642 |
| 17–24 | 54,488 / 91,512 | 10,291 / 105,853 |
| 25–32 | 61,900 / 96,988 | 11,524 / 91,732 |

| Support | Positive exposures | Negative exposures | Total | Positive fraction |
| --- | ---: | ---: | ---: | ---: |
| Manual | 230,340 | 387,868 | 618,208 | 37.26% |
| Pseudo | 40,620 | 389,748 | 430,368 | 9.44% |

Region 0 contributes manual `146,181+/139,781−` and pseudo `29,465+/208,861−`; region 1 contributes manual `84,159+/248,087−` and pseudo `11,155+/180,887−`. Separate support means and the fixed pseudo coefficient determine the loss; raw exposure totals do not create another weighting rule. Unknown outputs never participate. Image context outside these output masks can still affect supported predictions.

The unconstrained per-output optimum of the full preservation objective is `p*=(s+q)/2`. In the last window the student-minus-optimum full-loss gaps are `0.074159` for A and `0.083938` for B, versus initializer gaps `0.021660` and `0.026409`. These are mathematical output-space diagnostics; the shared model is not guaranteed to attain these independent optima, and they say nothing about generalization.

## Verification and practical limit

The root-executed audit passed. Reconstructed logged scalars differ by at most `1.25×10⁻⁷`, well within the declared `2×10⁻⁶` tolerance. The maximum stepwise decomposition residual is `2.16×10⁻¹⁶`. Saved preservation-teacher logits are bitwise identical between arms, support/class counts match the training receipts, and first-step student/teacher loss differences are exactly zero. The audit took `0.331 s` and peaked at `46.1 MiB` RSS, with zero checkpoint loads, model calls, backwards, optimizer calls or network calls.

Independent objective reconstruction also passed: all 64 arm-step snapshots, 512 strata, 64 CSV rows and 14,841 scalar comparisons were checked. The separate first-step probe verification passed all six branches, with maximum stable-term error `2.78×10⁻¹⁷` and FP32-loss error `1.03×10⁻⁷`. Its parameter review covers 1,584 reported summary rows and selected retained coordinates, not independent replay of every parameter coordinate: complete raw gradients and updated states were not retained.

Ninth-run comparisons remain **scalar-receipt evidence** because its full preupdate logits were not retained. In the last window A_preserve’s base is `0.000198` lower than A_smooth’s receipt and B_preserve’s base is `0.005043` lower than B_smooth’s receipt, while both remain above the same-input initializer. This distinction prevents a relative improvement over an earlier adapted trajectory from being mistaken for an improvement over initialization.

The evidence warrants revisiting the short-run optimization recipe before another adaptation claim. It does not distinguish Adam state, clipping, shared-parameter coupling, step size or minibatch changes as the cause. Step 32 is saved before update 32; neither a final-model training loss nor an immediate before/after update was measured here. No new training, inference, data acquisition or threshold fitting was performed for this audit.

## Separate first-step probe

A separate root-executed probe measures immediate before/after losses on the same first batch, resetting initialization and the optimizer for each of three predeclared step sizes. Its saved results show that base and optimized loss must be distinguished even in this local comparison.

![Separate root first-step probe](step-probe-figures/first-step-loss-change.png)

At `1e-6`, base/full changes are `−0.006745/−0.005870` for A and `−0.007344/−0.006496` for B. At `1e-5`, base falls (`−0.018163`, `−0.021086`) while total rises (`+0.024002`, `+0.026476`). At the original `1e-4`, both base and total rise: `+0.014596/+0.118165` for A and `+0.020080/+0.145220` for B. This establishes local first-batch step-size sensitivity; it does not identify an optimum or a learning rate that works across batches, longer training or evaluation regions. The plot reads only the separately saved probe JSON, without another model call.

## Receipts

- Root final plan: `final-plan/audit-plan.json`, SHA-256 `8dd7332395c656de89dd1407930bda97fe64800bbea193bf7d8d4a4cc8b59866`.
- Saved numerical result: `run/audit.json`, SHA-256 `f6116a851a418d2982b025f04b595af919eca037dd1f90001eb3be580e217a80`.
- Source package sealed unchanged: `code-freeze.json`, SHA-256 `e2616dc66d72c97098e320b7ba1a6eb4d193bc28210c0ef9f80cd4de8b5448b2`. This additive seal was written after execution; the proposal, independent review and root ready plan had already bound the same source bytes.
- Figure input/script/output identities: `figures/figure-receipt.json`.
- Separate first-step figure identities: `step-probe-figures/figure-receipt.json` (root probe inputs A `234ee87243e25ee2f78158399b1bff7e3909757bdb9956fd6459dc8d9823a5ce`, B `1ecbfb91d97c0fdc42a50241c2582a707acf7d55fc5514922f8c099f42d14410`).
- Independent objective verification: `../numeric-review/completed-review/independent-verification.json`, SHA-256 `0491a9e26c8629283ca909894568fff80d8d0deed8bfdcba2ac3aca5a0165aed`.
- Independent first-step verification: `../loss-review/output-verification-corrected/verification.json`, SHA-256 `3e4d4c78da97cd4266075de495cf75aba44492dba110a16329a6585bb169a65e`.

This report and its plots read only the saved root numerical results and independent verification receipts. Source files and the source seal remain unchanged; final reporting identities are in `reporting/RECEIPTS.json`.
