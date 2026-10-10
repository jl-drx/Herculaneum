# CT-only texture depth prototype: mechanically valid, insufficient for automatic recentering

**The fixed rule passed its synthetic gate and found repeatable texture transitions on about half the real pixels, but an additional stress test exposed search-edge censoring. Retain it as an experimental structural cue; do not treat its depth as a physical ink surface or use it to crop/recenter the baseline input.**

This is development work motivated by earlier inspection of the same four Frag6/PHerc51 contexts. No ink labels, optical images, model predictions or new source data were used. The result is not independent ink accuracy or a model improvement.

## Fixed rule and controls

At every historical CT layer, compute squared central spatial gradients of the existing 3×3 spatial means, then average energy in a 7×7 coarse window. Remove a four-pixel halo, yielding120×120 pixels per context. Compare five planes before/after each candidate boundary between layers5/6 through58/59, normalized by the maximum five-plane energy on that ray. Select greatest absolute contrast; accept only contrast≥0.5 and no separated competitor at≥0.8 of the peak. Otherwise abstain. Sign describes increasing/decreasing texture, not anatomical orientation. Thresholds are fixed engineering choices, not calibrated confidence.

Seven mechanics fixtures passed, including exact gradient energy, boundary position, depth translation/reversal, multiple-interface abstention, constant abstention, affine intensity invariance and nonfinite rejection. The41-case synthetic gate then passed: all8 single-boundary and8 sloped-boundary cases had100% coverage and100% within their respective1/2-layer tolerance; all8 double-interface,8 noise-only,8 ramp and1 flat cases had zero accepted pixels. These simple controls do not establish validity on complex papyrus structure.

The first attempt computed synthetic cases but failed while serializing a NumPy boolean. Its partial `.writing` file and failure receipt remain in `run-v1/`. Recovery changes only native-boolean conversion and output/receipt names; all scientific parameters and inputs remain frozen. The successful gate and first real-map evaluation are in `run-v2/`. No real context was evaluated in the failed attempt.

## Every real context

| Context | Original coverage | Original median layer boundary | Jointly accepted after smoothing | Joint depth MAE | Original endpoint flags |
|---|---:|---:|---:|---:|---:|
| r544-c4128 | 53.40% (7690/14400) | 28.5 | 6875 | 0.097 | 841 |
| r2080-c3616 | 48.06% (6920/14400) | 18.5 | 6259 | 0.095 | 622 |
| r4640-c1056 | 51.96% (7482/14400) | 29.5 | 6362 | 0.133 | 491 |
| r6688-c1568 | 51.62% (7433/14400) | 28.5 | 6645 | 0.108 | 592 |

The predeclared perturbation was a three-plane depth moving mean. Jointly accepted estimates were within two layers in99.89–99.99% of cases. **That conditional statistic omits pixels whose acceptance changed:**6049/57600 pixels changed acceptance, and only26123/57600 (45.35%) were jointly accepted and within two layers. These are stability measures, not accuracy. Pixel counts are dependent spatial samples, not independent experimental units. No variant was selected as a winner. Positive affine intensity invariance passed mechanically for all maps; this does not establish cross-scanner generalization.

Full distributions and maps are retained. They contain both early transitions and transitions around layers30–35, rather than a single smooth global surface. The requested physical side and ink-bearing depth cannot be inferred from these distributions.

## Search-edge limitation and prospective stress extension

After seeing estimates at the earliest searchable boundary5.5, a separate extension was frozen before execution:22 noiseless step-energy fixtures at11 specified depths and both orientations. An independent analytic calculation of window/step intersection lengths verified every output. The original rule accepted6 transitions whose true boundary lay outside its searchable interval. Thus a sharp maximum is not sufficient evidence that the true transition is observable.

An advisory censor flag marks accepted estimates exactly at5.5 or58.5. It rejects all6 false out-of-range acceptances in this suite, and conservatively rejects true transitions exactly at the search endpoints too. This flag is posthoc development; it does not replace the original maps or establish correctness elsewhere. On the original real baseline it flags2546/29525 accepted pixels (8.62%);26979/57600 (46.84%) are accepted away from the endpoints. All2546 flags are at5.5. The search-edge result does not prove that every flagged real estimate is wrong.

## Verification, resources and review

Separate direct-offset arithmetic checked all7,257,600 baseline/smoothed energy values (maximum absolute difference2.68e-7 in squared digital units),128 fixed scalar boundary/abstention decisions, all115200 variant support pixels, every histogram, depth quantile and smoothing comparison. Synthetic gate arithmetic was reconstructed from its41 records; synthetic images were not independently regenerated. Verification shares the same input data, not an independent physical reference.

Root inspected the all-context figure and retained broad depth structure, visible abstention and endpoint pileups in the interpretation. The large conditional agreement is reported alongside acceptance changes. No root/agent review supplies anatomical truth. The earlier user-requested agent review concerns phase35 optical/CT landmarks, not these new depth maps.

Successful worker:3.055seconds,179957760bytes peak RSS. Independent numerical verification:2.803seconds,100794368bytes peak RSS. Censor stress:0.088seconds,49221632bytes peak RSS. No network requests, model forwards, optimizer steps, GPU use, R2 uploads, publication or submission occurred. £0 GPU spend; £10 total first-pilot cap remains. Original model, primary evaluation and all earlier findings are unchanged.

## Decision for Plinius

Do not promote this proxy to an ink-depth estimator or automatic input recentering rule. A change in texture can reflect fibres, inclusions, reconstruction, noise or a material interface; stability cannot distinguish these causes. The current fixed rule also omits an independently verified ink-bearing surface.

Keep full-depth model evidence available. A worthwhile next design is to test whether uncertain structural proposals can aid depth aggregation while retaining an unguided path and explicit missing/censored/ambiguous flags. Before any training, challenge that representation with multiple layers, misleading strong texture changes and out-of-window ink targets on known synthetic geometry. Measure information retention and false certainty, rather than choosing a window from these labels. A later real model ablation requires prospectively fixed, nonoverlapping development support and must continue to disclose unknown teacher ancestry and unresolved optical truth.
