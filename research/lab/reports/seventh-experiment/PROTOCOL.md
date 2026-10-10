# Fixed cross-scroll reference panel — staged protocol

Declared 8 October 2026 before acquiring new annotation chunks, CT chunks or predictions. Root owns execution. This is an extension of the completed sixth experiment; all its inputs and outputs remain unchanged.

## Question and interpretation

Does Russo ex016's disadvantage against its matched manual control recur on other scrolls' supplied validation annotations, and do thresholds selected on the used PHerc1667/w029 development crop transfer? Fixed official seed42 and seed43 checkpoints remain additional references. There are no new model arms, fitted ensembles, depth searches or appearance-based selection.

Available released validation masks cover three physical specimens: PHerc1667, PHerc0139 and PHerc0814. Reuse the sixth w029 primary raw outputs for development. Reserve the other two specimens from our threshold selection. The released models and authors have already used these datasets during development; historical teacher/training ancestry is unresolved. The new cases are reserved from our current tuning, not globally model-held-out tests. Per-crop and per-specimen descriptive evidence is useful without claiming independent ink truth or generalization.

The reviewer's preferred larger panel is adapted to actual available sources: one used development crop plus at most four new contexts across two additional physical specimens. Two contexts of one specimen are not two independent scrolls. A minimum of one adequate new context per specimen completes the cross-scroll diagnostic; do not add correlated/training examples merely to meet a numeric region target. Native coarse-scan feasibility is a separate metadata audit. This pooled-only panel cannot test native acquisition effects.

## Fixed source and selection rules

New cases are PHerc0139-w016 and PHerc0814-46527. Their verified public surface level 2 is uint8 with ZYX shapes [109,7020,7220] and [109,2130,3455], respectively, and uncompressed chunks [109,128,128]. Corrected labels, public supervision and validation are separate 21-plane arrays annotated at Z10. Source depths [13,97) are averaged in groups of four using float32 means, nearest-even rounding and uint8 output; read existing level-2 XY directly. Nominal pooled spacing is 9.596 µm. The unchanged model input is pooled [2,19), forward.

Freeze 16 candidates per case using metadata alone: chunk-aligned 384 × 384 contexts ranked by potential validation-chunk count, then potential ink-label-chunk count, then source Y/X. Object sizes are ranking proxies, not proof of positive/negative labels. Inspect supplied annotations in that fixed order, accepting the first adequate candidate and then the next adequate candidate whose full 384² context does not overlap an accepted context in that representation. Accept at most two per case. Do not search beyond the fixed candidate list or relax support after seeing scores.

Each primary mask is supplied validation ∩ actually acquired source ∩ fixed 64-pixel margin, intersected after inference with common finite/evaluated support. Its maximum interior is 256². Require at least 256 supplied positive and 256 supplied negative pixels before and after inference. Keep supplied labels, supervision and validation separate. Raw supervision membership does not establish historical labeled exposure; do not subtract it blindly. Record all rejected candidates and counts, missing support and any alias uncertainty.

Whole context separation in flattened coordinates prevents repeated 2D model input, but does not prove disjoint 3D interpolation support or correctly placed papyrus faces. Different documented specimens provide the cross-scroll grouping. No authenticated geometry-validity mask is assumed. Observed zero profiles remain eligible; record nonzero occupancy separately as a diagnostic. No unobserved fill chunks are permitted in this bounded panel.

## Thresholds and outputs

Before any new reserved predictions, lock one threshold per model using only the existing sixth w029 primary support. Prediction is score ≥ threshold. Target at least 95% empirical supplied-negative specificity: permitted false positives are floor(number of negatives / 20). Exclude whole tied-score groups conservatively, using a representable threshold immediately above the boundary score when needed. Store exact thresholds, score dtype, model/input hashes, class counts and achieved development specificity/sensitivity. This is an empirical operating point, not a confidence bound or calibrated probability.

Run fixed FP32 CPU inference with the unchanged reference normalization, 128² tiles, stride64, Hann blending, batch1 and four threads. Each case declares model depth [2,19). Never use annotation or occupancy masks inside normalization. No model or threshold is refitted on new regions.

Report each new region's ROC AUC, average precision, prevalence, sensitivity and achieved specificity at the frozen development threshold. Also retain midpoint0.5 diagnostics for continuity, exact confusion counts, raw float32 maps, common support and ROC/PR curves. Report supplied-negative performance as annotation agreement. A transferred threshold that misses95% specificity has failed to transfer; do not call its sensitivity matched-specificity performance. Group summaries weight crops equally within specimen and specimens equally across the two new scrolls; preserve every row. No pixel-based uncertainty interval, significance test or claim of independent replication from overlapping masks. AP comparisons across regions must retain class prevalence.

## Resource limits and execution

At most four new 384² contexts: nine source chunks each, 16,072,704 raw source bytes per crop, at most64,290,816 source bytes total. A shared64MiB source response-body budget includes metadata and redirect/error overhead. Label acquisition has a separately frozen2MiB response-body cap unless exact planned object lengths demonstrate that an additive pre-execution revision is necessary. Each acquisition plan must pin exact metadata, allowed objects, expected lengths, candidate order, request budget and code before execution. Public HTTPS only, bounded explicit redirects, no credential use, no automatic retries, no bulk-volume retrieval.

At most16 new model runs; each has120seconds wall time, four CPU threads and bounded output. Reuse development maps without inference. Freeze code/data/thresholds before reserved scoring; one execution owner. Preserve failures and use additive revisions for corrections. Independent reviews may add reports, not edit frozen executing inputs.

No paid GPU, R2 write, public publication or new training is part of this panel. The £10 total first GPU pilot cap and unresolved scoped-worker credential delivery remain unchanged. The research goal leads; prize eligibility is recorded separately from actual source licenses and access conditions.

## Decision rule

Retain heterogeneous or negative results. Repeated ranking and transferred operating-point deficits motivate a controlled supervision-quality study, not a claim about its mechanism. Competitive ranking with failed threshold transfer favors operating-point research. A reversal across specimens rejects extrapolating the sixth crop's local disadvantage. Complete this bounded comparison without expanding candidates or checkpoints to obtain a preferred result. Separately assess whether genuine native references and reliable physical/annotation lineage can support the next training experiment.
