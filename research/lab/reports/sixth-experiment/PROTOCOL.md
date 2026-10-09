# Sixth experiment: a bounded dense-supervision diagnostic

Protocol decisions recorded before candidate checkpoint acquisition, reference scan acquisition or new predictions. Exact code, object lists, selected crop and metric masks will be frozen in staged plans after their metadata contracts are verified. This document is not a claim that those stages have already completed.

## Question and interpretation

Can the released Russo dense-supervision and matched-control checkpoints be reproduced through the existing verified inference implementation, and how do their ranking, threshold behaviour and spatial responses differ on identical corrected w029 validation support?

The scientific comparison is ex01675k versus its matched manual-label control75k. Existing official seed42/75k and seed43/75k are fixed reference baselines, not matched controls for every difference in the author's training run. No checkpoint will be selected or replaced after viewing this result. The author already reports lower ex016 AUC than control on the whole w029 validation region, so improvement is not assumed. A small annotation-selected crop is not a reproduction of the full reported metric or a generalization benchmark.

The current annotations may contain iterative/model-assisted lineage. Withheld pixels inside a trained segment, shared acquisition support and unresolved teacher ancestry limit independence. Claims will concern agreement with supplied annotation and implementation behaviour. No PHerc0800 response will become training truth, and no reading or translation is attempted.

## Staged acquisition and contract gates

- Use only the two exact public Russo75k checkpoints identified in the pinned metadata: combined declared payload 276,734,604 bytes. Verify object length and SHA-256, reuse only matching retained files, and perform no unrestricted pickle fallback. Strict tensor key/shape loading and relevant configuration comparison must pass before predictions. Freeze the downloader/auditor and object plan before execution; one attempt per object, retaining failure evidence.
- Verify the corrected PHerc1667/w029 source, label, supervision and validation contracts. The public surface has 109 source depth planes; its curated 21-plane input must follow the recorded Z crop `[13,97)`, four-plane mean and documented rounding, with increasing depth retained. Model input is exactly pooled planes `2:19`, with no padding, fitted resampling, contrast, sign search or offset adjustment.
- Freeze a bounded annotation-selection rule before label payload inspection. Select at most one 256-square crop using metadata, annotation coverage and source extent, never model outputs. Require at least 256 supplied positives and 256 supplied negatives in its fixed 64-pixel-margin validation interior before inference. If the bounded selection fails, preserve the negative outcome and stop rather than relax it automatically.
- Keep scan acquisition below 64 MiB response bodies, with a separately specified smaller label/metadata allowance, no whole-volume download and no automatic retries. Every used source chunk must be observed and hashed; unknown support is not negative ink. Explicitly audit whether the training loader subtracts validation from the supplied supervision mask before interpreting their intersection.

## Fixed execution

Use the current focused upstream model code and unchanged `run_ink_reference.py` normalization, tiling and blending functions through a new reviewed wrapper. CPU FP32, four threads, deterministic settings, center input `2:19`, 128-square tiles, stride64 and existing Hann blending. Run exactly the four fixed checkpoints. Cap each model at 120 seconds, with no automatic retries or paid compute. Any existing w013 integration check is reported separately from w029 and is never pooled into its metrics.

Freeze all active execution and analysis code before predictions. One execution owner runs the package; reviewers write additive records. On a code/input mismatch, preserve the failed attempt and version any correction separately.

## Evaluation

Primary support is the fixed 64-pixel-margin interior intersected with supplied validation, genuinely observed source and common evaluated/finite outputs, with direct training-target exclusions only where actual training semantics establish them. No independent geometry-validity mask is assumed. Report any missing/ambiguous geometry and full model-context overlap as limitations. Primary support must meet the predeclared two-class counts before prediction; later failures of support or class availability stop the primary analysis rather than alter the crop.

Observed zero selected-depth profiles remain distinguishable from unobserved source. Report their counts and occupancy-stratified diagnostics, rather than labelling zero intensity as geometry validity or as ink absence. Entire skipped tiles are unevaluated. The full supported validation mask is a predeclared secondary domain; crop edges have different context and blending support from full-surface inference.

Retain raw float32 score maps and masks. Report tie-aware ROC AUC, average precision and precision–recall curves, plus confusion counts, precision, recall and balanced accuracy at the fixed raw midpoint0.5. That midpoint is a diagnostic inherited from smoothed targets, not calibrated ink confidence or a threshold fitted to this crop. Undefined metrics must remain explicitly undefined, with denominators and class counts. Show paired model differences and source/annotation panels at fixed display ranges; all cases and models remain visible. Do not infer pixel-independent confidence intervals or a global model ranking from one selected crop.

## Decision

A successful strict load and bounded execution completes the integration gate. A positive or negative ex016/control result completes the developmental comparison if its scope and uncertainties are documented. Neither result by itself justifies training the target, a first-in-class claim or a prize claim. The next research decision concerns acquisition-domain mismatch, supervision reliability and physical reference evidence, as set out in [the charter](../../RESEARCH_CHARTER.md) and [Astra's review](STRATEGY_REVIEW.md).
