# Seventh-study strategy review

Read-only review of the charter and sixth-study results, 8 October 2026. This additive report is the reviewer's only file change. Root owns selection, protocol freeze and execution.

**GO for one bounded reference-panel experiment. Its immediate question is whether the sixth study's dense-model disadvantage and poor threshold transfer recur across physical regions. Independent ink recovery and acquisition-domain causality are separate, stronger questions.** The four fixed models are sufficient; add no checkpoints, ensembles or inference tuning.

## Concrete scope

- Preferred panel: **six 256 × 256 contexts**, with fixed 128 × 128 evaluated interiors: **two development regions and four reserved regions**. Reuse the sixth-study crop as one development region. Seek at least three physical segments and, if feasible, multiple scrolls. Regions are the reporting units, not pixels.
- Require at least **256 supplied positive and 256 supplied negative evaluated pixels** per region. This is a feasibility floor, not statistical power or authenticated truth. Retain class prevalence and annotation coverage. Select deterministically using geometry/annotation availability before scores; retain rejected candidates and the stopping rule. Avoid choosing unusually clean or dense text from imagery.
- Freeze all regions and roles together. Within each physical group, keep every representation, scan energy, surface alias and full model/sampling context on the same side of the development/reserved split. Coordinate distance alone does not establish independence. If physical mapping is unresolved, retain the uncertainty and call the cases separate reference representations rather than verified disjoint regions.
- Suggested hard scope: at most **eight distinct physical contexts**, **64 MiB new source/label response bodies**, and **8 MiB metadata**, subject to actual chunk contracts. Stop when the first adequate frozen panel is obtained; do not keep searching until a preferred result is likely. Existing inference timings make CPU execution appropriate. No paid GPU is necessary.

## Threshold and analysis rules

Use **one threshold per model**, learned only from development support. For each development region, find the lowest threshold satisfying empirical reference specificity ≥95%; take the maximum across development regions. With prediction defined as score ≥ threshold, handle tied negative scores explicitly: the permitted false-positive count is floor(0.05 × negative count), and a threshold immediately above a tied score may be necessary. Store the rule, thresholds, counts and achieved development specificity. This conservative rule is an engineering operating point, not a 95% confidence guarantee.

Freeze thresholds before reserved predictions. Report per reserved region: AUC, AP, prevalence, sensitivity and **achieved specificity at the frozen development threshold**. Do not call these reserved sensitivities “at matched 95% specificity” when achieved specificity differs. The transfer failure is itself an outcome; do not recalibrate on reserved labels. Report equal-region summaries as secondary descriptions, preserving all region rows. AP changes across regions are affected by prevalence. No pixel confidence intervals or pixel-level significance tests.

Preserve fixed-0.5 results only as secondary continuity diagnostics. The sixth study already showed that ex016 loses ranking as well as specificity, so recalibration alone cannot explain its complete local deficit. Do not use the seventh study to repeatedly optimize that same crop.

## Native-domain and label fallbacks

**Native plus pooled references with comparable annotations:** proceed, but distinguish an observational domain association from an acquisition effect. Different scrolls, surface quality, annotation sources and teacher ancestry can explain apparent domain differences. The strongest small design uses paired native and pooled observations of the same physical regions, with entire pairs assigned to development or reserved roles. A paired representation doubles inference cases without creating another independent physical region. Directly resampling a pooled scan does not create native acquisition evidence.

**Native teacher-only references:** still useful for teacher-agreement and threshold-transfer diagnostics. State “teacher-reference negative/positive” and do not promote specificity to true ink specificity. Keep different reference lineages in separate result tables. Do not silently mix manual-supported development negatives with teacher-derived negatives to claim a universally calibrated threshold. Teacher uncertainty/thresholding is part of the reference contract. An unavailable native label bridge is not a reason to download a whole corpus.

**No usable native label pair:** complete a pooled-only experiment, preferably two development plus four reserved regions; minimum fallback is **one development plus three reserved**. Report within-corpus annotation agreement and operating-point transfer. If only one training segment is available, physically separated regions remain useful but support only within-segment conclusions. Leave the acquisition-domain hypothesis untested. Do not repeat w029 fragments and count them as cross-scroll validation.

Known student-training segments or unresolved teacher ancestry do not automatically invalidate this developmental experiment. They prevent a held-out generalization claim. If a reference contract or alignment cannot be established at all, omit that candidate under the frozen rule; labels with unknown semantics cannot support even a meaningful agreement metric.

## Decision after this bounded study

If dense supervision repeatedly loses ranking or useful sensitivity at transferred thresholds, proceed to a small matched supervision ablation: manual-only, hard teacher targets, and reliability-aware targets, controlling coverage/sampling and initialization. Recurrence motivates that experiment; it does not establish the teacher-error mechanism. If only threshold transfer fails while ranking is competitive, prioritize calibration/domain transfer. If behavior varies substantially by region or reference lineage, investigate that heterogeneity before expanding model complexity. If the original disadvantage does not recur, retain it as a local negative result and reject its broad extrapolation.

This panel is research infrastructure plus a falsifiable diagnostic, not a novel method. Its completion should unlock an informative ablation rather than another open-ended audit. Prize eligibility is optional; source permissions, privacy commitments and the £10 total paid-pilot cap remain binding.
