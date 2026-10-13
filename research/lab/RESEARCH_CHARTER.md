# Herculaneum research charter

Updated 12 October 2026 after the user's approval of the broader research programme. This charter supersedes prize-first priorities in earlier briefs and experiment recommendations; those historical records remain unchanged.

## Current priority amendment, 12 October 2026

The [active research plan](../CURRENT_PLAN.md) and [ranked programme](../planning/plinius-frontier-programme-20261012/README.md) now govern work selection. Pursue correct-sheet recovery and broad native ink learning in parallel, supported by targeted annotation and fresh physically separated evaluation. Distinguish topology/connected coverage from local surface placement. Do not select geometry solely by an ink detector's response. Strong released models and adequate GPU-scale training are eligible hypotheses even when a small CPU proxy fails.

The native0139 nine-fit study, its full-depth inputs and locked evaluation remain unchanged. Its result is a baseline, not a ceiling or a universal admission gate. All already inspected evaluation support becomes development evidence for later studies. Reserve new confirmation support before viewing labels/scores; declare unknown upstream training and teacher ancestry.

New paid compute requires explicit approval after a concrete workload, cost bound, environment, result delivery and independent shutdown are ready. The £10 total first-pilot cap remains. Keep GitHub work private; no new PR is required for routine research commits. Selective private Git LFS snapshots and checkpoint backup branches now exist; use their receipts to establish what is actually backed up.

The evidence, immediate-action and H1–H3 sections below preserve the original 8 October starting position. Their sequence is historical and is superseded by the linked active plan; the objective and evidence-quality principles remain in force.

## Objective and roles

Develop leading models and techniques for recovering defensible information from Herculaneum CT. A first-in-class claim requires a precisely defined task, strong contemporary baselines, independently supported evaluation, reproducible gains and an audited prior-art comparison. It is an ambition, not a present result.

The user contributes architecture, research direction and high-level review. The lead agent owns implementation, experiment execution, evidence quality and decisions within authorized resource limits; specialist agents provide independent reviews when assigned. CPU training and inference are operational. An independently established improvement and readable new target text remain research goals.

Prizes are optional downstream applications. Grand Prize restrictions on target resolution, human input or submission format should not veto an otherwise scientifically valuable and lawful experiment. Record the eligibility consequences of a branch so a compatible subset can be identified later. Actual asset licenses, access conditions, privacy/publication commitments and the approved compute budget remain binding. This clarification does not itself authorize public disclosure or external outreach.

## Current evidence and immediate action

Six retained studies establish bounded scan acquisition, source/render correspondence, checkpoint inference, repeatability and a fixed supplied-annotation model comparison. They do not establish authentic target ink or a readable sequence. See [the sixth experiment](reports/sixth-experiment/RESULTS.md), [the fifth experiment](reports/fifth-experiment/RESULTS.md) and [the primary-source methods audit](reports/frontier-methods-audit-20261008/RESEARCH_PROGRAM.md).

The frozen Russo ex016/control comparison with existing seed42/seed43 baselines on corrected PHerc1667/w029 support is complete. On its small fixed primary crop, ex016 AUC was 0.912934 versus control 0.947220; retain the existing baselines. This was an integration and prediction-behaviour diagnostic. The author reports whole-region w029 AUC 0.8853 for ex016, 0.8945 for its matched control and 0.9298 for the released seed43/75k baseline; our selected crop does not reproduce those whole-region figures or rank generalization. The supplied validation mask, teacher ancestry, physical aliases and annotation creation limit independence. Next establish a physically distinct reference panel with separate development and reserved support, then compare ranking and sensitivity at matched specificity across acquisition domains before a controlled training study.

Do not require a favorable result to finish the diagnostic, or follow a negative result with unbounded checkpoint shopping. Preserve exact model, label, source and support identities, all raw scores and all outcomes. No target pseudo-training labels are created from the unresolved PHerc0800 responses.

## Research hypotheses after the integration benchmark

**H1: acquisition domain and supervision reliability interact.** Native coarse scans differ from pooled fine scans beyond voxel spacing. Test whether reliability-aware supervision transfers better than uniform hard teacher labels. The initial arms are manual-only, ordinary hard pseudo-labels, and a candidate that preserves unreliable teacher outputs as unknown or weights them using independently assessed reliability. Match training support, sampling and compute where possible; add coverage-only and target-quality ablations to distinguish more examples from better targets. This is a hypothesis, not an asserted novel contribution. Review prior work before designing or naming a new method.

**H2: explicitly modelling surface uncertainty improves robustness.** Measure changes under fixed physical normal/depth perturbations on supported positive and negative references before building a new geometry network. Existing shifted windows share 15 of 17 slices, so their correlation is weak evidence about independent geometric robustness. Synthetic perturbations yield a robustness curve, not proof of real sheet-placement error. Compare fixed-window inference with a predeclared uncertainty-aware alternative only after defining plausible perturbations from CT/mesh evidence; assess both missed ink and false positives.

**H3: supervision efficiency can lower the cost of useful training.** After H1's baseline is stable, vary depth width and jitter jointly with matched training. Keep model quality, physical context and data exposure distinct from throughput. A smaller input to an unchanged incompatible checkpoint is not a training ablation. Hardware choice should follow measured memory and throughput, not the prestige of an H100.

## Evaluation investment

Build a ledger of physical regions, scan energy/resolution, source/mesh versions, full sampled-volume and model-context support, annotation provenance, ambiguity and teacher/student lineage. Distinguish fragment optical evidence, specialist-reviewed intact-scroll annotations and model-derived references. No label type is silently promoted to independent truth. Lack of perfect intact-scroll ground truth should limit claims rather than prevent all prototyping.

Use development regions for protocol choices and reserve unused physical regions before model selection. Group splits and uncertainty estimates by physically distinct support, not pixels or aliases. Multiple seeds test optimization variation; additional physical regions test generalization. Teacher agreement, threshold metrics, stroke preservation and authentic source-supported recovery answer different questions and should be reported separately. Seek specialist assessment when a concrete candidate reading or annotation need exists, with the user's authorization for any outreach.

## Execution and resources

Freeze the hypothesis, selection rule, code/environment, data/label versions, metrics, resource cap and decision rule before each run. Use one execution owner after a freeze; reviewers add reports instead of changing executing inputs. Preserve failures and protocol revisions. Avoid making provenance tooling a substitute for informative experiments.

The first paid GPU pilot remains capped at £10 total, including failures, retries and attributable overhead. No paid job may start without the agreed frozen workload, pinned CUDA environment, scoped worker credentials, current pricing, measured feasibility and independent shutdown. The cap is not a commitment to fit a full research training program inside £10. A larger budget should be proposed using measured pilot costs and a concrete study.

Keep reusable code/configuration in private GitHub and use selective, checksummed artifact bundles and the existing checkpoint backup branch for portable retained evidence. R2 is another option once worker/runtime access is verified. A local artifact or saved startup instruction alone is not a backup; distinguish successful remote verification from an intended upload. An award application may follow good research, but research decisions are governed by information value, validity and feasible cost.
