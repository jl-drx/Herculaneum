# Review: the next matched soft-supervision learning comparison

11 October 2026. Read-only design review at repository cfb5775. No new data, model inference, training or paid compute. This is a recommendation, not a frozen run protocol.

## Decision

**Do not train a scientific comparison on the four site1 checking supports or pretend the three retained w040 CT cores are a qualified paired benchmark.** The current packet can support a finite target/operator comparison and optimizer implementation checks. It cannot decide whether uncertainty-aware supervision improves learning. The next actual learning study should compare ordinary soft targets against CT-averaged soft targets on native CT, with a matched smoothing control and a different reference on separate physical tissue. Use a modest complete learner trained to a measured endpoint, not another short continuation of the 34.5M-parameter released network.

This is not a recommendation for an open-ended registration campaign. Acquire only the finite training/evaluation packet needed by the one comparison; freeze its physical regions, source versions and acceptance rule before target inspection. If that packet cannot be qualified, stop this learning proposal and report the exact missing endpoint. A tiny current-data fit is not a substitute for that decision.

## Why the retained data are insufficient

- The teacher packet covers four neighboring 32×32 checked supports in one inspected training winding. Three carry teacher-positive pixels; one has none. The three positive supports are not independent physical replications, and random pixel/patch splitting leaks the same fibers, local structure and target smoothing across partitions.
- Site1's CT alternatives apply to their exact supports. They are not a displacement field qualified for the entire 256-square core. Do not interpolate those four uncertainty sets across the core without declaring and checking a new model.
- Site2 and site3 failed different held-out displacement-consistency checks. They cannot inherit site1's fit/uncertainty set merely because their CT arrays are retained. Their absent teacher pixels do not make them untouched test tissue; their structural outcomes have already influenced development.
- All existing targets here are outputs from the same fine78 teacher. Evaluating a learner against those outputs measures teacher imitation. Scoring each treatment against the target it was trained on is especially invalid: each treatment moves the scoring target. A common fit-target score is less tautological, but can still penalize useful smoothing and cannot establish ink accuracy.
- Repeated derivatives of the same winding cannot supply a physical-split uncertainty estimate. Millions of pixels would not fix this.

None of these arguments requires proving a perfect physical ink surface before every useful experiment. A native-only learner can train on explicitly uncertain pseudo-labels. The blockers are lack of adequate qualified training support and an evaluation target that is not just its own transformed training teacher, not a demand that every historical annotation be flawless.

## Do not confound softness with calibration

The old hard target is `1[score >= 48]` on w040. The ordinary raw soft target is `q = score / 255`. These have different semantics and decision pivots: thresholding raw q at 0.5 corresponds to score127.5, not48. The raw teacher is probability-style output with unknown calibration.

A hard-versus-soft comparison therefore changes target entropy, effective class balance, calibration and gradient magnitudes. It may be useful as an explicitly composite contrast, but it does not isolate registration robustness. Do not fit a calibration transform on inspected evaluation targets or silently make q=48/255 into q=0.5. If calibration is desired later, fit it only on a physically separate calibration set and apply exactly the same transform to every arm before spatial averaging.

For the next primary contrast, keep the same raw q scale in every arm and use unweighted masked BCE without Dice, class rebalancing or different threshold conventions. Report training target means, entropies, support and loss-gradient magnitudes; a change of target mean is an effect of the declared operation, not improved probability calibration.

For BCE, `mean_t BCE(p,q_t) = BCE(p,mean_t q_t)`. A uniform mean over the finite CT alternatives is a declared averaging treatment, not posterior marginalization, a new architecture or a calibrated geometry uncertainty model.

## One actual learning experiment to prepare

**Question:** on native113 inputs, does local CT-informed averaging improve transfer to reserved physical tissue beyond ordinary soft targets and equivalent nonlocal smoothing?

1. **Training tissue:** select a finite set of separated blocks from existing native-training windings, preferably at least three windings such as w040, w041 and w043. Selection must use geometry/CT validity, not teacher response. Keep entire physical windings together in the split, including aliases and alternate surfaces. The exact sites, eligible area and source-normal context must be frozen in a separate acquisition plan; these names alone do not authorize a physical split. All arms use the same accepted blocks, pixel mask, CT planes, patch order and draws. Site1 can remain a known development block. Include both narrow and broader CT ambiguity strata if supported; do not select only the visually best patches.
2. **Evaluation reference:** freeze separate native physical tissue before reading targets. Existing native manual releases for w035/w039/w044 are concrete candidate reference-agreement endpoints; they need current version pins, source/supervision semantics and overlap checks. They are transferred annotations with unresolved provenance, not automatically independent physical ground truth. They are not the numeric fine78 teacher used as the training target. Require corroboration on the released w024 fine59 reference with its own support/coordinate qualification, and preserve the author-designated w053 blank response control. If only same-fine78 teacher targets can be obtained, this study may report teacher compression, but it must not select a scientific winner or authorize a GPU on that basis.
3. **Model:** a fixed small fully convolutional native-input learner (roughly 10^5–10^6 parameters), no pretrained ancestry, retaining all native depth planes as channels. Use no position coordinates that permit memorizing the raster. This is a complete coarse-input learner, not a depth adapter or fusion of saved predictions. Its limited representation bounds the conclusion: a null cannot establish that full models or native CT cannot learn.
4. **Arms:** A ordinary soft target at the fixed CT fit; B uniform CT-alternative mean target; C matched global displacement-kernel mean, using the aggregate training-only displacement distribution. C must use the same transfer operator, source support and retained target-scale convention as B. All arms use the intersection of valid target footprints and report what area that excludes. If B beats A and C, only then add the stronger spatial-permutation control, preserving the training distribution of uncertainty sets while breaking association with the local CT. A treatment cannot earn credit solely by rejecting difficult pixels.
5. **Exposure:** matched paired initialization and optimizer states, batch/order/augmentations, and complete learning curves. Use at least three paired initializations if feasible on CPU. First measure optimization saturation on a training-only optimization partition with a fixed maximum exposure; freeze that common exposure before evaluating the external references. Then run every arm for that same exposure. Do not early-stop B on its different training-target loss or choose each arm's best reference checkpoint. If the control still makes substantial training progress at the cap, the comparison is optimization-limited; it is not a scientific null. CPU time itself is not a spending gate.
6. **Endpoint:** use identical frozen manual/reference labels for all arms. Report per-physical-region AP/AUC and proper scoring quantities only where their reference semantics support them, plus blank false response and retention. Calibration thresholds, if needed, come from a dedicated development partition shared across arms. For the fine59 model reference, call the score alternate-reference agreement. Do not combine it with manual labels into an alleged truth score. A common fine78 agreement score is allowed as a diagnostic, never as the sole success endpoint.

This isolates target treatment on one input acquisition. It does not simultaneously identify the real-native versus degraded-fine acquisition effect. That second factor should wait until the simpler target contrast is interpretable; adding it now doubles jobs and makes a null harder to diagnose.

## Stopping and decision rule

The completed sensitivity audit supplies engineering tolerances, not a training AP threshold or a sample-size calculation. Do not mechanically import +0.01 Paris AP or twice the spread of two released checkpoints as a new noise gate.

Before running, choose a practical effect size from the final endpoint and freeze it. A defensible pilot admission rule is: B must improve the same predeclared regional endpoint over both A and C in each paired initialization, the mean effect must exceed the frozen practical margin, no evaluation region may exceed the separately frozen regression allowance, and the alternate-reference effect must point in the same direction without increased blank false response. Report all paired regional/seed effects; three seeds and a few windings do not support population-wide significance claims. Any stronger inferential claim needs more independent physical units.

Stop the current experiment at the fixed matched endpoint. Do not add crops, choose a more favorable depth, retune score calibration or extend only the losing arm after inspecting evaluation results.

- **B matches C or gains disappear on the different reference:** local CT-informed uncertainty has not earned further model complexity. Prefer ordinary soft targets or simple smoothing; stop this uncertainty-learning branch at the tested scale.
- **B beats A and C, consistently across paired initializations and external physical regions, with the alternate-reference and blank checks intact:** run the predeclared uncertainty-permutation control. If B also beats that, the evidence supports a controlled larger native learner. It still does not prove authenticated ink or automatically authorize a paid GPU run.
- **All arms fail to learn the training target, or improvement continues strongly at the exposure cap:** representation/optimization is unresolved. Do not use that null to reject CT information or the target treatment. Diagnose once using training-only evidence; do not keep repeating weak continuations as a supposed evidence gate.
- **All arms fit training data but fail across physical tissue:** sparse coverage or teacher/reference mismatch dominates at this scale. Expand genuinely different training tissue or redirect to acquisition comparison, rather than inventing another depth/fusion adapter.
- **Reference provenance/geometry fails admission:** no ink-accuracy comparison is available. Preserve a reproducible compression/implementation result if useful, but do not announce a frontier learning gain.

## Immediate useful work without acquiring another CT voxel

From the retained arrays, construct A/B/C target operators and verify support, pooling, constant-field preservation and the BCE identity. Report their actual differences in the four fixed supports, including positive-union statistics. This settles whether B is numerically distinguishable from ordinary smoothing on the evidence that motivated it. It is a bounded prerequisite, not a training study, not a newly independent experiment, and not evidence of predictive benefit. If B and C are essentially identical here, ordinary soft supervision is the appropriate priority; there is no reason to launch a local-uncertainty architecture on this packet.

In parallel prepare the exact finite physical learning packet and reference admission manifest. No need to wait for another machine for these short tasks. This reviewer has not run or authorized them.

## Sources read

- research/planning/pherc0139-supervision-20261011/{README.md,PROTOCOL.md,provenance/REPORT.md,review/DESIGN_REVIEW.md}
- research/planning/plinius-research-review-20261011/{novelty/REVIEW.md,paired-data/REVIEW.md}
- research/lab/reports/thirtyseventh-experiment/RESULTS.md
- research/lab/reports/fortyfirst-experiment/RESULTS.md

The earlier matched-continuation findings already demonstrate that fit can improve without useful transfer. They do not establish optimization convergence, sufficient physical coverage, or the value of soft versus CT-averaged targets. This proposed comparison is not an equivalent repeat, provided it obtains the stated physical split and different-reference endpoint.
