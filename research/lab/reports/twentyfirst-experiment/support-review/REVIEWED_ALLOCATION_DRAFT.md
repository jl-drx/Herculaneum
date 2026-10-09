# Conditional CPU study: reallocate a fixed preservation budget

Status: pre-execution scientific draft awaiting the completed support/source seals, implementation and independent technical/numerical review. This file is not an execution GO. Root chose this narrower question after the phase21 source audit, before any new scientific model execution or outcome. The user authorized continued CPU research; no paid resource is involved.

## Why this question is admissible

The proposed full geometry-valid output mask cannot presently be authenticated. Keep that proposal blocked. The retained released supervision S and retired-validation V are disjoint, and their acquisition support is complete. The fixed training schedule includes 54,701 unique M=S pixels and 43,515 unique V pixels, plus 4,184 pixels with neither annotation role. Their repeated batch-output exposures are respectively 618,208, 409,448 and 20,920. These counts will be bound to independent support receipts before execution.

Instead of calling all observed pixels geometry-valid, test teacher consistency on the **explicitly retired supplied-reference participation mask V**. V ceased to be a validation set in the earlier declared adaptation work; using it here does not yield a new holdout. Its labels will not participate in objectives, teacher construction, selection or gates. Historical complete-array identity checks may still read label/pseudo bytes, and must not be misreported as zero label-byte reads. No Paris region or alias enters the training inputs, masks or teacher targets.

## Matched intervention

Use only the explicit-depth feature adapter from phase20: 5,296 trainable parameters, fresh identical forked-seed 20261008 initialization, random stem and exactly zero projection, nominal intensity/difference/index features, insertion before seed42's frozen 2D U-Net. Reuse no rejected learned state. Keep the original stem detached, decoder input gradients active, all original parameters/buffers frozen and eval, FP32 model execution, same exact two PHerc0139 contexts and 32 batch-two schedule, 17 planes, normalization, D4 transforms, manual target smoothing, AdamW and clip settings.

For manual M and retired reference V, let K_M and K_V be **separately normalized FP64 means** of KL(original seed42 || student). Let B_M be the unchanged FP32 smoothed manual BCE. Arms:

- `manual_double`: B_M + 2 K_M.
- `manual_retired`: B_M + K_M + K_V.

The total group coefficients are two in each arm. The intervention reallocates one unit from M to V; it halves manual KL protection and adds protection on retired reference outputs. It is not an unchanged-manual-strength intervention, a uniform mean over M union V, a full geometry-valid mask or an authenticated new supervision signal. Require disjoint nonempty M/V for every batch; no empty-group zero fallback. Outside M union V the objective logit gradient is exactly zero. Manual hard labels remain restricted to M.

Both arms start from the same logits; KL and its mathematical gradient are zero at initialization. Their first update should therefore agree under the specified numeric contract. Predeclare tolerances before model execution, not from observed differences. Retain matching original/cached/zero-projection output identity checks on all 32 inputs, source/state/mode/dtype cache binding and complete optimizer traces as in phase20. No independent checker should claim to replay convolution/autograd if it only checks retained numerical traces.

## Fixed measurements and decision rules

Run exactly 32 updates per arm, preserving immediate pre/post outputs and all-schedule measurements at updates 1, 8, 16 and 32. Save only final adapters. Measure B_M, K_M, K_V, each arm's own optimized full objective, and both common objective formulas on both arms. Report manual hard BCE and probability drift separately on M/V. Do not score retired V against its hard labels. Repeated/overlapping batch means are empirical training diagnostics, not independent samples.

Before execution, freeze these proposed mechanism criteria: each arm improves its own final full objective by at least 0.0001; `manual_retired` improves its own manual BCE by at least 0.0001; its final K_V is at least 0.0001 lower than `manual_double`; and its final manual BCE is no more than 0.001 higher than `manual_double`. Report every component and distance, without choosing intermediate iterations. Different arm objectives must not be directly compared as if they measured the same scalar risk.

After independent training verification, evaluate both final states on all six unchanged Paris DEVELOPMENT regions, even if the mechanism criteria fail. Preserve original seed42 as the reference, the literal cutoff 0.7558947205543518, 25 fixed tiles per region, baseline cache checks and existing blending/scoring. Candidate-vs-control differences are descriptive. A practical improvement screen requires candidate equal-fragment Paris2 AUROC and AP each to exceed seed42 by at least 0.005, and candidate losses versus seed42 no worse than 0.02 for either ranking metric in every region. Report all cutoff metrics regardless of ranking results. No automatic promotion follows even if these checks pass; reused development cannot support confirmation or a frontier claim.

The retained phase20 explicit-depth result (B_M + K_M) may be shown as a clearly historical descriptive reference. No new predictions or reoptimization of that arm are required. Its own objective differs from both new objectives; compare any historical training terms only where the same quantity and inputs are available.

## Execution boundaries

Use the existing Python 3.12.14 / NumPy 2.5.3 / Torch 2.12.0+cpu environment, deterministic algorithms, four intraop/one interop threads, no model-worker network. Retain phase20 limits: each train worker 600 s wall / 1,800 s CPU, each development worker 300 s wall / 900 s CPU; 4 GiB RSS / 12 GiB address space; controllers 1,380 s training / 660 s development; at most 1 GiB new study outputs. No automatic retries, sweeps, cutoff fitting or selection from development outcomes.

Before execution require a final scientific protocol, code/config/input pins, independent support and synthetic math checks, technical review, frozen numerical tolerances and a root GO. All earlier failures and evidence remain immutable. Record allocation tradeoffs or null findings without inventing a new architecture sweep. Paid GPU spend remains £0; the £10 TOTAL pilot cap and all launch prerequisites persist.
