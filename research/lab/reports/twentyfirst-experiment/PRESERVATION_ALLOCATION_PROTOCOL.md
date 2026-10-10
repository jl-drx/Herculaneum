# Conditional CPU study: reallocate a fixed preservation budget

Status: final root scientific protocol, fixed before new model execution. Support and synthetic math prerequisites have passed. Implementation, independent technical/numerical review and a separate source/plan GO are still required. This file alone is not an execution GO. Root chose this narrower question after the phase21 source audit, before any new scientific model execution or outcome. The user authorized continued CPU research; no paid resource is involved.

## Why this question is admissible

The proposed full geometry-valid output mask cannot presently be authenticated. Keep that proposal blocked. The retained released supervision S and retired-validation V are disjoint, and their acquisition support is complete. The fixed training schedule includes 54,701 unique M=S pixels and 43,515 unique V pixels, plus 4,184 pixels with neither annotation role. Their repeated batch-output exposures are respectively 618,208, 409,448 and 20,920. The support receipts below bind these counts and all seven independently reconstructed arrays.

Instead of calling all observed pixels geometry-valid, test teacher consistency on the **explicitly retired supplied-reference participation mask V**. V ceased to be a validation set in the earlier declared adaptation work; using it here does not yield a new holdout. Its labels will not participate in objectives, teacher construction, selection or gates. Historical complete-array identity checks may still read label/pseudo bytes, and must not be misreported as zero label-byte reads. No Paris region or alias enters the training inputs, masks or teacher targets.

## Matched intervention

Use only the explicit-depth feature adapter from phase20: 5,296 trainable parameters, fresh identical forked-seed 20261008 initialization, random stem and exactly zero projection, nominal intensity/difference/index features, insertion before seed42's frozen 2D U-Net. Reuse no rejected learned state. Keep the original stem detached, decoder input gradients active, all original parameters/buffers frozen and eval, FP32 model execution, same exact two PHerc0139 contexts and 32 batch-two schedule, 17 planes, normalization, D4 transforms, manual target smoothing, AdamW and clip settings.

For manual M and retired reference V, let K_M and K_V be **separately normalized FP64 means** of KL(original seed42 || student). Let B_M be the unchanged FP32 smoothed manual BCE. Arms:

- `manual_double`: B_M + 2 K_M.
- `manual_retired`: B_M + K_M + K_V.

The total group coefficients are two in each arm. The intervention reallocates one unit from M to V; it halves manual KL protection and adds protection on retired reference outputs. It is not an unchanged-manual-strength intervention, a uniform mean over M union V, a full geometry-valid mask or an authenticated new supervision signal. Require disjoint nonempty M/V for every batch; no empty-group zero fallback. Outside M union V the objective logit gradient is exactly zero. Manual hard labels remain restricted to M.

Both arms start from the same logits; KL and its mathematical gradient are zero at initialization. Finite-precision KL derivatives can contain tiny cancellation residuals (up to 2.7755575615628914e-17 in the preparatory fixtures). Do not require bitwise identical first updates based solely on the algebra. Independently check each arm's optimizer trace under the frozen numerical contract and report first-step inter-arm parameter/logit differences as diagnostics, without choosing a tolerance from real outputs. Retain matching original/cached/zero-projection output identity checks on all 32 inputs, source/state/mode/dtype cache binding and complete optimizer traces as in phase20. No independent checker should claim to replay convolution/autograd if it only checks retained numerical traces.

## Fixed measurements and decision rules

Run exactly 32 updates per arm, preserving immediate pre/post outputs and all-schedule measurements at updates 1, 8, 16 and 32. Save only final adapters. Measure B_M, K_M, K_V, each arm's own optimized full objective, and both common objective formulas on both arms. Report manual hard BCE and probability drift separately on M/V. Do not score retired V against its hard labels. Repeated/overlapping batch means are empirical training diagnostics, not independent samples.

The frozen mechanism criteria are: each arm improves its own final full objective by at least 0.0001; `manual_retired` improves its own manual BCE by at least 0.0001; its final K_V is at least 0.0001 lower than `manual_double`; and its final manual BCE is no more than 0.001 higher than `manual_double`. Report every component and distance, without choosing intermediate iterations. Different arm objectives must not be directly compared as if they measured the same scalar risk.

After independent training verification, evaluate both final states on all six unchanged Paris DEVELOPMENT regions, even if the mechanism criteria fail. Preserve original seed42 as the reference, the literal cutoff 0.7558947205543518, 25 fixed tiles per region, baseline cache checks and existing blending/scoring. Candidate-vs-control differences are descriptive. A practical improvement screen requires candidate equal-fragment Paris2 AUROC and AP each to exceed seed42 by at least 0.005, and candidate losses versus seed42 no worse than 0.02 for either ranking metric in every region. Report all cutoff metrics regardless of ranking results. No automatic promotion follows even if these checks pass; reused development cannot support confirmation or a frontier claim.

The retained phase20 explicit-depth result (B_M + K_M) may be shown as a clearly historical descriptive reference. No new predictions or reoptimization of that arm are required. Its own objective differs from both new objectives; compare any historical training terms only where the same quantity and inputs are available.

## Execution boundaries

Use the existing Python 3.12.14 / NumPy 2.5.3 / Torch 2.12.0+cpu environment, deterministic algorithms, four intraop/one interop threads, no model-worker network. Retain phase20 limits: each train worker 600 s wall / 1,800 s CPU, each development worker 300 s wall / 900 s CPU; 4 GiB RSS / 12 GiB address space; controllers 1,380 s training / 660 s development; at most 1 GiB new study outputs. No automatic retries, sweeps, cutoff fitting or selection from development outcomes.

Before execution require a final scientific protocol, code/config/input pins, independent support and synthetic math checks, technical review, frozen numerical tolerances and a root GO. All earlier failures and evidence remain immutable. Record allocation tradeoffs or null findings without inventing a new architecture sweep. Paid GPU spend remains £0; the £10 TOTAL pilot cap and all launch prerequisites persist.

## Literal configuration and source anchors

Optimizer: fresh adapter-only AdamW, learning rate 0.001, betas (0.9,0.999), epsilon 1e-8, weight decay 0, foreach=False, fused=False; no scheduler; global gradient norm clip1 with nonfinite failure. Smoothed participating manual targets are 0.25/0.75. Temperature-one teacher KL uses original seed42 logits at the same input and is computed in FP64. Each group mean is over its own participating batch pixels. Empty M or V is STOP.

Both arms use the exact phase20 explicit-depth feature construction, not an arm-name-dependent raw feature fallback. Root scientific controls require shared normalized inputs, targets, masks, augmentation, feature tensors and initial adapter state across the pair. Original checkpoint SHA e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab and original state SHA21c186000b9b3093835e185342668acd0d86147a719c37513e937924b77ea228 stay unchanged. Current teacher caches, model/source pins and data history remain those of phase20's plan; bind them literally in the new plan alongside this protocol.

All-schedule diagnostics are equal-batch means using math.fsum with the phase20 FP32/FP64 conventions. For each fixed measurement save manual base and hard BCE, K_M, K_V, probability drift on both masks, objective_A=B_M+2*K_M and objective_B=B_M+K_M+K_V. Retain the arm's own full scalar too. No full-scalar comparison across unlike objectives is an endpoint.

Preserve full logits, objective logit gradients, 5296-vector parameter/gradient/moment transitions, mask/source hashes, decoder-input backward evidence and state-isolated measurements. Loss gradients are zero outside M union V (and outside M in manual_double). Hidden-feature gradients may extend outside those masks by the decoder receptive field. The numerical checker must verify the appropriate mask semantics per arm.

Expected overlapping calls per training arm match phase20: full-model32, fusion32, 2D-network256, adapter224, backwards/decoder-input backwards/updates32. Per development arm: full-model150, fusion150, 2D-network300, adapter150, no backwards/updates. Four workers each load the retained backbone once; full/component counts overlap. If implementation requires additional calls, amend and review the explicit call contract before execution rather than silently exceeding it.

Primary mechanism has five components: both own-full improvements >=0.0001; candidate own-manual-BCE improvement >=0.0001; final K_V(control)-K_V(candidate)>=0.0001; final B_M(control)+0.001-B_M(candidate)>=0. The practical screen has14components: two candidate-vs-seed42 Paris2 macro gains>=0.005 and twelve per-region candidate-vs-seed42 AUROC/AP deltas>=-0.02. Always retain and report failed components. No ambiguity is resolved by rounding.

This narrower protocol supersedes the draft's pending prerequisites only for supplied-reference preservation allocation. The geometry-valid full-output hypothesis remains blocked. Root retains scientific review. Retired V is an explicitly revised teacher-consistency role on an already retired PHerc0139 training-domain reference; it is never called fresh validation. Annotation-unknown support is excluded. The phase21 structural/source research may continue independently and cannot alter these masks or scientific choices after freezing.

- support_contract: `preservation-support/audit-v2/support-contract.json`, SHA256 `762cb1e7ae7649d042358f80fea52712ce6471ebd5d4c4aaafc8f7a547694a8f`.
- support_source_seal: `preservation-support/retention-manifest-v2.json`, SHA256 `0caeb2f1d60ac644951db8728ae5a0f422f6af2bce43c6be34eb06e1145be96c`.
- independent_support_summary: `support-review/verification.json`, SHA256 `5dc755a577fbfaa0594392c4880e9adbdc214601179471e6eedfde0703954fae`.
- independent_support_seal: `support-review/retention-manifest.json`, SHA256 `fb945b48e9693c5969df110fddfbd74112cb88316eb944a4c0a72a61749d18c4`.
