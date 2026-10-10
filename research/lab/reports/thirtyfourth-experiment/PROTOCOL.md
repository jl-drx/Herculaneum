# Fixed cached CT contrast audit

Frozen after phase33's post hoc sensitivity findings, before computing any feature or profile in this phase. Four already model-scored PHerc51/Frag6 development contexts, one parent. This is a diagnostic, not an independent benchmark, new model, label correction or evidence of absent CT ink. Preserve all earlier primary results and the unchanged seed42 baseline.

## Inputs and complete family

Use the four exact phase32 raw-v2 uint16 arrays, each 63×384×384, and their verified historical-render source contract. Use phase31 core labels and phase33 original/interior (>4 coarse pixels from different/unknown labels) supports. No data acquisition, model call, training, label change, optical input in CT features or physical-depth claim. Other parent reserves stay untouched.

For each raw layer 1..63, sum each nonoverlapping spatial 3×3 block in exact integer arithmetic, giving S on a 128×128 canvas. Intensity is S/9 in retained TIFF uint16 units. Each nonoverlapping depth triplet 1..3, 4..6, ..., 61..63 uses the same spatial blocks: intensity is the 27-sample sum divided by 27. Triplet plot coordinates are source-layer centers 2,5,...,62. These are not calibrated attenuation values.

Local contrast is intensity minus its 9×9 coarse-pixel box mean (center included). Compute exact signed numerator 81*S_center - sum(S in 9×9 box), denominator 81*9 or 81*27. Use int64 summed-area tables; bounds fit int64. Exact numerators avoid accumulated floating-point tie changes. Score only central 64×64: the larger canvas supplies real margins, so no padding reaches a scored feature. Keep source geometry unchanged.

Two features (intensity, local contrast), two fixed polarities (+ feature and − feature), two scales (63 single layers, 21 triplets), two supports (original selected core, prior interior), four contexts: 2688 metric records / 5376 AP and AUROC values if all have both classes. Report every variant and explicit undefined outcomes. No winning depth, sign, threshold, context, model or warp may be selected; no optimistic best-score summary, hypothesis test or confidence interval.

For the probability-only metric adapter, map intensity to intensity/65535 and contrast to 0.5 + contrast/(2*65535); reverse polarity by 1-p. These fixed monotone encodings preserve ordering and are not calibration. Independently reconcile ranks directly from signed integer numerators, avoiding float encoding. No feature-dependent support exclusion.

For each unsigned feature, scale, depth and support, report positive/negative class counts, means, median, first and third quartiles. Summaries across depths use medians, interquartile ranges and counts with AUROC >0.5 and >0.6; these descriptive thresholds do not select a feature or establish significance. Neighboring depths/pixels are dependent. Always report prevalence; AP across supports is not a controlled comparison. Plot all signed AP/AUROC profiles and class distributions with all four contexts and fixed metric axes; no best-depth crop.

## Verification and resources

Freeze protocol, plan, executable code, tests, inputs and metric implementation before execution. Synthetic checks: constant/ramp fields, impulses, triplet dilution, uint16 overflow, margin exclusion, order/tie-preserving encoding and invalid dtype. Independent verification constructs spatial sums using nine strided slices, depth sums using three slices and box sums using 81 direct offsets, rather than reshape/summed-area formulations. Include scalar spot checks at fixed corners/center for each layer. Reconcile every numerator, distribution and metric; preserve old inputs/results by hash. Check retained pooled FP32 inputs against triplet intensity/257 once as a scale contract, not new inference.

One worker attempt: 120 seconds, 1 GiB peak RSS, 64 MiB new phase artifacts excluding cumulative retention manifest. Verification has its own 120-second/1-GiB ceiling. Retain failures and diagnose before retries. No network, model calls, optimizers, R2 uploads or paid resources. Fresh writable MPLCONFIGDIR and XDG_CACHE_HOME for figures. Root review is not independent-agent review.

## Decision and limits

Assess whether a coherent depth-local contrast hypothesis merits one prospectively frozen representation experiment. Simple-feature failure does not prove CT lacks ink. Associations could reflect paper structure, renderer artifacts, incomplete labels or alignment; they do not authenticate ink. Optical-label agreement remains potentially circular. Same-parent disjoint regions cannot resolve unknown model ancestry or independent accuracy.

If profiles suggest consistent signed local contrast at a common depth range across contexts, formulate one depth-conditioned spatial representation and fixed baseline/ablation on prospectively selected disjoint development support. No fitting on these contexts and calling it confirmation. If profiles are contradictory/inconclusive, prioritize an independent reference/geometry check instead of another checkpoint/depth sweep. No experiment beyond this feature audit is covered here.

GPU spend £0; £10 TOTAL pilot cap including failures/retries/overheads. Paid launch still needs frozen workload, pinned CUDA, current all-in price, independent shutdown and verified scoped R2 worker credentials. No maintainer request, outreach, publication or submission.
