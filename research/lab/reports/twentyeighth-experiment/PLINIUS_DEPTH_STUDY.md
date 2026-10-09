# Plinius H2: does small surface-position uncertainty matter for ink detection?

## Decision and hypothesis

The next scientific result should compare two fixed recipes on the same documented reference. The hypothesis is that a fixed average across nearby forward depth windows can reduce sensitivity to a small surface-position error without materially degrading performance at the nominal surface. This is a diagnostic comparator for the H2 research direction, not a proposed novel architecture or evidence that averaging will work. Seed42's retained training configuration already includes depth-window jitter up to2; depth robustness is not a capability absent from its training recipe.

Keep the unchanged forward seed42 model as baseline. Do not retrain, crop its depth width, reverse directions, pick a best depth per region, fit averaging weights, add a second checkpoint or choose a threshold from evaluation outcomes. Native reference results and the negative reversal result remain part of the motivation. We are testing local translations along an explicitly defined normal, not reversal of that normal or invariance to arbitrary surface changes.

## Exact comparison

Let delta be ONE validated effective rendered depth-plane interval in physical units. Its micrometre value and pooling rule must be bound to the selected reference before data acquisition/scoring; the bare3.24um scan name is insufficient. If that interval is implausible as a surface-error probe, revise this design prospectively with geometric evidence, never prediction outcomes.

Build21consecutive effective-depth planes with the same validated XY sampling, pooling and signed-depth convention as the reference input contract. Let p_k be seed42's raw probability mosaic from the17plane window starting at index k+2, where k is -2,-1,0,+1,+2. Thus windows are0:17,1:18,2:19,3:20,4:21, never an extra implicit center crop. Preserve physical spacing and full sampling/interpolation support when rendering these effective planes; they may require many more than21native source planes.

For imposed surface displacements j=-1,0,+1:

- Baseline A(j)=p_j.
- Fixed candidate B(j)=(p_(j-1)+p_j+p_(j+1))/3, averaging raw probabilities in float64 after identical spatial blending.
- Implementation control: (p_j+p_j+p_j)/3 must recover p_j within the predeclared arithmetic tolerance; this is not a new inference arm.

Exactly five unique windows per tile are needed for all comparisons. Each operational B output uses three model evaluations versus one for A. Report this cost; reuse of predictions for the paired analysis does not make B a one-pass model. Adjacent windows share16of17planes, and repeated depths/contexts are not independent samples.

Use the exact retained seed42 factory/state and robust-MAD normalization implementation, fixed17x128x128 model inputs, forward order and the already qualified FP32 evaluation path. Normalize each raw window with the standard recipe: this measures the operational effect of changing physical support, including any normalization change. Retain normalization parameters/input hashes to inspect that effect; do not silently freeze or change normalization to improve an arm. Same XY tile/blend schedule and identical primary support across all windows/arms. Existing byte-identical center outputs can be reused only with a complete input/state/support identity match.

## Reference and selection

Start with one documented exposed-fragment reference as a feasibility study. Frag6 is a candidate, not an admitted dataset. Select context positions from geometry/coverage and declared annotation support before model scores; retain every selected context and failures. Bind a common support requiring complete source coverage for the union of all five windows and trustworthy positive/negative/unknown annotation semantics. Record exclusions and area before inference. Contexts with only one known class remain coverage observations and cannot silently disappear from the selection record; no accuracy aggregate is released unless the predeclared usable-panel conditions are met.

A newly constructed reference is a valid alternative to reconstructing a historical recipe. It must bind the chosen source, explicitly verified surface coordinates, new rendering code/parameters, optical image, new or authenticated annotation, transform and held-out landmark residuals. Unknown upstream model exposure limits the claim; it does not make all descriptive prototyping impossible. Direct/model-derived labels support supplied-reference agreement, not independent optical accuracy. Reused Paris contexts remain development and are not a fallback confirmation set.

See REFERENCE_CONSTRUCTION_PLAN.md and DEPTH_STUDY_CONTRACT.json for remaining inputs. Exact reference identities, physical delta, masks, coordinates, annotation review, finite resource bounds, model input/output code and decision margins must be frozen before execution. No scientific inference is authorized by an incomplete JSON record.

## Measurements and decision rule

Primary descriptive quantities: per-context raw-score AP for each arm/displacement, and paired delta AP. Average the three displacement values within context; average contexts within parent; report parents separately before any equal-parent macro. Also report j=0clean AP, AUROC, complete-support coverage and raw score changes. Keep AP prevalence and label-class counts visible. Do not pool pixels to simulate independent replication or apply pixel-bootstrap significance.

False-positive evaluation requires a disjoint, documented calibration support. If available, derive each arm's cutoff using the SAME predeclared conservative95%-specificity rule on calibration, then freeze that cutoff across all evaluation displacements. Report attained specificity, sensitivity and confusion counts together: empirical cutoffs do not guarantee matched evaluation specificity. If calibration/support is missing, omit threshold-performance claims and report why. Optical accuracy and supplied-label agreement must not be merged.

This initial study is descriptive and does not promote a new model. Before any model-scoring release, use reference-only annotation/registration reliability and intended operating cost to freeze a minimum worthwhile robust-AP improvement and a clean-AP non-inferiority margin. Those values are currently null rather than invented from an unknown panel. Never set them after examining arm scores. No ranking sweep follows a negative comparison.

If the fixed candidate worsens both robustness and clean performance, retire it as a baseline comparator. If results are mixed, characterize the tradeoff without selecting regions or weights. If it meets the prospectively bound practical criteria, replicate on separate untouched physical support before proposing learned uncertainty weighting. A one-parent improvement is feasibility evidence; repeated seeds do not replace additional parents. Translation or letter identification is not an endpoint of this study.

## Resource and claim boundaries

First estimate the actual number of tiles and5unique model inputs per tile. Prefer the existing CPU path if the frozen small panel fits its measured budget. The£10TOTAL GPU pilot is conditional on useful work, pinned CUDA image, current all-in price, runtime cap/independent shutdown and scoped worker credential delivery. It buys a bounded feasibility/throughput run, not a promised full training programme. No new GPU spending authorization is requested here.

An uncertainty-conditioned Plinius model remains a later hypothesis. Equal averaging is common methodology; no novelty claim follows from implementing it. Any first-in-class claim will need contemporary prior-art comparison, controlled training ablations, external reference support and replication. Research merit leads; prize eligibility is reviewed separately if a submission becomes concrete.
