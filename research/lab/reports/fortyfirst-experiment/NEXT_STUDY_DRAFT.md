# Proposed next experiment: Paris4 observation-quality comparison

This is a proposal, not an executed or frozen training study. Its purpose is to distinguish recoverable signal under a specified learner from acquisition mismatch. It replaces further recipe/crop searches as the main research direction. Read PAIRED_ACQUISITION_REVIEW.md and research/PAIR_FEASIBILITY.json first.

## One bounded comparison

Use PHercParis4 modern2.4µm volume20260411134726 and older7.91µm volume20230205180739, beginning with the publicly paired geometry for segment20231016151002. The catalogue contains both directed transforms and meshes in both coordinate systems; raw metadata is accessible. Local correspondence, mesh sampling equivalence and independent reference support remain unverified.

Select separate physical blocks for training, protocol selection and a locked within-parent comparison, with at least two blocks per role if the reference permits. Keep all representations of a block in the same split. Guard bands must cover the full physical model context, native-volume sampling and depth support under both transforms. Repeatedly inspected Paris regions cannot become a new untouched test. Paris4 is also upstream-exposed for seed42. A small cleanly initialized conventional learner is the primary test of local learnability; pretrained models are qualified descriptive references.

Define three observations: H native modern CT; S one physically specified blur/sampling/noise degradation of H; R registered real older CT. Choose the degradation from acquisition properties or training-only calibration, not the locked results. Report its mismatch from the real scan. Use identical physical field of view, depth support, scored reference masks and model capacity across primary arms. A common rendering grid may contain interpolated coarse observations, but those remain explicitly coarse. Preserve physically relevant depth rather than shrinking the window to reuse an incompatible tensor contract.

| Trained observation | Fixed evaluation observations | Purpose |
| --- | --- | --- |
| H | H,S,R | Within-domain positive control and transfer under changed observations |
| S | S,R | Matched synthetic learnability and synthetic-to-real mismatch |
| R | R | Matched real-coarse learnability |

This is three fits and six comparisons if real pairing qualifies, or two fits/three comparisons if it does not. Freeze one adequate CPU update budget per fit after a training-only fit check; no development-driven extensions, sweeps or early checkpoint selection. Use matched optimizer/update exposure and a common initial state for the primary capacity-controlled learner. A suitable contemporary native model and a coarse model may be retained as references, but different families cannot substitute for this controlled comparison.

## Admission and stopping

Limit one admission exercise to at most256MiB of CT chunks and32MiB of geometry/reference material, with a fixed candidate-block list and no prediction-based selection. Existing chunk sizes are2MiB. Inspect the published mapping's local residuals using non-ink structural landmarks excluded from any correction fit. Freeze the physical error tolerance against the intended output pixel size and neighboring-wrap spacing before reading candidate model results. Require valid common support and explicit wrap identity. Numerical matrix inversion alone does not pass this gate.

An independent reference subset must include directly reviewed deposits and defensible non-ink substrate; mark uncertain pixels unknown. Dense DINO/detector outputs may support training or a separately labelled reference-agreement diagnostic, but cannot be the sole test truth for their own descendants. Record annotation confidence and sensitivity to registration uncertainty before selecting a winner. If neither independent positive nor negative evidence can be admitted, stop the ink-recovery comparison and state the missing evidence instead of training against an unqualified proxy.

At the admission deadline, either freeze the usable native/synthetic/real panel or drop the unqualified real arm and complete the smaller comparison. Do not turn this into repeated metadata-only searches. If no defensible native reference can be constructed in the bounded exercise, choose another documented material pair; do not consume the reserved parents automatically.

Once admitted, stop at the frozen updates or an integrity/nonfinite/resource failure. A training-only fit failure blocks scientific interpretation and further scaling. Score all physical regions with AP,AUROC,coverage and sensitivity at a selection-block-fixed operating point. Paired regions, not pixels, are the units for descriptive uncertainty. Two blocks per role cannot establish broad cross-scroll generalization.

## Results that change direction

- H-to-H works and H-to-S fails, but S-to-S recovers: acquisition/representation mismatch is demonstrated within this synthetic setting; degradation alone did not eliminate all usable signal.
- H-to-H works but S-to-S remains weak after a successful fit check: the degradation/learner loses useful evidence; this is not an impossibility result for coarse CT.
- S-to-S works and S-to-R fails, but R-to-R recovers: prioritize real-domain matched training and acquisition differences.
- R-to-R works: test training-only fine-observation distillation next, against both ordinary supervision and a coarse-teacher control, keeping labels and student exposure fixed.
- H-to-H remains weak on disjoint blocks despite training fit: improve the reference, representation or material coverage before introducing a privileged teacher.
- Only geometry-corrected observations work: first prefer correcting the rendering; uncertainty-aware geometry becomes a separate controlled hypothesis when residual uncertainty remains.

Preserve the practical +.01 AP gain and -.02 regional retention limits for a prospectively specified matched-input baseline/panel. Diagnostic patterns need not pass a promotion gate to be informative. Keep old studies and their failures unchanged; the old eight-region panel remains reused regression evidence wherever the input contract is applicable. Do not force an incompatible native model onto that panel and call the comparison fair.

No privileged teacher, geometry model or GPU job is part of this first comparison. CPU spend is unrestricted; GPU work remains deferred and the £10 total pilot cap persists. A later repeat on another physical parent is required before claiming broad transfer or investing in large training.
