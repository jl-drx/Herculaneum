# A discriminating next research direction

Review date: 11 October 2026. This is a prospective research assessment, not a new experiment or a change to study52. I read retained primary model cards, training/evaluation code and the negative studies. No model inference, scan chunks, weights, network downloads, paid resources or active-study files were used or changed. [SOURCES.json](SOURCES.json) records exact local-source hashes and pinned upstream links.

**Recommendation:** qualify PHerc0139 as a physically paired benchmark before constructing another model. The useful hypothesis is not “a high-resolution teacher helps a coarse student”: two released systems already implement that. It is **whether independently measured correspondence uncertainty explains and can reduce transfer error, beyond ordinary label smoothing or selecting easier patches**. Do not implement uncertainty-aware training unless the geometry audit actually finds consequential, recoverable same-sheet uncertainty. The newly identified same-scan reconstruction variants offer an unusually informative supporting control, potentially ahead of a training intervention.

## What is established prior art

1. **Hecate already combines fine-to-coarse distillation, native paired renders and depth selection.** Its [pinned model card](https://huggingface.co/scrollprize/hecate/blob/9cb86e500e944b11a06a7020403cde5dffb5bcb2/README.md), “The 9.6 µm model”, states that fine-model 2D/3D targets supervise degraded fine CT first and aligned native coarse CT later, with blur/contrast augmentation; failed native alignment falls back to fine-derived inputs. It already has volumetric ink output and learned depth attention, and explicitly warns that neighboring sheets can contaminate predictions. Generic distillation, depth attention, and “learn the middle sheet” therefore cannot be our novelty claims. The public release does not provide the full training alignment procedure or a complete physical-region ancestry, so this review cannot establish whether our precise proposed correspondence treatment is absent from unpublished development.
2. **Nieuwlaar already ran a native-supervision treatment and matched continuation.** The [pinned README](https://github.com/Nieuwlaar/ink9um-dense-native/blob/217e7521670db49cde97872ea6a8bdfb80c449f1/README.md), section 3, adds ten PHerc0139 native-113-keV segments using resized 78-keV teacher maps and manual-label precedence. The public [label builder](https://github.com/Nieuwlaar/ink9um-dense-native/blob/217e7521670db49cde97872ea6a8bdfb80c449f1/training/scripts/build_native_dense_labels.py) constructs supervision from native validity and teacher-positive support, then thresholds the teacher; it does not estimate a local registration posterior. Its benchmark excludes title/w024/w047/w053 from those native training configurations. These are useful author-reported exclusions, not a complete ancestry audit of every upstream teacher or Hecate.
3. **A short local fine-tune is a weak discriminator.** Our studies37/40/41 show improved fit, small transfer changes and exposure/supervision tradeoffs, not convergence or incapacity to learn. Nieuwlaar independently reports substantial degradation from sparse single-scroll fine-tuning. Repeating that family would not distinguish acquisition mismatch, uncertain correspondence and wrong-sheet supervision. Study42 also establishes that a public coordinate transform and plausible meshes do not alone verify local cross-acquisition correspondence.

## The identifiable question

Write the observed input schematically as `X = render(reconstruct(projections), mesh, normalization)`. A change between fine78-keV and native113-keV inputs changes several factors together: acquisition energy, spatial sampling and point-spread behavior, reconstruction, geometry and exported intensity scaling. Deliberately degrading fine CT changes only an approximation to some of them. Therefore:

- A native-versus-degraded comparison estimates an **acquisition-bundle transfer effect**, not the effect of voxel size alone.
- A fixed model that prefers one input does not show that another input contains no recoverable information; matched training at each input quality is needed for that question.
- Agreement with the model that generated training targets measures reference agreement. It cannot, by itself, distinguish better ink recovery from more faithful teacher imitation.
- Resizing to the same pixel dimensions is not a registration validation. Globally convincing correlation can coexist with local warp errors or another neighboring sheet.

The training hypothesis worth testing is narrower: **on the same physical sheet, CT-derived local correspondence uncertainty allows a coarse-input student to use imperfect fine supervision more effectively than a single correspondence or generic smoothing.** The proposed mechanism predicts an advantage concentrated in uncertain-but-identifiable correspondence regions, little advantage in very well registered regions, and failure/abstention where sheet identity itself is unresolved. This is a testable hypothesis, not a present novelty or performance claim.

## An important mathematical limitation

Let `q(Tu)` be a fine teacher probability mapped to coarse pixel `u` by possible transform `T`, and let `pi(T | CT)` be a distribution obtained without using ink predictions. With binary cross-entropy,

`E_T BCE(p(u), q(Tu)) = BCE(p(u), E_T q(Tu))`.

Thus averaging BCE across uncertain alignments is exactly training against an averaged teacher target. A constant translation distribution is just spatial smoothing. Calling this a latent-geometry architecture would exaggerate it. Any contribution must show that **spatially varying, independently supported correspondence information** matters beyond a blur with the same overall amount of smoothing. The distribution is an empirical set of plausible correspondences until calibrated; correlation-peak width is not automatically a Bayesian posterior or confidence interval.

Do not marginalize across different papyrus wraps merely because they have similar CT texture. Mixing true ink on different sheets can create plausible but physically wrong target strokes. Require independently supported sheet identity first; otherwise mark the target unknown and preserve the excluded fraction in the result. Geometry/model predictions may propose a sheet, but cannot certify it independently when sharing the same learned lineage.

## Prospective design, with controls that identify the mechanism

This is a design to freeze after qualification, not authorization for four training jobs. Every arm must share architecture, initializer, optimizer, update count, patch sequence, augmentation draws, physical training regions, positive/negative loss convention and effective supervised support. Keep the mixed-scroll retention data fixed; adding the treatment must not silently reduce original-corpus exposure again. Reference probabilities and manual targets need a declared hierarchy; teacher-zero/absent render values are not automatic negative ink labels.

| Factor | Level 0 | Level 1 | What the contrast can establish |
|---|---|---|---|
| Input acquisition | Fine CT deliberately degraded to coarse sampling | Registered real native coarse CT | Response to the acquisition bundle at matched physical support and exposure |
| Teacher correspondence | One fixed same-sheet correspondence | CT-derived mixture of same-sheet correspondences | Value of uncertain correspondence, conditional on support and label smoothing |

The resulting four cells are a useful eventual factorial, evaluated on *both* input acquisitions at a predeclared endpoint. Report the input-by-correspondence interaction as well as both main contrasts. A two-arm budget can test the correspondence contrast on native CT only; it cannot also identify the full factorial or attribute benefits specifically to acquisition mismatch. Additional controls should be staged only if that contrast merits them:

- **Fixed smoothing control:** the single-correspondence target convolved with a globally fixed kernel whose uncertainty distribution is set on training data. Match the treatment's overall smoothing and support; do not choose it on evaluation scores.
- **Uncertainty-permutation control:** permute local uncertainty descriptions between spatially separated, comparable training blocks, preserving their marginal distribution. If this performs equally, local CT-derived uncertainty may be unnecessary. This is stronger than simply comparing soft versus hard labels.
- **Coverage-only control when relevant:** use the treatment's exact accepted mask and sample sequence with single-correspondence targets. Otherwise improved performance could come entirely from discarding hard examples. Both arms report total excluded area.
- **Intentional misalignment positive control:** perturb a verified pair by known local displacements and check whether the estimator responds. This tests sensitivity of the qualification machinery, not benefit on naturally misregistered tissue.
- **Wrong-sheet negative control:** preserve a neighboring sheet's texture/context while deliberately assigning the incorrect correspondence, on regions with identified sheet identity. The method must reject/mark unknown; apparent improvement in a shared teacher score is not success.

Frozen Hecate and dense_native are useful external comparators. A treatment against a different architecture/checkpoint cannot replace a matched training control. If the pilot budget affords only one matched arm pair, use it as a pilot and reserve a second paired initialization/physical replication for any stronger claim. A historical $4 run is not a current cost estimate. There is no scientific reason to require CPU AP improvement from an intentionally tiny training budget first.

## Supervision and evaluation that are defensible

**Training:** use fine-scan teacher probabilities only as pseudo-labels; respect supplied supervision masks and retain manual-label precedence where provenance supports it. Build all correspondences, uncertain sets and acceptance masks from geometry/CT before inspecting model outputs. Do not fit transforms to teacher/student agreement. Physical exclusion operates across acquisitions, alias segment names, neighboring windows and model context; pixel masks alone are insufficient. In particular, the earlier audit's w028/w044 alias is a warning against name-based splitting.

**Evaluation:** reproduce the published native benchmark unchanged as a descriptive baseline, then separately report a qualified-support view. Its [actual code](https://github.com/Nieuwlaar/ink9um-dense-native/blob/217e7521670db49cde97872ea6a8bdfb80c449f1/native-eval/native_bench.py) uses area-resized teacher maps, confidence quantiles, native nonzero validity, and tie-aware AUC. The code's large shift/rotation controls test whether correspondence matters at all; they do not establish subpixel local registration. Do not retroactively replace its score while retaining its published benchmark name.

Use the separate59-keV/model reference on w024 as a required cross-check, with an explicit record of its ancestry and overlap. It is a different observation/reference, **not automatically independent ground truth**. A teacher-agreement ratio below the author's1.02 flag is a heuristic, not proof against shared bias. w053 remains an author-designated blank control; report flagged area at a predeclared recall operating point rather than asserting chemically proven absence of ink. Do not use test reference values to select smoothing, geometry, thresholds or checkpoints.

The currently evidenced benchmark has few physical segments, and the alternate59-keV comparison is especially narrow. Millions of correlated pixels do not create millions of scientific replications. Publish paired per-segment/block effects and uncertainty conditional on those blocks; do not claim population generalization or a calibrated training-seed variance from two released checkpoints. If training ancestry cannot exclude a candidate region, designate it a reused development diagnostic. A new model claim needs an honestly disjoint physical evaluation and, ultimately, independent ink/papyrological evidence.

## What can be done on CPU before training

A bounded CT/geometry qualification can establish:

1. Correct axis/order/spacing/normal conventions, same physical support, transform direction, context and interpolation bounds.
2. Local correspondence residuals on fixed geometry-selected blocks, with blind non-ink structures and reserved structural checks. Repeated correlations at nearby offsets are not independent confirmations. Include competing depth/sheet matches and identify ambiguity explicitly.
3. Whether a correspondence interval captures injected known offsets and withheld landmarks, whether uncertainty varies enough to justify a spatial treatment, and what fraction of tissue remains usable.
4. Whether the support of mapped teacher targets changes materially within the accepted correspondence uncertainty. Calculate this only after the CT-only geometry choices are fixed; do not adjust geometry to improve target agreement.
5. A physical-region ancestry/exposure manifest and a feasible streaming plan before any GPU launch.

This does **not** establish that native CT is unlearnable, that teacher targets are true ink, that the proposed student will improve, or that local uncertainty is calibrated outside the tested blocks. Small optimizer/gradient sanity checks can later qualify implementation; they should not recreate the old CPU-performance admission gate.

**Stop/redirect before training** if identity cannot be distinguished from a neighboring wrap, held-out structural checks fail, uncertainty is negligible relative to mapped features, or almost all accepted pairs already fall in the same narrow high-confidence stratum. Those observations respectively favor geometry work or ordinary matched acquisition learning, not a more elaborate uncertainty model. If the native acquisition differs materially but correspondence is already precise, use the released native-supervision recipe as the primary learning baseline.

**What would support advancement:** at an endpoint fixed before training, a matched treatment improves a predeclared practical reference-agreement effect on reserved physical regions, with corroborating direction on the alternate acquisition/reference, controlled blank false responses and no material retention regression. Benefits should exceed fixed/permuted smoothing at the same support/exposure before crediting local uncertainty. Set the practical effect size from an explicit task decision and attainable benchmark precision, not “twice the spread of two released seeds.” The old+0.01 Paris AP threshold cannot be transported mechanically to a different native AUC endpoint.

**What would falsify the proposed mechanism:** well-supported same-sheet uncertainty exists, yet at sufficient matched exposure the mixture is no better than the single transform or matched smoothing; gains vanish on the different-acquisition reference; gains occur equally with permuted uncertainty; or they are entirely explained by rejecting difficult areas. A one-seed null with inadequate sensitivity is inconclusive, not a reason to add arbitrary depth/fusion adapters.

## A stronger supporting control found in the existing catalog

The parent's inventory identified three PHerc0139 reconstructions from the **same scan20250720065842**, all113keV/9.362µm: usual volume20250728140407, pag0 volume20251107132835, and pag50 volume20251107135911. I checked their retained catalog entries directly; the [exact excerpt](PHERC0139_RECONSTRUCTION_CATALOG_EXTRACT.json) preserves them.

This is scientifically valuable because it holds the acquisition/specimen/nominal physical coordinates fixed while varying reconstruction/export. The publisher reports identity alignment from full-resolution3D block correlation, including an off-axis check; pag50 agrees with pag0 at r=0.98 at zero shift. These are publisher claims pending local checks. Neither sibling is marked for automatic segment rendering (`render_segments=false`); existing surface renders are not established by this evidence.

Important qualifications:

- The pag0 entry mentions disabled detector-distortion correction. Do not assume the only difference is one Paganin setting.
- The masked uint8 conversion windows differ: usual−0.03…0.145, pag0−0.09…0.4, pag50−0.16…0.75. Their metadata originally carried a misleading default; the catalog explicitly corrects it. Linear remapping cannot recover clipped information.
- The usual volume has20974z planes; siblings have20961. All have6621×6621XY. Both siblings have reported mask faults at z1144–1310 and from20448 onward. Identity coordinates and r=0.98 do not excuse these exclusions.
- These are reconstructions of the same measured rays, not independent observations or newly acquired information. A model could recover useful signal differently, but cross-reconstruction agreement is not an independent ink truth.

**Priority:** qualify one preregistered interior physical crop across these variants as a supporting control if rendering it with the existing geometry/chunk tooling is inexpensive. Freeze the mesh, normals, support, resampling and display; independently verify local identity and assess intensity/clipping/frequency changes before any new model maps. A later frozen-model comparison would ask whether a reconstruction/export change alters source-localized responses, with fine-scan reference and blank controls; a matched training comparison would be needed to separate input information from model familiarity. Do not optimize reconstruction choice on PHerc0800 output brightness.

This route can isolate a specific part of the acquisition bundle more cleanly than fine-versus-native comparisons. It should **supplement**, not replace, PHerc0139 pairing qualification: it has no independent labels by itself and still requires rendering. If correspondence to the original scan is locally verified but alternate reconstructions change responses strongly, reconstruction mismatch becomes a concrete competing explanation worth testing before new uncertainty architecture. If they scarcely affect the evidence, deprioritize that factor and proceed with the simpler matched learning design.

## Decision summary

Finish the unchanged study52. In parallel, complete metadata/geometry qualification of PHerc0139 and bound the same-scan reconstruction control. Choose the next intervention from those results: **precise pairing → matched acquisition training; material same-sheet uncertainty → controlled correspondence treatment; ambiguous sheet identity → geometry/data work.** None of these conclusions follows from another small unpowered fine-tune or from inspecting brighter maps. A publishable contribution may be a well-controlled account of which acquisition/geometry factors matter, before it becomes a new model.
