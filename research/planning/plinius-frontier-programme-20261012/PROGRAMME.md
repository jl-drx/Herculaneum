# Ranked research programme and discriminating experiments

This is a decision document dated 12 October 2026. All experiments below are
**proposed and unrun**. The current PHerc0139 study is explicitly excluded from
changes. Source identifiers refer to [SOURCES.md](SOURCES.md).

## The bottlenecks are different

| Stage | Scientific task | What does not establish success |
|---|---|---|
| Acquisition/reconstruction | Preserve information about fibres, sheets and ink | Smaller nominal voxels or a sharper-looking image alone |
| Surface detection | Locate sheet/recto evidence in native 3D CT | Voxel Dice alone; a thick mask may merge sheets |
| Tracing and topology | Keep sheet identity, connect correct parts, represent tears and separate neighbours | A smooth mesh, internally consistent graph, or a forced global spiral alone |
| Flattening | Map the recovered sheet with controlled distortion and no spurious overlap | An attractive 2D canvas |
| Placement and rendering | Sample the intended face with appropriate normals, spacing and interpolation | Reproducing a historical renderer or finding a brighter depth |
| Ink learning | Distinguish ink from fibres/artifacts and transfer across specimens/acquisitions | Training fit, teacher agreement, or consistency among correlated models alone |
| Text recovery | Establish source-supported readable strokes and specialist-supported readings | A language model's plausible completion |

A normal offset changes where an existing mesh samples. It cannot add an omitted
sheet, disentangle a merged prediction, fix a wrong winding assignment, or join
correct fragments. Conversely, a topologically coherent reconstruction can
still sit on the wrong face or be locally misplaced. An intrinsic quad-grid
topology does not by itself prevent its 3D embedding from crossing another sheet.

## 1. Correct-sheet recovery and topology

**Hypothesis.** Independently supported fibre/winding evidence and selective
repair can recover more connected *correct* sheet area than current tracing,
without hiding the gain in a higher wrong-wrap rate or deleting difficult areas.
This addresses missing/incorrect geometry, not merely depth placement.

**Limit and mechanism.** Binary recto fields merge close sheets and contain holes.
Local growth accumulates identity errors. A globally self-consistent wrong
assignment can survive cycle tests, and a diffeomorphic spiral may smoothly
interpolate through an unobserved or physically torn area. Local probability,
non-intersection, fibre continuation and independent winding anchors supply
different constraints; calibrated abstention can stop error propagation.

**Prior work / strongest baselines.** m7 is a released *component* of the Kaggle
winner, not the full winning ensemble. Include Socratic F0 with its stated TTA
and threshold, and the fine recto predictor where acquisition permits. For meshes
use current VC3D GrowPatch/copy, Stevens' conflict-filtered patch assembly, and
ScrollFiesta's documented pipeline; Lasagna/spiral fitting supplies the stronger
multi-sheet/global comparator. PLatS is a released instance/prompted baseline.
Winding-cycle checks, fibre constraints, topology losses and confidence-aware
neural tracing already exist [S01–S10]. None is a novelty claim for us.

**Smallest useful experiment G1.** Preselect 12 development CT blocks from at
least two scrolls, four each of open, curved, and crowded/damaged structure;
include touching-sheet continuations and true tears. Use public instance labels
and CT adjudication with ink maps hidden. If available, reuse published surface
fields. On exactly the same seeds and CT, compare m7 versus F0 with the *same*
tracer, then two released tracers with the *same* field. This separates detector
from mesher effects. Add a fixed, equal prompt budget for PLatS where it is an
eligible comparator; do not compare prompted output to unprompted output as if
their information were equal. A long connected path through adjacent blocks is
necessary for a tracing claim; isolated cube Dice is only a local diagnostic.

Only after finding repeatable failures, add **one** intervention: independent
fibre/winding anchors on flagged joins with uncertainty-based abstention. Compare
against the best eligible development baseline and simple pruning at matched
retained coverage. Withhold checking anchors from fitting. Include a small
reviewed-correction oracle on development data to estimate whether fixing that
failure could matter. Synthetic injected switches qualify the metric, not the
real-world method.

**Data / compute.** Native CT, m7/F0 probabilities or masks, meshes, surface/instance
labels, local fibre/relative-winding checks. A 256³ uint8 cube is 16 MiB before
compression; context, multiple fields and tracing across cubes multiply this.
Plan an exact chunk list before payloads; start at a few hundred MiB to a few GiB,
not a whole scroll. CPU meshing/audit is suitable locally; new 3D predictions
likely merit a 24 GB GPU. Large retraining/global fitting is a later request.

**Success / stop.** Pilot advance: at least 20% more unique, reference-supported,
connected correct-sheet area at matched wrong-sheet error and prompt budget,
with no additional confirmed catastrophic switch/self-intersection, or the same
area with at least 25% less measured correction time. Report coverage-error
curves, surface distance in µm, normal error, split/merge counts, tears, UV
distortion and time; do not let extra thickness count as area. Require gains in
both development scrolls. If too few actual errors exist to discriminate, call
the pilot uninformative rather than successful. Stop the proposed repair if
matched pruning explains its benefit, independent anchors cannot be verified,
or it bridges tears/wraps. Confirmation requires new physical blocks and blind
reference checks; zero observed errors in a small pilot is not zero population
error. A positive result justifies a multi-scroll error-controlled geometry study
and targeted surface training, not immediate whole-scroll deployment.

## 2. Broad native ink learning before another architecture

**Hypothesis.** Adequate native acquisition coverage and useful supervision can
improve transferable ink recovery more than further narrow continuation. This
is our strongest immediately trainable model hypothesis, independent of G1.

**Limit and mechanism.** A model trained on pooled fine CT can learn the wrong
appearance statistics for real 8–9 µm scans. Sparse single-scroll updates can
overfit texture or damage, and replacing old-source samples confounds diversity
with reduced exposure. Native 0139 alone cannot establish cross-scroll learning.

**Prior work / strongest baseline.** Reader v2 uses dense native supervision
across six objects plus the original corpus with the existing hybrid 3D/2D
architecture; dense-native and Hecate already cover native/fine transfer [S11–S14].
Keep all three, plus a predeclared official checkpoint/ensemble comparator.
There is no verified universal winner: Hecate leads Reader v2 on the reported
0841 human endpoint; dense-native is slightly ahead on the reported 0139 w047
teacher endpoint. Their published test sets are development comparisons for our
method selection, not a fresh Plinius confirmation set.

**Smallest experiment I1.** First reproduce a small fixed common-support released
comparison at each model's correct physical input contract. Then train a paired
coverage comparison with one released hybrid architecture, initializer, objective,
optimizer and schedule. Keep the original-source patch draws *identical* between
arms. Fill equal additional slots with repeated restricted-support samples in A
and broader native regions in B, with supervised class/exposure accounting and
the same annotation tier. This tests extra physical coverage rather than merely
less 0139 adaptation. Report unavoidable class/support differences rather than
calling an unmatched experiment causal. Use native-frame annotations where
available; teacher-derived targets retain their uncertainty and are scored
separately. Do not make the six failed fine/native transfers pixel-accurate truth.

**Data / compute.** Development regions from multiple acquisition bundles,
including native 116 keV material, approved mixed-scroll retention data and
known annotation-negative controls. A paired 20k-update continuation is a
reasonable *pilot candidate* given released recipes; Reader v2's full native
recipe is 40k at batch 64. Match effective batch/patch exposure, not just update
counts. Expect GPU-scale work, provisionally several to tens of GPU-hours on
24 GB hardware; measure it. One arm pair is exploratory; later use at least
three paired seeds and more physical parents. Tiny CPU failure does not veto it.

**Success / stop.** Proposed pilot gate: >=0.02 absolute AP improvement in
equal-parent native annotation agreement over the matched control, no parent
regression worse than 0.02, and no >2 percentage-point increase in independently
annotated negative false-positive rate at a development-fixed recall operating
point. Report AUC, calibration and all parents too. These are practical draft
margins, not significance estimates. Any final gate must be fixed after measuring
development precision, before confirmation. A gain only against the generating
teacher or caused by changed source exposure does not pass the mechanism test.
One useful pair justifies replication, not a frontier claim. Training still
improving at a resource cap is optimization-inconclusive, not absent signal.

## 3. Targeted annotation quality and evaluation coverage

**Hypothesis.** A modest amount of well-placed independent correction yields more
improvement per annotation hour than additional noisy pseudo-labels or uniformly
sampled labels. Surface and ink annotation have separate reference protocols.

**Limit and mechanism.** Mesh-derived surface labels may be off-face or cross
sheets. Sparse scroll ink labels may contain teacher ancestry; unsupervised zeros
are not negatives. Human review of model suggestions can reinforce common errors.
Correcting hard *development* regions, retaining ambiguity, and explicitly
annotating negative/blank structure can change the learning signal itself.

**Prior work / baseline.** The paper's fragment/manual plus iterative pseudo-label
loop, nnInteractive sheet refinement, dense-native/Reader supervision, curated
surface/ink datasets, and active learning already cover the general idea
[S01, S11–S15]. Use ordinary dense/soft teacher targets with manual precedence
as baseline. Novelty would require a measured label-selection/reliability benefit,
not a new dataset name or treating model disagreement as truth.

**Smallest experiment A1.** On development tissue only, allocate a fixed total
review budget (e.g. two reviewers × two hours). Compare uncertainty-stratified
selection with equal-time random selection, matched by physical support and
acquisition strata. Record corrected positives, negatives, unknowns, sheet identity,
agreement, time and adjudication. Keep an untouched independently annotated
checking subset. If an informative correction yield exists, train equal-compute,
equal-support ordinary versus corrected-target arms; add a coverage-matched
control if the correction policy changes which regions are admitted.

**Data / compute.** Development CT and meshes; optical evidence only where
registration is verified; specialist-reviewed or explicitly provisional scroll
marks. Annotation and CPU target construction first, then a GPU arm pair.
Expert annotation time is a real dependency; an agent agreeing with another
agent is not a papyrological reference. No outreach is initiated by this plan.

**Success / stop.** Advance annotation selection if it provides >=25% more
independently accepted corrections per hour than random at no worse disagreement,
then require the I1 transfer/negative-control gate for a learned benefit.
If disagreement remains irreducible, preserve unknown support and stop hard-label
transfer there. A quality-control benefit can be reported independently of a
model gain. Do not continually search new crops until a favourable correction
appears. Larger next step: an exposed-development and sealed-confirmation
multi-scroll reference with separate annotation provenance tiers.

## 4. Geometry-conditioned ink with explicit sheet identity

**Hypothesis.** Supplying independently estimated sheet identity and geometry
uncertainty can reduce neighbouring-wrap ink contamination while preserving
target-sheet ink, beyond Hecate's learned depth attention.

**Limit and mechanism.** Multiple sheets can enter one normal slab. A 2D label
does not identify the depth of ink; depth attention may select a wrong but ink-rich
neighbour. Our failed texture-depth proxy is not an admissible locator. Geometry
can constrain *which sheet* is scored without being selected by detector response.

**Prior work / baseline.** Hecate is the primary volumetric/attention baseline.
Lasagna, current neural sheet tracers and PLatS already model sheet geometry.
Generic multi-task networks, distillation, depth fusion and BCE averaged over
uncertain coordinates are prior art or ordinary averaged targets [S07–S14].

**Smallest experiment J1.** Use a fixed set of development examples with CT-reviewed
adjacent sheets, including target-blank/neighbour-ink and target-ink/neighbour-blank
cases where the labels support those distinctions. Hold rendering and Hecate
weights fixed. Compare its ordinary projection against projection restricted by
an independent target-sheet mask, a uniform-depth projection, and a support- and
smoothness-matched mask/control. Include a deliberately wrong-neighbour mask.
All outputs must share a denominator; masking away hard target pixels is not a
gain. This is a mechanism/upper-bound test, not yet a new learned model.

If the oracle-style geometry helps, train a geometry-conditioned student with
the same coarse CT at deployment, measured-uncertainty augmentation, and geometry
inputs available at deployment. Compare to a capacity-matched geometry-blind
model and spatially shuffled geometry. Keep an independent geometry loss/constraint;
never let ink response alone move the surface.

**Data / compute.** Valid sheet instances plus interpretable ink support in native
3D or independently aligned reference; Paris4 may supply development deposits,
but cannot stand in for every coarse scroll. Hecate 3D inference and training
merit a GPU; geometry-mask calculations are CPU work. Availability of independent
sheet/ink references, not a tiny CPU AP test, is the admission constraint.

**Success / stop.** >=25% relative reduction in wrong-sheet positives at matched
target-sheet recall, with no >0.01 target AP loss, across at least two development
parents. Benefit must exceed uniform/matched smoothing and persist with estimated,
not just oracle, geometry. Stop if the candidate merely lowers all responses,
its support shrinks, geometry identity remains unresolved, or smoothing explains
it. Advance to native training only after this result. This is materially
different from the rejected output-residual/depth adapters.

## 5. Acquisition and reconstruction robustness

**Hypothesis.** Some failures arise from reconstruction/intensity/acquisition
mismatch that can be corrected by matched training or measured augmentation,
rather than by changing geometry or inferring higher resolution from a coarse
image.

**Limit and prior work.** The 2026 paper uses a mixed morphology/phase/absorption
regime, not a universal topographic signal. The profilometry paper is a separate
modality proof of concept. Hecate and Reader already train on coarse native inputs;
blur/contrast augmentation and fine-teacher transfer are established [S01,
S11–S14, S16]. Our three-site reconstruction result motivates this track but
does not identify which reconstruction best preserves ink.

**Smallest experiment R1.** Use same-acquisition reconstruction/export variants
on physically identical development tissue. Verify coordinate support with CT
landmarks, then compare one frozen credible detector and basic intensity/blur
controls against the same independent native annotations. Extend to matched
training on the variants only if the difference is reproducible. A compact 2×2
train/test matrix distinguishes representation mismatch from an inference-only
failure. For fine/degraded/native comparisons later, maintain equal tissue,
supervision, training exposure and physical FOV and score both observation types.

**Data / compute.** Existing retained variants enable cheap CPU structural
statistics; larger fixed detector comparisons and matched training may need
1–several GPU-hours for inference and longer for training. Raw-projection
reconstruction/phase-retrieval work would require verified projections and an
imaging collaborator; it is a separate large project, not something justified
by uint8 surface stacks alone.

**Success / stop.** Advance matched reconstruction/augmentation if native AP gains
>=0.02 with the I1 retention/negative gates on new development blocks and no
geometric-support change. A pure inference difference only motivates the training
test. Stop a restoration method that wins only on visual sharpness, its teacher,
or synthetic degradation. Generative super-resolution cannot serve as recovered
historical evidence. Do not restart profilometry or the same six-site alignment
search; a new test needs different, consequential information.

## 6. Representation, objectives and compute efficiency

**Hypothesis.** Broad pretrained 3D CT representations or better use of an
adequate training budget may beat a freshly trained small model when labelled
native data is scarce. This is plausible even if the current CPU model fails.

**Prior work / baseline.** Start from a tuned hybrid U-Net/Reader, Hecate's
volumetric network, and released Dinovol features. The 2026 paper already uses
DINO-guided volumetric ink; current Dinovol has DINO/iBOT/Gram and point-supervision
paths, with incomplete general fine-tuning support. nnU-Net Revisited cautions
against weak U-Net baselines [S01, S13, S17]. Do not claim that adding a transformer,
SSL, a new loss or a larger context is novel by itself.

**Smallest experiment M1.** Extract fixed features on a bounded development cohort
with deployment-matched spacing, compare a simple regularized head with a similarly
budgeted learned CNN, and keep the labels and physical folds identical. A negative
linear probe closes that *probe*, not all fine-tuning. If features give useful
low-label transfer, compare equal-data fine-tuning with a tuned U-Net at both
equal exposure and equal measured GPU time. BCE/Dice/teacher-confidence changes
must be separate arms; gradient scale, sampling and annotation support must be
accounted for. Pseudo-label confidence is not calibrated ink correctness.

**Efficiency sub-study M2.** After choosing a credible model, test depth and jitter
together, with the full-depth baseline retained and physical context explicit.
Camillo already tested 17/13/9/5-layer variants; five layers with large jitter
failed, and the single-seed/one-rater narrow-window result is not a universal
equivalence claim [S18]. Use a small factorial full/narrow × proportional/fixed
jitter and report throughput, memory, transfer and neighbouring-sheet failures.
Do not change the current 28-plane native study or feed incompatible slices to
a frozen checkpoint. Faster Hecate kernels require output-parity verification
before being treated as a pure engineering improvement.

**Data / compute.** GPU feature extraction and fine-tuning, generally 24–48 GB
depending on backbone/context; large 3D pretraining from scratch may require
many GPU-days and is deferred. Local CPU tests verify sampling/gradients, not
the full representation hypothesis. No H100 assumption is made.

**Success / stop.** M1 advances with the I1 transfer gate across parents and a
useful label-efficiency gain, not just training fit. M2 advances at >=1.5× measured
throughput with <=0.01 AP degradation per parent and no material blank/geometry
regression. Failure of full-depth input contracts or truncated/no-fit budgets
makes the test uninformative. Do not begin broad architecture search or train a
foundation model until a released representation and strong U-Net have been
tested adequately.

## 7. Placement, normals, anti-aliasing and render stability

**Hypothesis.** Independently measured face/normal errors and sampling artifacts
explain some instability and can be reduced without altering sheet connectivity.
This is a supporting direction with lower novelty than correct-sheet recovery.

**Prior work / baseline.** The author's VC3D/PHerc343 render, ordinary mesh-normal
sampling, Lasagna snapping, robust model preprocessing and overlap blending are
baselines. Our exact-render, M7-offset, full-window, reversal, three-depth and
four-origin studies already cover important controls. Repeating them is not a
new test [EVIDENCE.md].

**Smallest experiment P1.** On CT-reviewed development surfaces with known sheet
identity, freeze a small normal-angle/placement perturbation set in physical
units and compare original normals with one independently estimated normal field.
Verify the renderer against a reference sampler; keep model, physical FOV,
interpolation and tile origin fixed. Separately test tile/normalization consistency
only on an annotated positive/negative development panel, with equal-cost
ordinary multi-origin averaging as comparator. Never select offset/normal by
maximal ink response. Hold the full-depth observation for the reference.

**Data / compute.** Existing CT/geometry for mechanics; independently reviewed
placement sites for a physical claim; modest CPU geometry and bounded detector
inference. GPU inference can be proposed if CPU runtime would be long.

**Success / stop.** Require >=25% lower median placement/normal error without
worse p90 or any added sheet switch, then a useful ink endpoint (I1 gate) before
claiming better recovery. A stability-only improvement must retain AP within
0.01 and be reported as robustness, not ink accuracy. If original placements
are already within reference uncertainty, or a smoother image alone explains
the result, close the branch. No new broad depth sweep of the same PHerc0800
region is recommended.

## Larger experiments justified by positive pilots

| Pilot evidence | Next investment | Claim still requiring confirmation |
|---|---|---|
| G1 increases correct connected area at controlled identity error | Several scrolls, longer traces, targeted surface labels; GPU surface fine-tuning only if field errors dominate | Generalizable surface recovery and intervention efficiency |
| I1 improves native transfer under matched exposure | Three or more paired training seeds, more acquisition bundles, reserved external parent(s) | Better native ink recovery rather than teacher imitation |
| A1 yields valuable reliable corrections | Structured annotation effort plus matched label-quality training | Benefit per annotation hour and downstream transfer |
| J1 suppresses known wrong-sheet ink beyond simple controls | Geometry-conditioned training with measured uncertainty and capacity controls | A joint geometry/ink mechanism |
| R1 shows correctable real reconstruction mismatch | Matched acquisition matrix; collaborate on reconstruction only with raw-data access | Robustness to genuine acquisition changes |
| M1/M2 gives transfer/efficiency benefit | Broader pretrained-feature adaptation or economical multi-seed training | Architectural/representation advantage at fair compute |

No single pilot, no two public checkpoints, and no collection of correlated
pixels establishes a frontier claim. The next contribution should earn its
scale-up through an informative mechanism test, not through a requirement to
win an intentionally inadequate CPU proxy.
