# Native PHerc0139 learnability comparison

Written 11 October 2026 before real optimization or held-out predictions.
`GO.json` seals the executable code, this protocol, PLAN, execution qualification,
environment, annotation objects, CT objects, receipts and exact physical folds.

## Question and scope

Can a fresh full-depth native CT learner predict the released native annotations
on an excluded winding better than a simple per-pixel depth-profile model? Does
it fit its training support, and does that fit transfer? This is a baseline for
the proposed paired-acquisition research. It is not a new architecture claim,
an ink authentication exercise, a return to the rejected depth/fusion adapters,
or another fine/native registration diagnostic.

Use all 265 previously admitted mask-selected cores: 2,825,834 supervised pixels
in PHerc0139 w035/w039/w044, native 9.362 µm, 113 keV. The retained 536 chunks
provide all 28 depth planes. No new source download or pretrained weights.
The supervision mask defines valid labels: nonzero ink outside it is ignored,
not treated as negative. The 67 cores without supervised positives and one
without supervised negatives remain. Stored annotation plane14 is not an ink
depth estimate. Native-frame annotation correspondence is sufficient here;
the fine-scan teacher's uncertain registration is not used.

Evaluation uses the previously fixed complete cores; train context exclusions
use the audited conservative XYZ boxes, full 28 planes and 256-square context.
Held-out core counts are 109/54/102; corresponding training counts 139/196/163.
Read their exact ordered IDs from the pinned geometry result. No endpoint
selection, normalization or parameter fitting may use held-out model outputs.
Different folds can reuse tissue and are not independent specimens. All three
windings belong to one scroll, with incomplete historical annotation ancestry.
The model starts from scratch, but this is not a pristine public-data benchmark.

## Arms and objective

One training-prevalence constant baseline and one 29-parameter logistic depth
profile baseline per fold. The latter sees the normalized 28 intensity values
at each scored pixel, trained on every valid supervised training pixel. L2 is
0.001 on its 28 slopes, none on the intercept. Deterministic L-BFGS-B from zero
slopes/training-logit intercept, maximum1000 iterations, gtol1e-7/ftol1e-12,
maxls40. Require final gradient infinity norm <=1e-5; otherwise stop and retain
the failed optimization, rather than compare with an unconverged reference.

The spatial learner is an ordinary 122,513-parameter 2D U-Net with 28 input
channels, widths16/32/64, two average pools, two 3x3 Conv/GroupNorm4/SiLU layers
per block, bilinear upsampling with skips, and a single logit output. Score only
the central128 of a full256 context. GroupNorm's statistics stay within that
audited input context. No coordinates, depth shifts, teacher input, depth
reversal, pretrained weights, output fusion or probability threshold fitting.
The context is intentionally common to both data preparation and footprint
audit even though the profile model uses only each pixel's depth vector.

Constant per-channel means and standard deviations come only from the admitted
training context occurrences, including valid unlabeled training context.
Repeated context occurrences count repeatedly; held-out intensities never enter
these moments. Divide uint8 by255; std floor1/255. Use the same constants for
the profile and spatial models and every held-out input.

CNN seeds17/43/89, three folds each: nine fits. Batch4; select training cores
with replacement in proportion to their supervised pixel counts, then average
each core's unweighted masked BCE and average over the batch. Its expectation
is the uniform supervised-pixel objective used by the profile baseline. Joint
D4 rotations/reflections of the whole XY context and target/mask; never reorder
depth. No positive oversampling, Dice or positive class weights. Adam lr1e-3,
default betas/epsilon, no weight decay; clip total gradient norm at5. All updates
use CPU float32, deterministic PyTorch operations and four threads.

## Optimization budget and endpoint, fixed before evaluation

Evaluate the complete unaugmented **training** set every256 updates, plus step0.
Keep all checkpoint receipts. A training BCE decrease of at least0.0005 resets
the significant-improvement counter. Four checks without that decrease lower
lr by0.3 (floor1e-5) and reset the counter. At the floor, eight checks without
such a decrease stop the run, but never before4096 updates. Ceiling16384 updates
per arm. The sole endpoint is the checkpoint with the lowest measured training
BCE, earliest on an exact tie. AP is logged but never used for stopping or model
selection. No held-out predictions, including baseline predictions, until all
nine training endpoints are committed together in `evaluation/ENDPOINTS.json`.

The ceiling is finite execution control, not a proof of adequate optimization.
At the ceiling, best-training-BCE improvement >=0.0005 over the preceding1024
updates is explicitly `ceiling_still_improving`; otherwise the ceiling still
does not establish the declared floor plateau. Both are unresolved optimization
for a negative transfer conclusion. Report updates, sampled core exposures,
unique training cores, supervised pixel exposures, learning rates and all
training curves. Minimum/maximum draws per arm are16,384/65,536, versus139–196
eligible training cores. Sampling is unequal, so average exposures are not a
claim that every core was seen equally often. CPU duration is not a spend cap.

## Evaluation and practical decisions

Save raw floating-point logits, ordered core IDs and checksums. Compute pooled
annotation AP/AUC and BCE per winding plus every core, mean probabilities and
response fraction at the fixed probability0.5. AP uses grouped ties; single-class
core AP/AUC are undefined and remain in the loss/response summaries. Report
each seed and its range, not only their mean. The primary contrast is the mean
of the three individual CNN APs minus profile AP in each winding (not an
ensemble prediction). Three seed results describe optimization variation, not
a population confidence interval. No pixel-bootstrap significance claims.

Prospective practical advancement requires all of the following:

* Mean CNN AP minus profile AP >=0.02 in **each** winding; every individual seed
  exceeds the profile baseline in that winding.
* Each selected model's training AP >=0.85 and training BCE at least20% below
  the optimal training constant's binary entropy. These are explicit fit
  adequacy heuristics, not an assertion that the labels are physically perfect.
* For each fold/seed, pooled mean probability over the fully supervised-negative
  cores is no more than0.05 above the profile model. These are zero-positive
  **annotation cores**, not authenticated blank papyrus. Report per-core results
  as well; this guard does not establish a particular false-positive rate.

The0.02 AP margin is a practical research-priority threshold fixed for this
dataset; it is **not** the old Paris gate, a measured noise floor or a power
calculation. A positive result supports a broader matched ordinary-soft versus
CT-averaged versus smoothing paired-supervision study. If the fit is adequate
and demonstrably plateaued but transfer fails, prioritize tissue coverage or
reference/domain mismatch. If fit is poor at a confirmed plateau, permit one
subsequently frozen training-only capacity/optimizer diagnosis; do not declare
coarse CT unreadable. A ceiling-limited null is unresolved optimization. A
blank-response failure warrants calibration/transfer review, not automatic
advancement. All fold and seed results remain visible regardless of decision.

## Execution and preservation

Qualify asymmetric chunk assembly, final-plane use, mask semantics, training-only
normalization, exact fold IDs, grouped metric ties, synthetic positive control,
plateau decisions, held-out evaluation lock, corrupted/partial checkpoint rejection
and exact interruption/resume. No real-study optimization before sealing GO.
Runtime benchmark is synthetic and reports overhead exclusions explicitly.

Each256-update checkpoint includes model, Adam, NumPy sampler/augmentation RNG,
PyTorch RNG, LR/plateau/history/exposure state and GO/fold/normalization binding.
Write/fsync/rename data, then publish its checksum receipt last. Unreceipted
files are ignored; mismatches halt. A worker lock prevents concurrent writers.
Private Git LFS backup generations include frozen code/GO, baseline artifacts,
current and best checkpoints and completed endpoints. Publish approximately
every10 minutes, with explicit final backup when the worker exits. Model input
archives already reside in private LFS and are separately pinned; do not duplicate
them each generation. Test restore and checksum readback before reporting backup
as usable. Runtime restarts are not automatic; resume using the documented
command after restoring files and the qualified runtime. A status message does
not prove that a background process survived an environment restart.

No paid GPU, Runpod resource, GitHub Actions job, public publication or external
VM assignment belongs to this study. Existing £10 total GPU pilot conditions
remain unchanged. A favorable result is not automatic GPU launch authorization.
