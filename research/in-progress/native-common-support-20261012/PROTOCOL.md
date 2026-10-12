# Fixed common-support native baseline pilot

12 October 2026. Approved research continuation; no paid compute is authorized.
This protocol precedes opening these segments' annotation array payloads or CT
for this comparison. The completed native0139 experiment is not amended.

## Cohort and claim

Use all four published Reader human-reference segments: PHerc0841
20260220213127, 20260220214732, 20260221022814, and PHerc0009B 20250919125754.
The metadata intake fixes their exact native volumes and fine-canvas labels.
These are exposed, transferred/model-reviewed development references; upstream
training/selection exposure is not ruled out. Agreement is not physical ink
accuracy, authenticated text, or independent generalisation. PHerc0175 A/B and
PHerc1203 remain reserved and unopened. No new source-offset correction is used.

## Selection before ink or CT

Decode supervision (S) and validation (V) separately at label level 0. Validate
the complete sharded objects, CRC32C indexes, codec headers and decoded bytes
against installed Zarr. Record histograms, overlap and hashes. Canonical fill
means no participation, not a negative label or geometric validity.

Match the pinned Reader evaluator: binarize each mask at >0, multiply by 255,
resize the entire fine canvas with OpenCV INTER_AREA, and threshold >=128.
For 0841, union the separately resized S and V; for 0009B, union on the fine
canvas before resizing. Preserve both alternatives and quantify their difference
without selecting between them by detector performance. This corrects the prior
intake's overly general statement about union ordering.

Enumerate complete 128x128 native cores with origins that are multiples of 128.
Require >=4096 participating native pixels and complete source context for all
contributing model tiles, including Hecate's interpolation footprints. Select six
cores per segment: first nearest the arithmetic centroid of eligible core
origins; thereafter maximize squared distance to the nearest already selected
origin. Every tie breaks by (y,x) ascending. Scores, ink values, CT intensities,
teacher maps and native0139 outcomes do not enter selection. If fewer than six
eligible cores exist, report a blocked segment; do not replace it or lower the
threshold. Cores cannot overlap because of the fixed grid; contexts may overlap.

Freeze selected IDs, tile lists and the exact native chunk union before ink
payload or CT acquisition. Six per segment is an implementation/development
pilot, not a powered final evaluation. Archive all eligible counts, selected
support and a coverage image showing the deterministic selection.

## Sampling and inference contract

Retain all 28/31 native planes. Use only the unpadded full-render origin, forward
depth order, no mirror TTA, no depth/normal shift, no origin search. All models
score the same native core pixels: published participation AND native centre
plane >0. This last mask reproduces the evaluator's array support; it does not
verify physical registration. Retain every selected core even if native support
or the subsequently opened labels make a metric undefined.

Reader v2, dense-native, seed42 and seed43 use their pinned 17x128x128 hybrid
models with robust MAD preprocessing and stride 64/Hann accumulation. The
released upper-centre convention selects planes 6..22 for depth 28, 7..23 for
depth 31. Hecate uses the pinned 9.6um model, 16x64x64, divide by 255, stride 32,
its own Hann/support accumulation. Use the already qualified PHerc343 preparation
algorithm on global canvas coordinates, with the same nominal centre (14/15).
Its source z positions are centre+(i-7.5)*9.6/native_spacing. XY output size is
round(native_size*native_spacing/9.6); use INTER_LINEAR at 9.366um (ratio >=.95),
INTER_AREA at 8.64um (ratio <.95), exactly as that algorithm specifies. These
different physical windows are declared model recipes, not an architecture-only
ablation. The choice does not certify the true ink surface or reading side.

The complete full-render grid determines which tiles contribute, including the
final boundary-aligned tile. No local tensor resize/reset is allowed. Unknown
unacquired CT cannot enter a contributing tile. Check source support with a
separate known mask, not nonzero CT. Hecate input resampling and output linear
resize are both included in the source footprint. Per-model native-grid output
is quantized as in the released pipelines (hybrid truncation; Hecate rounding on
its own grid before float resize). Retain unquantized outputs where available;
do not fit calibration on this panel.

Initial CPU implementation: fp32, batch 1, four intra-op threads, one inter-op
thread, no compilation. Pin exact runtime/checkpoints before real inference.
Preserve tile-level logits with input hashes and immutable completion receipts;
restart only from verified logits. Validate sparse/full-grid sampling without
real model inference first. This intake alone does not claim inference parity.

## Evidence and decisions

Report per-core, per-segment and equal-parent AP/AUC, prevalence, calibration and
annotation-negative response. Treat the two seed checkpoints as a control for
released-checkpoint variability, not a training-seed confidence interval. Show
all models and parent disagreements. No model can be declared a general winner
from 24 cores on two exposed parents. An opposite ranking between parents or a
gain smaller than checkpoint variation blocks that ranking claim; it does not
close adequately powered broader-coverage training. A useful pilot justifies
expansion over this predeclared development suite and a matched-coverage GPU
proposal, followed by fresh physical-parent confirmation.

Stop on metadata/source identity mismatch, incomplete context, decoder or
sampling disagreement, nonfinite output, or failed logit replay. Do not silently
repair thresholds, select a different core, or retry using a different model
recipe. Keep failed attempts and explain any implementation-only correction.

Acquisition is public read-only HTTPS with explicit per-object/cumulative caps,
durable reservation before body reads and SHA256 receipts. Annotation objects
use If-Match from HEAD; raw uncompressed native chunks have exact metadata-derived
lengths and first-read ETag/SHA receipts. No whole scans or fine CT are downloaded.
Private GitHub/LFS is the portable backup. New GPU work requires a concrete cost,
pinned environment, delivery/shutdown plan and user approval within the standing
£10 all-in first-pilot limit.
