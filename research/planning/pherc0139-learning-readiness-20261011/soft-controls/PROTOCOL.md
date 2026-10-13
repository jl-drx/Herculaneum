# Matched soft-target operators: finite pre-learning control

Frozen 11 October 2026 before computing these contrasts. Parent repository
cfb5775d4dfc38014b731201ee32712d6a60c2c0. Existing target values and previous
sensitivity results have already been inspected. This is a development control,
not a prospective held-out learning result or another numbered model experiment.

## Question and scope

Does local CT-candidate averaging make meaningfully different soft supervision
from a fixed pooled displacement kernel, and how much of its change is explained
by moving the sampling location to the candidates' mean projected coordinate?
This finite comparison can remove an unnecessary intervention; it cannot show
improved prediction or select a learning winner. No training, new teacher pixels,
CT downloads, model inference, GPU, Actions dispatch or external VM is required.

Reuse only the checksummed previous correspondence/supervision packets. Never
decode beyond the retained teacher/crop.npy. Its fine origin is [9084,7088] and
shape [780,788]. The four 32-square supports remain 4,5,6,7, including the
teacher-negative support. The local candidate counts remain 11,6,22,4; no target
based candidate selection, weighting or registration refitting is permitted.
Primary plane j=14 and operator mean16_bilinear; mandatory secondary controls
are all j=10..17 and point_bilinear, reported separately. Keep all native CT
depth planes in any future learner, independently of this 2D target convention.

## Four operators on identical support

All target scores use raw float64 teacher/255, with no fitted calibration,
rounding, Dice, positive weighting, or transformation of threshold48 to 0.5.

A: ordinary soft target at the fixed CT fit ZYX=(0,-1,0).

B: uniform mean of transferred targets over that support's frozen local CT
candidate set. These are descriptive alternatives, not posterior probabilities.

C: fixed global smoothing control. Apply every support's candidate kernel at
every support and average the four kernel means equally (1/4 per support and
1/n within each kernel), preserving absolute ZYX shifts relative to the common
fit. Do not pool all43 candidates with equal weight: that overweights support6.
The global control deliberately breaks local association while preserving the
aggregate distribution. It is geometry-conditioned resampling, not a claim of
a stationary Gaussian filter in the native image or a CT-independent prior.

D: sample the teacher once at the unweighted mean fine-group2 [u,v] coordinate
of the local candidate queries, using the same resampling operator as A/B/C.
This is a centroid control, not an estimated physical surface or prescription
to recenter CT depth. B minus D isolates the nonlinear effect of averaging
scores rather than taking the score at their mean coordinate; it is conditional
on this teacher's smoothness and this coordinate representation.

For every support/plane/operator use ONE common mask: native max(all28 CT)>0,
valid geometry and finite in-bounds teacher samples for A, every unique global
candidate and D, and teacher>0 for every such sample, preserving the previous
supervision convention. Also report geometry-only support separately. No zero
fill, reflection, boundary extrapolation, per-arm support or silent removal of
the fourth support. Out-of-crop requests are excluded from all four arms and
counted. No fetch or expansion is allowed to repair a failed support check.

## Outputs and interpretation

Save A/B/C/D targets, the unique candidate transfers/weights and all masks.
For A-B, A-C, B-C and B-D report MAE, RMSE, signed difference and p95/max absolute
difference on common support and on the common positive union (any fit, global
candidate or D >=48 on the original uint8 scale). Empty sets give null metrics.
Report each arm's target mean/entropy, threshold48 positive counts, and threshold
disagreements only as diagnostics. None is AP, calibrated probability or ink
accuracy. Report exact displacement means/covariances of every kernel and C to
expose recentering as a possible cause of difference. Keep each of the64 cases;
do not pool them as independent replicates or pick the most favorable depth.

For primary j14/mean16, report whether B-C and B-D active-union MAE are each
below 1/255 (one source quantization interval); report the exact values and
secondary cases. This is a transparent descriptive scale chosen for this
already-inspected packet, not a powered learning gate, noise floor or proof
that a smaller target change cannot affect training. No automatic training/GPU
admission follows either outcome. No extra supports/depths/threshold search.

## Verification and stopping

Before real analysis: test constant and asymmetric ramp fields; shift sign;
nonzero crop origin; equal-kernel versus equal-candidate weighting; missing
geometry/source/native support; empty positives; and the exact masked BCE
identity mean BCE(p,q_i)=BCE(p,mean q_i) at fixed p. Tests must detect a wrongly
weighted C and a zero-filled invalid sample. On real arrays independently
recompute all summaries and at least primary transfers from retained source,
using a separate implementation and retaining floating-point discrepancies.
Stop after64 cases and the separate reference/data-readiness audit. If a
defensible physical evaluation split is absent, report it explicitly rather
than training four tiny patches or treating teacher imitation as ink benefit.
