# Site 1: finite sensitivity of transferred teacher scores

11 October 2026. Frozen before acquiring or decoding target pixels. This is
a supporting study on previously inspected training tissue, not a test set.
It changes no previous correspondence gate, experiment, or model.

## Question and permitted conclusion

How much does a declared transfer of the released 2D fine-scan teacher change
over the existing CT-supported alternatives? Measure the change before deciding
whether correspondence uncertainty deserves a learning intervention. Scores are
model pseudo-labels, not authenticated ink or calibrated probabilities.

The entrance check establishes the **declared full fine-raster coordinate
contract** using the exact catalog asset, TIFF header and pinned v42 inference
implementation. Historical teacher-byte identity and all teacher-weight ancestry
are not available. Retain current-object validators and hashes of every acquired
range. Do not claim a complete-file hash from partial acquisition. The current
manual release differs in pixel count from the historical native-dense report;
therefore this test is teacher-only, before manual overrides, not a replay of
the actual historical merged training labels. No manual pixels will be fetched.

The teacher aggregates fine layers [23,85) into a 2D raster. It has no identified
ink-depth plane. Sampling it at a geometry query assumes its score is constant
along a fine normal ray. This can measure **conditional projected-XY sensitivity**;
it cannot resolve the effect of a wrong ink depth, face or neighboring wrap.
In particular, five candidates in support 6 have dz=1. Preserve these candidates
and flag their physical normal uncertainty even if the projected score is stable.

## Fixed supports and alternatives

Use only the four original 32-square checking supports at native centres
(2432,1840), (2352,1920), (2512,1920), (2432,2000). Native core origin is
(2304,1792), retained halo origin (2296,1784). Use all candidates in
CT_CANDIDATES.json: 11,6,22,4, with common fit ZYX=(0,-1,0).
The retained rule NCC >= NCC_at_fit-0.05 was already chosen from CT alone.
These are finite descriptive sets, not confidence regions or probabilities.
No weighting, pruning, new shift fitting, replacement support, or target-based
selection is allowed. Four neighboring supports are not independent replicates.

Use the checksummed geometry inverse queries [u,v,k] from the prior study.
Positive shift means query at native [j+dz,y+dy,x+dx]. Compare each alternative
against the common fit at the SAME j. Primary j=14 follows the documented native
annotation-layout index, not a claim that plane 14 is the surface (it is +0.5
voxel). Repeat at every j=10,...,17, the eight planes used for the CT comparison;
report each and the worst separately. Do not average different j values into a
target or regard their agreement as depth localization.

## Two fixed resampling operators

Primary: unrounded float64 mean of each aligned 4x4 fine teacher block, followed
by bilinear interpolation at the retained fine group2 (u,v) query. This fixed
anti-aliasing approximation is not exact integration over a curved native pixel.
It does not reproduce the twice-rounded CT pyramid or released uint8 INTER_AREA.

Mandatory control: bilinear point sampling of the original full-resolution
teacher at (4u+1.5,4v+1.5), without pre-pooling. Preserve both operators and do
not select the one with smaller changes. Stability conditional on a smoothed
operator alone cannot rule out resampling sensitivity.

For each operator retain three SEPARATE contrasts: naive full-raster centre
mapping versus geometry-zero; geometry-zero versus the CT fit; CT fit versus
every retained alternative. The naive centre equation is
x_fine=(x_native+0.5)*31040/7980-0.5, similarly Y with 24880/6400.
This is a same-operator geometry control, NOT an exact released INTER_AREA replay.
The first contrast includes the source-frame correction and inverse geometry;
neither belongs in the residual-CT-uncertainty metric.

## Support, scores and finite routing rules

Threshold >=48 on the 0..255 scale, fixed from the pinned **w040 own-calibration**
report. The default for other windings is 47; do not substitute it. Float scores
are not rounded before thresholding in this diagnostic. Native eligible support
uses retained max_z(native CT)>0; missing source pixels never become zeros.
Report geometric/array support separately from the recipe's teacher>0 convention.
Teacher>0 is a supervision convention, not proof of valid material or ink absence.

For each support, depth and operator preserve per-candidate masks/counts and
scores. On the intersection of candidate/fit supervision support report MAE
against fit, range mean/p95, threshold-positive union/intersection, changed-label
counts and variable-positive/positive-union fraction. Also report common/union
support, support-change fraction, and each candidate's gained/lost positives.
Repeat continuous metrics within the positive union so background cannot conceal
changes. Empty denominators are null, never a perfect score. Keep all four
supports; pooled means do not clear a failing one. No AP/AUC, pixel-independent
confidence interval, letter identification, or accuracy claim.

Predeclared engineering routing flags (not downstream-performance thresholds):
support-change >1%, variable-positive/positive-union >5%, or worst candidate
positive-union MAE >0.05 normalized (12.75/255-scale units) means operationally
sensitive. Fewer than 100 positive-union OR 100 negative-common pixels means
insufficient class support for a stability conclusion. Preserve numerical
sensitivity even when class support is insufficient. Otherwise the numerical
result is only 'within declared conditional tolerance'. Physical depth uncertainty
and provenance limits remain separately unresolved under every outcome.

Sensitive results motivate a separately designed uncertainty intervention with
matched support, smoothing and permuted-uncertainty controls. Stable lateral
results would reduce that priority on these supports only. Neither outcome admits
training, all w040, a new benchmark, or GPU use; all original CT failures remain.

## Acquisition, verification and stopping

Plan exact fine-pixel footprints and necessary TIFF tiles from geometry before
target access. Maximum 8 MiB compressed body bytes, no full teacher download,
new CT, teacher inference, training, GPU, Actions, PR, or external VM. Require
206 Content-Range, exact response length and unchanged ETag on each range; reject
silent full downloads. Save compressed bytes and decoded hashes, plus a checksum
manifest. Decode only complete known uint8 Deflate tiles with supported TIFF
semantics. Stop if declared coordinate, byte, geometry or source support checks
fail; do not reinterpret a failure as a negative target.

Before target access test shift sign/nonzero crop origin, bilinear and mean
operators with asymmetric synthetic patterns, threshold behavior, missing support
and empty positives. Independently recompute real candidate metrics after the
fixed analysis. Save all arrays, per-candidate metrics and fixed-display panels.
Stop after the four supports. A disappointing result does not authorize crop,
depth, threshold or operator search.
