# Cached reference-sensitivity diagnostic

This diagnostic is frozen after observing phase32's negative averaging result and before computing the diagnostics below. It is post hoc development analysis of the same four PHerc51 contexts. Preserve every phase32 primary score and its decision; this is neither new confirmation nor a reference correction.

## Inputs and scope

Use the exact phase31 128×128 conservative labels and central64×64 cores, phase32 clean baseline and candidate probability maps, and the retained optical PNG. No acquisition, model call, training, threshold fitting, checkpoint/depth/crop selection or paid resource. All four contexts remain included, even if a stratum lacks a class.

Add one fixed descriptive optical-darkness score: for the same 384×384 source crop, use the red channel after verifying equal RGB channels, sum each3×3block inuint32, then compute `1 - sum / float64(9*255)`. Do not quantize, fit a threshold, invert after seeing results or optimize a warp. Require opaque support wherever any diagnostic labels are scored. This proxy can assess agreement between the supplied optical image and supplied annotations; they may share provenance, so it is not independent truth or a competitive CT model.

## Label translations

Evaluate every(dy,dx) in{-1,0,+1}², in row-major order. Keep probabilities fixed at the original64×64 coordinates. Use `L_shift[y,x]=L128[32+y-dy,32+x-dx]`; positive dy/dx moves label content down/right. The larger label canvas supplies all pixels: no padding, wrapping, interpolation or boundary guessing. A unit is one pooled raster pixel, equal to three original raster pixels; do not claim a measured physical registration error.

For the main shift table, use the identical intersection of known-label support over all nine shifts and all predictors. Report each shift's class counts, prevalence, AP and AUROC for baseline, candidate and optical proxy. Report the common-mask zero-shift result separately from the original-support zero-shift result: dropping pixels can change metrics even without moving labels.

Explicitly count labels that change relative to zero shift on common support. Report class-stable support where every shift agrees, and score it once at zero shift. If common support removes all shifting boundaries, an unchanged score is uninformative about registration robustness; do not call it proof of alignment.

Also retain a secondary table using each shift's own known support. Report pixels added/removed versus the original support and class/prevalence changes. These scores combine label displacement and support changes. They are coverage-confounded diagnostics and cannot identify a true warp. Report all nine variants and ranges, never select an optimal shift or replace labels. No hypothesis test, confidence interval or model promotion follows.

## Fixed boundary-distance bands

At zero shift, calculate Euclidean distance in pooled pixel-center coordinates to the nearest different class OR unknown pixel. Pad the outside of the larger128×128label canvas as unknown; scoring remains on its central64×64 core. The fixed bands are `0<distance<=2`, `2<distance<=4`, and `distance>4` on known labels. Report all three predictors, class counts and prevalence in every band. Unknowns remain excluded. Require both classes for AP/AUROC; retain explicit undefined results rather than substitute a region or change a distance.

Distances describe this conservative label mask, including its unknown boundary strips, not an independently measured ink edge. Different bands have different supports and prevalences; their APs do not form a controlled model comparison. Exact band memberships will be cross-checked with direct lattice-neighborhood comparisons, separately from the distance-transform implementation.

## Verification and interpretation

Verify zero-shift original-support CT metrics against the sealed phase32 records. Independently recompute label translations by scalar indexing, compare band membership without a distance transform, check optical pooling with scalar3×3sums, and reconcile defined AP/AUROC through the retained separate phase32 ranking formulation. Counts and unknown-class outcomes must also agree. Inspect all four optical/reference/prediction panels, plus all-shift tables. No model forwards are added.

Freeze sources/executable hashes and a 120-second diagnostic limit,1GiBRSS,64MiBnew phase disk excluding the cumulative retention manifest. Verification has its own120-second/1GiBRSS ceiling and no network. Plotting uses fresh writable caches. No automatic rerun after a failed diagnostic; retain any failure and diagnose first. GPU spend£0 and £10total first-pilot cap unchanged. No outreach, R2 upload, publication, translation or submission.

Interpret only this bounded family. If weak ranking persists on both-class interiors and throughout the small-shift range, a one-pixel raster perturbation is insufficient to explain it under this reference. That does not exclude larger/nonrigid misregistration, incomplete annotations or renderer/input issues. If common support makes the test uninformative, state that. Any future learned method or genuine registration correction needs a separately designed experiment and evidence; do not fit to a favored diagnostic outcome here.
