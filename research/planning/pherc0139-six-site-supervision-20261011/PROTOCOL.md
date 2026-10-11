# Six-site transferred-supervision sensitivity

Frozen before reading these target crops. This extends the earlier single-site
diagnostic; it is not a training experiment or a fresh accuracy benchmark.

## Question and fixed population

Do the saved CT alignment alternatives materially change teacher supervision,
and is averaging those alternatives distinguishable from ordinary local
smoothing? Include all six scorable sites of the eight-site CT packet, all four
24×24 checking patches per site, and all native planes 10–17. Plane14 is primary
because of the released label-container convention, not a verified ink face.
Keep the two source-support failures in the accounting; do not replace them.
The previous strict structural-registration failures remain failures.

For each patch, retain every integer shift in the saved ±6 Z/Y/X search whose
CT NCC is at least that patch's NCC at the site's fitted shift minus0.05. Include
the fitted shift exactly once. No weights, teacher-driven pruning, threshold
adjustment, new correspondence fit, or new region selection. These are finite
CT-supported alternatives, not probabilities or confidence sets.

## Transfer and controls

For native pixel (j,y,x), use the saved inverse query at
(j+dz,y+dy,x+dx), preserving the sign of the original CT experiment. For each
alternative use the same j; never average different depths into one target.
Primary transfer: unrounded float64 mean of each aligned 4×4 full-fine teacher
block, followed by bilinear interpolation at saved group2 (u,v). Mandatory
operator control: bilinear original teacher at (4u+1.5,4v+1.5), without pooling.
This is a conditional 2D projection, not exact integration over a native voxel
or reproduction of the released uint8 INTER_AREA resizing pipeline.

Compare the fitted shift with every alternative and with zero residual shift.
Also compare three soft targets, with equal weights over the unique alternatives:
(a) mean of sampled alternative targets; (b) target sampled at mean query UV;
(c) a matched stationary smoothing control. For (c), let U_i(p) be an alternative
query, M(p) its mean over alternatives, and K_i the spatial mean over all 576
patch pixels of U_i(p)-M(p). Sample at M(p)+K_i and average. This preserves the
alternative count, weights, mean coordinate and spatially averaged displacement
kernel, while removing its spatial variation. It does not match every pixel's
covariance. Target pixels do not determine this kernel. Report errors of (a)
versus (b) and (c); no learning benefit or novelty follows from averaging alone.

## Full input footprint before target access

Include every interpolation contributor of both operators, all alternatives,
fit, zero, mean-query and stationary controls, at every plane. Conservatively
enclose their source pixels per site, enumerate every overlapping v42
256×256/stride128 tile including the special border tile, and qualify the entire
rectangle enclosing that tile union at fine planes [23,85). Recheck renderer
coordinate/normal stencils, source-volume extent, all332 reserved native
contexts, and computable conditional w024 fine-reference envelopes. This
rectangle may reject extra space; a pass clears every included teacher tile.
Any failure blocks target acquisition for that site, remains in the record, and
does not cause a replacement. This footprint screen does not establish correct
registration or independent teacher training ancestry.

## Provenance and bounded acquisition

Use only the catalog-declared 2.399µm/78keV canonical v42 teacher for each winding.
Require its exact filename to match the author report pinned at commit
217e7521670db49cde97872ea6a8bdfb80c449f1, its declared input raster to match the
fine geometry, and its uint8 TIFF canvas to match that raster. Teacher source is
v42 d4615b134ce1b62368bed1029ed47e5569ca03fb; source provenance is not a historical
binary/weight checksum. Thresholds from that report: w040≥48, w041≥46, w043≥47.
No manual overrides or native annotation pixels will be fetched: this measures
teacher-only target coverage and sensitivity, not final historical training labels.
Teacher ancestry is not established to be independent of reserved annotations.

Read only HEAD and precise TIFF structural ranges before freezing the image
request plan. Require HTTP206, exact Content-Range, unchanged ETag, bounded body
reads and successful bounded deflate decoding. Abort on format or identity
change. The finite image request plan lists all TIFF tiles and is committed
privately before image GETs. Cap: 32MiB compressed image bodies total, 256 tiles;
header-body cap: 256KiB total. Never fetch a whole TIFF as fallback. Retain hashes
for headers, compressed tiles, decoded bytes and every crop. Replay offline.

## Measures and decision rules

Native eligibility is max over the saved full28 native CT planes >0 at unshifted
output pixels. Geometric support, teacher>0 supervision convention and ink
evidence remain separate. Report complete source support and teacher nonzero
union/common counts. On common nonzero support across all alternatives report
threshold-positive union/intersection, variable positive fraction, common
negatives, per-candidate gained/lost positives and MAE to fitted target, plus
mean/p95 score range. Report continuous errors both on all common supervision
and on the finite-set positive union, so blanks cannot conceal changes.

Carry forward the earlier operational flags: teacher support variation >1%,
variable positive fraction >5%, or worst candidate MAE on the finite-set positive
union >0.05 normalized (12.75/255) is material sensitivity. Fewer than100 positive
union or100 common negative pixels makes class evidence insufficient; still
report numeric sensitivity. Empty denominators are null, never evidence of
stability. Aggregate counts descriptively, not as independent pixel replicates.
Report all24 primary patch results and all192 depth/operator cases per operator.

If well-supported cases repeatedly change hard targets while continuous scores
remain much steadier, prefer soft-target and matched-smoothing comparisons in
any future learning design; do not admit hard aligned truth. If support or class
evidence is mostly insufficient, explicitly limit conclusions and do not search
for brighter replacement crops. If uncertainty averaging matches stationary
smoothing, make that the required baseline, not a proposed novel contribution.
Neither outcome alone admits paired training or proves the ink-bearing depth.

## Verification and resource boundary

Test nonzero origins, asymmetric shifts, both interpolation operators, thresholds,
empty denominators and stationary kernels synthetically before acquisition.
Independently recompute transfers with alternate arithmetic and metrics from
saved source pixels; disclose threshold ties instead of hiding rounding effects.
Review fixed-scale panels for all24 patches. Package a self-contained private
replay with checksums. One low-priority CPU process, no new CT, inference,
training, GPU, Actions dispatch, external messages or spare-VM delegation.
Keep the active nine-fit native-learning study unchanged and held-out predictions
sealed until all endpoints are complete.
