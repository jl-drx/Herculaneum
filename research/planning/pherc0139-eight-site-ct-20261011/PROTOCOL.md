# Eight fixed PHerc0139 CT correspondence sites

11 October 2026. Frozen before new CT pixels or geometry inversion results.
This extends the earlier three-site w040 structural test to the eight unchanged
survivors of the fine-geometry pilot. The four preceding geometry failures
remain failures, without replacements. No targets, annotations, teacher maps,
neural inference, new training or GPU resources. The native CPU study continues
unchanged with its held-out predictions locked.

## Question and finite acquisition

Do these geometry-selected 128-square sites on w040/w041/w043 show consistent
local CT correspondence, and how often do they satisfy the earlier strict
structural-transfer screens? This is a development diagnostic on one specimen,
not eight independent specimens or an estimate of ink accuracy. Selection on
geometry limits generalization to the admitted geometry stratum. These smaller
supports differ from the earlier 256-square experiment; do not pool its pass
rates or call this a power-matched replication.

Use the unchanged NEXT_CT_PILOT_PLAN.json: 72 pinned objects, exactly 86,114,304
CT bytes (82.125 MiB), all 28 native and 109 fine depth planes. Conditional GET
with If-Match, exact length/ETag verification and retained SHA256 receipts.
No automatic retries, replacement crops, request expansion or raw-scan fallback.
Stop on source identity or length failure. Freeze code, geometry query hashes,
support checks and this protocol before GET. Read only public CT bodies in this
step; existing source metadata and coordinates are already retained.

## Geometry, representation and support

Reuse the previously checked representative-field inverse: average all 16
fine group0 renderer positions/normals represented by each group2 pixel,
transform to native XYZ, bilinearly interpolate the fields without normal
renormalization, and solve for native renderer locations. Use the generalized
source-derived mapping for each winding; pooled-pixel centers are at 4u+1.5.
Native center depth is 13.5, fine center depth 54; nominal spacing is
9.362/2.399. Retain native depths 4:24 on the core plus an 8-pixel XY halo.
The existing bounded Newton solver keeps its 0.01 native-voxel residual,
10-update, conditioning, orientation, +/-2 pooled-XY / +/-8 fine-depth basin
and step limits. Geometry is never fitted to CT values. Query failures remain
invalid and are recorded, including failures outside scoring supports.

Fine group2 has 9.596 micrometre XY and 2.399 micrometre depth sampling.
Apply the earlier fixed depth Gaussian sigma 1.5, radius 5, with no padded
values contributing to queries. This is a structural comparison surrogate,
not a measured scanner PSF or separation of resolution from acquisition energy.
Use trilinear sampling, with complete geometry, full stencil bounds and finite
values. Require positive native CT values in all scored templates and positive
fine group2 source values in every interpolation/Gaussian contributor used by
any scored translation. This conservative support test may exclude legitimate
zero-valued material; failure means unqualified support, not absent ink.
Record full-depth zero counts separately, without selecting on intensities.
The masked source exports do not provide a constituent mask in this finite
packet: positive pooled pixels cannot certify every underlying fine voxel was
unmasked. Geometry support and pooled nonzero support are separate claims.

## Fit and separate checking

Use four 24-square fitting supports centered at core XY (24,24), (104,24),
(24,104), (104,104), and four checking supports at (64,24), (24,64),
(104,64), (64,104). All use native depths 10:18. The 24-square width leaves
separation on the smaller core; it is fixed before the new data. Verify that
the complete fine group2 interpolation footprints over all translations are
disjoint between every pair of supports before CT acquisition. These are
separate sampled locations, not statistically independent texture samples.
Source scanner PSFs and nearby spatial dependence remain unmeasured.
If the footprint or geometry check fails, retain the site as unscorable; do
not reduce patch size or move it. Require the scoring query footprints to lie
within the already frozen fine CT request envelope, including depth smoothing.

Search the same 2,197 integer shifts: each of Z,Y,X from -6 to +6 native
sample steps. Positive shift compares native[z,y,x] with remapped
fine[z+dz,y+dy,x+dx]. Baseline is zero CT translation after source geometry.
Fit a single surface-coordinate shift per site by mean separately normalized
Pearson correlation on the four fitting patches. No affine/nonrigid fitting,
per-patch corrections or subvoxel refinement. Independently optimal checking
shifts are diagnostics and never applied to choose a correction.

Keep the earlier thresholds unchanged: non-boundary fit and checking optima;
all four checking correlations at the fit >=0.5; each exceeds the best
competitor farther than 2 native voxels by >=0.05; checking-versus-fit
tangential discrepancy median <=0.5 and maximum <=1 voxel; normal discrepancy
<=1 voxel. Use the native geometry basis at each support center, and a 1e-6
voxel arithmetic tolerance at distance boundaries. Preserve each criterion,
all landscapes, normal competitors and shifts in physical units.

## Verification, views and decisions

Before acquisition, test known shifts under intensity scale/offset changes,
boundary peaks, periodic depth ambiguity, constant/nonfinite rejection, an
analytic representative-field inverse and source-footprint separation.
Independently check fixed real geometry queries directly from coordinate
stencils, and independently recompute every CT score by spatial covariance.
Use one numerical CPU thread. Preserve corrections and their timing relative
to acquisition; never modify a frozen implementation silently.

Inspect every site with the same views: native and remapped fine mean of
layers 10:18 at zero and fitted CT shifts; fixed signed differences; native
full28 and fine full109 center-row/center-column depth sections. Intensities
use 0..255 and signed differences -64..64, without fitted contrast. Invalid
support is displayed explicitly. A correlation pass is only local structural
consistency at checked patches. Traceable lamina and nonrepeating landmarks
must be considered before a same-sheet claim; inability to identify them
leaves sheet identity unknown. No threshold is tuned after seeing images.

Report the number passing among all eight and retain the preceding four
geometry failures. One pass would justify a separately frozen, local target
provenance/sensitivity audit on that support; it does not admit all 102
candidates. If CT structure corresponds but strict localization fails, retain
the uncertainty and compare it with the earlier w040 finding. If source support
or correspondence fails, stop transfer at those sites. Do not add another
geometry/depth search merely to manufacture a passing region. The broader
decision combines this result with the unchanged native-learning comparison;
no GPU qualification or pointwise supervision is granted by this test alone.
