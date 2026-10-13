# PHerc0139 w040: paired-acquisition correspondence check

11 October 2026. **There is useful, directly observed correspondence between
these fine and real coarse CT observations. Exact pointwise supervision is not
yet qualified.** A source-derived rendering correction resolved a substantial
coordinate pitfall; the fixed CT test then found strong agreement, with residual
local uncertainty that differs by location. This supports continuing the paired
acquisition direction instead of abandoning PHerc0139 for lack of correspondence.

No ink targets or model outputs were used in this study. No new inference,
training, GPU resource, GitHub Actions run, or public publication. The active
PHerc0800 study 52 was preserved. These are inspected development locations,
not an untouched benchmark. This supporting study does not advance the formal
cumulative experiment seal beyond study 50/register v46.

## What changed scientifically

The fine78 and native113 surface volumes share the same normalized surface,
but their mesh-transform pipelines resample that surface onto different grids.
Composing the exact mesh and renderer conventions introduces approximately
**7.43 native pixels per axis** relative to ordinary full-canvas resizing.
Pinned source code predicts it; an independent 25-site coordinate check confirms
that the corrected mapping reduces median XYZ discrepancy from 10.506 to 0.057
native voxels (24 valid sites, one retained invalid site). Remaining normal-ray
and curvature differences reach roughly 1 voxel, so a scalar resize/depth ratio
is insufficient. This does **not** claim that an author's separately aligned
training data or released model is defective, nor that7.43 is a CT-estimated
acquisition error. It is a rendering-coordinate correction for these two assets.

We constructed a geometry-only inverse that matches native renderer positions
to representative positions of the fine pooled pixels. All 4,439,040 queries passed
the frozen support, conditioning, orientation, bounded-basin and 0.01 voxel
numerical residual checks. A separate scalar implementation verified 81 fixed
queries directly from the coordinate TIFFs. Matching representative positions
is not equivalent to matching the scanners' point-spread functions.

## Fixed CT comparison

The three 256-square cores are centred at native XY(2432,1920),(3968,3200),
(5632,4480) on training winding `20250831000000-w040_2025083102`.
Their positions were fixed before pixels were read. We acquired exactly 75
uncompressed surface chunks, 70,238,208 bytes (66.98MiB), within the 192 MiB cap.
Fine 109/native 28 source layers are retained; fitting uses the central 8 native
layers with a ±6-layer search and complete context. Fine XYpooling and a fixed
ZGaussian are structural approximations, not a calibrated scanner simulation.

Each site fits one integer shift using four 32-square supports and checks it on
four separate supports. Positive ZYX means native[z,y,x] is compared to the
geometry-remapped fine[z+dz,y+dy,x+dx]. The translation is in curved surface
coordinates; per-support local bases convert discrepancies to physical units.

| Site | Fitted Z,Y,X shift | Mean fit NCC, zero→fitted | Four separate checks at the fit | Frozen decision |
|---|---|---|---|---|
| 1 | (0,−1,0) | 0.933→0.951 | 0.949,0.868,0.938,0.926 | Fail uniqueness-margin screen only |
| 2 | (+2,+1,+1) | 0.529→0.945 | 0.846,0.960,0.900,0.866 | Fail margins and lateral consistency |
| 3 | (0,+2,+2) | 0.669→0.842 | 0.677,0.834,0.794,0.854 | Fail margins and lateral consistency |

All 12 checking supports exceed the prospective 0.5 NCC threshold. None of the
three entire cores passes every frozen screen; **none is admitted as precise
pixelwise teacher-transfer support**. The thresholds were not relaxed after
seeing these results. Checking optima are diagnostics and were not applied as
extra local registrations.

Site 1 is particularly informative. Three checking optima equal the fitted
shift; the fourth differs by 0.986 native voxel tangentially and 0.014 normally.
Two checks miss the 0.05 distinct-competitor margin, at 0.035 and 0.031. Those
competitors are primarily lateral shifts. Competitors more than 2 voxels away
in the local normal direction are worse by 0.245–0.600 NCC. An independent
post-result description found connected near-fit score ridges, not evidence
of a distinct distant alternative. That description does not replace the
frozen gate or provide calibrated uncertainty probabilities.

Sites 2–3 demonstrate why a one shared surface-coordinate translation cannot simply be
assumed. Site 2's checking preferences differ laterally by about 2–3 voxels;
site 3 additionally has approximately 1 voxel normal variation. These are
correlation-peak discrepancies, not independently known registration errors.

## Interpretation and next decision

All six fixed panels were inspected by the principal researcher and an
independent agent. The same local material structures, channels and void
boundaries are recognizable across acquisitions. Site 2's depth correction
visibly restores the correspondence. Full-depth context also shows fine
internal structures blended in the coarse scan. Those bands are not by
themselves evidence of different wraps. We have not certified an individual
ink-bearing face, an ink label, or absence/presence of writing.

The next useful experiment is **a bounded uncertainty-impact audit on the
unchanged site 1 supports**, with the current CT mapping and score landscapes
frozen before looking at any target. First verify the released target's own
coordinate convention and provenance. Compare the single CT-fit transfer
with transfers over the plausible local shifts retained by the CT evidence;
quantify how much supervision changes. Treat the source honestly as a model
pseudo-label if that is what it is. Do not tune registration against the target.

If the target is stable across the CT-supported alternatives, a simple matched
real-native versus degraded-fine training comparison becomes defensible on
that checked support once target coordinates, shared physical support and
supervision provenance are confirmed under a separately declared supervision
tolerance. Target stability would not retroactively turn the current failures
into passes. If the target changes materially, the research question
is explicitly registration-aware supervision, with matched smoothing/support
controls. Averaging ordinary BCE over uncertain transforms is equivalent to
BCE against the average transformed target; that alone is not a new method.
If target coordinates cannot be verified or distinct sheets remain plausible,
stop pointwise transfer there rather than blend them. This limited audit does
not admit all w040, establish training power, or automatically authorize GPU use.

This result also changes the useful question: it is no longer merely whether
these acquisitions contain corresponding local structure. They demonstrably
do. The next issue is whether the remaining uncertainty is small enough for
the particular supervision and learning objective.

## Verification and retained evidence

- Source-derived geometry was reviewed independently;81scalar query checks
  passed, maximum residual 0.00773 native voxel.
- Known-shift, intensity-affine, periodic-depth and constant-support fixtures
  passed. Periodic structure retains multiple equivalent peaks.
- All 52,728 real NCC values were independently recomputed by direct spatial
  covariance, maximum difference 5.04e−13; all decisions agreed.
- Input byte counts, 75 chunk SHA-256 hashes and source/query receipts were checked.
- All 6 panels use fixed 0–255 displays and the same predetermined sections.
- Pre-CT corrections (support-local physical bases and 1e−6 voxel arithmetic
  tolerance at boundaries) retain their earlier code/request-plan versions.
  No earlier draft scored CT.

Start with `ct-study/PROTOCOL.md`, `ct-study/results/RESULTS.json`,
`geometry-review/CONTRACT.md`, `geometry-review/VISUAL_REVIEW.md` and
`method-review/score-check/REVIEW.md`. The portability guide specifies the
complete retained inputs, disposable derived arrays, checksum verification
and offline numerical replay. No credentials are included.
