# Eight-site PHerc0139 CT correspondence result

11 October 2026. **The broader comparison confirms strong local structural
correspondence, while qualifying zero sites for strict pointwise transfer.**
Six of eight fixed sites were scorable; all 24 separate checking correlations
are 0.828–0.973. Two sites fail the conservative source-support screen. All
sites and the four preceding geometry failures are retained without replacement.

This extends the earlier w040 observation to w041 and w043. It is useful
evidence for paired-acquisition learning with explicit uncertainty, not a
reason to treat the scans as exactly registered or to declare the idea failed.

## Fixed comparison

The protocol and code were pushed in commit `9d18d0e` before acquiring CT.
Lock SHA256: `a88804be567164ed6fc69fdd82658fea30f4610581643cfe7af7c28518765c22`.
All 72 conditional GETs matched their pinned ETags and lengths, totaling
86,114,304 bytes (82.125 MiB). All28 native and all109 fine planes are retained.
Fine group2 pools XY only; its Z sampling remains 2.399 µm. No new targets,
model predictions, inference, training or paid resources were used.

Each 128-square core uses four 24-square fitting patches and four separate
checking patches. All search/interpolation footprints are disjoint in sampled
fine group2 pixels. The search and thresholds match the earlier study, but
the smaller patches are not a power-matched replication. Physical texture
dependence, common specimen and geometry-based selection limit generalization.

| Site | Fitted Z,Y,X | Fit NCC: geometry → CT fit | Checking NCC range | Frozen result |
|---|---|---|---|---|
| w040-x2368-y5440 | Unscored | — | — | Incomplete nonzero source support |
| w040-x6720-y4672 | -1,+1,+1 | 0.815 → 0.862 | 0.877–0.945 | Margin and lateral consistency fail |
| w041-x7488-y3136 | 0,-1,0 | 0.929 → 0.929 | 0.882–0.919 | Median lateral consistency fails |
| w041-x4672-y576 | 0,0,0 | 0.934 → 0.934 | 0.843–0.973 | Margin and lateral consistency fail |
| w041-x320-y576 | +2,-1,0 | 0.567 → 0.924 | 0.862–0.920 | Margin, lateral and normal consistency fail |
| w043-x7488-y2880 | Unscored | — | — | Incomplete nonzero source support |
| w043-x4928-y5440 | 0,+3,+2 | 0.687 → 0.910 | 0.828–0.927 | Margin, lateral and normal consistency fail |
| w043-x3648-y2624 | +2,+1,0 | 0.391 → 0.903 | 0.863–0.945 | Margin and maximum lateral consistency fail |

All six fitted optima and all checking optima are inside the search boundary.
The fitted correlation exceeds the strongest competitor displaced more than
two local normal voxels at every checking patch, by 0.137–0.643 NCC. This argues
against a similarly strong distant-normal alternative *within this finite
search and representation*. It does not identify the ink face, rule out every
neighboring sheet, or establish a calibrated registration confidence interval.

The closest case, **w041-x7488-y3136**, passes every screen except median
lateral agreement. Three checking patches prefer zero shift, while the fit
prefers Y=-1; the mean fitting advantage is only **0.000300 NCC**. Checking
tangential differences are 0–0.990 native voxel. The strict 0.5-voxel median
criterion fails, but this is not evidence of a grossly wrong physical pairing.
Integer-grid optima and local geometry make borderline distances sensitive to
discretization. We preserve the failure rather than certify subpixel precision.

## Source support and visual assessment

All 3,317,760 geometry queries are retained: 93 fail in unused outer context,
and every scored query passes. The independent scalar check passes 216 fixed
queries. Complete finite-search footprints remain within the prior request.

At w040-x2368-y5440, three native template voxels are zero and 16 fine search
queries touch zero-valued interpolation/Gaussian support. At w043-x7488-y2880,
12 such fine queries fail in one checking search. The protocol requires every
contributor to be nonzero, so neither site was scored. This is a conservative
operational exclusion, not evidence the material is unavailable or contains no
ink. Positive pooled source values also cannot certify every underlying fine
voxel was unmasked; no constituent mask was acquired.

All 16 fixed panels were inspected. Curved interfaces, void boundaries and
localized bright features recur across observations, including the two
unscored sites. Fine images resolve internal interfaces blurred in the native
scan. The [site-by-site visual review](VISUAL_REVIEW.md) records what is visible
and why no particular ink-bearing face is certified. No fitted contrast or
new display/region selection was used.

## Research decision

Close this bounded correspondence pilot without expanding the search or
moving to another set of crops. The recurring finding is now **corresponding
material with consequential local uncertainty**, not lack of acquisition
overlap. More near-identical registration screens would have diminishing value.

The unchanged native-learning study remains the main learning result to obtain:
it can distinguish fitting capacity from transfer on its declared native
annotation references. Do not open its held-out predictions early. This CT
pilot does not replace that experiment or establish a model improvement.

For paired learning, the next preparation should be one frozen supervision
sensitivity/coverage audit on the six scorable sites, using their saved CT
landscapes and the source-qualified v42 teacher. Compare a single-fit soft
target against plausible-transfer alternatives and a matched smoothing control;
check available native annotation coverage and keep teacher-only regions
distinct. Fix the ambiguity set and tolerances before opening target values.
The earlier four-patch w040 study supplies the method; this extension would
test whether its conclusion generalizes to the broader training distribution,
not claim a new algorithm. No additional CT acquisition is needed for that audit.

If target uncertainty remains material, the learning study must represent or
exclude it explicitly; if stable, use the simple soft-target baseline. Neither
outcome retroactively passes these strict CT gates, proves the ink face, or
admits all102 geometry candidates. A trained paired learner still needs adequate
coverage, matched inputs/capacity, honest teacher ancestry and native reference
evaluation. GPU admission remains separate; no new universal tiny-CPU-AP gate
is introduced.

## Verification and portability

235 pre-acquisition checks include known shifts, intensity-affine changes,
periodic-depth ambiguity, constant/nonfinite rejection, source-mask propagation,
geometry inversion and independently marked disjoint footprints. All **105,456
real correlations** were recomputed using direct spatial covariance, with maximum
error 1.10e-12. Scalar physical-distance calculations reproduce every decision.
Every assembled source array matches raw chunks; 216 sampled filtered/interpolated
CT values were independently checked. These are different implementations in
the same runtime, not an independent team or physical ground-truth assessment.

The private archive contains the complete selected source coordinates and
metadata, code, protocol/lock, raw CT chunks, query fields, results and panels.
See [PORTABILITY.md](PORTABILITY.md). The previous intake is unchanged. This
supporting diagnostic does not advance the formal experiment seal beyond50.
