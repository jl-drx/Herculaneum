# Frozen local CT correspondence comparison — PHerc0139 w040

11 October 2026. This protocol is fixed before reading any new CT pixels.
Scope: training winding `20250831000000-w040_2025083102`, fine78 volume
20260102150214 and native113 volume20250728140407. No model, target/ink map,
optical annotation, teacher inference, new training or GPU. Study52 is unchanged.

## Question and representation

Do three geometry-selected sites support a locally consistent structural
correspondence between acquisitions, after accounting for published surface
sampling? Site1 develops/checks implementation conventions; sites2–3 are fixed
checking sites. No replacement or parameter tuning on their image results.
They are development observations, not unexposed evaluation regions.

Native XY centres: (2432,1920), (3968,3200), (5632,4480). Each core is256square,
with8pixel XY halo (272square). These approximate the30%,50%,70% diagonal
locations of the native raster, rounded to128; selection uses no CT or labels.
Keep every site including failures. Geometry validity can stop a site before CT.

Use all retained fine109/native28 layers as source evidence. Registration
queries use native layers4 through23. Fine group2 is a twice-rounded XYmean
pyramid (9.596µm XY,2.399µm Z), not an isotropic coarse scan. Smooth fine in Z
only with a Gaussian sigma1.5 fine layers, radius5; no padding contributes to
any query. This is one declared structural surrogate, not an identified native
scanner PSF or clean isolation of resolution from energy/reconstruction.

The source-derived UV mapping composes mesh half-pixel resampling and renderer
coordinates. Ordinary full-raster resizing is displaced by approximately7.43
native pixels per axis. Each fine group2 pixel represents16 fine renderer
samples: average their positions and cached normals to define a representative
coordinate field. Interpolate these fields in group2 UV; apply the public
fine-to-native affine. Solve the geometry-only inverse for each native sample,
starting at the source UV map. The inverse matches representative locations,
not individual CT samples or PSFs. Numerical residual<=0.01native voxel is an
implementation screen, not a claim of physical registration accuracy.

Require conservative constituent position/normal support, finite nonsingular
Jacobian, retained local orientation, bounded Newton basin and steps, and
convergence. Preserve conditioning and residual distributions and all exclusions.
No CT-dependent warp. Freeze exact query files/hashes and source chunk list
before acquisition. Bound the source request body total to192MiB. No request
expansion or raw-volume fallback is authorized by this protocol. A geometry
failure is a documented stop for that support, with a separate future fallback
design if scientifically useful.

## Fit, checking, and controls

Within each256square core,32square fitting supports have centres(48,48),
(208,48),(48,208),(208,208). The checking supports have centres(128,48),
(48,128),(208,128),(128,208). All use native layers10:18 (8layers), centred
symmetrically about the mesh. The central support is reserved for fixed views.
Spatial supports and shifted/interpolated source footprints remain disjoint.

Search integer translation in Z,Y,X, each from−6 to+6 native sample steps.
Positive translation means native[z,y,x] is compared with remapped
fine[z+dz,y+dy,x+dx]. Baseline is the published geometry with zero residual
CT translation. Fit one translation per site by maximising the mean of the
four separately centred/normalised Pearson correlations. No local affine,
nonrigid or patch-specific correction. No subvoxel fit in this first check.

Retain all2197 scores per support, the fit landscape, zero-shift/fit/check
scores, best independently checking translations, boundaries, and strongest
competing normal positions. Checking optima are diagnostics, never applied as
new corrections. Report shift distances in native voxel units and micrometres
using the measured local native renderer Jacobian; canvas steps are not assumed
to be an exactly orthonormal physical frame.

Before reading CT, fixtures must recover known integer shifts under positive
scale/offset changes, preserve ambiguity for periodic depth structure, and reject
constant support. They establish implementation sensitivity only.

## Prospective operational screens and decisions

A strict structural-consistency screen requires complete support, a non-boundary
fit and checking optimum, checking NCC at the fit>=0.5, and at least0.05 margin
over the best candidate outside a2native-voxel physical neighbourhood of that
fit. Check-vs-fit tangential discrepancy must have median<=0.5 and maximum<=1
native voxel; normal discrepancy<=1. All four checks must pass; report each
criterion rather than hide failures in a single average. Integer-grid rounding
limits the precision of these screens. They are conservative operational
criteria, not calibrated physical truth or a power calculation.

A passing core is provisionally consistent at its checked supports, not certified
everywhere. Fixed whole-depth cross-sections and competing normal positions
must also expose a traceable lamina and a nonrepeating feature before any
same-sheet claim. Absence of clear discriminators means unknown; correlation
alone never certifies sheet identity. Distinct adjacent wraps cannot be averaged
into a teacher target.

If correspondence is clear, develop a matched-tissue real-native versus
synthetically degraded fine comparison with honest supervision ancestry.
If same sheet is plausible but residual uncertainty is material, record that
uncertainty before opening targets and measure whether it changes transferred
supervision. If ambiguous or failed, stop transfer on these fixed supports.
Do not replace sites, choose a prettier projection or tune filters to rescue them.
The next decision must distinguish an inadequate representation/support from
proof that paired learning cannot work. None of these CT screens is an ink
accuracy score or a reason to revive a circular tiny-CPU-AP gate for GPU work.

## Pre-acquisition geometry clarification

Normal directions vary substantially over these curved cores. Distinct-peak
neighbourhoods and checking discrepancies therefore use the independently
measured basis at each support centre, retained in LOCAL_BASES.json. The core
basis is used only to report the fit displacement at the core centre. The fit
is a single shift in surface-raster/layer coordinates, not one rigid world-XYZ
translation. All scores and thresholds above remain fixed. The earlier
core-basis scoring draft is preserved; it never scored CT.

Independent pre-CT review found FP32 unit-normal norm errors near1e-8.
Physical distance boundary tests use a1e-6 native-voxel arithmetic tolerance:
competitors are strictly farther than2+1e-6; discrepancy limits permit limit+1e-6.
This avoids classifying exactly two depth steps as farther than two because of
roundoff. It changes no biological/registration threshold. The pre-tolerance
code and request-plan version are preserved and were never used to score CT.
