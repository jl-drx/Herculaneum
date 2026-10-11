# Independent design review: bounded w040 correspondence qualification

11 October 2026. This is a prospective methodological review, not a frozen acquisition plan or a claim that w040 is registered. It uses existing reports and the current metadata-only catalog extract. It fetched no network object, viewed no new CT or labels, called no model, and did not inspect or change study52.

**Recommendation:** proceed with the bounded CT-only check, but first verify the common surface parameterization. The efficient first route is the two existing surface volumes, with their complete depth and explicit sampling geometry. If their canvas/depth mapping cannot be recovered, a small pair of raw-volume blocks is the correct fallback. An unconstrained 2D registration of their projections would hide precisely the normal/sheet ambiguity the experiment needs to test.

## What the earlier failure does and does not teach us

Study42 fixed six Paris4 sites and tested a published affine using four disjoint fitting and four checking structural supports. Four analyzable sites failed the checking-shift screen; two did not have the required acquired fine-volume context. Even the site with fit NCC 0.649 had inconsistent checking peaks. Those are differences between correlation maxima, not independently measured physical registration errors. The experiment demonstrated ambiguous local structural matching and an acquisition-placement limitation; it did not demonstrate that paired-acquisition learning is impossible.

The useful parts to retain are fixed geometry-selected locations, separate fit/check support, preserved score landscapes, explicit depth alternatives, and refusal to convert high correlation into ink truth. Improve the spatial support: the old 6-cubed checking cubes were very small and can be dominated by a repeated lamina or a low-information patch. Compute the exact transformed interpolation and filtering footprint before acquiring, so missing context is a deliberate exclusion rather than a surprise halfway through analysis. Do not simply repeat the previous six-site all-or-nothing gate on another sample.

## Surface canvases: promising common ancestry, unverified sample correspondence

The new w040 extract establishes that both transformed meshes derive from the same `mesh/intermediate/tifxyz_normalized/` source. The fine mesh is produced by a 4I transform of that source. The native mesh is produced by the supplied affine on the normalized coordinates. Their mesh metadata both report scale 0.05. Both render requests use `flip=-1`, `flip-normals=true`, and no crop or rotation. This makes a common parameterization plausible and testable.

It is not yet proven. The geometry transforms and renderers have different pinned revisions: fine rendering uses Villa `650bd56663efb7225fdba73fe267ba86b8e29368`; native rendering uses `530584554c031c982ba3aa243e1ea39c4bef9818`. A shared source filename, matching array proportions, root translations of zero, or dividing the widths does not verify the pixel map. Read the exact coordinate TIFFs and renderer conventions at those revisions. In particular:

- Compare the native XYZ mesh against the published affine applied to the fine XYZ mesh at corresponding **source parameter coordinates**. Preserve mesh invalid points and interpolation support. Report the actual residual distribution and any index/order reversal.
- Recover how normalized mesh coordinates become each output canvas, including output size rounding, half-pixel conventions, original scale, axis order, flips, and fine-level2 pooling. Do not assume pixel j at pyramid2 represents fine pixel 4j rather than the center of its pooled footprint.
- Recover layer zero and depth direction. Fine has 109 layers; native has 28. Even versus odd length matters, and nominal layer middles are not evidence of an anatomical surface.
- Compare the *sample locations off the mesh*, not just the mesh itself. An affine transforms positions with A and normals with the inverse transpose; rendering separately from transformed meshes need not reproduce transformed normal rays under non-similarity transforms. Quantify this with the actual local Jacobian/normal conventions before using a separable XY resize and Z resample.
- Fine level2 is 9.596 micrometres in XY but 2.399 in Z. Native is 9.362 in XYZ. A full-depth level2 cube is an inexpensive registration representation, not an isotropic coarse input or a calibrated scanner simulation.

A near-zero mesh residual establishes agreement with the **publisher's transform**, not physical registration between acquisitions. The published fine/native transform already has a median 2.05-native-voxel residual on its own fit landmarks. That median is not a local normal offset and cannot be subtracted from every surface.

**Engineering exit:** once a single explicit pullback maps the frozen native output sample coordinates to the fine surface frame and can be replayed against coordinate geometry, proceed. If a surface-derived sample is more than 0.1 native voxel from its intended mapped 3D point, or falls outside reconstructible support, mark it unknown. This 0.1 threshold is a proposed numerical implementation tolerance, not physical registration accuracy; freeze or replace it before pixels are scored, based on the actual renderer precision. If the map cannot be recovered, stop this route rather than fitting image-specific warps to conceal the discrepancy.

## A minimum finite comparison

Prefer **three** separated, geometry-selected 256-square native-scale cores if the declared object budget permits: one convention-development site and two reserved checking sites. Choose by mesh coverage, nondegenerate local tangent frame and adequate context alone, before CT/teacher viewing. Keep all chosen sites and record failure or unknown support without replacement. Two sites are still useful if the budget only supports one development and one check; four are not necessary merely to inflate the sample count.

Acquire each core's exact complete physical context, including all available depths and the mapped filter/search halos. A 32-pixel tangential halo is a plausible bound, but the actual footprint must be calculated from the pullback. Store the request list before fetching pixels. Native 28-plane depth is the tight dimension: do not invent additional depth by padding, edge replication or treating an invalid voxel as zero-valued tissue.

Use one fixed fine-to-native structural representation: fine XY pyramid2 with an explicitly recorded Z low-pass and physical resampling onto the native frame. Choose the filter once from the sampling ratio, not whichever filter gives the highest correlation. Keep the full unfiltered saved inputs for cross-section review. This is a specified structural surrogate, not an identified native point-spread function; cross-energy/reconstruction differences may lower correlation even for a correct pair.

For a tractable local check, use eight separated 32-square tangential supports within each core, centered at coordinates in {48,128,208}. Fit on the four corner positions (48,48), (208,48), (48,208), (208,208); check on the four edge-midpoint positions (128,48), (48,128), (208,128), (128,208). Reserve the central support for a geometry review. These coordinates are a suggested design to freeze before acquisition, not authority to overwrite a different frozen plan. The gap leaves ample room for bounded filters and shifts; preserve their exact nonoverlap after mapping and interpolation.

Retain at least seven physically spaced depth samples about the geometry-defined zero for each structural support. Search a single translation within a fixed native-coordinate box, for example tangential plus/minus 6 voxels and normal plus/minus 6 voxels, only if the full sample/filter footprint fits the 28-layer acquisition. Obtain one fit translation from the four fitting supports jointly. No local affine, elastic warp, depth-specific filter or intensity sign flip. Report the zero-shift score, fit score, full score landscape, boundary condition, and all checking results. A bounded quadratic subvoxel refinement may be used if fixed beforehand and tested on synthetic shifts.

On each checking support, compare the fit translation with that support's independent optimum, while retaining the entire landscape. Independent checking optima diagnose consistency; they do not become four new local corrections. Render the same fixed sagittal/tangential views at the public mapping, fitted mapping and best physically distinct competing normal candidate. Use one predetermined display convention and expose validity explicitly.

## A practical numerical screen and the separate sheet decision

Before real correspondence scoring, a small same-volume synthetic shift fixture should recover prescribed tangential and normal shifts, including fractional shifts if claimed, to at most 0.5 native voxel. Include wrong-sign normal and pixel-center fixtures. This tests implementation/sensitivity, not success on real acquisitions. Injected ambiguity from two repeated layers should remain ambiguous; a confidence measure that returns a unique confident answer in that fixture is not usable.

For each real checking support, a proposed strict structural screen is:

1. Complete mapped support and a non-boundary optimum; no constant/zero-variance correlation.
2. Checking NCC at the **fit** translation at least 0.5, rather than only NCC at that check's independently refitted optimum.
3. Margin at least 0.05 over a distinct competing optimum outside a 2-native-voxel physical neighborhood; retain all smaller margins as ambiguous. Explicitly inspect competitors separated mainly along the normal, since a tangentially nearby wrong sheet can be dangerous.
4. Checking optimum versus fit translation: tangential median at most 0.5 native voxel and maximum at most 1 voxel over the four checks; report normal discrepancies separately, requiring at most 1 voxel for a strict single-surface transfer claim.

These thresholds are **proposed operational screens**, inherited partly from study42 for comparability. They are not calibrated probabilities of correct sheet identity, not a power calculation, and not proof that every intervening pixel is aligned. Freeze them before real scores; do not tune them to rescue a favorable-looking crop. If the physical frame is non-orthogonal, compute residuals through the local mesh Jacobian and report micrometres, not raw canvas distances.

Do not pool one failure away by averaging over sites or millions of voxels. Report all sites/supports with the same rules. A failing fixed core is excluded as a core; alternatively a predeclared rule may retain only its actually checked structural neighborhoods, with their coverage reported, but this must not be called full-core qualification. A passing site does not certify all w040 or the benchmark regions.

Numerical success yields **locally consistent CT correspondence**. To use the stronger phrase **same-sheet correspondence**, the fixed cross-normal views must reveal a traceable lamina plus some nonrepeating discriminator (junction, tear, local termination, or related 3D structural feature) that follows that same lamina in both acquisitions. Inspect both sides and all competing plausible normal positions. A high-NCC feature that could be either of two nearby sheets is unknown, even if the best peak wins by 0.05. A CT-only agent review is evidence for a qualified structural interpretation, not a papyrological or chemical authentication.

No ink label is required to conduct this qualification. CT-dense structures are not automatically ink, zero intensity may be masking/air, and an apparent substrate match does not establish identical ink supervision. Teacher maps and sparse manual labels remain unopened until the CT choices and uncertainty mask are fixed. Any later training uses those sources with their actual pseudo/manual status and ancestry.

## When raw volumes are the better finite fallback

Use raw CT if the surface canvases cannot be put into a verified common frame, or if their finite normal range cannot display the competing sheets. Keep the same geometry-selected centers rather than searching for attractive replacements. For each center, specify a small coarse physical cube and transform **all eight footprint corners plus interpolation/filter context** to fine native coordinates using the public affine. Fetch only intersected native chunks, with a fixed total cap, then render both cubes in the same local tangent/normal frame. This removes the pre-rendered-canvas ambiguity while preserving the uncertain acquisition alignment.

A native raw cube of 32-cubed voxels already spans about 300 micrometres and may provide useful through-sheet context; a 48- or 64-cubed cube allows a larger fixed search and structural support. Its fine mapped footprint is roughly 4 times larger per dimension, so enumerate chunk costs before choosing the size. These raw cubes diagnose local 3D correspondence; they do not by themselves cover a 256-square training patch. If the budget cannot support the selected size, record a feasibility stop rather than silently cropping away context. Do not launch a surface route, raw route and expanding-search route all at once.

## Finite decisions after the check

- **Clear local correspondence, unique sheet:** admit only the checked physical support to development of the matched acquisition comparison. Keep the same target hierarchy, support and exposure between real-native and degraded-fine training arms. No uncertainty architecture is needed merely because the global affine had nonzero landmark residuals.
- **Unique sheet but material bounded residual uncertainty:** retain the uncertainty interval and, only after freezing CT choices, measure how much the transferred target would change over it. Compare against fixed/permuted smoothing and support-matched controls before crediting an uncertainty method.
- **Ambiguous sheet, unidentifiable frame or unsuccessful fixed checking supports:** stop pixelwise transfer on that support. Preserve failures. Use the one explicitly budgeted raw-volume fallback if it addresses the specific missing evidence; otherwise close the qualification as inconclusive. No endless crop replacement, depth sweep or teacher-agreement registration.

The output is an admission map and a reasoned next decision, not a new ink detector. Failure of a stringent alignment screen must not revive the old rule that tiny CPU fine-tuning has to gain AP before a powered GPU study is allowed. The useful subsequent question remains the matched-tissue acquisition comparison; its supervision and reserved physical evaluation must be specified honestly at their evidenced level.

## Sources inspected

- `/workspace/herculaneum-lab/reports/fortysecond-experiment/ALIGNMENT_PROTOCOL.md` and `RESULTS.md`.
- `/workspace/Herculaneum/research/planning/plinius-research-review-20261011/README.md`, `paired-data/REVIEW.md`, `metadata/RESULTS.md`, and `novelty/REVIEW.md`.
- `/workspace/herculaneum-lab/reports/pherc0139-correspondence-20261011/metadata/CATALOG_EXTRACT.json` and the two transformed-mesh metadata responses.

The root experiment author should resolve renderer/frame conventions and choose one feasible frozen design before acquiring new CT. This review deliberately does not authorize changing an already frozen plan or the active PHerc0800 experiment.
