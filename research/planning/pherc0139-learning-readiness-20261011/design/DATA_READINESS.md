# Matched-learning data readiness

11 October 2026. Read-only audit of retained metadata, published source code and retained mesh geometry. No CT intensities or additional teacher/annotation pixels were viewed; no network access, model inference, training, or Git mutation. The companion scripts reproduce the metadata/context arithmetic and the mesh-only separation check.

**The currently justified next step is the finite target-operator comparison, not fitting four patches.** Existing targets can compare numerical supervision operators on strict common support. They do not supply an independent ink-accuracy endpoint or a defensible generalization test. Freeze any pooled shift-kernel support before execution: applying one support's shifts elsewhere may leave the old per-support footprint even when it stays inside the stored bounding crop. Report support loss separately; never convert out-of-crop values to zero or silently select easier pixels.

## Retained observations and target access

All three previously selected w040 cores are 256 × 256 native pixels with an eight-pixel halo (272 × 272). Their centres are site1 (2432,1920), site2 (3968,3200), site3 (5632,4480). Each has 16 native chunks, totalling 7,340,032 bytes, and nine fine chunks, totalling 16,072,704 bytes. Total retained CT is 70,238,208 bytes. Native source has all 28 planes; fine source has all 109 planes.

The fine CT is **pyramid level2: 9.596 µm XY, 2.399 µm Z**, not native 2.399 µm XY high-resolution input. Native is 9.362 µm XYZ. A comparison using these arrays tests these retained representations and acquisition differences; it is not a native full-resolution high-versus-low-resolution comparison.

Only site1's four 32-square checking supports have been analysed against teacher scores. The saved crop is full-fine x[9084,9872), y[7088,7868), shape780 ×788; its bounding rectangle contains unanalysed pixels. Storage or earlier decoding of the enclosing rectangle does not make these pixels an admitted experimental region. Four saved Deflate TIFF tiles (indices194,195,225,226) cover x[8192,10240), y[6144,8192), total428,618 compressed bytes. They could supply the whole retained site1 geometry footprint without new download, but enlarging the analysis still requires its own frozen scope.

The teacher is a quantized model score on the declared fine full canvas, aggregating fine Z[23,85). Its output is a two-dimensional projection without an identified ink-depth plane. Its historical complete-image hash and full teacher training ancestry are unavailable. It is not an independently authenticated label.

Existing inverse queries cover **native j=4,...,23 only**, with j10,...,17 used in the previous structural/target check. All28 native planes are available for a future native-only learner. A supposedly matched all28-plane fine-derived learner would need new geometry-only queries and filter-boundary qualification. In particular, the retained five-plane fine Gaussian radius may exceed source depth near the extreme native planes. No boundary padding or silent depth truncation should be introduced to manufacture a match.

## Teacher context prevents a casual four-support split

The pinned v42 inference code uses global tile256/stride128 grids and overlap-add predictions. One tile is approximately65.8 native pixels wide. A typical pixel can inherit a384-fine-pixel union of input context (98.7 native pixels), before grouping the target support. Conservative bounding rectangles of all potentially contributing teacher tiles over the frozen j10:17 alternatives are:

| Support | Native output box, half open | Fine teacher CT input bound, half open |
|---|---|---|
|4|x2416:2448,y1824:1856|x9216:9728,y6912:7424|
|5|x2336:2368,y1904:1936|x8960:9472,y7168:7808|
|6|x2496:2528,y1904:1936|x9600:10112,y7168:7808|
|7|x2416:2448,y1984:2016|x9216:9728,y7552:8064|

|Pair|Shared possible inference tiles|Overlapping fine CT input pixels in conservative boxes|
|---|---:|---:|
|4,5|1: start(9216,7168)|65,536|
|4,6|0|32,768|
|4,7|0|0|
|5,6|0|0|
|5,7|1: start(9216,7552)|65,536|
|6,7|0|32,768|

These are conservative context bounds from output coordinate footprints and source code, not proof of every model neuron's receptive field or every historical mask-admitted tile. They deliberately retain possible overlap. Disjoint tile IDs alone do not establish disjoint input CT. Only opposite support pairs4/7 and5/6 clear this bounded context test; all supports are already inspected, and selecting5/6 because7 lacks positives would be post-result development selection. Their small size and shared winding do not provide regional replication. Do not fit these four patches and report the remaining one as an untouched test.

## Prospective physical split and minimal data expansion

The three native 272-square domains are geometrically separate. Using only the retained native mesh, all voxel centres along their full28-plane rays lie in disjoint XYZ boxes; boxes expanded by0.5 voxel have separation lower bounds1461.93 native voxels for site1/site2,3144.44 for site1/site3,1644.44 for site2/site3. These are about13.69mm,29.44mm,15.40mm. This establishes separation of the sampled native coordinate domains, not local fine/native registration or teacher-history independence. The sites remain previously inspected same-winding development tissue.

Once an evaluation endpoint has separately qualified, a feasible frozen acquisition recipe is:

1. Assign **whole site1 to training, site2 to validation, site3 to final prospective scoring**, before new teacher/manual pixel access. Keep every alternate rendering, depth, patch, and repeated sample from a site in that site's group. These roles hold out future target access/training, not prior CT inspection or upstream model exposure.
2. Use a fixed central128-square score/target block per site and its enclosing256-square native CT context, keeping the retained eight-pixel halo for bounded transfers. Centre offsets and geometry margins are chosen before labels. A finite-field student must have output dependencies contained in that declared context; no global canvas normalization using evaluation regions. Do not tune score blocks after discovering sparse annotation coverage or weak positives; report endpoint inadequacy and stop that comparison.
3. Freeze student input acquisition/depth, receptive field, target resampler, loss support, finite geometric alternatives and all arm-independent split IDs. A native-only supervised-target comparison can retain all28native channels now. A fine/native acquisition comparison additionally needs matched physical sampling/depth support; current queries do not qualify all28.
4. Require common geometric/array/support coverage for every arm and target alternative. Fit normalization only on training. Score identical prospective pixels against the independent endpoint; include separate foreground/negative support and per-site results. One validation and one test block cannot sustain a frontier/generalization claim even if useful as a small pilot.

For a smaller two-block preliminary comparison, site1 versus site2 suffices to avoid shared sampled tissue/context, but any tuning consumes site2 as validation; it is then not a final independent test. The three-block version avoids inventing a test by repeatedly reusing validation. Neither is currently admitted for ink-accuracy training/evaluation because its endpoint and prior CT failures remain unresolved.

Current TIFF header permits precise **compressed-byte estimates only**, with no new download:

| Retained geometry domain | Conservative full-fine teacher bound | TIFF tiles | Additional compressed bytes |
|---|---|---|---:|
|site1|x8952:10020,y6960:8028|194,195,225,226|0|
|site2|x14932:15996,y11936:13000|355,356,386,387|584,743|
|site3|x21404:22468,y16912:17976|516,517,547,548|361,019|

Thus both new teacher domains need at most945,762 current-object compressed bytes for these bounds, with **zero new CT bytes for the original retained domains**. These are x4 expansions of original finelevel2 source bounds, not frozen authorisation for new shifted training footprints. Before access, compute exact transfer/context bounds, add operator halo, enforce sourcecanvas/ETag/version/range checks, and recalculate tile sets. Existing TIFF ETag/body length remain version validators, not historical whole-object hashes. Manual-label compressed bytes cannot be stated from the retained array metadata alone.

## What could evaluate the comparison fairly

There is **no already-qualified independent ink reference in this packet**. A common fixed teacher target at the CT-fit transform is arm-independent as software input, but it shares the training teacher and privileging its point transform can favour the fit-target arm. It measures reference agreement, not ink accuracy. The finite CT-alternative ensemble can measure sensitivity/robustness, but weighting its candidates uniformly does not turn them into a calibrated truth distribution. A method choosing that same mean target would have a tautological advantage if evaluated only against that mean.

Current sparse manual native annotations are a plausible **separately qualified** endpoint: retain them entirely from all training targets/selection, pin the current bytes and declared native frame, preselect evaluation regions, measure current support, and audit physical mapping provenance. Their metadata reports745,657 ink pixels, whereas the pinned historical native-dense report says745,601. Their unavailable transfer reports and absent pixelwise mapping-validity mask preclude treating them as exact historical overrides or proven aligned truth. We have fetched no manual pixels. A current annotation endpoint could support carefully qualified annotation agreement; independent physical ink accuracy would still require stronger adjudication/provenance. An alternate scan/teacher may add evidence after qualification but remains model output, with exposure overlap to investigate.

All original local CT correspondence gates remain FAIL. Site1's practical teacher uncertainty measurement did not pass those gates or establish a correct physical ink face. Sites2/3 have larger local-fit inconsistency, so their convenient availability must not admit precise transferred supervision by default. The immediate A/B/C operator analysis may establish whether CT-specific alternatives differ from ordinary softening/smoothing on the already declared supports. It cannot itself supply the missing independent endpoint or justify a frontier architecture/GPU pilot.

Reproduction files: `audit_readiness_metadata.py`, `DATA_READINESS_METADATA.json`, `audit_native_geometry_disjointness.py`, `NATIVE_GEOMETRY_DISJOINTNESS.json`. Sources: retained supervision `PROTOCOL.md`, `SOURCE_FOOTPRINT.json`, `teacher/REQUEST_LEDGER.json`, `provenance/REPORT.md`, `provenance/TEACHER_TIFF_HEADER.json`, pinned `v42-inference.py`; correspondence `ct-study/REQUEST_PLAN.json` and retained geometry/pullback implementation; project `paired-data/REVIEW.md`.
